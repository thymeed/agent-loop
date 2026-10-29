"""Read-only local dashboard for runs and evals.

    python -m agent_loop.dashboard          # prints the URL to open

It only reads files under runs/. It listens on 127.0.0.1 and nothing else, requires a random
per-launch token (kept in a cookie after the first visit), checks the Host header against
DNS rebinding, answers GET only, sends no CORS headers, and serves a page whose CSP blocks
inline scripts. Everything a model wrote is data in JSON and shown as text, never as HTML.
"""
import argparse
import hmac
import json
import os
import re
import secrets
import socket
import sys
import threading
import time
import webbrowser
from http import cookies
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlsplit

from . import evals, mailbox, runs

WEB_DIR = Path(__file__).resolve().parent / "web"
STATIC = {
    "/": ("index.html", "text/html; charset=utf-8"),
    "/app.js": ("app.js", "text/javascript; charset=utf-8"),
    "/logic.js": ("logic.js", "text/javascript; charset=utf-8"),
    "/app.css": ("app.css", "text/css; charset=utf-8"),
}
CSP = ("default-src 'none'; script-src 'self'; style-src 'self'; connect-src 'self'; "
       "img-src 'none'; base-uri 'none'; form-action 'none'; frame-ancestors 'none'")
COOKIE = "dash"
PAGE_EVENTS = 200            # events per /api/trace response
MAX_POST_BYTES = 4096        # the one write path takes a tiny JSON body
POST_INTERVAL = 1.0          # seconds between writes the server will even consider
STALE_AFTER = 15 * 60        # a run with no end and no activity this long is "stopped"
MAX_PATCH_BYTES = 500_000

_RUN_ID = re.compile(r"[A-Za-z0-9._-]+(?:/[A-Za-z0-9._-]+){0,3}")
_EVAL_FILE = re.compile(r"[A-Za-z0-9._-]+\.json")


def trace_path(root: Path, ident: str) -> Path | None:
    """runs/<id>/trace.jsonl for a plain id, or None. The id never leaves `root`."""
    if not _RUN_ID.fullmatch(ident or "") or ".." in ident:
        return None
    path = (root / ident / "trace.jsonl").resolve()
    if root.resolve() not in path.parents or not path.is_file():
        return None
    return path


def read_events(path: Path) -> list[dict]:
    """Complete, valid lines only: a line still being written is skipped until it is whole."""
    data = path.read_bytes()
    complete = data[: data.rfind(b"\n") + 1]
    events = []
    for raw in complete.split(b"\n"):
        if not raw.strip():
            continue
        try:
            events.append(json.loads(raw))
        except ValueError:
            continue
    return events


def summarize_run(ident: str, path: Path) -> dict:
    events = read_events(path)
    start = next((e for e in events if e["event"] == "run_start"), {})
    end = next((e for e in reversed(events) if e["event"] == "run_end"), None)
    failed = next((e for e in reversed(events) if e["event"] == "run_error"), None)
    last = events[-1] if events else {}
    open_calls = 0
    for e in events:
        open_calls += {"llm_start": 1, "llm_end": -1, "llm_error": -1}.get(e["event"], 0)
    age = time.time() - path.stat().st_mtime
    if end:
        status = "finished"
    elif failed:
        status = "error"
    elif last.get("event") == "pause":
        status = "paused" if age < STALE_AFTER else "stopped"
    else:
        status = "running" if age < STALE_AFTER else "stopped"
    return {
        "id": ident, "status": status, "result": end.get("result") if end else None,
        "kind": start.get("kind"), "task": (start.get("task") or "")[:200],
        "models": start.get("models"), "started": events[0]["ts"] if events else None,
        "updated": path.stat().st_mtime, "events": len(events),
        "calls": sum(1 for e in events if e["event"] == "llm_end"),
        "thinking": open_calls > 0 and status == "running",
    }


class _Cache:
    """Summaries keyed by (mtime, size) so unchanged runs are not re-parsed on every poll."""

    def __init__(self):
        self._items: dict[Path, tuple[tuple[int, int], dict]] = {}

    def get(self, ident: str, path: Path) -> dict:
        stat = path.stat()
        key = (stat.st_mtime_ns, stat.st_size)
        hit = self._items.get(path)
        if hit and hit[0] == key and hit[1]["status"] in ("finished", "error"):
            return hit[1]
        value = summarize_run(ident, path)
        self._items[path] = (key, value)
        return value


def list_runs(root: Path, cache: _Cache) -> list[dict]:
    found = sorted([*root.glob("*/trace.jsonl"), *root.glob("evals/*/*/trace.jsonl")],
                   key=lambda p: p.stat().st_mtime, reverse=True)[:200]
    return [cache.get(p.parent.relative_to(root).as_posix(), p) for p in found]


def list_eval_reports(root: Path) -> list[dict]:
    out = []
    for path in sorted((root / "evals").glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True):
        try:
            report = json.loads(path.read_text(encoding="utf-8"))
        except ValueError:
            continue
        results = report.get("results", [])
        overall = evals.summarize(results)[-1] if results else {}
        out.append({"name": path.name, "label": report.get("label"), "started": report.get("started"),
                    "finished": report.get("finished"), "attempts": report.get("attempts"),
                    "done": len(results), "passes": overall.get("passes", 0), "n": overall.get("n", 0),
                    "rate": overall.get("rate", 0.0), "mean_tokens": overall.get("mean_tokens"),
                    "mean_seconds": overall.get("mean_seconds")})
    return out


def fence_for(text: str) -> str:
    """A backtick fence longer than any backtick run in `text`, so the text cannot close it early."""
    longest = max((len(run) for run in re.findall(r"`+", text)), default=0)
    return "`" * max(3, longest + 1)


def _block(text, lang: str = "") -> str:
    text = str(text)
    fence = fence_for(text)
    return f"{fence}{lang}\n{text}\n{fence}\n"


def export_filename(ident: str) -> str:
    name = re.sub(r"\.{2,}", ".", re.sub(r"[^A-Za-z0-9._-]+", "_", ident))
    return (name[:80].strip("._") or "run") + ".md"


def render_report(ident: str, events: list[dict], patch: str | None) -> str:
    """A run as Markdown. Everything a model wrote sits inside a code fence the text cannot
    close, so the file is safe to open in any viewer and carries no active content."""
    lines = [f"# Run {export_filename(ident)[:-3]}", ""]
    start = next((e for e in events if e.get("event") == "run_start"), {})
    end = next((e for e in reversed(events) if e.get("event") == "run_end"), None)
    lines += [f"- kind: {start.get('kind', '?')}", f"- result: {end.get('result') if end else 'not finished'}"]
    if end:
        lines += [f"- rejections: {end.get('rejections')}", f"- pauses: {end.get('pauses')}"]
    lines += ["", "## Task", "", _block(start.get("task", "")), "## Models", "",
              _block(json.dumps(start.get("models", {}), indent=2), "json"), "## Timeline", ""]
    for number, e in enumerate((e for e in events if e.get("event") in
                                {"llm_start", "llm_end", "llm_error", "node", "pause", "owner_call", "run_error"}), 1):
        kind = e["event"]
        if kind == "llm_start":
            lines += [f"### {number}. {e.get('role')} asked ({e.get('model')})", "", _block(e.get("prompt", ""))]
        elif kind == "llm_end":
            tokens = (e.get("tokens") or {}).get("total_tokens", 0)
            lines += [f"### {number}. {e.get('role')} replied ({e.get('seconds')}s, {tokens} tokens)", "", _block(e.get("reply", ""))]
        elif kind == "llm_error":
            lines += [f"### {number}. {e.get('role')} failed", "", _block(e.get("error", ""))]
        elif kind == "node":
            lines += [f"### {number}. node {e.get('node')} ({e.get('seconds')}s)", "",
                      _block(json.dumps(e.get("update", {}), indent=2, ensure_ascii=False), "json")]
        elif kind == "pause":
            lines += [f"### {number}. paused for the owner", "",
                      _block(json.dumps(e.get("info", {}), indent=2, ensure_ascii=False), "json")]
        elif kind == "owner_call":
            lines += [f"### {number}. owner replied ({e.get('source', 'terminal')})", "", _block(e.get("text", ""))]
        else:
            lines += [f"### {number}. run crashed", "", _block(e.get("error", ""))]
    if patch:
        lines += ["## Patch", "", _block(patch, "diff")]
    return "\n".join(lines)


def load_eval_report(root: Path, name: str) -> dict | None:
    if not _EVAL_FILE.fullmatch(name or "") or name not in {r["name"] for r in list_eval_reports(root)}:
        return None
    return json.loads((root / "evals" / name).read_text(encoding="utf-8"))


class Dashboard(ThreadingHTTPServer):
    daemon_threads = True
    allow_reuse_address = False  # on Windows SO_REUSEADDR lets another local program bind the same port

    def __init__(self, root: Path, port: int = 0, token: str | None = None):
        self.root = Path(root)
        self.token = token or secrets.token_urlsafe(24)
        self.cache = _Cache()
        self.csrf = secrets.token_urlsafe(24)  # echoed by the page on writes; a cross-site page cannot read it
        self._post_lock = threading.Lock()
        self._last_post = 0.0
        super().__init__(("127.0.0.1", port), _Handler)  # loopback only; there is no option to change it

    def allow_post(self) -> bool:
        """At most one write attempt per POST_INTERVAL, however many clients ask."""
        with self._post_lock:
            now = time.monotonic()
            if now - self._last_post < POST_INTERVAL:
                return False
            self._last_post = now
            return True

    def server_bind(self):
        if os.name == "nt":
            self.socket.setsockopt(socket.SOL_SOCKET, socket.SO_EXCLUSIVEADDRUSE, 1)
        super().server_bind()

    @property
    def port(self) -> int:
        return self.server_address[1]


class _Handler(BaseHTTPRequestHandler):
    server: Dashboard
    server_version = "agent-loop-dashboard"
    sys_version = ""

    def log_message(self, *args):  # request lines can carry the login token; log nothing
        pass

    # --- plumbing ---------------------------------------------------------------------
    def _send(self, status: int, body: bytes, content_type: str, extra: dict | None = None) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("Content-Security-Policy", CSP)
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("X-Frame-Options", "DENY")
        self.send_header("Referrer-Policy", "no-referrer")
        self.send_header("Cross-Origin-Resource-Policy", "same-origin")
        for key, value in (extra or {}).items():
            self.send_header(key, value)
        self.end_headers()
        self.wfile.write(body)

    def _json(self, obj, status: int = 200) -> None:
        self._send(status, json.dumps(obj, ensure_ascii=False).encode("utf-8"), "application/json; charset=utf-8")

    def _deny(self, status: int, message: str = "forbidden") -> None:
        self._send(status, message.encode("ascii"), "text/plain; charset=utf-8")

    def _host_ok(self) -> bool:
        return self.headers.get("Host", "") in (f"127.0.0.1:{self.server.port}", f"localhost:{self.server.port}")

    def _same(self, given: str) -> bool:
        return hmac.compare_digest(given.encode("utf-8"), self.server.token.encode("utf-8"))

    def _authed(self) -> bool:
        jar = cookies.SimpleCookie()
        try:
            jar.load(self.headers.get("Cookie", ""))
        except cookies.CookieError:
            return False
        return COOKIE in jar and self._same(jar[COOKIE].value)

    # --- routing ----------------------------------------------------------------------
    def do_GET(self):
        if not self._host_ok():
            return self._deny(403)
        url = urlsplit(self.path)
        query = parse_qs(url.query)
        if not self._authed():
            if url.path == "/" and self._same(query.get("t", [""])[0]):
                return self._send(302, b"", "text/plain", {
                    "Location": "/", "Set-Cookie": f"{COOKIE}={self.server.token}; HttpOnly; SameSite=Strict; Path=/"})
            return self._deny(403, "open the address printed in the terminal where the dashboard started")
        try:
            self._route(url.path, query)
        except Exception:  # never echo paths or internals to the client
            self._json({"error": "internal error"}, 500)

    def _method_not_allowed(self):
        self._send(405, b"read-only", "text/plain; charset=utf-8", {"Allow": "GET"})

    do_PUT = do_DELETE = do_PATCH = do_OPTIONS = do_HEAD = _method_not_allowed

    def _origin_ok(self) -> bool:
        port = self.server.port
        return self.headers.get("Origin", "") in (f"http://127.0.0.1:{port}", f"http://localhost:{port}")

    def do_POST(self):
        """The only write: leave an owner reply for a run that is waiting at a pause.

        Everything must hold before anything is written: login cookie, Host, exact Origin,
        the per-launch CSRF header, a JSON body of limited size, a run that is paused right now
        with this nonce, and an acceptable reply text. The server then writes one small file
        inside that run's own folder and nothing else."""
        if urlsplit(self.path).path != "/api/owner_reply":
            return self._method_not_allowed()
        if not self._host_ok() or not self._authed() or not self._origin_ok():
            return self._deny(403)
        if not hmac.compare_digest(self.headers.get("X-CSRF", "").encode("utf-8"), self.server.csrf.encode("utf-8")):
            return self._deny(403)
        if not self.headers.get("Content-Type", "").lower().startswith("application/json"):
            return self._json({"error": "send JSON"}, 415)
        try:
            length = int(self.headers.get("Content-Length", ""))
        except ValueError:
            return self._json({"error": "length required"}, 411)
        if not 0 < length <= MAX_POST_BYTES:
            return self._json({"error": "bad size"}, 413)
        if not self.server.allow_post():
            return self._json({"error": "slow down"}, 429)
        try:
            data = json.loads(self.rfile.read(length))
        except ValueError:
            return self._json({"error": "bad request"}, 400)
        if not isinstance(data, dict) or not all(isinstance(data.get(k), str) for k in ("id", "nonce", "text")):
            return self._json({"error": "bad request"}, 400)
        found = trace_path(self.server.root, data["id"])
        if found is None:
            return self._json({"error": "not found"}, 404)
        events = read_events(found)
        pauses = [i for i, e in enumerate(events) if e.get("event") == "pause"]
        waiting = bool(pauses) and events[pauses[-1]].get("nonce") == data["nonce"] and not any(
            e.get("event") in ("owner_call", "run_end", "run_error") for e in events[pauses[-1] + 1:])
        if not waiting:
            return self._json({"error": "that run is not waiting for this answer"}, 409)
        try:
            mailbox.write_reply(found.parent, data["nonce"], data["text"])
        except ValueError:
            return self._json({"error": "not an acceptable reply"}, 400)
        except FileExistsError:
            return self._json({"error": "an answer is already waiting"}, 409)
        except OSError:
            return self._json({"error": "could not save the reply"}, 500)
        self._json({"ok": True})

    def _route(self, path: str, query: dict) -> None:
        def arg(name: str) -> str:
            return query.get(name, [""])[0]

        root = self.server.root
        if path in STATIC:
            name, ctype = STATIC[path]
            return self._send(200, (WEB_DIR / name).read_bytes(), ctype)
        if path == "/api/runs":
            return self._json({"runs": list_runs(root, self.server.cache), "now": time.time()})
        if path == "/api/trace":
            found = trace_path(root, arg("id"))
            after = arg("after") or "0"
            if not after.isdigit():
                return self._json({"error": "bad request"}, 400)
            if found is None:
                return self._json({"error": "not found"}, 404)
            events = read_events(found)
            start = int(after)
            return self._json({"events": events[start:start + PAGE_EVENTS], "next": min(start + PAGE_EVENTS, len(events)),
                               "total": len(events)})
        if path == "/api/patch":
            found = trace_path(root, arg("id"))
            patch = found.parent / "changes.patch" if found else None
            text = None
            if patch and patch.is_file() and patch.stat().st_size <= MAX_PATCH_BYTES:
                text = patch.read_bytes().decode("utf-8", errors="replace")
            return self._json({"patch": text})
        if path == "/api/session":
            return self._json({"csrf": self.server.csrf})
        if path == "/api/export":
            found = trace_path(root, arg("id"))
            if found is None:
                return self._json({"error": "not found"}, 404)
            patch_file = found.parent / "changes.patch"
            patch = patch_file.read_bytes().decode("utf-8", errors="replace") if (
                patch_file.is_file() and patch_file.stat().st_size <= MAX_PATCH_BYTES) else None
            report = render_report(arg("id"), read_events(found), patch)
            return self._send(200, report.encode("utf-8"), "text/markdown; charset=utf-8", {
                "Content-Disposition": f'attachment; filename="{export_filename(arg("id"))}"'})
        if path == "/api/evals":
            return self._json({"reports": list_eval_reports(root)})
        if path == "/api/eval":
            report = load_eval_report(root, arg("name"))
            if report is None:
                return self._json({"error": "not found"}, 404)
            return self._json({"report": report, "summary": evals.summarize(report.get("results", []))})
        if path == "/api/compare":
            a, b = load_eval_report(root, arg("a")), load_eval_report(root, arg("b"))
            if a is None or b is None:
                return self._json({"error": "not found"}, 404)
            return self._json({"text": evals.compare(a, b)})
        self._json({"error": "not found"}, 404)


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--port", type=int, default=8765, help="port on 127.0.0.1 (0 picks a free one)")
    ap.add_argument("--runs", default=str(runs.RUNS_DIR), help="folder to read (default: runs)")
    ap.add_argument("--open", action="store_true", help="open the page in your browser")
    args = ap.parse_args()
    try:
        server = Dashboard(Path(args.runs), args.port)
    except OSError as exc:
        raise SystemExit(f"could not listen on 127.0.0.1:{args.port} ({exc}); try --port 0")
    url = f"http://127.0.0.1:{server.port}/?t={server.token}"
    print(f"dashboard (read-only, this machine only): {url}\nCtrl+C to stop", flush=True)
    if args.open:
        webbrowser.open(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
