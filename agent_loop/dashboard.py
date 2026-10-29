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
import time
import webbrowser
from http import cookies
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlsplit

from . import evals, runs

WEB_DIR = Path(__file__).resolve().parent / "web"
STATIC = {
    "/": ("index.html", "text/html; charset=utf-8"),
    "/app.js": ("app.js", "text/javascript; charset=utf-8"),
    "/app.css": ("app.css", "text/css; charset=utf-8"),
}
CSP = ("default-src 'none'; script-src 'self'; style-src 'self'; connect-src 'self'; "
       "img-src 'none'; base-uri 'none'; form-action 'none'; frame-ancestors 'none'")
COOKIE = "dash"
PAGE_EVENTS = 200            # events per /api/trace response
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
        out.append({"name": path.name, "label": report.get("label"), "started": report.get("started"),
                    "finished": report.get("finished"), "attempts": report.get("attempts"),
                    "done": len(report.get("results", []))})
    return out


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
        super().__init__(("127.0.0.1", port), _Handler)  # loopback only; there is no option to change it

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

    do_POST = do_PUT = do_DELETE = do_PATCH = do_OPTIONS = do_HEAD = _method_not_allowed

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
