"""Append-only trace of a run: what each model was asked and what it answered.

One JSON object per line in runs/<thread>/trace.jsonl, read by the dashboard. The files hold
your prompts (including project code), so they stay local under the gitignored runs/ and
key-shaped strings are redacted before anything is written.
"""
import json
import threading
import time
from pathlib import Path

from .config import CONFIG
from .workspace import SECRET_CONTENT

MAX_TEXT = 50_000       # one prompt or reply; longer text is cut
MAX_UPDATE_TEXT = 4_000  # one field of a node update
_lock = threading.Lock()
_sink: Path | None = None  # one run at a time per process, evals included


def begin(path: Path | None) -> None:
    """Send events to `path` (None turns tracing off)."""
    global _sink
    _sink = path
    if path is not None:
        path.parent.mkdir(parents=True, exist_ok=True)


def clean(value, limit: int = MAX_TEXT) -> str:
    text = SECRET_CONTENT.sub("[redacted]", str(value))
    if len(text) > limit:
        text = text[:limit] + f"\n... ({len(text)} chars, cut)"
    return text


def emit(event: str, **fields) -> None:
    if _sink is None:
        return
    line = json.dumps({"ts": round(time.time(), 3), "event": event, **fields}, ensure_ascii=False)
    with _lock:
        # newline="\n": one complete line per event, even on Windows, so a reader can tell
        # a finished line from one still being written.
        with open(_sink, "a", encoding="utf-8", newline="\n") as f:
            f.write(line + "\n")


def llm_call(role: str, model, prompt: str):
    """model.invoke(prompt), recorded as llm_start / llm_end (or llm_error)."""
    emit("llm_start", role=role, model=CONFIG["models"].get(role, "?"), prompt=clean(prompt))
    started = time.monotonic()
    try:
        resp = model.invoke(prompt)
    except Exception as exc:
        emit("llm_error", role=role, seconds=round(time.monotonic() - started, 2),
             error=clean(f"{type(exc).__name__}: {exc}", 400))
        raise
    meta = getattr(resp, "usage_metadata", None) or {}
    emit("llm_end", role=role, seconds=round(time.monotonic() - started, 2), reply=clean(resp.content),
         tokens={k: int(meta.get(k) or 0) for k in ("input_tokens", "output_tokens", "total_tokens")})
    return resp


# Fields already recorded elsewhere in the trace, or too bulky to repeat on every node.
_SKIP = {"tokens", "work", "code", "plan"}


def summarize_update(update) -> dict:
    """A node's state update, trimmed for the trace."""
    out = {}
    for key, value in (update or {}).items():
        if key in _SKIP:
            continue
        if key == "sources":
            out[key] = [p.get("id", "?") for p in value]
        elif isinstance(value, str):
            out[key] = clean(value, MAX_UPDATE_TEXT)
        elif isinstance(value, list):
            out[key] = [clean(v, 500) for v in value][:50]
        else:
            out[key] = value
    return out
