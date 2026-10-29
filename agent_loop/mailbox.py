"""Owner-call mailbox: how the dashboard answers a run that is paused at the rejection cap.

The run puts a random nonce in its `pause` trace event and waits. The dashboard writes
runs/<thread>/owner_reply.json = {"nonce": ..., "text": ...} with that nonce. The run reads it
once, deletes it, and treats the text exactly like a line typed in the terminal. A reply with
the wrong nonce, a stale one, an oversize one or one with control characters is discarded.
"""
import json
import os
import secrets
import uuid
from pathlib import Path

REPLY_FILE = "owner_reply.json"
MAX_TEXT = 2000
MAX_FILE_BYTES = 8192
_ALLOWED_CONTROL = {"\n", "\t"}


def new_nonce() -> str:
    return secrets.token_urlsafe(12)


def reply_path(thread_dir: Path) -> Path:
    return Path(thread_dir) / REPLY_FILE


def clean_text(text) -> str | None:
    """The reply text if it is acceptable, else None: 1-2000 characters, no control characters."""
    if not isinstance(text, str):
        return None
    text = text.strip()
    if not text or len(text) > MAX_TEXT:
        return None
    if any((ord(ch) < 32 and ch not in _ALLOWED_CONTROL) or ord(ch) == 127 for ch in text):
        return None
    return text


def clear(thread_dir: Path) -> None:
    """Drop a reply left over from an earlier pause."""
    try:
        reply_path(thread_dir).unlink()
    except FileNotFoundError:
        pass


def write_reply(thread_dir: Path, nonce: str, text: str) -> None:
    """Leave a reply for the run. Raises ValueError for a bad reply, FileExistsError if one is
    already waiting. The content appears all at once and never overwrites an earlier reply."""
    cleaned = clean_text(text)
    if cleaned is None or not isinstance(nonce, str) or not 8 <= len(nonce) <= 64:
        raise ValueError("bad reply")
    final = reply_path(thread_dir)
    tmp = Path(thread_dir) / f".owner_reply.{uuid.uuid4().hex}.tmp"
    tmp.write_text(json.dumps({"nonce": nonce, "text": cleaned}), encoding="utf-8", newline="\n")
    try:
        os.link(tmp, final)  # atomic, and fails if a reply is already there
    finally:
        tmp.unlink(missing_ok=True)


def consume_reply(thread_dir: Path, nonce: str) -> str | None:
    """The waiting reply's text if it is valid for this pause (and it is then deleted); otherwise
    None. A reply that does not belong to this pause is deleted so it cannot clog the next one."""
    path = reply_path(thread_dir)
    try:
        if path.stat().st_size > MAX_FILE_BYTES:
            path.unlink(missing_ok=True)
            return None
        raw = path.read_text(encoding="utf-8")
    except FileNotFoundError:
        return None
    except (OSError, UnicodeDecodeError):
        path.unlink(missing_ok=True)
        return None
    path.unlink(missing_ok=True)
    try:
        data = json.loads(raw)
    except ValueError:
        return None
    if not isinstance(data, dict) or data.get("nonce") != nonce:
        return None
    return clean_text(data.get("text"))
