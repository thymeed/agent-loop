"""Per-run records and checkpoint storage under runs/ (gitignored)."""
import json
import re
import time
from pathlib import Path

RUNS_DIR = Path(__file__).resolve().parent.parent / "runs"
DB_PATH = RUNS_DIR / "checkpoints.sqlite"


def new_thread_id(label: str) -> str:
    """Unique per run, so a shared checkpoint DB never mixes two runs in one thread."""
    slug = re.sub(r"[^A-Za-z0-9_-]+", "-", label).strip("-")[:40] or "run"
    return f"{slug}-{time.strftime('%Y%m%d-%H%M%S')}"


def result_label(values: dict) -> str:
    if values.get("outcome"):
        return values["outcome"].split(":")[0]
    if values.get("approved") and values.get("tests_passed"):
        return "approved"
    return "unfinished"


def write_record(runs_dir: Path, thread: str, values: dict, pauses: int) -> Path:
    """Save the outcome, final answer and token table of one run as runs/<thread>.json."""
    runs_dir.mkdir(parents=True, exist_ok=True)
    record = {
        "thread": thread,
        "finished_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "result": result_label(values),
        "pauses": pauses,
        **{k: values.get(k) for k in (
            "task", "type", "approved", "tests_passed", "rejections", "outcome",
            "work", "review", "test_output", "tokens")},
    }
    path = runs_dir / f"{thread}.json"
    path.write_text(json.dumps(record, indent=2, ensure_ascii=False), encoding="utf-8")
    return path
