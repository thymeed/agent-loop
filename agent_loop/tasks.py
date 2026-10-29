from dataclasses import dataclass
from pathlib import Path

import yaml


@dataclass
class Task:
    task: str
    kind: str = "code"                    # "code" (asserts in Docker) or "research" (arXiv)
    tests: list[str] | None = None        # code only
    query: str | None = None              # research only
    max_papers: int = 8


def _text(data: dict, key: str, path: str) -> str:
    value = data.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{path}: '{key}' must be a non-empty string")
    return value.strip()


def load_task(path: str) -> Task:
    """Read a task file. `type: code` (default) needs `tests`; `type: research` needs `query`."""
    data = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"{path}: expected a mapping")
    task = _text(data, "task", path)
    kind = data.get("type", "code")

    if kind == "code":
        tests = data.get("tests")
        if not isinstance(tests, list) or not tests or not all(isinstance(t, str) and t.strip() for t in tests):
            raise ValueError(f"{path}: 'tests' must be a non-empty list of strings")
        return Task(task, "code", tests=tests)
    if kind == "research":
        max_papers = data.get("max_papers", 8)
        if not isinstance(max_papers, int) or isinstance(max_papers, bool) or not 1 <= max_papers <= 30:
            raise ValueError(f"{path}: 'max_papers' must be an integer from 1 to 30")
        return Task(task, "research", query=_text(data, "query", path), max_papers=max_papers)
    raise ValueError(f"{path}: 'type' must be 'code' or 'research', got {kind!r}")
