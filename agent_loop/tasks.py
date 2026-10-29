from dataclasses import dataclass, field
from pathlib import Path

import yaml

from .sandbox import validate_requirements
from .workspace import DEFAULT_PROTECTED


@dataclass
class Task:
    task: str
    kind: str = "code"                    # "code" (asserts), "research" (arXiv) or "project"
    tests: list[str] | None = None        # code only
    query: str | None = None              # research only
    max_papers: int = 8
    project: str | None = None            # project only: absolute path of the original
    test_command: str = "pytest -q"
    requirements: list[str] = field(default_factory=list)
    # globs the worker may not write; empty only when the task sets `allow_test_edits: true`
    protected: list[str] = field(default_factory=lambda: list(DEFAULT_PROTECTED))


def _text(data: dict, key: str, path: str) -> str:
    value = data.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{path}: '{key}' must be a non-empty string")
    return value.strip()


def load_task(path: str) -> Task:
    """Read a task file. `type: code` (default) needs `tests`, `research` needs `query`,
    `project` needs `project` (a directory) and may set `test_command` and `requirements`."""
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
    if kind == "project":
        project = (Path(path).resolve().parent / _text(data, "project", path)).resolve()
        if not project.is_dir():
            raise ValueError(f"{path}: project directory not found: {data['project']}")
        requirements = data.get("requirements", [])
        if not isinstance(requirements, list):
            raise ValueError(f"{path}: 'requirements' must be a list of strings")
        command = data.get("test_command", "pytest -q")
        if not isinstance(command, str) or not command.strip():
            raise ValueError(f"{path}: 'test_command' must be a non-empty string")
        try:
            requirements = validate_requirements(requirements)
        except ValueError as exc:
            raise ValueError(f"{path}: {exc}") from exc
        extra = data.get("protected", [])
        if not isinstance(extra, list) or not all(isinstance(g, str) and g.strip() for g in extra):
            raise ValueError(f"{path}: 'protected' must be a list of glob strings")
        allow = data.get("allow_test_edits", False)
        if not isinstance(allow, bool):
            raise ValueError(f"{path}: 'allow_test_edits' must be true or false")
        protected = [] if allow else [*DEFAULT_PROTECTED, *(g.strip() for g in extra)]
        return Task(task, "project", project=str(project), test_command=command.strip(),
                    requirements=requirements, protected=protected)
    raise ValueError(f"{path}: 'type' must be 'code', 'research' or 'project', got {kind!r}")
