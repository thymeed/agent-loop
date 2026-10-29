from typing import Annotated, Any, NotRequired, TypedDict


def merge_usage(left: dict, right: dict) -> dict:
    """Reducer: merge per-node token counts, summing repeated visits of a node."""
    out = {node: dict(counts) for node, counts in (left or {}).items()}
    for node, counts in (right or {}).items():
        acc = out.setdefault(node, {})
        for key, val in counts.items():
            acc[key] = acc.get(key, 0) + val
    return out


class State(TypedDict):
    task: str                  # always there
    plan: str
    review: NotRequired[str]   # only the truly optional ones
    work: str
    code: str
    tests_passed: bool
    test_output: str
    tests: list[str]
    type: NotRequired[str]     # "code" (default) or "research"
    query: NotRequired[str]
    max_papers: NotRequired[int]
    sources: list[dict]        # research: fetched arXiv papers
    project_dir: NotRequired[str]    # project: the original, never written
    workspace: NotRequired[str]      # project: the copy the worker edits
    test_command: NotRequired[str]
    requirements: NotRequired[list[str]]
    protected: NotRequired[list[str]]    # globs the worker may not write (the tests)
    changed: list[str]               # project: files written so far
    apply_errors: list[str]          # project: blocks rejected on the last lap (lap fails)
    apply_notes: list[str]           # project: protected-file edits ignored on the last lap
    approved: bool
    escalated: bool            # stronger model already tried at this cap
    rejections: int
    outcome: str
    # node name -> {"input": n, "output": n, "total": n, "calls": n}, summed
    tokens: Annotated[dict[str, dict[str, int]], merge_usage]


def usage(resp: Any) -> dict[str, int]:
    """Token counts for one model call, straight off the AIMessage."""
    meta = getattr(resp, "usage_metadata", None) or {}
    return {
        "input": int(meta.get("input_tokens") or 0),
        "output": int(meta.get("output_tokens") or 0),
        "total": int(meta.get("total_tokens") or 0),
        "calls": 1,
    }
