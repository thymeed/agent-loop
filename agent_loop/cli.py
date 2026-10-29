import argparse
import sys
from typing import Any

from langgraph.types import Command

from .checks import IMPOSSIBLE_TESTS, TESTS
from .graph import graph
from .tasks import Task, load_task


def print_step(node: str, update: dict) -> None:
    shown = {}
    for key, value in update.items():
        if key == "sources":
            shown[key] = [f"{p['id']} {p['title'][:60]}" for p in value]
        elif isinstance(value, str) and len(value) > 700:
            shown[key] = value[:700] + f"... ({len(value)} chars)"
        else:
            shown[key] = value
    print(f"[{node}] {shown}\n")


def print_tokens(tokens: dict[str, dict[str, int]]) -> None:
    print("Token usage per node (calls, then input / output / total; summed over visits):")
    grand = {"input": 0, "output": 0, "total": 0, "calls": 0}
    for node, counts in tokens.items():
        print(f"  {node:10s} calls={counts.get('calls', 0):>2d}  "
              f"{counts.get('input', 0):>6d} / {counts.get('output', 0):>5d} / "
              f"{counts.get('total', 0):>6d}")
        for key in grand:
            grand[key] += counts.get(key, 0)
    print(f"  {'TOTAL':10s} calls={grand['calls']:>2d}  {grand['input']:>6d} / "
          f"{grand['output']:>5d} / {grand['total']:>6d}\n")


def run(task: str, *, tests: list[str] | None = None,
        kind: str = "code", query: str | None = None, max_papers: int = 8,
        resume_with: str | None = None, thread: str = "1",
        max_pauses: int | None = None) -> dict:
    """Drive the graph through interrupts until END; returns the final state."""
    cfg = {"configurable": {"thread_id": thread}}
    # Tests travel in state, so a --cap run cannot leak into later runs.
    inputs: Any = {"task": task, "rejections": 0, "type": kind}
    if kind == "research":
        inputs.update(query=query, max_papers=max_papers)
    else:
        inputs["tests"] = tests or TESTS
    pauses = 0

    while True:
        for step in graph.stream(inputs, cfg):
            for node, update in step.items():
                if node == "__interrupt__":
                    pauses += 1
                    print(f"PAUSED #{pauses} - owner call needed: {update[0].value}\n")
                else:
                    print_step(node, update)

        state = graph.get_state(cfg)
        if not state.next:
            break  # reached END
        if max_pauses is not None and pauses >= max_pauses:
            print(f"stopping after {pauses} pause(s); next={state.next} "
                  "(resume with Command(resume=...))\n")
            break
        ans = resume_with if resume_with is not None else input("Owner call: ").strip()
        inputs = Command(resume=ans)

    values = graph.get_state(cfg).values
    print_tokens(values.get("tokens", {}))
    print(f"pauses={pauses} approved={values.get('approved')} "
          f"rejections={values.get('rejections')} tests_passed={values.get('tests_passed')} "
          f"outcome={values.get('outcome') or '-'}")
    return values


def main() -> None:
    # Paper titles and model output are not cp1252-safe on Windows consoles.
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    ap = argparse.ArgumentParser()
    ap.add_argument("task_file", nargs="?", default=None,
                    help="YAML file with `task` and `tests` (asserts); "
                         "default is the built-in palindrome task")
    ap.add_argument("--cap", action="store_true",
                    help="swap in an impossible assert to force the 3-rejection cap")
    ap.add_argument("--resume", metavar="TEXT", default=None,
                    help="owner call used at the interrupt instead of prompting; "
                         "'stop' or 'accept' ends the run")
    ap.add_argument("--thread", default="1")
    ap.add_argument("--max-pauses", type=int, default=None,
                    help="stop after N interrupts instead of looping forever")
    args = ap.parse_args()
    if args.cap and args.task_file:
        ap.error("--cap only applies to the built-in palindrome task")

    if args.task_file:
        spec = load_task(args.task_file)
    else:
        spec = Task("Write is_palindrome(s) in Python that ignores case and spaces.",
                    tests=IMPOSSIBLE_TESTS if args.cap else None)

    run(
        spec.task,
        tests=spec.tests,
        kind=spec.kind,
        query=spec.query,
        max_papers=spec.max_papers,
        resume_with=args.resume,
        thread=args.thread,
        max_pauses=args.max_pauses,
    )
