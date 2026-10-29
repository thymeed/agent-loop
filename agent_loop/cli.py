import argparse
import sys
from pathlib import Path
from typing import Any

from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.types import Command

from .checks import IMPOSSIBLE_TESTS, TESTS
from . import runs
from .graph import compile_graph, graph
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


def run(task: str | None, *, tests: list[str] | None = None,
        kind: str = "code", query: str | None = None, max_papers: int = 8,
        resume_with: str | None = None, thread: str = "1",
        max_pauses: int | None = None, app=None,
        runs_dir: Path | None = None, continue_thread: bool = False) -> dict:
    """Drive the graph through interrupts until END; returns the final state.

    `app` is the compiled graph (default: in-memory). With `runs_dir` set, the final
    state is saved there. `continue_thread` picks up a saved thread instead of
    starting one (after a crash, or a run stopped with --max-pauses).
    """
    app = app or graph
    cfg = {"configurable": {"thread_id": thread}}
    inputs: Any
    if continue_thread:
        snap = app.get_state(cfg)
        if not snap.next:
            raise SystemExit(f"nothing to resume: thread {thread!r} has no unfinished run")
        paused = any(t.interrupts for t in snap.tasks)
        inputs = Command(resume=_owner_call(resume_with)) if paused else None
        print(f"continuing {thread!r} at {snap.next}\n")
    else:
        # Tests travel in state, so a --cap run cannot leak into later runs.
        inputs = {"task": task, "rejections": 0, "type": kind}
        if kind == "research":
            inputs.update(query=query, max_papers=max_papers)
        else:
            inputs["tests"] = tests or TESTS
    pauses = 0

    while True:
        for step in app.stream(inputs, cfg):
            for node, update in step.items():
                if node == "__interrupt__":
                    pauses += 1
                    print(f"PAUSED #{pauses} - owner call needed: {update[0].value}\n")
                else:
                    print_step(node, update)

        state = app.get_state(cfg)
        if not state.next:
            break  # reached END
        if max_pauses is not None and pauses >= max_pauses:
            print(f"stopping after {pauses} pause(s); next={state.next}; "
                  f"continue with --continue {thread}\n")
            break
        inputs = Command(resume=_owner_call(resume_with))

    values = app.get_state(cfg).values
    print_tokens(values.get("tokens", {}))
    print(f"pauses={pauses} approved={values.get('approved')} "
          f"rejections={values.get('rejections')} tests_passed={values.get('tests_passed')} "
          f"outcome={values.get('outcome') or '-'}")
    if runs_dir is not None and values:
        print(f"saved {runs.write_record(runs_dir, thread, values, pauses)}")
    return values


def _owner_call(resume_with: str | None) -> str:
    return resume_with if resume_with is not None else input("Owner call: ").strip()


def expand_task_files(paths: list[str]) -> list[str]:
    """Files stay as given; a directory becomes its *.yaml files, sorted."""
    files: list[str] = []
    for p in paths:
        path = Path(p)
        found = sorted(str(f) for f in path.glob("*.yaml")) if path.is_dir() else [p]
        if not found:
            raise SystemExit(f"no .yaml task files in {p}")
        files += found
    return files


def print_batch_summary(rows: list[tuple[str, dict]]) -> None:
    print("\nBatch summary:")
    for name, values in rows:
        total = sum(c.get("total", 0) for c in values.get("tokens", {}).values())
        print(f"  {name:32s} {runs.result_label(values):18s} "
              f"rejections={values.get('rejections', 0)}  tokens={total}")


def main() -> None:
    # Paper titles and model output are not cp1252-safe on Windows consoles.
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    ap = argparse.ArgumentParser()
    ap.add_argument("task_files", nargs="*",
                    help="YAML task files or directories of them (run one after another); "
                         "default is the built-in palindrome task")
    ap.add_argument("--cap", action="store_true",
                    help="swap in an impossible assert to force the 3-rejection cap")
    ap.add_argument("--resume", metavar="TEXT", default=None,
                    help="owner call used at the interrupt instead of prompting; "
                         "'stop' or 'accept' ends the run")
    ap.add_argument("--thread", default=None,
                    help="thread id (default: <task>-<timestamp>)")
    ap.add_argument("--continue", dest="continue_thread", metavar="THREAD", default=None,
                    help="resume a saved, unfinished thread from runs/checkpoints.sqlite")
    ap.add_argument("--max-pauses", type=int, default=None,
                    help="stop after N interrupts instead of looping forever")
    args = ap.parse_args()
    if args.cap and args.task_files:
        ap.error("--cap only applies to the built-in palindrome task")
    if args.continue_thread and (args.task_files or args.cap or args.thread):
        ap.error("--continue takes a thread id and nothing else")
    if args.thread and len(args.task_files) > 1:
        ap.error("--thread only makes sense for a single task")

    runs.RUNS_DIR.mkdir(exist_ok=True)
    with SqliteSaver.from_conn_string(str(runs.DB_PATH)) as saver:
        app = compile_graph(saver)
        common = dict(app=app, runs_dir=runs.RUNS_DIR, resume_with=args.resume,
                      max_pauses=args.max_pauses)

        if args.continue_thread:
            run(None, thread=args.continue_thread, continue_thread=True, **common)
            return

        if args.task_files:
            jobs = [(Path(f).stem, load_task(f)) for f in expand_task_files(args.task_files)]
        else:
            jobs = [("palindrome", Task(
                "Write is_palindrome(s) in Python that ignores case and spaces.",
                tests=IMPOSSIBLE_TESTS if args.cap else None))]

        rows = []
        for label, spec in jobs:
            print(f"\n=== {label} ===")
            thread = args.thread or runs.new_thread_id(label)
            values = run(spec.task, tests=spec.tests, kind=spec.kind, query=spec.query,
                         max_papers=spec.max_papers, thread=thread, **common)
            rows.append((label, values))
        if len(rows) > 1:
            print_batch_summary(rows)
