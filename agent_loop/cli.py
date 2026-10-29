import argparse
import functools
import queue
import sys
import threading
import time
from pathlib import Path
from typing import Any

from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.types import Command

from .checks import IMPOSSIBLE_TESTS, TESTS
from . import mailbox, runs, trace, workspace
from .config import CONFIG
from .graph import compile_graph, graph
from .tasks import Task, load_task


def print_step(node: str, update: dict) -> None:
    shown = {}
    for key, value in (update or {}).items():  # no-op nodes stream None
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


def _traced(fn):
    """Record the run to runs/<thread>/trace.jsonl when it has a runs_dir; always release the sink."""
    @functools.wraps(fn)
    def wrapper(*args, **kw):
        runs_dir, thread = kw.get("runs_dir"), kw.get("thread", "1")
        trace.begin(Path(runs_dir) / thread / "trace.jsonl" if runs_dir is not None else None)
        try:
            return fn(*args, **kw)
        except BaseException as exc:
            trace.emit("run_error", error=trace.clean(f"{type(exc).__name__}: {exc}", 400))
            raise
        finally:
            trace.begin(None)
    return wrapper


@_traced
def run(task: str | None, *, tests: list[str] | None = None,
        kind: str = "code", query: str | None = None, max_papers: int = 8,
        resume_with: str | None = None, thread: str = "1",
        max_pauses: int | None = None, app=None,
        runs_dir: Path | None = None, continue_thread: bool = False,
        project: str | None = None, test_command: str = "pytest -q",
        requirements: list[str] | None = None,
        protected: list[str] | None = None,
        timings: dict[str, float] | None = None,
        max_tokens: int | None = None) -> dict:
    """Drive the graph through interrupts until END; returns the final state.

    `app` is the compiled graph (default: in-memory). With `runs_dir` set, the final
    state is saved there. `continue_thread` picks up a saved thread instead of
    starting one (after a crash, or a run stopped with --max-pauses). `timings`, if
    given, is filled with wall seconds per node. `max_tokens` stops the run at the
    first node boundary after the total passes it (a node already running finishes).
    """
    app = app or graph
    cfg = {"configurable": {"thread_id": thread}}
    thread_dir = Path(runs_dir) / thread if runs_dir is not None else None
    nonce = None
    inputs: Any
    if continue_thread:
        snap = app.get_state(cfg)
        if not snap.next:
            raise SystemExit(f"nothing to resume: thread {thread!r} has no unfinished run")
        waiting = [t.interrupts[0].value for t in snap.tasks if t.interrupts]
        print(f"continuing {thread!r} at {snap.next}\n")
        trace.emit("resumed", thread=thread, next=list(snap.next))
        if waiting:
            nonce = _open_pause(thread_dir, 0, waiting[0])
            inputs = Command(resume=_ask_owner(resume_with, thread_dir, nonce))
        else:
            inputs = None
    else:
        # Tests travel in state, so a --cap run cannot leak into later runs.
        inputs = {"task": task, "rejections": 0, "type": kind}
        if kind == "research":
            inputs.update(query=query, max_papers=max_papers)
        elif kind == "project":
            if runs_dir is None:  # the workspace has to live somewhere that gets looked after
                raise ValueError("project tasks need a runs_dir: the workspace copy is kept there")
            inputs.update(project_dir=project, workspace=str(Path(runs_dir) / thread / "workspace"),
                          test_command=test_command, requirements=requirements or [],
                          protected=list(workspace.DEFAULT_PROTECTED) if protected is None else protected)
        else:
            inputs["tests"] = tests or TESTS
        trace.emit("run_start", thread=thread, kind=kind, task=trace.clean(task or "", 4000),
                   models=dict(CONFIG["models"]), max_rejections=CONFIG["max_rejections"])
    pauses = 0
    used = 0
    over_budget = False

    while True:
        clock = time.monotonic()  # reset per stream so an owner's thinking time is not a node's
        for step in app.stream(inputs, cfg):
            now = time.monotonic()
            for node, update in step.items():
                if node == "__interrupt__":
                    pauses += 1
                    print(f"PAUSED #{pauses} - owner call needed: {update[0].value}\n")
                    nonce = _open_pause(thread_dir, pauses, update[0].value)
                else:
                    print_step(node, update)
                    trace.emit("node", node=node, seconds=round(now - clock, 2),
                               update=trace.summarize_update(update))
                    if timings is not None:
                        timings[node] = timings.get(node, 0.0) + (now - clock)
                    used += sum(c.get("total", 0) for c in (update or {}).get("tokens", {}).values())
            clock = now
            if max_tokens is not None and used > max_tokens:
                over_budget = True
                break
        if over_budget:
            print(f"stopping: {used} tokens used, over the budget of {max_tokens}\n")
            break

        state = app.get_state(cfg)
        if not state.next:
            break  # reached END
        if max_pauses is not None and pauses >= max_pauses:
            print(f"stopping after {pauses} pause(s); next={state.next}; "
                  f"continue with --continue {thread}\n")
            break
        inputs = Command(resume=_ask_owner(resume_with, thread_dir, nonce))

    values = app.get_state(cfg).values
    if over_budget:
        values = {**values, "outcome": f"over token budget: used {used} of {max_tokens}"}
    print_tokens(values.get("tokens", {}))
    print(f"pauses={pauses} approved={values.get('approved')} "
          f"rejections={values.get('rejections')} tests_passed={values.get('tests_passed')} "
          f"outcome={values.get('outcome') or '-'}")
    if runs_dir is not None and values.get("workspace"):
        patch = workspace.make_patch(Path(values["project_dir"]), Path(values["workspace"]))
        patch_file = runs_dir / thread / "changes.patch"
        patch_file.parent.mkdir(parents=True, exist_ok=True)
        patch_file.write_bytes(patch.encode("utf-8"))  # bytes: keep LF so git apply accepts it
        values = {**values, "patch": f"{thread}/changes.patch"}
        print(f"changes: {len(values.get('changed', []))} file(s); review {patch_file.name} in "
              f"runs/{thread}, then from your project: git apply <path to changes.patch>")
    if runs_dir is not None and values:
        print(f"saved {runs.write_record(runs_dir, thread, values, pauses)}")
    trace.emit("run_end", result=runs.result_label(values), approved=bool(values.get("approved")),
               tests_passed=bool(values.get("tests_passed")), rejections=values.get("rejections"),
               outcome=values.get("outcome"), pauses=pauses, changed=values.get("changed"),
               patch=values.get("patch"), tokens=values.get("tokens", {}))
    return values


class _Terminal:
    """One reader thread per process. input() cannot be cancelled, so a single thread feeds a
    queue and a wait can give up on the terminal (when the dashboard answers first) without
    leaving a second reader behind to steal the next line."""

    def __init__(self, read_line=input):
        self.read_line, self.lines, self.thread, self.eof = read_line, queue.Queue(), None, False

    def start(self) -> None:
        if self.thread is None or (not self.thread.is_alive() and not self.eof):
            self.thread = threading.Thread(target=self._pump, daemon=True)
            self.thread.start()

    def _pump(self) -> None:
        while True:
            try:
                line = self.read_line()
            except (EOFError, OSError):
                self.eof = True
                return
            self.lines.put(line.strip())

    def drain(self) -> int:
        dropped = 0
        while True:
            try:
                self.lines.get_nowait()
                dropped += 1
            except queue.Empty:
                return dropped


_terminal = _Terminal()


def _open_pause(thread_dir: Path | None, number: int, info) -> str:
    """Start a pause: a fresh nonce for the dashboard to echo back, recorded in the trace."""
    nonce = mailbox.new_nonce()
    if thread_dir is not None:
        mailbox.clear(thread_dir)  # a reply left from an earlier pause must not answer this one
    trace.emit("pause", number=number, info=trace.summarize_update(info), nonce=nonce)
    return nonce


def _wait_for_owner(thread_dir: Path, nonce: str) -> tuple[str, str]:
    """The first answer to arrive: a line typed here, or a reply the dashboard left for this pause."""
    _terminal.start()
    dropped = _terminal.drain()  # lines typed while no question was open are not answers
    if dropped:
        print(f"(ignored {dropped} line(s) typed while no question was open)")
    if _terminal.eof:
        print("No terminal input here; waiting for an answer from the dashboard (Ctrl+C to stop).")
    else:
        print("Owner call (type here, or answer in the dashboard): ", end="", flush=True)
    while True:
        try:
            return _terminal.lines.get(timeout=0.5), "terminal"
        except queue.Empty:
            pass
        text = mailbox.consume_reply(thread_dir, nonce)
        if text is not None:
            print("\n(answered from the dashboard)")
            return text, "dashboard"


def _ask_owner(resume_with: str | None, thread_dir: Path | None, nonce: str | None) -> str:
    """The owner's call at a pause, from --resume, the terminal, or (with a runs folder) the dashboard.
    The text is used exactly like terminal input, so ask_tai's exact stop/accept rule is unchanged."""
    if resume_with is not None:
        answer, source = resume_with, "flag"
    elif thread_dir is None or nonce is None:
        answer, source = input("Owner call: ").strip(), "terminal"
    else:
        answer, source = _wait_for_owner(thread_dir, nonce)
    trace.emit("owner_call", text=trace.clean(answer, 500), source=source)
    return answer


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
    for given in (args.thread, args.continue_thread):
        if given and not runs.valid_thread(given):
            ap.error("thread ids may only use letters, digits, '.', '_' and '-' (max 80)")
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
                         max_papers=spec.max_papers, project=spec.project,
                         test_command=spec.test_command, requirements=spec.requirements,
                         protected=spec.protected,
                         thread=thread, **common)
            rows.append((label, values))
        if len(rows) > 1:
            print_batch_summary(rows)
