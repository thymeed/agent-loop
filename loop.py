"""Lead -> worker -> check -> reviewer loop with a 3-rejection cap, on local Ollama."""
import argparse
import re
import subprocess
import sys
from typing import Annotated, Any, TypedDict

from langchain_ollama import ChatOllama
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command, interrupt

lead_llm = ChatOllama(model="gemma3:4b")
worker_llm = ChatOllama(model="gemma3:4b")    # swap per role later
reviewer_llm = ChatOllama(model="gemma3:4b")

MAX_REJECTIONS = 3
TEST_TIMEOUT = 15
APPROVED_TAG = "APPROVED"

# Fixed asserts the worker's code has to satisfy. `--cap` swaps these out for an
# impossible one so the 3-rejection path can be exercised on purpose.
TESTS = [
    'assert is_palindrome("A man a plan a canal Panama") is True',
    'assert is_palindrome("") is True',
    'assert is_palindrome("ab") is False',
]

IMPOSSIBLE_TESTS = [
    'assert is_palindrome("ab") is True  # impossible: "ab" is not a palindrome',
]

# Owner calls at the interrupt that end the run instead of asking for another lap.
STOP_WORDS = {"stop", "quit", "end", "halt", "abort"}
ACCEPT_WORDS = {"accept", "approve", "ship"}


def merge_usage(left: dict, right: dict) -> dict:
    """Reducer: merge per-node token counts, summing repeated visits of a node."""
    out = {node: dict(counts) for node, counts in (left or {}).items()}
    for node, counts in (right or {}).items():
        acc = out.setdefault(node, {})
        for key, val in counts.items():
            acc[key] = acc.get(key, 0) + val
    return out


class State(TypedDict, total=False):
    task: str
    plan: str
    work: str
    code: str
    tests_passed: bool
    test_output: str
    tests: list[str]
    review: str
    approved: bool
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


def lead(s: State):
    msg = (
        f"Task: {s['task']}\n"
        "Write a short numbered plan (max 6 steps) to solve the task. "
        "No code, no preamble."
    )
    resp = lead_llm.invoke(msg)
    return {"plan": resp.content.strip(), "tokens": {"lead": usage(resp)}}


def worker(s: State):
    msg = (
        f"Task: {s['task']}\n"
        f"Plan: {s['plan']}\n"
        f"Last review: {s.get('review', 'none')}\n"
        "Write the solution."
    )
    resp = worker_llm.invoke(msg)
    return {"work": resp.content, "tokens": {"worker": usage(resp)}}


_FENCE = re.compile(r"```(?:python|py)?[ \t]*\r?\n(.*?)```", re.DOTALL | re.IGNORECASE)


def extract_code(text: str) -> str:
    """First ```python fenced block, else everything from the first `def `."""
    match = _FENCE.search(text or "")
    if match:
        return match.group(1).strip()
    idx = (text or "").find("def ")
    return (text[idx:] if idx >= 0 else (text or "")).strip()


def run_tests(code: str, tests: list[str] | None = None) -> tuple[bool, str]:
    """Run `code` plus the asserts in a throwaway subprocess; never raises."""
    if not code.strip():
        return False, "no Python code found in the worker output"
    body = "\n".join(tests if tests is not None else TESTS)
    script = f"{code}\n\n\n{body}\nprint('ALL TESTS PASSED')\n"
    try:
        proc = subprocess.run(
            [sys.executable, "-I", "-c", script],
            capture_output=True,
            text=True,
            timeout=TEST_TIMEOUT,
            check=False,  # a red test suite is a result, not an exception
        )
    except subprocess.TimeoutExpired:
        return False, f"timeout after {TEST_TIMEOUT}s"
    except OSError as exc:  # interpreter missing / not executable
        return False, f"could not start subprocess: {exc}"
    output = (proc.stdout + proc.stderr).strip()
    return proc.returncode == 0, (output[-1500:] or "(no output)")


def check(s: State):
    """Deterministic gate: actually execute the worker's code against the tests."""
    code = extract_code(s.get("work", ""))
    passed, output = run_tests(code, s.get("tests"))
    return {"code": code, "tests_passed": passed, "test_output": output}


def reviewer(s: State):
    tests = s.get("tests") or TESTS
    msg = (
        f"Task: {s['task']}\n"
        f"Plan:\n{s['plan']}\n"
        f"Solution:\n{s['work']}\n\n"
        f"Automated tests run by the harness:\n" + "\n".join(tests) + "\n"
        f"Test result: {'PASSED' if s.get('tests_passed') else 'FAILED'}\n"
        f"Test output:\n{s.get('test_output', '')}\n\n"
        "Check the solution against the task. Your reply MUST start with the "
        f"single word {APPROVED_TAG} or REJECTED, then a short reason."
    )
    resp = reviewer_llm.invoke(msg)
    text = resp.content.strip()
    model_ok = text.upper().startswith(APPROVED_TAG)
    tests_ok = bool(s.get("tests_passed"))
    if model_ok and not tests_ok:
        text += "\n[check] Overridden: automated tests failed."
    ok = model_ok and tests_ok
    return {
        "review": text,
        "approved": ok,
        "rejections": s.get("rejections", 0) + (0 if ok else 1),
        "tokens": {"reviewer": usage(resp)},
    }


def ask_tai(s: State):
    choice = interrupt({
        "review": s["review"],
        "rejections": s["rejections"],
        "test_output": s.get("test_output", ""),
        "options": "reply exactly 'stop' or 'accept' to end the run; "
                   "anything longer is guidance and retries",
    })
    text = (choice or "").strip()
    cmd = text.lower().strip(" .!?")
    plan = s["plan"] + "\nOwner call: " + text

    # Whole-reply match only: "accept the assert for now" is guidance, not a decision.
    if cmd in STOP_WORDS:
        return {"plan": plan, "outcome": f"stopped by owner: {text}"}
    if cmd in ACCEPT_WORDS:
        return {"plan": plan, "outcome": f"accepted by owner despite failing checks: {text}"}
    return {"plan": plan, "rejections": 0}


def after_ask(s: State):
    """Owner can end the run here; otherwise send it back for another lap."""
    return END if s.get("outcome") else "worker"


def route(s: State):
    # Tests are the hard gate: a model "APPROVED" cannot override a red run.
    if s.get("approved") and s.get("tests_passed"):
        return END
    if s.get("rejections", 0) >= MAX_REJECTIONS:
        return "ask_tai"
    return "worker"


g = StateGraph(State)
g.add_node("lead", lead)
g.add_node("worker", worker)
g.add_node("check", check)
g.add_node("reviewer", reviewer)
g.add_node("ask_tai", ask_tai)
g.add_edge(START, "lead")
g.add_edge("lead", "worker")
g.add_edge("worker", "check")
g.add_edge("check", "reviewer")
g.add_conditional_edges("reviewer", route, ["worker", "ask_tai", END])
g.add_conditional_edges("ask_tai", after_ask, ["worker", END])
graph = g.compile(checkpointer=InMemorySaver())


def print_step(node: str, update: dict) -> None:
    shown = {}
    for key, value in update.items():
        if isinstance(value, str) and len(value) > 700:
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
        resume_with: str | None = None, thread: str = "1",
        max_pauses: int | None = None) -> dict:
    """Drive the graph through interrupts until END; returns the final state."""
    cfg = {"configurable": {"thread_id": thread}}
    # Tests travel in state, so a --cap run cannot leak into later runs.
    inputs: Any = {"task": task, "rejections": 0, "tests": tests or TESTS}
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


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--cap", action="store_true",
                    help="swap in an impossible assert to force the 3-rejection cap")
    ap.add_argument("--resume", metavar="TEXT", default=None,
                    help="owner call used at the interrupt instead of prompting; "
                         "'stop' or 'accept' ends the run")
    ap.add_argument("--thread", default="1")
    ap.add_argument("--max-pauses", type=int, default=None,
                    help="stop after N interrupts instead of looping forever")
    args = ap.parse_args()

    run(
        "Write is_palindrome(s) in Python that ignores case and spaces.",
        tests=IMPOSSIBLE_TESTS if args.cap else None,
        resume_with=args.resume,
        thread=args.thread,
        max_pauses=args.max_pauses,
    )
