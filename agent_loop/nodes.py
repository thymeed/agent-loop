from langgraph.graph import END
from langgraph.types import interrupt

from .checks import TESTS, extract_code, run_tests
from .config import CONFIG, make_llm
from .state import State, usage

MAX_REJECTIONS = CONFIG["max_rejections"]
APPROVED_TAG = "APPROVED"

# Owner calls at the interrupt that end the run instead of asking for another lap.
STOP_WORDS = {"stop", "quit", "end", "halt", "abort"}
ACCEPT_WORDS = {"accept", "approve", "ship"}

# One chat model per role; tests swap entries for fakes.
MODELS = {role: make_llm(spec) for role, spec in CONFIG["models"].items()}


def lead(s: State):
    msg = (
        f"Task: {s['task']}\n"
        "Write a short numbered plan (max 6 steps) to solve the task. "
        "No code, no preamble."
    )
    resp = MODELS["lead"].invoke(msg)
    return {"plan": resp.content.strip(), "tokens": {"lead": usage(resp)}}


def worker(s: State):
    msg = (
        f"Task: {s['task']}\n"
        f"Plan: {s['plan']}\n"
        f"Last review: {s.get('review', 'none')}\n"
        "Write the solution."
    )
    resp = MODELS["worker"].invoke(msg)
    return {"work": resp.content, "tokens": {"worker": usage(resp)}}


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
    resp = MODELS["reviewer"].invoke(msg)
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
