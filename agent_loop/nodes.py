import os
import re
import shutil
from pathlib import Path

from langgraph.graph import END
from langgraph.types import interrupt

from . import research, sandbox, trace, workspace
from .checks import TESTS, extract_code, run_tests
from .config import CONFIG, make_llm
from .state import State, usage

MAX_REJECTIONS = CONFIG["max_rejections"]
APPROVED_TAG = "APPROVED"
MAX_PATCH_CHARS = 6000

# Owner calls at the interrupt that end the run instead of asking for another lap.
STOP_WORDS = {"stop", "quit", "end", "halt", "abort"}
ACCEPT_WORDS = {"accept", "approve", "ship"}

# One chat model per role; tests swap entries for fakes.
MODELS = {role: make_llm(spec, CONFIG["model_options"].get(role)) for role, spec in CONFIG["models"].items()}


_THINK_BLOCK = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)
_THINK_OPEN = re.compile(r"<think>.*", re.DOTALL | re.IGNORECASE)  # never closed: a token cap cut it off
_THINK_CLOSE = re.compile(r"</think>", re.IGNORECASE)


def strip_think(text: str) -> str:
    """The reply without the model's reasoning.

    Thinking models (qwen3 and friends) return their reasoning inside <think> tags unless the
    client separates it. Reasoning often contains draft code blocks, so it must never be parsed
    for answers: file blocks inside it would otherwise be written to the workspace."""
    text = _THINK_BLOCK.sub("", text or "")
    text = _THINK_OPEN.sub("", text)
    parts = _THINK_CLOSE.split(text)  # a lone closing tag: everything before it was reasoning
    return parts[-1].strip()


def answer_text(resp) -> str:
    """What a model actually answered: its reply with the reasoning removed."""
    return strip_think(str(resp.content))


def lead(s: State):
    if "no_plan" in CONFIG["prompt_variants"]:  # experiment: the worker gets no plan, and no call is made
        return {"plan": "none", "tokens": {"lead": {"input": 0, "output": 0, "total": 0, "calls": 0}}}
    msg = (
        f"Task: {s['task']}\n"
        "Write a short numbered plan (max 6 steps) to solve the task. "
        "No code, no preamble."
    )
    resp = trace.llm_call("lead", MODELS["lead"], msg)
    return {"plan": answer_text(resp), "tokens": {"lead": usage(resp)}}


def search(s: State):
    """Research mode: fetch the papers the answer has to be grounded in."""
    papers = research.fetch_papers(s["query"], s.get("max_papers", 8))
    return {"sources": papers}


def is_research(s: State) -> bool:
    return s.get("type") == "research"


def is_project(s: State) -> bool:
    return s.get("type") == "project"


def after_lead(s: State):
    if is_research(s):
        return "search"
    return "prepare" if is_project(s) else "worker"


def prepare(s: State):
    """Project mode: copy the project into the workspace once; the original is never written."""
    if not is_project(s):
        return {}
    dest = Path(s["workspace"])
    if not dest.exists():  # a resumed run keeps the files the worker already changed
        partial = dest.with_name(dest.name + ".partial")
        shutil.rmtree(partial, ignore_errors=True)
        skipped = workspace.copy_project(Path(s["project_dir"]), partial,
                                         max_bytes=CONFIG["max_project_bytes"])
        os.replace(partial, dest)
        if skipped:
            print("note: not copied into the workspace (look like credentials): " + ", ".join(skipped))
    return {}


def _protected(s: State):
    """The task's protected globs; a missing key means the defaults, an empty list means none."""
    protected = s.get("protected")
    return workspace.DEFAULT_PROTECTED if protected is None else protected


def apply(s: State):
    """Project mode: write the worker's file blocks into the workspace, safely."""
    if not is_project(s):
        return {}
    result = workspace.apply_blocks(Path(s["workspace"]), s.get("work", ""),
                                    protected=_protected(s))
    return {"changed": sorted(set(s.get("changed", [])) | set(result.written)),
            "apply_errors": result.errors, "apply_notes": result.skipped}


def _example_path(s: State) -> str:
    """The file-block example's path: the first workspace .py file the worker may edit (not an
    __init__.py), else pkg/module.py."""
    if "real_path_example" in CONFIG["prompt_variants"]:
        for name in workspace.project_files(Path(s["workspace"])):
            if (name.endswith(".py") and Path(name).name != "__init__.py"
                    and not workspace.is_protected(name, _protected(s))):
                return name
    return "pkg/module.py"


def _worker_prompt(s: State) -> str:
    if is_project(s):
        context = workspace.build_context(Path(s["workspace"]), budget=CONFIG["max_context_chars"])
        return (
            f"Task: {s['task']}\n"
            f"Plan: {s['plan']}\n\n"
            f"{context}\n"
            f"Last review: {s.get('review', 'none')}\n"
            f"Last test run:\n{s.get('test_output', 'none')}\n\n"
            "Change the project so the task is done and its tests pass. Reply with one fenced "
            "block per file you change or create. The fence's info string is the file path "
            "and the block holds the COMPLETE new file, for example:\n"
            f"```{_example_path(s)}\n<entire file>\n```\n"
            "Use paths relative to the project root, only files inside the project, and no "
            "other fenced blocks."
            + ("\nTest files and test configuration (test_*.py, tests/, conftest.py, pytest.ini, "
               "pyproject.toml, ...) are read-only: fix the code so the existing tests pass."
               if _protected(s) else "")
        )
    if is_research(s):
        return (
            f"Task: {s['task']}\n"
            f"Plan: {s['plan']}\n"
            f"Papers (your ONLY allowed source):\n{research.format_papers(s['sources'])}\n\n"
            f"Last review: {s.get('review', 'none')}\n"
            f"Last citation check: {s.get('test_output', 'none')}\n"
            "Answer the task using only these papers. Support every claim by citing the paper "
            'as [arXiv:ID] and copying an exact phrase of at least 4 words from its abstract '
            'in double quotes in the same paragraph. Every paper you cite needs at least one '
            "such quote. Do not use outside knowledge and do not cite other papers."
        )
    return (
        f"Task: {s['task']}\n"
        f"Plan: {s['plan']}\n"
        f"Last review: {s.get('review', 'none')}\n"
        "Write the solution."
    )


def worker(s: State):
    resp = trace.llm_call("worker", MODELS["worker"], _worker_prompt(s))
    return {"work": answer_text(resp), "tokens": {"worker": usage(resp)}}


def escalate(s: State):
    """One attempt by the stronger model at the cap; its tokens are reported separately."""
    resp = trace.llm_call("escalation", MODELS["escalation"], _worker_prompt(s))
    return {"work": answer_text(resp), "escalated": True, "tokens": {"escalate": usage(resp)}}


def check(s: State):
    """Deterministic gate: run the code against the tests, or verify the citations."""
    if is_research(s):
        passed, output = research.check_citations(s.get("work", ""), s["sources"])
        return {"tests_passed": passed, "test_output": output}
    if is_project(s):
        if s.get("apply_errors"):  # a reply that could not be applied is never tested as if it had been
            return {"tests_passed": False, "test_output":
                    "your reply could not be applied:\n- " + "\n- ".join(s["apply_errors"])}
        # Belt and braces: whatever got past apply, edited tests never count as a pass.
        tampered = workspace.protected_changes(Path(s["project_dir"]), Path(s["workspace"]),
                                               _protected(s))
        if tampered:
            return {"tests_passed": False, "test_output":
                    "protected test/config files differ from the original: " + ", ".join(tampered)}
        passed, output = sandbox.run_project_tests(
            Path(s["workspace"]), s.get("test_command", "pytest -q"), s.get("requirements", []))
        if s.get("apply_notes"):  # first, so the output tail cannot cut it off
            output = ("note: ignored edits to protected files (fix the code, not the tests):\n- "
                      + "\n- ".join(s["apply_notes"]) + "\n" + output)
        return {"tests_passed": passed, "test_output": output}
    code = extract_code(s.get("work", ""))
    passed, output = run_tests(code, s.get("tests"))
    return {"code": code, "tests_passed": passed, "test_output": output}


def reviewer(s: State):
    if is_project(s):
        patch = workspace.make_patch(Path(s["project_dir"]), Path(s["workspace"]))
        if len(patch) > MAX_PATCH_CHARS:
            patch = patch[:MAX_PATCH_CHARS] + "\n... (diff truncated)\n"
        msg = (
            f"Task: {s['task']}\n"
            f"Changes made to the project (unified diff):\n{patch or '(no changes)'}\n\n"
            f"Project tests: {'PASSED' if s.get('tests_passed') else 'FAILED'}\n"
            f"{s.get('test_output', '')}\n\n"
            "Judge whether the change does what the task asks, without breaking anything "
            "else. Your reply MUST start with the single word "
            f"{APPROVED_TAG} or REJECTED, then a short reason."
        )
    elif is_research(s):
        msg = (
            f"Task: {s['task']}\n"
            f"Answer:\n{s['work']}\n\n"
            f"Citation check: {'PASSED' if s.get('tests_passed') else 'FAILED'}\n"
            f"{s.get('test_output', '')}\n\n"
            "Judge whether the answer addresses the task and whether the claims follow "
            "from the quoted evidence. Your reply MUST start with the single word "
            f"{APPROVED_TAG} or REJECTED, then a short reason."
        )
    else:
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
    resp = trace.llm_call("reviewer", MODELS["reviewer"], msg)
    text = answer_text(resp)
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
    return {"plan": plan, "rejections": 0, "escalated": False}


def after_ask(s: State):
    """Owner can end the run here; otherwise send it back for another lap."""
    return END if s.get("outcome") else "worker"


def route(s: State):
    # Tests are the hard gate: a model "APPROVED" cannot override a red run.
    if s.get("approved") and s.get("tests_passed"):
        return END
    if s.get("rejections", 0) >= MAX_REJECTIONS:
        if "escalation" in MODELS and not s.get("escalated"):
            return "escalate"
        return "ask_tai"
    return "worker"
