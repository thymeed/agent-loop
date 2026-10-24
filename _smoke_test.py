"""Offline checks: owner stop/accept, guidance retries, lap counting, tokens.

Run with pytest (`python -m pytest _smoke_test.py -q`) or directly
(`python _smoke_test.py`).
"""
from types import SimpleNamespace

from langgraph.types import Command

import loop

GOOD = """```python
def is_palindrome(s):
    cleaned = "".join(c.lower() for c in s if c.isalnum())
    return cleaned == cleaned[::-1]
```"""

BROKEN = """```python
def is_palindrome(s):
    return True
```"""


class FakeLLM:
    """Says the same thing every call; reports fixed token usage."""

    def __init__(self, reply):
        self.reply = reply
        self.calls = 0

    def invoke(self, _msg):
        self.calls += 1
        return SimpleNamespace(
            content=self.reply,
            usage_metadata={"input_tokens": 10, "output_tokens": 5, "total_tokens": 15},
        )


def install(*, code=GOOD, review="APPROVED: fine"):
    """Point the graph at fake models; returns the (worker, reviewer) fakes."""
    loop.lead_llm = FakeLLM("1. Do it")
    loop.worker_llm = FakeLLM(code)
    loop.reviewer_llm = FakeLLM(review)
    return loop.worker_llm, loop.reviewer_llm


def drive(thread, *, resume=None, stop_after_pauses=None, tests=None):
    """Run the graph, resuming with `resume` each time it pauses."""
    cfg = {"configurable": {"thread_id": thread}}
    inputs = {
        "task": "is_palindrome",
        "rejections": 0,
        "tests": loop.IMPOSSIBLE_TESTS if tests is None else tests,
    }
    nodes = []
    while True:
        for step in loop.graph.stream(inputs, cfg):
            for node in step:
                nodes.append(node)
        state = loop.graph.get_state(cfg)
        if not state.next:
            return nodes, state
        if stop_after_pauses is not None and nodes.count("__interrupt__") >= stop_after_pauses:
            return nodes, state
        inputs = Command(resume=resume)


def test_extract_code_and_run_tests():
    code = loop.extract_code(GOOD)
    assert code.startswith("def is_palindrome"), code
    ok, out = loop.run_tests(code)
    assert ok and "ALL TESTS PASSED" in out, (ok, out)

    ok, out = loop.run_tests(loop.extract_code(BROKEN))
    assert not ok and "AssertionError" in out, (ok, out)

    ok, out = loop.run_tests("")
    assert not ok and "no Python code" in out, (ok, out)


def test_guidance_reply_retries():
    """A whole sentence must be treated as guidance, not a decision."""
    worker, _ = install()
    nodes, state = drive("guidance",
                         resume="Accept the assert for now; explain the reasoning",
                         stop_after_pauses=2)
    assert nodes.count("__interrupt__") == 2, nodes
    assert "ask_tai" in nodes, nodes
    assert state.values.get("outcome") is None, state.values
    assert worker.calls == 2 * loop.MAX_REJECTIONS, worker.calls


def test_whole_reply_stop_and_accept_end_run():
    cases = [
        ("stop", "stopped by owner"),
        ("Stop.", "stopped by owner"),
        ("  accept  ", "accepted by owner"),
        ("SHIP!", "accepted by owner"),
    ]
    for i, (cmd, expected) in enumerate(cases):
        thread = f"end-{i}"
        worker, _ = install()
        nodes, state = drive(thread, resume=cmd)
        assert nodes[-1] == "ask_tai", (cmd, nodes)
        assert state.values["outcome"].startswith(expected), (cmd, state.values)
        assert worker.calls == loop.MAX_REJECTIONS, (cmd, worker.calls)
        assert loop.graph.get_state({"configurable": {"thread_id": thread}}).next == ()


def test_stop_does_not_add_a_lap():
    worker, reviewer = install()
    nodes, state = drive("stop-once", resume="stop")
    assert nodes.count("__interrupt__") == 1, nodes
    assert worker.calls == loop.MAX_REJECTIONS, worker.calls
    assert reviewer.calls == loop.MAX_REJECTIONS, reviewer.calls
    assert state.values["rejections"] == loop.MAX_REJECTIONS
    assert state.values["tokens"]["worker"]["calls"] == loop.MAX_REJECTIONS
    assert state.values["tokens"]["lead"]["calls"] == 1


def test_token_counts_sum_across_retries():
    worker, reviewer = install()
    laps = 2 * loop.MAX_REJECTIONS
    nodes, state = drive("retry", resume="more", stop_after_pauses=2)
    assert nodes.count("__interrupt__") == 2, nodes
    assert worker.calls == laps and reviewer.calls == laps
    assert state.values["tokens"]["worker"] == {
        "input": 10 * laps, "output": 5 * laps, "total": 15 * laps, "calls": laps,
    }, state.values["tokens"]


def test_cap_run_does_not_leak_tests_into_later_runs():
    before = list(loop.TESTS)
    install()
    stopped = loop.run("is_palindrome", tests=loop.IMPOSSIBLE_TESTS,
                       resume_with="stop", thread="leak")
    assert stopped["outcome"].startswith("stopped by owner")
    assert loop.TESTS == before, loop.TESTS

    install()
    passed = loop.run("is_palindrome", thread="leak-after")
    assert passed["approved"] is True and passed["tests_passed"] is True


if __name__ == "__main__":
    checks = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for check in checks:
        check()
        print(f"{check.__name__}: ok")
    print(f"{len(checks)} offline checks passed")
