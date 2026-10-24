"""Offline checks: owner stop/accept, lap counting, token call counts."""
from types import SimpleNamespace

from langgraph.types import Command

import loop

GOOD = """```python
def is_palindrome(s):
    cleaned = "".join(c.lower() for c in s if c.isalnum())
    return cleaned == cleaned[::-1]
```"""


class FakeLLM:
    def __init__(self, reply):
        self.reply = reply
        self.calls = 0

    def invoke(self, _msg):
        self.calls += 1
        return SimpleNamespace(
            content=self.reply,
            usage_metadata={"input_tokens": 10, "output_tokens": 5, "total_tokens": 15},
        )


def drive(inputs, thread, resume=None, stop_after_pauses=None):
    cfg = {"configurable": {"thread_id": thread}}
    events = []
    while True:
        for step in loop.graph.stream(inputs, cfg):
            for node, update in step.items():
                events.append((node, update))
        state = loop.graph.get_state(cfg)
        if not state.next:
            return events, state
        if stop_after_pauses is not None and \
                sum(1 for n, _ in events if n == "__interrupt__") >= stop_after_pauses:
            return events, state
        inputs = Command(resume=resume)


def reset(resume_reply="APPROVED: fine"):
    loop.TESTS = loop.IMPOSSIBLE_TESTS
    loop.lead_llm = FakeLLM("1. Do it")
    loop.worker_llm = FakeLLM(GOOD)
    loop.reviewer_llm = FakeLLM(resume_reply)


# --- owner "stop" ends the run --------------------------------------------
reset()
events, state = drive({"task": "t", "rejections": 0}, "stop", resume="stop")
nodes = [n for n, _ in events]
print("stop run:", nodes)
assert nodes[-1] == "ask_tai", nodes
assert loop.graph.get_state({"configurable": {"thread_id": "stop"}}).next == ()
assert state.values["outcome"] == "stopped by owner: stop", state.values
assert loop.worker_llm.calls == loop.MAX_REJECTIONS, loop.worker_llm.calls
assert loop.reviewer_llm.calls == loop.MAX_REJECTIONS
assert state.values["rejections"] == loop.MAX_REJECTIONS
assert state.values["tokens"]["worker"]["calls"] == loop.MAX_REJECTIONS
assert state.values["tokens"]["lead"]["calls"] == 1

# --- owner "accept" also ends, marked differently -------------------------
reset()
events, state = drive({"task": "t", "rejections": 0}, "accept", resume="accept")
assert state.values["outcome"].startswith("accepted by owner")
assert not loop.graph.get_state({"configurable": {"thread_id": "accept"}}).next
print("accept outcome:", state.values["outcome"])

# --- any other owner call retries for exactly 3 more laps -----------------
reset()
events, state = drive({"task": "t", "rejections": 0}, "more", resume="more", stop_after_pauses=2)
nodes = [n for n, _ in events]
assert nodes.count("__interrupt__") == 2, nodes
assert loop.worker_llm.calls == 2 * loop.MAX_REJECTIONS, loop.worker_llm.calls
assert state.values["rejections"] == loop.MAX_REJECTIONS
assert state.values["tokens"]["worker"]["calls"] == 2 * loop.MAX_REJECTIONS
assert state.values["tokens"]["reviewer"]["calls"] == 2 * loop.MAX_REJECTIONS
assert state.values["tokens"]["worker"]["total"] == 15 * 2 * loop.MAX_REJECTIONS
print("retry run: worker laps =", loop.worker_llm.calls,
      "| pauses =", nodes.count("__interrupt__"))
loop.print_tokens(state.values["tokens"])
print("SMOKE TEST OK")
