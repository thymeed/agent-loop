from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph

from .nodes import (after_ask, after_lead, apply, ask_tai, check, escalate, lead, prepare,
                    reviewer, route, search, worker)
from .state import State

g = StateGraph(State)
g.add_node("lead", lead)
g.add_node("search", search)
g.add_node("prepare", prepare)
g.add_node("worker", worker)
g.add_node("escalate", escalate)
g.add_node("apply", apply)  # a no-op unless the task is a project
g.add_node("check", check)
g.add_node("reviewer", reviewer)
g.add_node("ask_tai", ask_tai)
g.add_edge(START, "lead")
g.add_conditional_edges("lead", after_lead, ["search", "prepare", "worker"])
g.add_edge("search", "worker")
g.add_edge("prepare", "worker")
g.add_edge("worker", "apply")
g.add_edge("escalate", "apply")
g.add_edge("apply", "check")
g.add_edge("check", "reviewer")
g.add_conditional_edges("reviewer", route, ["worker", "escalate", "ask_tai", END])
g.add_conditional_edges("ask_tai", after_ask, ["worker", END])


def compile_graph(saver):
    return g.compile(checkpointer=saver)


graph = compile_graph(InMemorySaver())  # default for tests; the CLI swaps in SQLite
