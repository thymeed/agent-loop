from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph

from .nodes import after_ask, after_lead, ask_tai, check, lead, reviewer, route, search, worker
from .state import State

g = StateGraph(State)
g.add_node("lead", lead)
g.add_node("search", search)
g.add_node("worker", worker)
g.add_node("check", check)
g.add_node("reviewer", reviewer)
g.add_node("ask_tai", ask_tai)
g.add_edge(START, "lead")
g.add_conditional_edges("lead", after_lead, ["search", "worker"])
g.add_edge("search", "worker")
g.add_edge("worker", "check")
g.add_edge("check", "reviewer")
g.add_conditional_edges("reviewer", route, ["worker", "ask_tai", END])
g.add_conditional_edges("ask_tai", after_ask, ["worker", END])
graph = g.compile(checkpointer=InMemorySaver())
