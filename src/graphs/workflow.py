from langgraph.graph import StateGraph, END
from src.utils.state import AgentStateLLM
from src.agents.reasoning import reasoning_node_llm
from src.agents.action import action_node_llm

workflow_llm = StateGraph(AgentStateLLM)
workflow_llm.add_node("reasoning",reasoning_node_llm)
workflow_llm.add_node("action", action_node_llm)

workflow_llm.set_entry_point("reasoning")
workflow_llm.add_conditional_edges("reasoning", lambda s: s["next_action"],
                                   {"action":"action", "end": END} )
workflow_llm.add_edge("action", "reasoning")

app_llm = workflow_llm.compile()
