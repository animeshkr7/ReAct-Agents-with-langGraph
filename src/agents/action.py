from src.utils.state import AgentStateLLM
from src.llms.gemini import llm_tool

def action_node_llm(state: AgentStateLLM):
    last_thought = state["messages"][-1]
    query = last_thought.replace("Though: QUERY:", "").strip()
    result = llm_tool(query)
    return {
        "messages": [f"Action: query('{query}')", f"Observation: {result}"],
        "next_action": "reasoning",
        "iteration_count": state.get("iteration_count",0)+1
    }
