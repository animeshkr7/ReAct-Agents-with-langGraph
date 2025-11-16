from src.utils.state import AgentStateLLM
from src.llms.gemini import client

def reasoning_node_llm(state: AgentStateLLM):
    iteration_count = state.get("iteration_count",0)
    if iteration_count >= 3:
        return {
            "messages": ["Thought: I have gathered enough information"],
            "next_action": "end", "iteration_count": iteration_count
        }
    
    history = "\n".join(state["messages"])
    prompt = f""" You are an AI Agent answering: "Tell me about Tokyo and Japan"

Conversation so far:
{history}

Queries completed: {iteration_count}/3

You MUST make exactly 3 queries to gather information.
Respond ONLY with: QUERY: <your specific question>

Do NOT be conversational. Do not thank the user. Only output: QUERY: <question>"""
    
    decision = client.models.generate_content(
        model='gemini-2.5-flash',
        contents=prompt
    ).text


    if decision.startswith("QUERY:"):
        return {"messages": [f"Thought: {decision}"], "next_action": "action",
                "iteration_count": iteration_count}
    return {"messages": [f"Thought: {decision}"], "next_action": "end",
            "iteration_count":iteration_count}
