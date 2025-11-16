from langgraph.graph import StateGraph, END
from typing import TypedDict, Annotated
import operator
import os

from google import genai
from google.genai import types



# from openai import OpenAI
# client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

class AgentStateLLM(TypedDict):
    messages: Annotated[list, operator.add]
    next_action: str
    iteration_count: int

def llm_tool(query: str) -> str:
    response = client.models.generate_content(
        model='gemini-2.5-flash',
        contents=[f"Answer this query briefly: {query}"]
    )

    return response.text

# LLM reasoning

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
    

    ### gpt
    # decision = client.chat.completions.create(
    #     model="gpt-4o", max_tokens=100,
    #     messages=[{"role": "user", "content": prompt}]
    # ).choices[0].message.content.strip()

    ### gemini
    decision = client.models.generate_content(
        model='gemini-2.5-flash',
        contents=prompt
    ).text


    if decision.startswith("QUERY:"):
        return {"messages": [f"Thought: {decision}"], "next_action": "action",
                "iteration_count": iteration_count}
    return {"messages": [f"Thought: {decision}"], "next_action": "end",
            "iteration_count":iteration_count}

## Action Exection

def action_node_llm(state: AgentStateLLM):
    last_thought = state["messages"][-1]
    query = last_thought.replace("Though: QUERY:", "").strip()
    result = llm_tool(query)
    return {
        "messages": [f"Action: query('{query}')", f"Observation: {result}"],
        "next_action": "reasoning",
        "iteration_count": state.get("iteration_count",0)+1
    }

## Graph construction

workflow_llm = StateGraph(AgentStateLLM)
workflow_llm.add_node("reasoning",reasoning_node_llm)
workflow_llm.add_node("action", action_node_llm)

workflow_llm.set_entry_point("reasoning")
workflow_llm.add_conditional_edges("reasoning", lambda s: s["next_action"],
                                   {"action":"action", "end": END} )
workflow_llm.add_edge("action", "reasoning")

app_llm = workflow_llm.compile()


result_llm = app_llm.invoke({
    "messages": ["User: Tell me about Tokyo and Japan"],
    "next_action": "",
    "iteration_count": 0
})


print("\n=== ReAct Flow ===")
for msg in result_llm["messages"]:
    print(msg)

