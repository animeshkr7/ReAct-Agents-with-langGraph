from src.graphs.workflow import app_llm

result_llm = app_llm.invoke({
    "messages": ["User: Tell me about Tokyo and Japan"],
    "next_action": "",
    "iteration_count": 0
})


print("\n=== ReAct Flow ===")
for msg in result_llm["messages"]:
    print(msg)
