from typing import TypedDict, Annotated
import operator

class AgentStateLLM(TypedDict):
    messages: Annotated[list, operator.add]
    next_action: str
    iteration_count: int
