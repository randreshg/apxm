"""
Type Verification Workflow - LangGraph Implementation (INTENTIONALLY BROKEN)

Contains the SAME bug as workflow.ais: using an undefined key.
LangGraph discovers this at RUNTIME, after the first LLM call.
"""

from typing import TypedDict
from langgraph.graph import StateGraph, START, END
from langchain_core.messages import SystemMessage, HumanMessage
from llm_instrumentation import get_llm, HAS_OLLAMA
from prompt_config import get_system_prompt_or_none


class BrokenState(TypedDict):
    """State for broken workflow."""
    result: str
    output: str


def get_llm_instance():
    """Get the configured LLM instance."""
    return get_llm()


def first_step(state: BrokenState) -> dict:
    """First step - succeeds and makes an LLM call.

    This call SUCCEEDS, costing tokens/money.
    The error will only be discovered in the second step.
    """
    llm = get_llm_instance()

    messages = []
    system_prompt = get_system_prompt_or_none("ask")
    if system_prompt:
        messages.append(SystemMessage(content=system_prompt))
    messages.append(HumanMessage(content="Do something useful"))
    response = llm.invoke(messages)
    return {"result": response.content}


def second_step(state: BrokenState) -> dict:
    """Second step - FAILS due to undefined key.

    This is the same bug as in workflow.ais.
    But LangGraph only discovers it HERE, after the first LLM call.
    """
    # BUG: 'undefined_var' key doesn't exist in state
    # This raises KeyError at RUNTIME
    undefined_value = state["undefined_var"]  # KeyError!

    llm = get_llm_instance()
    messages = []
    system_prompt = get_system_prompt_or_none("ask")
    if system_prompt:
        messages.append(SystemMessage(content=system_prompt))
    messages.append(HumanMessage(content="Use this: " + undefined_value))
    response = llm.invoke(messages)
    return {"output": response.content}


def build_graph() -> StateGraph:
    """Build the broken graph."""
    builder = StateGraph(BrokenState)

    builder.add_node("first", first_step)
    builder.add_node("second", second_step)

    builder.add_edge(START, "first")
    builder.add_edge("first", "second")  # Error occurs HERE
    builder.add_edge("second", END)

    return builder.compile()


# Compile the graph
graph = build_graph()


def run() -> dict:
    """Execute the broken workflow.

    This will FAIL at runtime with KeyError.
    But only AFTER the first LLM call has been made.
    """
    initial_state = {
        "result": "",
        "output": "",
        # Note: 'undefined_var' is NOT in the state
    }
    return graph.invoke(initial_state)


if __name__ == "__main__":
    try:
        result = run()
        print(f"Output: {result['output']}")
    except KeyError as e:
        print(f"Runtime Error: KeyError - {e}")
        print("Error discovered AFTER first LLM call was made!")
