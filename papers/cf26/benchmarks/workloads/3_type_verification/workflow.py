"""
Type Verification Workflow - LangGraph Implementation (INTENTIONALLY BROKEN)

Contains the SAME bug as workflow.ais: using an undefined key.
LangGraph discovers this at RUNTIME, after the first LLM call.
"""

import os
from typing import TypedDict
from langgraph.graph import StateGraph, START, END

# Try to import Ollama for real LLM calls
try:
    from langchain_ollama import ChatOllama
    HAS_OLLAMA = True
except ImportError:
    HAS_OLLAMA = False

# Ollama model configuration
OLLAMA_MODEL = (
    os.environ.get("APXM_BENCH_OLLAMA_MODEL")
    or os.environ.get("OLLAMA_MODEL")
    or "phi3:mini"
)


class BrokenState(TypedDict):
    """State for broken workflow."""
    result: str
    output: str


def get_llm():
    """Get the LLM instance (Ollama or mock)."""
    if HAS_OLLAMA:
        return ChatOllama(model=OLLAMA_MODEL, temperature=0)
    return None


def first_step(state: BrokenState) -> dict:
    """First step - succeeds and makes an LLM call.

    This call SUCCEEDS, costing tokens/money.
    The error will only be discovered in the second step.
    """
    llm = get_llm()

    if llm:
        prompt = "Do something simple: say 'Hello World'"
        response = llm.invoke(prompt)
        return {"result": response.content}
    return {"result": "First step completed successfully"}


def second_step(state: BrokenState) -> dict:
    """Second step - FAILS due to undefined key.

    This is the same bug as in workflow.ais.
    But LangGraph only discovers it HERE, after the first LLM call.
    """
    # BUG: 'undefined_var' key doesn't exist in state
    # This raises KeyError at RUNTIME
    undefined_value = state["undefined_var"]  # KeyError!
    return {"output": f"Using: {undefined_value}"}


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
