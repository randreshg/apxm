"""
Chain Fusion Workflow - LangGraph Implementation

CANNOT fuse - each node is a separate LLM call.
Compare with workflow.ais where the compiler fuses 5 calls into 1.
"""

from typing import TypedDict
from langgraph.graph import StateGraph, START, END
from llm_instrumentation import get_llm, HAS_OLLAMA


class ChainState(TypedDict):
    """State for chain fusion workflow."""
    step1: str
    step2: str
    step3: str
    step4: str
    summary: str


def get_llm_instance():
    """Get the configured LLM instance."""
    return get_llm()


def define_quantum(state: ChainState) -> dict:
    """Step 1: Define quantum computing."""
    llm = get_llm_instance()

    prompt = "Define quantum computing in 1-2 sentences."
    response = llm.invoke(prompt)
    return {"step1": response.content}


def explain_qubits(state: ChainState) -> dict:
    """Step 2: Explain qubits using previous context."""
    llm = get_llm_instance()
    context = state["step1"]

    prompt = f"Using this context: {context}\n\nExplain qubits in 1-2 sentences."
    response = llm.invoke(prompt)
    return {"step2": response.content}


def explain_superposition(state: ChainState) -> dict:
    """Step 3: Explain superposition using previous context."""
    llm = get_llm_instance()
    context = state["step2"]

    prompt = f"Using this context: {context}\n\nExplain superposition in 1-2 sentences."
    response = llm.invoke(prompt)
    return {"step3": response.content}


def explain_entanglement(state: ChainState) -> dict:
    """Step 4: Explain entanglement using previous context."""
    llm = get_llm_instance()
    context = state["step3"]

    prompt = f"Using this context: {context}\n\nExplain quantum entanglement in 1-2 sentences."
    response = llm.invoke(prompt)
    return {"step4": response.content}


def summarize(state: ChainState) -> dict:
    """Step 5: Summarize all concepts."""
    llm = get_llm_instance()
    context = state["step4"]

    prompt = f"Summarize all the quantum computing concepts discussed: {context}"
    response = llm.invoke(prompt)
    return {"summary": response.content}


def build_graph() -> StateGraph:
    """Build the chain fusion graph.

    Note: LangGraph CANNOT fuse these calls.
    Each node triggers a separate LLM request.
    """
    builder = StateGraph(ChainState)

    # Add nodes - each is a separate LLM call
    # Node names must differ from state keys
    builder.add_node("node_step1", define_quantum)
    builder.add_node("node_step2", explain_qubits)
    builder.add_node("node_step3", explain_superposition)
    builder.add_node("node_step4", explain_entanglement)
    builder.add_node("node_summary", summarize)

    # Sequential chain - no fusion possible
    builder.add_edge(START, "node_step1")
    builder.add_edge("node_step1", "node_step2")
    builder.add_edge("node_step2", "node_step3")
    builder.add_edge("node_step3", "node_step4")
    builder.add_edge("node_step4", "node_summary")
    builder.add_edge("node_summary", END)

    return builder.compile()


# Compile the graph
graph = build_graph()


def run() -> dict:
    """Execute the chain fusion workflow."""
    initial_state = {
        "step1": "",
        "step2": "",
        "step3": "",
        "step4": "",
        "summary": "",
    }
    return graph.invoke(initial_state)


if __name__ == "__main__":
    result = run()
    print(f"Summary: {result['summary']}")
