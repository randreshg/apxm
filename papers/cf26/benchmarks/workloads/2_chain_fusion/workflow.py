"""
Chain Fusion Workflow - LangGraph Implementation

CANNOT fuse - each node is a separate LLM call.
Compare with workflow.ais where the compiler fuses 5 calls into 1.
"""

from typing import TypedDict
from langgraph.graph import StateGraph, START, END

# Try to import Ollama for real LLM calls
try:
    from langchain_ollama import ChatOllama
    HAS_OLLAMA = True
except ImportError:
    HAS_OLLAMA = False

# Ollama model configuration
OLLAMA_MODEL = "phi3:mini"


class ChainState(TypedDict):
    """State for chain fusion workflow."""
    step1: str
    step2: str
    step3: str
    step4: str
    summary: str


def get_llm():
    """Get the LLM instance (Ollama or mock)."""
    if HAS_OLLAMA:
        return ChatOllama(model=OLLAMA_MODEL, temperature=0)
    return None


def define_quantum(state: ChainState) -> dict:
    """Step 1: Define quantum computing."""
    llm = get_llm()

    if llm:
        prompt = "Define quantum computing in 1-2 sentences."
        response = llm.invoke(prompt)
        return {"step1": response.content}
    return {"step1": "Quantum computing uses quantum-mechanical phenomena..."}


def explain_qubits(state: ChainState) -> dict:
    """Step 2: Explain qubits using previous context."""
    llm = get_llm()
    context = state["step1"]

    if llm:
        prompt = f"Using this context: {context}\n\nExplain qubits in 1-2 sentences."
        response = llm.invoke(prompt)
        return {"step2": response.content}
    return {"step2": f"Based on {context[:50]}..., qubits are..."}


def explain_superposition(state: ChainState) -> dict:
    """Step 3: Explain superposition using previous context."""
    llm = get_llm()
    context = state["step2"]

    if llm:
        prompt = f"Using this context: {context}\n\nExplain superposition in 1-2 sentences."
        response = llm.invoke(prompt)
        return {"step3": response.content}
    return {"step3": f"Based on {context[:50]}..., superposition means..."}


def explain_entanglement(state: ChainState) -> dict:
    """Step 4: Explain entanglement using previous context."""
    llm = get_llm()
    context = state["step3"]

    if llm:
        prompt = f"Using this context: {context}\n\nExplain quantum entanglement in 1-2 sentences."
        response = llm.invoke(prompt)
        return {"step4": response.content}
    return {"step4": f"Based on {context[:50]}..., entanglement is..."}


def summarize(state: ChainState) -> dict:
    """Step 5: Summarize all concepts."""
    llm = get_llm()
    context = state["step4"]

    if llm:
        prompt = f"Summarize all the quantum computing concepts discussed: {context}"
        response = llm.invoke(prompt)
        return {"summary": response.content}
    return {"summary": f"Summary of all concepts: {context[:100]}..."}


def build_graph() -> StateGraph:
    """Build the chain fusion graph.

    Note: LangGraph CANNOT fuse these calls.
    Each node triggers a separate LLM request.
    """
    builder = StateGraph(ChainState)

    # Add nodes - each is a separate LLM call
    builder.add_node("step1", define_quantum)
    builder.add_node("step2", explain_qubits)
    builder.add_node("step3", explain_superposition)
    builder.add_node("step4", explain_entanglement)
    builder.add_node("summary", summarize)

    # Sequential chain - no fusion possible
    builder.add_edge(START, "step1")
    builder.add_edge("step1", "step2")
    builder.add_edge("step2", "step3")
    builder.add_edge("step3", "step4")
    builder.add_edge("step4", "summary")
    builder.add_edge("summary", END)

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
