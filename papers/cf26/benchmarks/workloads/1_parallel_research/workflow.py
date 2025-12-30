"""
Parallel Research Workflow - LangGraph Implementation

Demonstrates explicit parallelism with Send API.
Compare with workflow.ais which achieves the same with automatic dataflow parallelism.
"""

import os
from typing import TypedDict, List
from langgraph.graph import StateGraph, START, END
from langgraph.constants import Send

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


class ResearchState(TypedDict):
    """State for parallel research workflow."""
    topic: str
    background: str
    advances: str
    impact: str
    combined: str
    report: str


def get_llm():
    """Get the LLM instance (Ollama or mock)."""
    if HAS_OLLAMA:
        return ChatOllama(model=OLLAMA_MODEL, temperature=0)
    return None


def research_background(state: ResearchState) -> dict:
    """Research domain background."""
    llm = get_llm()
    topic = state["topic"]

    if llm:
        prompt = f"Explain the domain background of {topic} in 2-3 sentences."
        response = llm.invoke(prompt)
        return {"background": response.content}
    return {"background": f"Background research on {topic}..."}


def research_advances(state: ResearchState) -> dict:
    """Research recent advances."""
    llm = get_llm()
    topic = state["topic"]

    if llm:
        prompt = f"What are the recent advances in {topic}? Answer in 2-3 sentences."
        response = llm.invoke(prompt)
        return {"advances": response.content}
    return {"advances": f"Recent advances in {topic}..."}


def research_impact(state: ResearchState) -> dict:
    """Research societal impact."""
    llm = get_llm()
    topic = state["topic"]

    if llm:
        prompt = f"What is the societal impact of {topic}? Answer in 2-3 sentences."
        response = llm.invoke(prompt)
        return {"impact": response.content}
    return {"impact": f"Societal impact of {topic}..."}


def merge_results(state: ResearchState) -> dict:
    """Merge parallel results."""
    combined = (
        f"{state['background']}\n"
        f"{state['advances']}\n"
        f"{state['impact']}"
    )
    return {"combined": combined}


def synthesize(state: ResearchState) -> dict:
    """Synthesize final report."""
    llm = get_llm()
    combined = state["combined"]

    if llm:
        prompt = f"Synthesize this research into a coherent report:\n\n{combined}"
        response = llm.invoke(prompt)
        return {"report": response.content}
    return {"report": f"Synthesized report from: {combined}"}


def fan_out_research(state: ResearchState) -> List[Send]:
    """Fan out to parallel research nodes."""
    return [
        Send("background", state),
        Send("advances", state),
        Send("impact", state),
    ]


def build_graph() -> StateGraph:
    """Build the parallel research graph."""
    builder = StateGraph(ResearchState)

    # Add nodes
    builder.add_node("background", research_background)
    builder.add_node("advances", research_advances)
    builder.add_node("impact", research_impact)
    builder.add_node("merge", merge_results)
    builder.add_node("synthesize", synthesize)

    # Fan-out for parallel execution (requires explicit Send API)
    builder.add_conditional_edges(START, fan_out_research)

    # All parallel branches merge
    builder.add_edge("background", "merge")
    builder.add_edge("advances", "merge")
    builder.add_edge("impact", "merge")

    # Sequential synthesis after merge
    builder.add_edge("merge", "synthesize")
    builder.add_edge("synthesize", END)

    return builder.compile()


# Compile the graph
graph = build_graph()


def run(topic: str = "quantum computing") -> dict:
    """Execute the parallel research workflow."""
    initial_state = {
        "topic": topic,
        "background": "",
        "advances": "",
        "impact": "",
        "combined": "",
        "report": "",
    }
    return graph.invoke(initial_state)


if __name__ == "__main__":
    result = run("quantum computing")
    print(f"Report: {result['report']}")
