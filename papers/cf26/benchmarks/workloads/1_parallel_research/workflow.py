"""
Parallel Research Workflow - LangGraph Implementation

Demonstrates explicit parallelism with Send API.
Compare with workflow.ais which achieves the same with automatic dataflow parallelism.
"""

from typing import TypedDict, List
from langgraph.graph import StateGraph, START, END
from langgraph.constants import Send
from langchain_core.messages import SystemMessage, HumanMessage
from llm_instrumentation import get_llm, HAS_OLLAMA
from prompt_config import get_system_prompt_or_none


class ResearchState(TypedDict):
    """State for parallel research workflow."""
    topic: str
    background: str
    advances: str
    impact: str
    combined: str
    report: str


def get_llm_instance():
    """Get the configured LLM instance."""
    return get_llm()


def research_background(state: ResearchState) -> dict:
    """Research domain background."""
    llm = get_llm_instance()
    topic = state["topic"]

    messages = []
    system_prompt = get_system_prompt_or_none("ask")
    if system_prompt:
        messages.append(SystemMessage(content=system_prompt))
    messages.append(HumanMessage(content=f"Explain in 2 sentences the domain background of {topic}"))
    response = llm.invoke(messages)
    return {"background": response.content}


def research_advances(state: ResearchState) -> dict:
    """Research recent advances."""
    llm = get_llm_instance()
    topic = state["topic"]

    messages = []
    system_prompt = get_system_prompt_or_none("ask")
    if system_prompt:
        messages.append(SystemMessage(content=system_prompt))
    messages.append(HumanMessage(content=f"Give me 2 recent advances in {topic}"))
    response = llm.invoke(messages)
    return {"advances": response.content}


def research_impact(state: ResearchState) -> dict:
    """Research societal impact."""
    llm = get_llm_instance()
    topic = state["topic"]

    messages = []
    system_prompt = get_system_prompt_or_none("ask")
    if system_prompt:
        messages.append(SystemMessage(content=system_prompt))
    messages.append(HumanMessage(content=f"Explain in 1 sentence what is the societal impact of {topic}"))
    response = llm.invoke(messages)
    return {"impact": response.content}


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
    llm = get_llm_instance()
    combined = state["combined"]

    messages = []
    system_prompt = get_system_prompt_or_none("ask")
    if system_prompt:
        messages.append(SystemMessage(content=system_prompt))
    messages.append(HumanMessage(content=f"Synthesize into a 3 sentences report: {combined}"))
    response = llm.invoke(messages)
    return {"report": response.content}


def fan_out_research(state: ResearchState) -> List[Send]:
    """Fan out to parallel research nodes."""
    return [
        Send("research_background", state),
        Send("research_advances", state),
        Send("research_impact", state),
    ]


def build_graph() -> StateGraph:
    """Build the parallel research graph."""
    builder = StateGraph(ResearchState)

    # Add nodes
    builder.add_node("research_background", research_background)
    builder.add_node("research_advances", research_advances)
    builder.add_node("research_impact", research_impact)
    builder.add_node("merge", merge_results)
    builder.add_node("synthesize", synthesize)

    # Fan-out for parallel execution (requires explicit Send API)
    builder.add_conditional_edges(START, fan_out_research)

    # All parallel branches merge
    builder.add_edge("research_background", "merge")
    builder.add_edge("research_advances", "merge")
    builder.add_edge("research_impact", "merge")

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
