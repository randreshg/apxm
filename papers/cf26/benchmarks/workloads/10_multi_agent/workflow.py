"""
Multi-Agent Workflow - LangGraph Implementation

Demonstrates multi-agent collaboration with subgraphs.
Compare with workflow.ais which has native agent definitions
and communicate operations for inter-agent messaging.
"""

from typing import TypedDict
from langgraph.graph import StateGraph, START, END
from llm_instrumentation import get_llm, HAS_OLLAMA


class MultiAgentState(TypedDict):
    """State for multi-agent workflow."""
    topic: str
    research_result: str
    critique_result: str
    final_report: str


def get_llm_instance():
    """Get the configured LLM instance."""
    return get_llm()


def researcher_agent(state: MultiAgentState) -> dict:
    """Researcher agent: conducts deep research.

    Note: In LangGraph, this is just a function node.
    A-PXM has native agent spawning with spawn operator.
    """
    llm = get_llm_instance()
    topic = state["topic"]

    prompt = f"Conduct thorough research on this topic and provide key findings:\n\n{topic}"
    response = llm.invoke(prompt)
    return {"research_result": response.content}


def critic_agent(state: MultiAgentState) -> dict:
    """Critic agent: reviews and critiques research.

    Note: In LangGraph, agents are just nodes.
    No native agent lifecycle management.
    """
    llm = get_llm_instance()
    research = state["research_result"]

    prompt = f"Critically analyze this research and identify weaknesses or gaps:\n\n{research}"
    response = llm.invoke(prompt)
    return {"critique_result": response.content}


def coordinator_synthesize(state: MultiAgentState) -> dict:
    """Coordinator synthesizes final report from agent outputs."""
    llm = get_llm_instance()
    research = state["research_result"]
    critique = state["critique_result"]

    prompt = f"""Synthesize a final report combining research and critique.

Research:
{research}

Critique:
{critique}

Provide a balanced final report addressing the critique."""
    response = llm.invoke(prompt)
    return {"final_report": response.content}


def build_graph() -> StateGraph:
    """Build the multi-agent graph.

    Note: LangGraph requires:
    - Manual subgraph composition
    - Shared state dict (no native message passing)
    - No automatic agent lifecycle management
    - Sequential execution (parallel requires Send API)
    """
    builder = StateGraph(MultiAgentState)

    # Add agent nodes
    builder.add_node("researcher", researcher_agent)
    builder.add_node("critic", critic_agent)
    builder.add_node("synthesize", coordinator_synthesize)

    # Sequential flow (no native parallel agent spawning)
    builder.add_edge(START, "researcher")
    builder.add_edge("researcher", "critic")
    builder.add_edge("critic", "synthesize")
    builder.add_edge("synthesize", END)

    return builder.compile()


# Compile the graph
graph = build_graph()


def run(topic: str = "The future of renewable energy") -> dict:
    """Execute the multi-agent workflow."""
    initial_state = {
        "topic": topic,
        "research_result": "",
        "critique_result": "",
        "final_report": "",
    }
    return graph.invoke(initial_state)


if __name__ == "__main__":
    result = run()
    print(f"Topic: {result['topic']}")
    print(f"\nResearch: {result['research_result'][:200]}...")
    print(f"\nCritique: {result['critique_result'][:200]}...")
    print(f"\nFinal Report: {result['final_report'][:200]}...")
