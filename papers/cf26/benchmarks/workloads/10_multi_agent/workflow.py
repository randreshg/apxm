"""
Multi-Agent Workflow - LangGraph Implementation

Demonstrates multi-agent collaboration with subgraphs.
Compare with workflow.ais which has native agent definitions
and communicate operations for inter-agent messaging.
"""

from typing import TypedDict
from langgraph.graph import StateGraph, START, END
from langchain_core.messages import SystemMessage, HumanMessage
from llm_instrumentation import get_llm, HAS_OLLAMA
from prompt_config import get_system_prompt_or_none


class MultiAgentState(TypedDict):
    """State for multi-agent workflow."""
    topic: str
    research_result: str
    critique_prep: str
    critique_result: str
    final_report: str


def get_llm_instance():
    """Get the configured LLM instance."""
    return get_llm()


def researcher_agent(state: MultiAgentState) -> dict:
    """Researcher agent: conducts deep research.

    Note: In LangGraph, this is just a function node.
    """
    llm = get_llm_instance()
    topic = state["topic"]

    messages = []
    system_prompt = get_system_prompt_or_none("ask")
    if system_prompt:
        messages.append(SystemMessage(content=system_prompt))
    messages.append(HumanMessage(content=f"Conduct deep research on: {topic}"))
    response = llm.invoke(messages)
    return {"research_result": response.content}


def critic_prepare(state: MultiAgentState) -> dict:
    """Critic agent: prepares initial critique questions."""
    llm = get_llm_instance()
    topic = state["topic"]

    messages = []
    system_prompt = get_system_prompt_or_none("ask")
    if system_prompt:
        messages.append(SystemMessage(content=system_prompt))
    messages.append(HumanMessage(content=f"Prepare initial critique questions for: {topic}"))
    response = llm.invoke(messages)
    return {"critique_prep": response.content}


def critic_agent(state: MultiAgentState) -> dict:
    """Critic agent: reviews and critiques research.

    Note: In LangGraph, agents are just nodes.
    No native agent lifecycle management.
    """
    llm = get_llm_instance()
    research = state["research_result"]

    messages = []
    system_prompt = get_system_prompt_or_none("ask")
    if system_prompt:
        messages.append(SystemMessage(content=system_prompt))
    messages.append(HumanMessage(content=f"Critically analyze and identify weaknesses: {research}"))
    response = llm.invoke(messages)
    return {"critique_result": response.content}


def coordinator_synthesize(state: MultiAgentState) -> dict:
    """Coordinator synthesizes final report from agent outputs."""
    llm = get_llm_instance()
    research = state["research_result"]
    critique_prep = state["critique_prep"]
    critique = state["critique_result"]

    messages = []
    system_prompt = get_system_prompt_or_none("ask")
    if system_prompt:
        messages.append(SystemMessage(content=system_prompt))
    messages.append(HumanMessage(content=f"Synthesize into final report. Research: {research} | Prepared questions: {critique_prep} | Critique: {critique}"))
    response = llm.invoke(messages)
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
    builder.add_node("critic_prepare", critic_prepare)
    builder.add_node("critic", critic_agent)
    builder.add_node("synthesize", coordinator_synthesize)

    builder.add_edge(START, "researcher")
    builder.add_edge("researcher", "critic_prepare")
    builder.add_edge("critic_prepare", "critic")
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
        "critique_prep": "",
        "critique_result": "",
        "final_report": "",
    }
    return graph.invoke(initial_state)


if __name__ == "__main__":
    result = run()
    print(f"Topic: {result['topic']}")
    print(f"\nResearch: {result['research_result'][:200]}...")
    print(f"\nCritique Prep: {result['critique_prep'][:200]}...")
    print(f"\nCritique: {result['critique_result'][:200]}...")
    print(f"\nFinal Report: {result['final_report'][:200]}...")
