"""
Scalability Workflow - LangGraph Implementation

Tests parallelism efficiency at different levels (N = 2, 4, 8).
Requires explicit Send API for parallel execution.
"""

import operator
from typing import Annotated, TypedDict, List
from langgraph.graph import StateGraph, START, END
from langgraph.constants import Send
from langchain_core.messages import SystemMessage, HumanMessage
from llm_instrumentation import get_llm, HAS_OLLAMA
from prompt_config import get_system_prompt_or_none


class ScalabilityState(TypedDict):
    """State for scalability tests."""
    results: Annotated[List[str], operator.add]  # Aggregates concurrent updates
    final: str


def get_llm_instance():
    """Get the configured LLM instance."""
    return get_llm()


TASK_PROMPTS = [
    "Provide a brief fact about science",     # task_0
    "Provide a brief fact about history",     # task_1
    "Provide a brief fact about geography",   # task_2
    "Provide a brief fact about technology",  # task_3
    "Provide a brief fact about art",         # task_4
    "Provide a brief fact about music",       # task_5
    "Provide a brief fact about sports",      # task_6
    "Provide a brief fact about nature",      # task_7
]


def create_task_node(prompt: str):
    """Create a task node that makes a real LLM call."""
    def task_fn(state: ScalabilityState) -> dict:
        llm = get_llm_instance()

        messages = []
        system_prompt = get_system_prompt_or_none("ask")
        if system_prompt:
            messages.append(SystemMessage(content=system_prompt))
        messages.append(HumanMessage(content=prompt))
        response = llm.invoke(messages)
        return {"results": [response.content]}

    return task_fn


def merge_results(state: ScalabilityState) -> dict:
    """Merge all parallel results."""
    return {"final": "\n".join(state["results"])}


def build_parallel_graph(n: int) -> StateGraph:
    """Build a graph with N parallel branches.

    Requires explicit Send API - more verbose than A-PXM's automatic parallelism.
    """
    builder = StateGraph(ScalabilityState)

    # Fan-out node
    def fan_out(state: ScalabilityState) -> List[Send]:
        return [Send(f"task_{i}", state) for i in range(n)]

    # Add task nodes
    for i in range(n):
        builder.add_node(f"task_{i}", create_task_node(TASK_PROMPTS[i]))

    # Add merge node
    builder.add_node("merge", merge_results)

    # Parallel: START -> fan_out -> [task_0, task_1, ...] -> merge -> END
    builder.add_conditional_edges(START, fan_out)
    for i in range(n):
        builder.add_edge(f"task_{i}", "merge")
    builder.add_edge("merge", END)

    return builder.compile()


# Pre-build graphs for common sizes
graph_2 = build_parallel_graph(2)
graph_4 = build_parallel_graph(4)
graph_8 = build_parallel_graph(8)


def run(n: int = 4) -> dict:
    """Execute scalability test with N parallel operations."""
    graph = {2: graph_2, 4: graph_4, 8: graph_8}.get(n, build_parallel_graph(n))

    initial_state = {
        "results": [],
        "final": "",
    }
    return graph.invoke(initial_state)


if __name__ == "__main__":
    for n in [2, 4, 8]:
        print(f"\nN={n} Parallel Result")
        result = run(n)
        print(result["final"])
