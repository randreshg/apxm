"""
Tool Invocation Workflow - LangGraph Implementation

Demonstrates tool/function calling with LangChain tools.
Compare with workflow.ais which has native capability system.
"""

from typing import TypedDict, Any
from langgraph.graph import StateGraph, START, END
from langchain_core.messages import SystemMessage, HumanMessage
from llm_instrumentation import get_llm, HAS_OLLAMA
from prompt_config import get_system_prompt_or_none


class ToolState(TypedDict):
    """State for tool invocation workflow."""
    query: str
    search_results: str
    answer: str


def get_llm_instance():
    """Get the configured LLM instance."""
    return get_llm()


def mock_search(query: str) -> str:
    """Mock search tool implementation."""
    return f"Search results for '{query}': Found 3 relevant documents about the topic."


def invoke_tool(state: ToolState) -> dict:
    """Execute the tool (search)."""
    # Simulate tool invocation
    search_results = mock_search(state["query"])
    return {"search_results": search_results}


def synthesize_answer(state: ToolState) -> dict:
    """Synthesize final answer from tool results."""
    llm = get_llm_instance()
    query = state["query"]
    results = state["search_results"]

    messages = []
    system_prompt = get_system_prompt_or_none("ask")
    if system_prompt:
        messages.append(SystemMessage(content=system_prompt))
    messages.append(HumanMessage(content=f"Given search results: {results}, answer this query: {query}"))
    response = llm.invoke(messages)
    return {"answer": response.content}


def build_graph() -> StateGraph:
    """Build the tool invocation graph.

    Note: LangGraph requires:
    - Explicit tool binding at runtime
    - Manual state management
    - No compile-time tool validation

    """
    builder = StateGraph(ToolState)

    builder.add_node("invoke_tool", invoke_tool)
    builder.add_node("synthesize", synthesize_answer)

    builder.add_edge(START, "invoke_tool")
    builder.add_edge("invoke_tool", "synthesize")
    builder.add_edge("synthesize", END)

    return builder.compile()


# Compile the graph
graph = build_graph()


def run(query: str = "What is quantum computing?") -> dict:
    """Execute the tool invocation workflow."""
    initial_state = {
        "query": query,
        "search_results": "",
        "answer": "",
    }
    return graph.invoke(initial_state)


if __name__ == "__main__":
    result = run("What is quantum computing?")
    print(f"Search results: {result['search_results']}")
    print(f"Answer: {result['answer']}")
