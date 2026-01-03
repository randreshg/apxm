"""
Tool Invocation Workflow - LangGraph Implementation

Demonstrates tool/function calling with LangChain tools.
Compare with workflow.ais which has native capability system.
"""

from typing import TypedDict, Any
from langgraph.graph import StateGraph, START, END
from llm_instrumentation import get_llm, HAS_OLLAMA


class ToolState(TypedDict):
    """State for tool invocation workflow."""
    query: str
    tool_decision: str
    search_results: str
    answer: str


def get_llm_instance():
    """Get the configured LLM instance."""
    return get_llm()


def mock_search(query: str) -> str:
    """Mock search tool implementation."""
    return f"Search results for '{query}': Found 3 relevant documents about the topic."


def decide_tool(state: ToolState) -> dict:
    """Reason about which tool to use."""
    llm = get_llm_instance()
    query = state["query"]

    prompt = f"Analyze this query and decide which tool is needed. Just respond with a brief tool recommendation.\n\nQuery: {query}"
    response = llm.invoke(prompt)
    return {"tool_decision": response.content}


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

    prompt = f"Given these search results:\n{results}\n\nAnswer this query concisely: {query}"
    response = llm.invoke(prompt)
    return {"answer": response.content}


def build_graph() -> StateGraph:
    """Build the tool invocation graph.

    Note: LangGraph requires:
    - Explicit tool binding at runtime
    - Manual state management
    - No compile-time tool validation
    """
    builder = StateGraph(ToolState)

    # Add nodes
    builder.add_node("decide_tool", decide_tool)
    builder.add_node("invoke_tool", invoke_tool)
    builder.add_node("synthesize", synthesize_answer)

    # Sequential flow
    builder.add_edge(START, "decide_tool")
    builder.add_edge("decide_tool", "invoke_tool")
    builder.add_edge("invoke_tool", "synthesize")
    builder.add_edge("synthesize", END)

    return builder.compile()


# Compile the graph
graph = build_graph()


def run(query: str = "What is quantum computing?") -> dict:
    """Execute the tool invocation workflow."""
    initial_state = {
        "query": query,
        "tool_decision": "",
        "search_results": "",
        "answer": "",
    }
    return graph.invoke(initial_state)


if __name__ == "__main__":
    result = run("What is quantum computing?")
    print(f"Tool decision: {result['tool_decision']}")
    print(f"Search results: {result['search_results']}")
    print(f"Answer: {result['answer']}")
