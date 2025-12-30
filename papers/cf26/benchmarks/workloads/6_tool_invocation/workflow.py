"""
Tool Invocation Workflow - LangGraph Implementation

Demonstrates tool/function calling with LangChain tools.
Compare with workflow.ais which has native capability system.
"""

import os
from typing import TypedDict, Any
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


class ToolState(TypedDict):
    """State for tool invocation workflow."""
    query: str
    tool_decision: str
    search_results: str
    answer: str


def get_llm():
    """Get the LLM instance (Ollama or mock)."""
    if HAS_OLLAMA:
        return ChatOllama(model=OLLAMA_MODEL, temperature=0)
    return None


def mock_search(query: str) -> str:
    """Mock search tool implementation."""
    return f"Search results for '{query}': Found 3 relevant documents about the topic."


def decide_tool(state: ToolState) -> dict:
    """Reason about which tool to use."""
    llm = get_llm()
    query = state["query"]

    if llm:
        prompt = f"Analyze this query and decide which tool is needed. Just respond with a brief tool recommendation.\n\nQuery: {query}"
        response = llm.invoke(prompt)
        tool_decision = response.content
    else:
        tool_decision = f"Need search tool for: {query}"

    return {"tool_decision": tool_decision}


def invoke_tool(state: ToolState) -> dict:
    """Execute the tool (search)."""
    # Simulate tool invocation
    search_results = mock_search(state["query"])
    return {"search_results": search_results}


def synthesize_answer(state: ToolState) -> dict:
    """Synthesize final answer from tool results."""
    llm = get_llm()
    query = state["query"]
    results = state["search_results"]

    if llm:
        prompt = f"Given these search results:\n{results}\n\nAnswer this query concisely: {query}"
        response = llm.invoke(prompt)
        answer = response.content
    else:
        answer = f"Based on {results}, the answer to '{query}' is synthesized."

    return {"answer": answer}


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
