"""
Conditional Routing Workflow - LangGraph Implementation

Demonstrates dynamic routing based on LLM classification.
Compare with workflow.ais which uses dataflow-based parallel preparation.
A-PXM automatically parallelizes independent response preparation.
"""

import os
from typing import TypedDict, Literal
from langgraph.graph import StateGraph, START, END
from llm_instrumentation import get_ollama_llm, HAS_OLLAMA

# Ollama model configuration
OLLAMA_MODEL = (
    os.environ.get("APXM_BENCH_OLLAMA_MODEL")
    or os.environ.get("OLLAMA_MODEL")
    or "phi3:mini"
)


class RoutingState(TypedDict):
    """State for conditional routing workflow."""
    input: str
    category: str
    response: str


def get_llm():
    """Get the LLM instance (Ollama only)."""
    return get_ollama_llm(OLLAMA_MODEL)


def classify_input(state: RoutingState) -> dict:
    """Classify input into a category."""
    llm = get_llm()
    user_input = state["input"]

    prompt = f"""Classify this input into exactly one category.
Categories: technical, creative, factual

Input: {user_input}

Respond with only the category name (technical, creative, or factual)."""
    response = llm.invoke(prompt)
    category = response.content.strip().lower()
    # Normalize category
    if "technical" in category:
        category = "technical"
    elif "creative" in category:
        category = "creative"
    elif "factual" in category:
        category = "factual"
    else:
        category = "general"

    return {"category": category}


def route_by_category(state: RoutingState) -> str:
    """Route to appropriate handler based on category.

    Note: LangGraph uses runtime conditional edges.
    No compile-time validation of routes.
    """
    category = state["category"]
    if category == "technical":
        return "technical_response"
    elif category == "creative":
        return "creative_response"
    elif category == "factual":
        return "factual_response"
    else:
        return "general_response"


def technical_response(state: RoutingState) -> dict:
    """Handle technical queries."""
    llm = get_llm()
    user_input = state["input"]

    prompt = f"Provide a detailed technical explanation for: {user_input}"
    response = llm.invoke(prompt)
    return {"response": response.content}


def creative_response(state: RoutingState) -> dict:
    """Handle creative queries."""
    llm = get_llm()
    user_input = state["input"]

    prompt = f"Provide a creative, imaginative response for: {user_input}"
    response = llm.invoke(prompt)
    return {"response": response.content}


def factual_response(state: RoutingState) -> dict:
    """Handle factual queries."""
    llm = get_llm()
    user_input = state["input"]

    prompt = f"Provide accurate, factual information for: {user_input}"
    response = llm.invoke(prompt)
    return {"response": response.content}


def general_response(state: RoutingState) -> dict:
    """Handle general/default queries."""
    llm = get_llm()
    user_input = state["input"]

    prompt = f"Provide a helpful response for: {user_input}"
    response = llm.invoke(prompt)
    return {"response": response.content}


def build_graph() -> StateGraph:
    """Build the conditional routing graph.

    Note: LangGraph requires:
    - Runtime conditional edges (not compile-time)
    - Manual route function
    - No static route validation
    """
    builder = StateGraph(RoutingState)

    # Add nodes
    builder.add_node("classify", classify_input)
    builder.add_node("technical_response", technical_response)
    builder.add_node("creative_response", creative_response)
    builder.add_node("factual_response", factual_response)
    builder.add_node("general_response", general_response)

    # Entry point
    builder.add_edge(START, "classify")

    # Conditional routing
    builder.add_conditional_edges(
        "classify",
        route_by_category,
        {
            "technical_response": "technical_response",
            "creative_response": "creative_response",
            "factual_response": "factual_response",
            "general_response": "general_response",
        }
    )

    # All routes lead to END
    builder.add_edge("technical_response", END)
    builder.add_edge("creative_response", END)
    builder.add_edge("factual_response", END)
    builder.add_edge("general_response", END)

    return builder.compile()


# Compile the graph
graph = build_graph()


def run(user_input: str = "How does a neural network work?") -> dict:
    """Execute the conditional routing workflow."""
    initial_state = {
        "input": user_input,
        "category": "",
        "response": "",
    }
    return graph.invoke(initial_state)


if __name__ == "__main__":
    result = run()
    print(f"Input: {result['input']}")
    print(f"Category: {result['category']}")
    print(f"Response: {result['response'][:200]}...")
