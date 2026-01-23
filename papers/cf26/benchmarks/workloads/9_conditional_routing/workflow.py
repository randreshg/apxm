"""
Conditional Routing Workflow - LangGraph Implementation

Demonstrates dynamic routing based on LLM classification.
Compare with workflow.ais which uses dataflow-based parallel preparation.
A-PXM automatically parallelizes independent response preparation.
"""

from typing import TypedDict, Literal
from langgraph.graph import StateGraph, START, END
from langchain_core.messages import SystemMessage, HumanMessage
from llm_instrumentation import get_llm, HAS_OLLAMA
from prompt_config import get_system_prompt_or_none


class RoutingState(TypedDict):
    """State for conditional routing workflow."""
    input: str
    category: str
    response: str
    output: str


def get_llm_instance():
    """Get the configured LLM instance."""
    return get_llm()


def classify_input(state: RoutingState) -> dict:
    """Classify input into a category."""
    llm = get_llm_instance()
    user_input = state["input"]

    messages = []
    system_prompt = get_system_prompt_or_none("ask")
    if system_prompt:
        messages.append(SystemMessage(content=system_prompt))
    messages.append(
        HumanMessage(
            content="Classify this input into exactly one word: technical, creative, or factual. Input: "
            + user_input
        )
    )
    response = llm.invoke(messages)
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
    llm = get_llm_instance()
    user_input = state["input"]

    messages = []
    system_prompt = get_system_prompt_or_none("ask")
    if system_prompt:
        messages.append(SystemMessage(content=system_prompt))
    messages.append(
        HumanMessage(content="Provide a detailed technical explanation for: " + user_input)
    )
    response = llm.invoke(messages)
    return {"response": response.content}


def creative_response(state: RoutingState) -> dict:
    """Handle creative queries."""
    llm = get_llm_instance()
    user_input = state["input"]

    messages = []
    system_prompt = get_system_prompt_or_none("ask")
    if system_prompt:
        messages.append(SystemMessage(content=system_prompt))
    messages.append(
        HumanMessage(content="Provide a creative, imaginative response for: " + user_input)
    )
    response = llm.invoke(messages)
    return {"response": response.content}


def factual_response(state: RoutingState) -> dict:
    """Handle factual queries."""
    llm = get_llm_instance()
    user_input = state["input"]

    messages = []
    system_prompt = get_system_prompt_or_none("ask")
    if system_prompt:
        messages.append(SystemMessage(content=system_prompt))
    messages.append(HumanMessage(content="Provide accurate factual information for: " + user_input))
    response = llm.invoke(messages)
    return {"response": response.content}


def general_response(state: RoutingState) -> dict:
    """Handle general/default queries."""
    llm = get_llm_instance()
    user_input = state["input"]

    messages = []
    system_prompt = get_system_prompt_or_none("ask")
    if system_prompt:
        messages.append(SystemMessage(content=system_prompt))
    messages.append(HumanMessage(content="Provide a helpful general response for: " + user_input))
    response = llm.invoke(messages)
    return {"response": response.content}


def refine_response(state: RoutingState) -> dict:
    """Refine and summarize the routed response."""
    llm = get_llm_instance()
    routed_response = state["response"]

    messages = []
    system_prompt = get_system_prompt_or_none("ask")
    if system_prompt:
        messages.append(SystemMessage(content=system_prompt))
    messages.append(HumanMessage(content="Refine and summarize this response: " + routed_response))
    response = llm.invoke(messages)
    return {"output": response.content}


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
    builder.add_node("refine", refine_response)

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

    # All routes lead to refine step
    builder.add_edge("technical_response", "refine")
    builder.add_edge("creative_response", "refine")
    builder.add_edge("factual_response", "refine")
    builder.add_edge("general_response", "refine")

    # Refine leads to END
    builder.add_edge("refine", END)

    return builder.compile()


# Compile the graph
graph = build_graph()


def run(user_input: str = "How does a neural network work?") -> dict:
    """Execute the conditional routing workflow."""
    initial_state = {
        "input": user_input,
        "category": "",
        "response": "",
        "output": "",
    }
    return graph.invoke(initial_state)


if __name__ == "__main__":
    result = run()
    print(f"Input: {result['input']}")
    print(f"Category: {result['category']}")
    print(f"Output: {result['output'][:200]}...")
