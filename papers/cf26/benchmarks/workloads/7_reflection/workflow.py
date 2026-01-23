"""
Reflection Workflow - LangGraph Implementation

Demonstrates self-critique and improvement loop.
Compare with workflow.ais which has native reflect operation.
"""

from typing import TypedDict
from langgraph.graph import StateGraph, START, END
from langchain_core.messages import SystemMessage, HumanMessage
from llm_instrumentation import get_llm, HAS_OLLAMA
from prompt_config import get_system_prompt_or_none


class ReflectionState(TypedDict):
    """State for reflection workflow."""
    task: str
    initial_answer: str
    reflection: str
    improved_answer: str


def get_llm_instance():
    """Get the configured LLM instance."""
    return get_llm()


def initial_attempt(state: ReflectionState) -> dict:
    """Make initial attempt at solving the task."""
    llm = get_llm_instance()
    task = state["task"]

    messages = []
    system_prompt = get_system_prompt_or_none("ask")
    if system_prompt:
        messages.append(SystemMessage(content=system_prompt))
    messages.append(HumanMessage(content="Solve this task: " + task))
    response = llm.invoke(messages)
    return {"initial_answer": response.content}


def reflect(state: ReflectionState) -> dict:
    """Reflect on the initial answer.

    Note: LangGraph has no native reflection operation.
    Must use custom prompting.
    """
    llm = get_llm_instance()
    answer = state["initial_answer"]

    messages = []
    # Use "reflect" system prompt for reflection operation
    system_prompt = get_system_prompt_or_none("reflect")
    if system_prompt:
        messages.append(SystemMessage(content=system_prompt))
    messages.append(HumanMessage(content=answer))
    response = llm.invoke(messages)
    return {"reflection": response.content}


def improve(state: ReflectionState) -> dict:
    """Improve the answer based on reflection."""
    llm = get_llm_instance()
    task = state["task"]
    reflection = state["reflection"]

    messages = []
    system_prompt = get_system_prompt_or_none("ask")
    if system_prompt:
        messages.append(SystemMessage(content=system_prompt))
    messages.append(
        HumanMessage(
            content="Given this feedback: " + reflection + ", improve the answer to: " + task
        )
    )
    response = llm.invoke(messages)
    return {"improved_answer": response.content}


def build_graph() -> StateGraph:
    """Build the reflection graph.

    Note: LangGraph requires:
    - Custom prompts for reflection (no native REFL)
    - Manual state management for improvement loop
    - No structured reflection output format
    """
    builder = StateGraph(ReflectionState)

    # Add nodes
    builder.add_node("initial", initial_attempt)
    builder.add_node("reflect", reflect)
    builder.add_node("improve", improve)

    # Sequential flow
    builder.add_edge(START, "initial")
    builder.add_edge("initial", "reflect")
    builder.add_edge("reflect", "improve")
    builder.add_edge("improve", END)

    return builder.compile()


# Compile the graph
graph = build_graph()


def run(task: str = "Explain the concept of recursion in programming") -> dict:
    """Execute the reflection workflow."""
    initial_state = {
        "task": task,
        "initial_answer": "",
        "reflection": "",
        "improved_answer": "",
    }
    return graph.invoke(initial_state)


if __name__ == "__main__":
    result = run()
    print(f"Task: {result['task']}")
    print(f"\nInitial: {result['initial_answer'][:200]}...")
    print(f"\nReflection: {result['reflection'][:200]}...")
    print(f"\nImproved: {result['improved_answer'][:200]}...")
