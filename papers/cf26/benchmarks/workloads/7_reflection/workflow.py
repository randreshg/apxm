"""
Reflection Workflow - LangGraph Implementation

Demonstrates self-critique and improvement loop.
Compare with workflow.ais which has native reflect operation.
"""

import os
from typing import TypedDict
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


class ReflectionState(TypedDict):
    """State for reflection workflow."""
    task: str
    initial_answer: str
    reflection: str
    improved_answer: str


def get_llm():
    """Get the LLM instance (Ollama or mock)."""
    if HAS_OLLAMA:
        return ChatOllama(model=OLLAMA_MODEL, temperature=0)
    return None


def initial_attempt(state: ReflectionState) -> dict:
    """Make initial attempt at solving the task."""
    llm = get_llm()
    task = state["task"]

    if llm:
        prompt = f"Solve this task concisely:\n\n{task}"
        response = llm.invoke(prompt)
        answer = response.content
    else:
        answer = f"Initial solution for: {task}"

    return {"initial_answer": answer}


def reflect(state: ReflectionState) -> dict:
    """Reflect on the initial answer.

    Note: LangGraph has no native reflection operation.
    Must use custom prompting.
    """
    llm = get_llm()
    answer = state["initial_answer"]

    if llm:
        prompt = f"""Critically analyze this answer and identify improvements:

Answer: {answer}

Provide specific, actionable feedback for improvement."""
        response = llm.invoke(prompt)
        reflection = response.content
    else:
        reflection = f"Reflection: The answer could be more detailed."

    return {"reflection": reflection}


def improve(state: ReflectionState) -> dict:
    """Improve the answer based on reflection."""
    llm = get_llm()
    task = state["task"]
    reflection = state["reflection"]

    if llm:
        prompt = f"""Given this feedback:
{reflection}

Provide an improved answer to the original task:
{task}"""
        response = llm.invoke(prompt)
        improved = response.content
    else:
        improved = f"Improved solution incorporating: {reflection}"

    return {"improved_answer": improved}


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
