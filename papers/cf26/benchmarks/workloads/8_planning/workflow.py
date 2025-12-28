"""
Planning Workflow - LangGraph Implementation

Demonstrates multi-step task decomposition and execution.
Compare with workflow.ais which has native PLAN operation.
"""

from typing import TypedDict, List
from langgraph.graph import StateGraph, START, END

# Try to import Ollama for real LLM calls
try:
    from langchain_ollama import ChatOllama
    HAS_OLLAMA = True
except ImportError:
    HAS_OLLAMA = False

# Ollama model configuration
OLLAMA_MODEL = "phi3:mini"


class PlanningState(TypedDict):
    """State for planning workflow."""
    goal: str
    steps: List[str]
    step_results: List[str]
    final_result: str


def get_llm():
    """Get the LLM instance (Ollama or mock)."""
    if HAS_OLLAMA:
        return ChatOllama(model=OLLAMA_MODEL, temperature=0)
    return None


def create_plan(state: PlanningState) -> dict:
    """Decompose goal into steps.

    Note: LangGraph has no native PLAN operation.
    Must use chain-of-thought prompting.
    """
    llm = get_llm()
    goal = state["goal"]

    if llm:
        prompt = f"""Break down this goal into 3 concrete, actionable steps:

Goal: {goal}

Respond with exactly 3 steps, one per line, numbered 1-3."""
        response = llm.invoke(prompt)
        # Parse steps from response
        lines = response.content.strip().split('\n')
        steps = [line.strip() for line in lines if line.strip()][:3]
    else:
        steps = [
            f"Step 1: Research {goal}",
            f"Step 2: Analyze {goal}",
            f"Step 3: Synthesize {goal}",
        ]

    return {"steps": steps}


def execute_steps(state: PlanningState) -> dict:
    """Execute each step and collect results.

    Note: LangGraph cannot automatically parallelize steps.
    This runs sequentially unless using Send API.
    """
    llm = get_llm()
    steps = state["steps"]
    results = []

    for i, step in enumerate(steps):
        if llm:
            prompt = f"Execute this step concisely (1-2 sentences):\n\n{step}"
            response = llm.invoke(prompt)
            results.append(response.content)
        else:
            results.append(f"Result for {step}")

    return {"step_results": results}


def synthesize(state: PlanningState) -> dict:
    """Synthesize final result from step results."""
    llm = get_llm()
    goal = state["goal"]
    results = state["step_results"]

    if llm:
        results_text = "\n".join([f"- {r}" for r in results])
        prompt = f"""Given these step results:
{results_text}

Provide a concise final answer for the original goal:
{goal}"""
        response = llm.invoke(prompt)
        final = response.content
    else:
        final = f"Final synthesis of {len(results)} steps for: {goal}"

    return {"final_result": final}


def build_graph() -> StateGraph:
    """Build the planning graph.

    Note: LangGraph requires:
    - Custom prompts for planning (no native PLAN)
    - Sequential step execution (manual Send API for parallel)
    - Manual result aggregation
    """
    builder = StateGraph(PlanningState)

    # Add nodes
    builder.add_node("plan", create_plan)
    builder.add_node("execute", execute_steps)
    builder.add_node("synthesize", synthesize)

    # Sequential flow
    builder.add_edge(START, "plan")
    builder.add_edge("plan", "execute")
    builder.add_edge("execute", "synthesize")
    builder.add_edge("synthesize", END)

    return builder.compile()


# Compile the graph
graph = build_graph()


def run(goal: str = "Build a simple web application") -> dict:
    """Execute the planning workflow."""
    initial_state = {
        "goal": goal,
        "steps": [],
        "step_results": [],
        "final_result": "",
    }
    return graph.invoke(initial_state)


if __name__ == "__main__":
    result = run()
    print(f"Goal: {result['goal']}")
    print(f"\nSteps:")
    for i, step in enumerate(result['steps'], 1):
        print(f"  {i}. {step}")
    print(f"\nFinal: {result['final_result'][:200]}...")
