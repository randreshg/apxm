"""
Planning Workflow - LangGraph Implementation

Demonstrates multi-step task decomposition and execution.
Compare with workflow.ais which has native PLAN operation.
"""

from typing import TypedDict, List
from langgraph.graph import StateGraph, START, END
from langchain_core.messages import SystemMessage, HumanMessage
from llm_instrumentation import get_llm, HAS_OLLAMA
from prompt_config import get_system_prompt_or_none


class PlanningState(TypedDict):
    """State for planning workflow."""
    goal: str
    steps: List[str]
    step_results: List[str]
    final_result: str


def get_llm_instance():
    """Get the configured LLM instance."""
    return get_llm()


def create_plan(state: PlanningState) -> dict:
    """Decompose goal into steps.

    Note: LangGraph has no native PLAN operation.
    Must use chain-of-thought prompting.
    """
    llm = get_llm_instance()
    goal = state["goal"]

    messages = []
    # Use "plan" system prompt for planning operation
    system_prompt = get_system_prompt_or_none("plan")
    if system_prompt:
        messages.append(SystemMessage(content=system_prompt))
    messages.append(HumanMessage(content=goal))
    response = llm.invoke(messages)
    # Parse steps from response
    lines = response.content.strip().split('\n')
    steps = [line.strip() for line in lines if line.strip()][:3]

    return {"steps": steps}


STEP_PROMPTS = [
    "Execute step 1 - create detailed design:",
    "Execute step 2 - implement core features:",
    "Execute step 3 - testing and refinement:",
]


def execute_steps(state: PlanningState) -> dict:
    """Execute each step and collect results.

    Note: LangGraph cannot automatically parallelize steps.
    This runs sequentially unless using Send API.
    """
    llm = get_llm_instance()
    steps = state["steps"]
    plan_context = "\n".join(steps) if steps else ""
    results = []

    system_prompt = get_system_prompt_or_none("ask")
    for i in range(min(3, len(STEP_PROMPTS))):
        messages = []
        if system_prompt:
            messages.append(SystemMessage(content=system_prompt))
        prompt = f"{STEP_PROMPTS[i]} {plan_context}"
        messages.append(HumanMessage(content=prompt))
        response = llm.invoke(messages)
        results.append(response.content)

    return {"step_results": results}


def synthesize(state: PlanningState) -> dict:
    """Synthesize final result from step results."""
    llm = get_llm_instance()
    results = state["step_results"]

    combined = " ".join(results)
    messages = []
    system_prompt = get_system_prompt_or_none("ask")
    if system_prompt:
        messages.append(SystemMessage(content=system_prompt))
    messages.append(HumanMessage(content=f"Synthesize these step results into a final deliverable: {combined}"))
    response = llm.invoke(messages)
    return {"final_result": response.content}


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
