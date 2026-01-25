"""
Planning Workflow - LangGraph Implementation

Demonstrates multi-step task decomposition and execution.
Compare with workflow.ais which has native PLAN operation.
"""

from typing import TypedDict, Optional
from langgraph.graph import StateGraph, START, END
from langgraph.constants import Send
from langchain_core.messages import SystemMessage, HumanMessage
from llm_instrumentation import get_llm, HAS_OLLAMA
from prompt_config import get_system_prompt_or_none


class PlanningState(TypedDict):
    """State for planning workflow."""
    goal: str
    plan_result: str
    step1: str
    step2: str
    step3: str
    combined: str
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
    return {"plan_result": response.content}


def _run_step(llm, system_prompt: Optional[str], prompt: str, context: str) -> str:
    messages = []
    if system_prompt:
        messages.append(SystemMessage(content=system_prompt))
    # Mirror AIS-style `ask(prompt, plan_result)` by passing context separately.
    messages.append(HumanMessage(content=prompt))
    messages.append(HumanMessage(content=context))
    response = llm.invoke(messages)
    return response.content


def execute_step1(state: PlanningState) -> dict:
    """Execute step 1."""
    llm = get_llm_instance()
    plan_result = state["plan_result"]
    system_prompt = get_system_prompt_or_none("ask")
    prompt = "Execute step 1 - create detailed design:"
    return {"step1": _run_step(llm, system_prompt, prompt, plan_result)}


def execute_step2(state: PlanningState) -> dict:
    """Execute step 2."""
    llm = get_llm_instance()
    plan_result = state["plan_result"]
    system_prompt = get_system_prompt_or_none("ask")
    prompt = "Execute step 2 - implement core features:"
    return {"step2": _run_step(llm, system_prompt, prompt, plan_result)}


def execute_step3(state: PlanningState) -> dict:
    """Execute step 3."""
    llm = get_llm_instance()
    plan_result = state["plan_result"]
    system_prompt = get_system_prompt_or_none("ask")
    prompt = "Execute step 3 - testing and refinement:"
    return {"step3": _run_step(llm, system_prompt, prompt, plan_result)}


def fan_out_steps(state: PlanningState) -> list[Send]:
    """Fan out step execution in parallel."""
    return [
        Send("step1", state),
        Send("step2", state),
        Send("step3", state),
    ]


def merge_results(state: PlanningState) -> dict:
    """Merge step results."""
    combined = "\n".join([state["step1"], state["step2"], state["step3"]])
    return {"combined": combined}


def synthesize(state: PlanningState) -> dict:
    """Synthesize final result from step results."""
    llm = get_llm_instance()
    combined = state["combined"]
    messages = []
    system_prompt = get_system_prompt_or_none("ask")
    if system_prompt:
        messages.append(SystemMessage(content=system_prompt))
    # Mirror AIS-style `ask(prompt, combined)` by passing context separately.
    messages.append(HumanMessage(content="Synthesize these step results into a final deliverable:"))
    messages.append(HumanMessage(content=combined))
    response = llm.invoke(messages)
    return {"final_result": response.content}


def build_graph() -> StateGraph:
    """Build the planning graph.

    Note: LangGraph requires:
    - Custom prompts for planning (no native PLAN)
    - Manual Send API for parallel step execution
    - Manual result aggregation
    """
    builder = StateGraph(PlanningState)

    # Add nodes
    builder.add_node("plan", create_plan)
    builder.add_node("step1", execute_step1)
    builder.add_node("step2", execute_step2)
    builder.add_node("step3", execute_step3)
    builder.add_node("merge", merge_results)
    builder.add_node("synthesize", synthesize)

    # Plan -> parallel steps -> merge -> synthesize
    builder.add_edge(START, "plan")
    builder.add_conditional_edges("plan", fan_out_steps)
    builder.add_edge("step1", "merge")
    builder.add_edge("step2", "merge")
    builder.add_edge("step3", "merge")
    builder.add_edge("merge", "synthesize")
    builder.add_edge("synthesize", END)

    return builder.compile()


# Compile the graph
graph = build_graph()


def run(goal: str = "Build a simple web application") -> dict:
    """Execute the planning workflow."""
    initial_state = {
        "goal": goal,
        "plan_result": "",
        "step1": "",
        "step2": "",
        "step3": "",
        "combined": "",
        "final_result": "",
    }
    return graph.invoke(initial_state)


if __name__ == "__main__":
    result = run()
    print(f"Goal: {result['goal']}")
    print("\nPlan:")
    print(result["plan_result"][:200] + "...")
    print("\nStep Results:")
    print(f"  1. {result['step1'][:200]}...")
    print(f"  2. {result['step2'][:200]}...")
    print(f"  3. {result['step3'][:200]}...")
    print(f"\nFinal: {result['final_result'][:200]}...")
