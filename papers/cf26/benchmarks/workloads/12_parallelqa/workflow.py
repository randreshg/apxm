"""
ParallelQA Parallel Question Answering - LangGraph Implementation

For comparison with A-PXM workflow.ais
Demonstrates parallel sub-question handling.
"""

from typing import TypedDict, List
from langgraph.graph import StateGraph, START, END
from langgraph.constants import Send
from llm_instrumentation import get_llm, HAS_OLLAMA


class ParallelQAState(TypedDict):
    """State for ParallelQA workflow."""
    question: str
    decomposition: str
    sub_answer1: str
    sub_answer2: str
    combined_answers: str
    answer: str


def get_llm_instance():
    """Get the configured LLM instance."""
    return get_llm()


def decompose_question(state: ParallelQAState) -> dict:
    """Decompose question into parallel sub-questions."""
    llm = get_llm_instance()
    question = state["question"]
    
    prompt = f"Break down this question into independent sub-questions that can be answered in parallel: {question}"
    response = llm.invoke(prompt)
    return {"decomposition": response.content}


def answer_sub1(state: ParallelQAState) -> dict:
    """Answer first sub-question."""
    llm = get_llm_instance()
    decomposition = state["decomposition"]
    
    prompt = f"Answer the first sub-question from: {decomposition}"
    response = llm.invoke(prompt)
    return {"sub_answer1": response.content}


def answer_sub2(state: ParallelQAState) -> dict:
    """Answer second sub-question."""
    llm = get_llm_instance()
    decomposition = state["decomposition"]
    
    prompt = f"Answer the second sub-question from: {decomposition}"
    response = llm.invoke(prompt)
    return {"sub_answer2": response.content}


def combine_answers(state: ParallelQAState) -> dict:
    """Combine sub-answers."""
    combined = f"{state['sub_answer1']}\n{state['sub_answer2']}"
    return {"combined_answers": combined}


def synthesize_answer(state: ParallelQAState) -> dict:
    """Synthesize final answer from combined sub-answers."""
    llm = get_llm_instance()
    question = state["question"]
    combined = state["combined_answers"]
    
    prompt = (
        f"Given the original question: {question}\n"
        f"And the sub-answers: {combined}\n"
        "Provide the final answer. Be concise."
    )
    response = llm.invoke(prompt)
    return {"answer": response.content}


def fan_out_subquestions(state: ParallelQAState) -> List[Send]:
    """Fan out to parallel sub-question answering."""
    return [
        Send("answer_sub1", state),
        Send("answer_sub2", state),
    ]


def build_graph() -> StateGraph:
    """Build the ParallelQA graph."""
    builder = StateGraph(ParallelQAState)
    
    # Add nodes
    builder.add_node("decompose_question", decompose_question)
    builder.add_node("answer_sub1", answer_sub1)
    builder.add_node("answer_sub2", answer_sub2)
    builder.add_node("combine_answers", combine_answers)
    builder.add_node("synthesize_answer", synthesize_answer)
    
    # Flow: decompose -> parallel sub-answers -> combine -> synthesize
    builder.add_edge(START, "decompose_question")
    builder.add_conditional_edges("decompose_question", fan_out_subquestions)
    builder.add_edge("answer_sub1", "combine_answers")
    builder.add_edge("answer_sub2", "combine_answers")
    builder.add_edge("combine_answers", "synthesize_answer")
    builder.add_edge("synthesize_answer", END)
    
    return builder.compile()


# Export graph for runner
graph = build_graph()
