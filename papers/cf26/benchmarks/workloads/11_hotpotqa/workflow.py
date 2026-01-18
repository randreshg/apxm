"""
HotpotQA Multi-hop Question Answering - LangGraph Implementation

For comparison with A-PXM workflow.ais
"""

from typing import TypedDict
from langgraph.graph import StateGraph, START, END
from llm_instrumentation import get_llm, HAS_OLLAMA


class HotpotQAState(TypedDict):
    """State for HotpotQA workflow."""
    question: str
    analysis: str
    info_needed: str
    answer: str


def get_llm_instance():
    """Get the configured LLM instance."""
    return get_llm()


def analyze_question(state: HotpotQAState) -> dict:
    """Analyze question to identify entities and reasoning steps."""
    llm = get_llm_instance()
    question = state["question"]
    
    prompt = f"Analyze this question and identify the key entities and reasoning steps needed: {question}"
    response = llm.invoke(prompt)
    return {"analysis": response.content}


def gather_info(state: HotpotQAState) -> dict:
    """Simulate gathering information about entities."""
    llm = get_llm_instance()
    question = state["question"]
    
    prompt = (
        f"Based on the question '{question}', what information do we need to gather? "
        "Provide a brief summary of what facts we need to find."
    )
    response = llm.invoke(prompt)
    return {"info_needed": response.content}


def synthesize_answer(state: HotpotQAState) -> dict:
    """Reason about gathered information to answer."""
    llm = get_llm_instance()
    question = state["question"]
    info_needed = state["info_needed"]
    
    prompt = (
        f"Given the question: {question}\n"
        f"And the information needed: {info_needed}\n"
        "Provide a concise answer. The answer should be short (a few words or yes/no)."
    )
    response = llm.invoke(prompt)
    return {"answer": response.content}


def build_graph() -> StateGraph:
    """Build the HotpotQA graph."""
    builder = StateGraph(HotpotQAState)
    
    # Add nodes
    builder.add_node("analyze_question", analyze_question)
    builder.add_node("gather_info", gather_info)
    builder.add_node("synthesize_answer", synthesize_answer)
    
    # Sequential flow
    builder.add_edge(START, "analyze_question")
    builder.add_edge("analyze_question", "gather_info")
    builder.add_edge("gather_info", "synthesize_answer")
    builder.add_edge("synthesize_answer", END)
    
    return builder.compile()


# Export graph for runner
graph = build_graph()
