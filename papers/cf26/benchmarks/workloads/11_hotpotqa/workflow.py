"""
HotpotQA Multi-hop Question Answering - LangGraph Implementation

For comparison with A-PXM workflow.ais
"""

from typing import TypedDict
from langgraph.graph import StateGraph, START, END
from langchain_core.messages import SystemMessage, HumanMessage
from llm_instrumentation import get_llm, HAS_OLLAMA
from prompt_config import get_system_prompt_or_none


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

    messages = []
    system_prompt = get_system_prompt_or_none("ask")
    if system_prompt:
        messages.append(SystemMessage(content=system_prompt))
    messages.append(
        HumanMessage(
            content="Analyze this question and identify the key entities and reasoning steps needed: "
            + question
        )
    )
    response = llm.invoke(messages)
    return {"analysis": response.content}


def gather_info(state: HotpotQAState) -> dict:
    """Determine what information is needed based on the analysis."""
    llm = get_llm_instance()
    analysis = state["analysis"]

    messages = []
    system_prompt = get_system_prompt_or_none("ask")
    if system_prompt:
        messages.append(SystemMessage(content=system_prompt))
    messages.append(
        HumanMessage(
            content="Based on this analysis: "
            + analysis
            + "\nWhat specific facts do we need to find to answer the original question? "
            + "Provide a brief summary."
        )
    )
    response = llm.invoke(messages)
    return {"info_needed": response.content}


def synthesize_answer(state: HotpotQAState) -> dict:
    """Reason about gathered information to answer."""
    llm = get_llm_instance()
    question = state["question"]
    analysis = state["analysis"]
    info_needed = state["info_needed"]

    messages = []
    system_prompt = get_system_prompt_or_none("ask")
    if system_prompt:
        messages.append(SystemMessage(content=system_prompt))
    messages.append(
        HumanMessage(
            content="Question: "
            + question
            + "\nAnalysis: "
            + analysis
            + "\nRelevant facts: "
            + info_needed
            + "\n\nBased on your knowledge, provide the direct answer:"
            + "\n- Yes/no questions: answer 'yes' or 'no'"
            + "\n- Entity questions: give the specific name"
            + "\n- Do NOT say 'neither' or 'unknown' - make your best judgment"
            + "\n\nFinal answer:"
        )
    )
    response = llm.invoke(messages)
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
