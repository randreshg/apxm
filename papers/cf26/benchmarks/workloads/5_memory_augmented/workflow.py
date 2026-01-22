"""
Memory Augmented Workflow - LangGraph Implementation

Simulates memory operations using LangGraph's checkpoint system.
Compare with workflow.ais which has native qmem (query) and umem (update) operations
for 3-tier memory: STM (short-term), LTM (long-term), and Episodic.
"""

from typing import TypedDict, Dict, Any, List
from langgraph.graph import StateGraph, START, END
from langchain_core.messages import SystemMessage, HumanMessage

# Import LLM instrumentation for real LLM calls
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from llm_instrumentation import get_llm, HAS_OLLAMA
from prompt_config import get_system_prompt_or_none

# Try to import checkpoint support
try:
    from langgraph.checkpoint.memory import MemorySaver
    HAS_CHECKPOINT = True
except ImportError:
    HAS_CHECKPOINT = False


def get_llm_instance():
    """Get the configured LLM instance."""
    return get_llm()


class MemoryState(TypedDict):
    """State for memory-augmented workflow."""
    query: str
    # Simulated memory tiers (embedded in state)
    stm: Dict[str, Any]  # Short-term memory
    ltm: Dict[str, Any]  # Long-term memory
    episodic: List[str]  # Episodic memory (audit trail of answers)
    # Working values
    cached: str
    answer: str


def recall_ltm(state: MemoryState) -> dict:
    """Recall from long-term memory."""
    cached = state["ltm"].get("domain_knowledge", "No cached knowledge")
    return {"cached": cached}


def store_stm(state: MemoryState) -> dict:
    """Store in short-term memory."""
    stm = state["stm"].copy()
    stm["current_query"] = state["query"]
    return {"stm": stm}


def reason(state: MemoryState) -> dict:
    """Reason with context using LLM."""
    cached = state["cached"]
    query = state["query"]

    user_prompt = f"Use the retrieved memory to answer the query. {cached} {query}"

    llm = get_llm_instance()
    messages = []
    system_prompt = get_system_prompt_or_none("ask")
    if system_prompt:
        messages.append(SystemMessage(content=system_prompt))
    messages.append(HumanMessage(content=user_prompt))
    response = llm.invoke(messages)
    answer = response.content if hasattr(response, 'content') else str(response)

    return {"answer": answer}


def record_episodic(state: MemoryState) -> dict:
    """Record to episodic memory for audit trail."""
    episodic = state["episodic"].copy()
    episodic.append(state["answer"])
    return {"episodic": episodic}


def persist_ltm(state: MemoryState) -> dict:
    """Persist to long-term memory."""
    ltm = state["ltm"].copy()
    ltm["answer"] = state["answer"]
    return {"ltm": ltm}


def build_graph() -> StateGraph:
    """Build the memory-augmented graph.

    Note: LangGraph requires:
    - Manual state management for each memory tier
    - Checkpoint system for persistence (adds overhead)
    - No native episodic/audit support
    """
    builder = StateGraph(MemoryState)

    # Add nodes
    builder.add_node("recall_ltm", recall_ltm)
    builder.add_node("store_stm", store_stm)
    builder.add_node("reason", reason)
    builder.add_node("record_episodic", record_episodic)
    builder.add_node("persist_ltm", persist_ltm)

    # Sequential flow
    builder.add_edge(START, "recall_ltm")
    builder.add_edge("recall_ltm", "store_stm")
    builder.add_edge("store_stm", "reason")
    builder.add_edge("reason", "record_episodic")
    builder.add_edge("record_episodic", "persist_ltm")
    builder.add_edge("persist_ltm", END)

    # Optionally add checkpointing for persistence
    if HAS_CHECKPOINT:
        checkpointer = MemorySaver()
        return builder.compile(checkpointer=checkpointer)
    else:
        return builder.compile()


# Compile the graph
graph = build_graph()


def run(query: str = "What is quantum computing?") -> dict:
    """Execute the memory-augmented workflow."""
    initial_state = {
        "query": query,
        "stm": {},
        "ltm": {"domain_knowledge": "Quantum computing fundamentals"},
        "episodic": [],
        "cached": "",
        "answer": "",
    }

    config = {}
    if HAS_CHECKPOINT:
        config = {"configurable": {"thread_id": "memory-demo"}}

    return graph.invoke(initial_state, config)


if __name__ == "__main__":
    result = run("What is quantum computing?")
    print("Memory-Augmented Answer")
    print(result['answer'])
    print(f"Episodic log: {result['episodic']}")
    print(f"STM: {result['stm']}")
    print(f"LTM: {result['ltm']}")
