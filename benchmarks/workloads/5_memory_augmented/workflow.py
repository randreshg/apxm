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


def qmem_ltm(state: MemoryState) -> dict:
    """Query long-term memory (qmem)."""
    cached = state["ltm"].get("domain_knowledge", "No cached knowledge")
    return {"cached": cached}


def umem_stm(state: MemoryState) -> dict:
    """Update short-term memory (umem)."""
    stm = state["stm"].copy()
    stm["current_query"] = state["query"]
    return {"stm": stm}


def reason(state: MemoryState) -> dict:
    """Reason with context using LLM."""
    cached = state["cached"]
    query = state["query"]

    llm = get_llm_instance()
    messages = []
    system_prompt = get_system_prompt_or_none("ask")
    if system_prompt:
        messages.append(SystemMessage(content=system_prompt))
    # Mirror AIS-style `ask(prompt, cached, query)` by passing context separately.
    messages.append(HumanMessage(content="Use the retrieved memory to answer the query."))
    messages.append(HumanMessage(content=cached or ""))
    messages.append(HumanMessage(content=query or ""))
    response = llm.invoke(messages)
    answer = response.content if hasattr(response, 'content') else str(response)

    return {"answer": answer}


def umem_episodic(state: MemoryState) -> dict:
    """Update episodic memory (umem) for audit trail."""
    episodic = state["episodic"].copy()
    episodic.append(state["answer"])
    return {"episodic": episodic}


def umem_ltm(state: MemoryState) -> dict:
    """Update long-term memory (umem)."""
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
    builder.add_node("qmem_ltm", qmem_ltm)
    builder.add_node("umem_stm", umem_stm)
    builder.add_node("reason", reason)
    builder.add_node("umem_episodic", umem_episodic)
    builder.add_node("umem_ltm", umem_ltm)

    # Sequential flow
    builder.add_edge(START, "qmem_ltm")
    builder.add_edge("qmem_ltm", "umem_stm")
    builder.add_edge("umem_stm", "reason")
    builder.add_edge("reason", "umem_episodic")
    builder.add_edge("umem_episodic", "umem_ltm")
    builder.add_edge("umem_ltm", END)

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
