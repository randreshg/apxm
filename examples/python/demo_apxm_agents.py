#!/usr/bin/env python3
"""
APXM Flagship Demo — On-Premises Multi-Agent Council
=====================================================

Demonstrates APXM's program execution model using the On-Premises LLM backend:
- Multi-model council with parallel inference (3 models simultaneously)
- A2A v0.3 agent registry and discovery
- MCP 2025-11-05 tools exposure
- COMMUNICATE HTTP between agents

Requirements:
    pip install requests openai

Usage:
    # Set your API key
    export OCP_APIM_KEY="<your-key>"

    # Start apxm-server (in another terminal):
    APXM_SERVER_ADDR=127.0.0.1:18800 cargo run -p apxm-server

    # Run the demo
    python examples/demo_apxm_agents.py

    # Or run against a specific question
    python examples/demo_apxm_agents.py --question "What are the best hardware choices for LLM inference at scale?"
"""

import os
import sys
import json
import time
import argparse
import requests
from typing import Optional

try:
    import openai as _openai_mod
except ImportError:
    _openai_mod = None

# ─── Configuration ────────────────────────────────────────────────────────────

SERVER_URL = os.environ.get("APXM_SERVER_URL", "http://127.0.0.1:18800")
APXM_BASE_URL = os.environ.get("APXM_LLM_BASE_URL", "https://your-llm-gateway/v1")
APXM_API_KEY = "dummy"
APXM_MODEL = os.environ.get("APXM_LLM_MODEL", "gpt-4o-mini")  # set APXM_LLM_MODEL to your model name
APXM_OCP_KEY = os.environ.get("OCP_APIM_KEY", "")  # set via OCP_APIM_KEY env var
APXM_USER = os.environ.get("USERNAME", os.environ.get("USER", "apxm-demo"))

_ONPREM_CLIENT = None

def _get_onprem_client():
    global _ONPREM_CLIENT
    if _ONPREM_CLIENT is None:
        if _openai_mod is None:
            return None
        _ONPREM_CLIENT = _openai_mod.OpenAI(
            base_url=APXM_BASE_URL,
            api_key=APXM_API_KEY,
            default_headers={
                "Ocp-Apim-Subscription-Key": APXM_OCP_KEY,
                "user": APXM_USER,
            },
        )
    return _ONPREM_CLIENT

BANNER = """
╔══════════════════════════════════════════════════════════════╗
║       APXM — On-Premises Multi-Agent Council Demo            ║
║  Program Execution Model for AI Agents · Sprint 0           ║
╚══════════════════════════════════════════════════════════════╝
"""


# ─── On-Premises LLM Client ──────────────────────────────────────────────────

def call_onprem_llm(prompt: str, system: str = "", max_tokens: int = 200) -> Optional[str]:
    """Call On-Premises LLM directly via OpenAI-compatible API."""
    try:
        client = _get_onprem_client()
        if client is None:
            print("  [LLM ERROR] openai package not installed (pip install openai)")
            return None
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        resp = client.chat.completions.create(
            model=APXM_MODEL,
            max_completion_tokens=max_tokens,
            temperature=0.7,
            messages=messages,
        )
        return resp.choices[0].message.content
    except Exception as e:
        print(f"  [LLM ERROR] {e}")
        return None


# ─── APXM Server Client ───────────────────────────────────────────────────────

def check_server() -> bool:
    """Verify apxm-server is running."""
    try:
        resp = requests.get(f"{SERVER_URL}/health", timeout=3)
        data = resp.json()
        print(f"✓ apxm-server running — version {data.get('version', '?')}")
        return True
    except Exception:
        print(f"✗ apxm-server not reachable at {SERVER_URL}")
        print(f"  Start with: APXM_SERVER_ADDR=127.0.0.1:18800 cargo run -p apxm-server")
        return False


def register_capability(name: str, description: str, endpoint: str) -> bool:
    """Register an HTTP capability with the server."""
    resp = requests.post(f"{SERVER_URL}/v1/capabilities/register", json={
        "name": name,
        "description": description,
        "endpoint": endpoint,
        "timeout_ms": 30000,
    })
    return resp.status_code == 200


def register_agent(name: str, url: str, flows: list) -> bool:
    """Register an agent with the server registry."""
    resp = requests.post(f"{SERVER_URL}/v1/agents/register", json={
        "name": name,
        "url": url,
        "flows": flows,
    })
    return resp.status_code == 200


def execute_graph(graph: dict, args: list = None) -> Optional[dict]:
    """Submit a graph for execution on the APXM runtime."""
    resp = requests.post(f"{SERVER_URL}/v1/execute", json={
        "graph": graph,
        "args": args if args is not None else [],
    })
    if resp.status_code == 200:
        return resp.json()
    print(f"  [EXECUTE ERROR] {resp.status_code}: {resp.text[:200]}")
    return None


def send_a2a_task(message: str) -> Optional[str]:
    """Send a task via A2A v0.3 JSON-RPC."""
    resp = requests.post(f"{SERVER_URL}/a2a", json={
        "jsonrpc": "2.0",
        "id": 1,
        "method": "tasks/send",
        "params": {
            "message": {
                "role": "user",
                "parts": [{"text": message}],
            }
        },
    })
    if resp.status_code == 200:
        result = resp.json().get("result", {})
        return result.get("id")
    return None


def list_mcp_tools() -> list:
    """List available tools via MCP 2025-11-05."""
    resp = requests.post(f"{SERVER_URL}/v1/mcp", json={
        "jsonrpc": "2.0",
        "id": 1,
        "method": "tools/list",
        "params": {},
    })
    if resp.status_code == 200:
        return resp.json().get("result", {}).get("tools", [])
    return []


# ─── Council Graph Builder ────────────────────────────────────────────────────

def build_council_graph(question: str, num_experts: int = 3) -> dict:
    """
    Build a parallel council graph with N experts + synthesis.

    Graph topology:
        CONST_STR(question) → [ASK×N in parallel] → WAIT_ALL → ASK(synthesis)
    """
    next_token = 10
    nodes = []

    # Input node
    nodes.append({
        "id": 1,
        "op": "CONST_STR",
        "attributes": {"value": question},
        "input_tokens": [],
        "output_tokens": [next_token],
    })
    question_token = next_token
    next_token += 1

    # Expert ASK nodes (run in parallel — no shared dependencies)
    expert_tokens = []
    expert_systems = [
        "You are a hardware architect specializing in AI accelerators.",
        "You are a systems performance engineer with HPC expertise.",
        "You are a total-cost-of-ownership analyst for AI infrastructure.",
    ][:num_experts]

    for i, system in enumerate(expert_systems):
        tok = next_token
        next_token += 1
        nodes.append({
            "id": 10 + i,
            "op": "ASK",
            "attributes": {
                "system_prompt": system,
                "prompt": f"Answer concisely in 2-3 sentences: {question}",
            },
            "input_tokens": [question_token],
            "output_tokens": [tok],
        })
        expert_tokens.append(tok)

    # WAIT_ALL — fires when all experts complete
    wait_token = next_token
    next_token += 1
    nodes.append({
        "id": 20,
        "op": "WAIT_ALL",
        "attributes": {},
        "input_tokens": expert_tokens,
        "output_tokens": [wait_token],
    })

    # Build synthesis prompt
    expert_refs = "\n".join(
        f"Expert {i+1}: {{{{t{tok}}}}}"
        for i, tok in enumerate(expert_tokens)
    )
    synthesis_prompt = (
        f"Question: {question}\n\n"
        f"Three expert perspectives:\n{expert_refs}\n\n"
        f"Synthesize into one definitive answer, noting consensus and disagreements:"
    )

    # Synthesis ASK
    output_token = next_token
    nodes.append({
        "id": 30,
        "op": "ASK",
        "attributes": {
            "prompt": synthesis_prompt,
        },
        "input_tokens": [wait_token],
        "output_tokens": [output_token],
    })

    return {
        "name": "apxm_council",
        "nodes": nodes,
    }


# ─── Demo Steps ───────────────────────────────────────────────────────────────

def demo_direct_onprem(question: str):
    """Step 1: Direct on-premises LLM call (baseline single-model response)."""
    print("\n[Step 1] On-Premises LLM — Single Model Response")
    print("─" * 60)
    print(f"Question: {question}")
    start = time.time()
    response = call_onprem_llm(
        prompt=question,
        system="You are a helpful AI assistant.",
        max_tokens=200,
    )
    elapsed = time.time() - start
    if response:
        print(f"\nOn-Premises Response ({elapsed:.1f}s):\n{response}")
    else:
        print("  (on-premises backend unavailable — skipping)")


def demo_a2a_discovery(question: str):
    """Step 2: A2A v0.3 protocol demo."""
    print("\n[Step 2] A2A v0.3 — AgentCard + Task Submit")
    print("─" * 60)

    # Fetch agent card
    resp = requests.get(f"{SERVER_URL}/.well-known/agent.json")
    if resp.status_code == 200:
        card = resp.json()
        print(f"AgentCard: {card['name']} v{card.get('version', '?')}")
        print(f"  Protocol: A2A {card['protocolVersion']}")
        print(f"  Capabilities: {json.dumps(card['capabilities'])}")
    else:
        print("  (Agent card unavailable)")

    # Register a demo agent
    register_agent("DemoAgent", f"{SERVER_URL}", ["main"])
    print("\nRegistered DemoAgent in agent registry")

    # Send a task via A2A
    task_id = send_a2a_task(question)
    if task_id:
        print(f"A2A task submitted: id={task_id}")
    else:
        print("  (A2A endpoint unavailable)")


def demo_mcp_tools():
    """Step 3: MCP 2025-11-05 tool discovery."""
    print("\n[Step 3] MCP 2025-11-05 — Tool Discovery")
    print("─" * 60)

    # Register a demo capability first
    requests.post(f"{SERVER_URL}/v1/capabilities/register", json={
        "name": "apxm_llm",
        "description": "Query On-Premises LLM via APXM capability system",
        "static_response": "LLM response",
    })

    tools = list_mcp_tools()
    print(f"Available MCP tools ({len(tools)}):")
    for tool in tools[:5]:
        print(f"  • {tool['name']}: {tool['description'][:60]}")
    if len(tools) > 5:
        print(f"  ... and {len(tools) - 5} more")


def demo_council_graph(question: str):
    """Step 4: APXM multi-agent council via graph execution."""
    print("\n[Step 4] APXM Council Graph — 3-Expert Parallel Synthesis")
    print("─" * 60)
    print(f"Question: {question}")
    print("Building council graph (3 parallel experts + synthesis)...")

    graph = build_council_graph(question, num_experts=3)
    print(f"Graph: {len(graph['nodes'])} nodes")
    print("Executing via apxm-server...")

    start = time.time()
    result = execute_graph(graph)
    elapsed = time.time() - start

    if result:
        content = result.get("content") or "No text output"
        stats = result.get("stats", {})
        usage = result.get("llm_usage", {})
        print(f"\nCouncil Verdict ({elapsed:.1f}s):")
        print(content[:800])
        print(f"\nStats: {stats.get('executed_nodes', '?')} nodes, "
              f"{stats.get('duration_ms', '?')}ms")
        print(f"LLM Usage: {usage.get('input_tokens', '?')} in / "
              f"{usage.get('output_tokens', '?')} out tokens")
    else:
        print("  (Graph execution failed — check server logs)")
        print("  Note: Requires on-premises backend configured in apxm.toml")


def demo_memory(question: str):
    """Step 5: Store and retrieve from APXM long-term memory."""
    print("\n[Step 5] APXM Memory — Store + Retrieve")
    print("─" * 60)

    # Store a fact
    resp = requests.post(f"{SERVER_URL}/v1/memory/facts/store", json={
        "text": f"Council reviewed: {question}",
        "tags": ["demo", "council", "review"],
        "source": "demo_apxm_agents.py",
    })
    if resp.status_code == 200:
        fact_id = resp.json().get("id")
        print(f"Stored fact: id={fact_id}")

    # Search
    resp = requests.post(f"{SERVER_URL}/v1/memory/facts/search", json={
        "query": "council review",
        "limit": 3,
    })
    if resp.status_code == 200:
        results = resp.json()
        print(f"Retrieved {len(results)} fact(s) from LTM")
        for r in results[:2]:
            print(f"  • {r.get('text', '')[:80]}")


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="APXM On-Premises Multi-Agent Council Demo")
    parser.add_argument(
        "--question",
        default="What are the best hardware choices for LLM inference at scale?",
        help="Question to put to the council",
    )
    parser.add_argument(
        "--server",
        default=SERVER_URL,
        help=f"apxm-server URL (default: {SERVER_URL})",
    )
    parser.add_argument(
        "--skip-onprem",
        action="store_true",
        help="Skip direct on-premises LLM call (useful when not on the LLM network)",
    )
    args = parser.parse_args()

    global SERVER_URL
    SERVER_URL = args.server

    print(BANNER)
    print(f"Server: {SERVER_URL}")
    print(f"On-Premises Model: {APXM_MODEL} @ {APXM_BASE_URL}")
    print(f"Question: {args.question}\n")

    # Check server
    if not check_server():
        sys.exit(1)

    # Run demo steps
    if not args.skip_onprem:
        demo_direct_onprem(args.question)
    else:
        print("\n[Step 1] Skipping direct on-premises call (--skip-onprem)")

    demo_a2a_discovery(args.question)
    demo_mcp_tools()
    demo_council_graph(args.question)
    demo_memory(args.question)

    print("\n" + "═" * 60)
    print("DEMO COMPLETE")
    print("═" * 60)
    print("\nWhat you just saw:")
    print(f"  1. LLM backend — single model baseline ({APXM_MODEL})")
    print("  2. A2A v0.3 — AgentCard discovery + task submission")
    print("  3. MCP 2025-11-05 — tool listing via JSON-RPC")
    print("  4. APXM Council — 3 parallel inferences + synthesis")
    print("  5. APXM Memory — long-term fact storage + semantic search")
    print("\nNext steps:")
    print("  • Run examples/code_review_council.ais for code review demo")
    print("  • Run examples/multi_agent_communicate.ais for cross-agent messaging")
    print("  • See tasks/starter_protocol/plan/ for implementation roadmap")


if __name__ == "__main__":
    main()
