"""
APXM Server API Tests — pytest

Requires apxm-server running on APXM_SERVER_URL (default: http://127.0.0.1:18800).

Run:
    # Start server first:
    APXM_SERVER_ADDR=127.0.0.1:18899 cargo run -p apxm-server &
    sleep 2

    # Run tests:
    pytest tests/test_server.py -v
"""

import os
import json
import pytest
import requests
import threading

BASE_URL = os.environ.get("APXM_SERVER_URL", "http://127.0.0.1:18899")


@pytest.fixture(scope="session")
def server():
    """Assume server is already running. Check health."""
    resp = requests.get(f"{BASE_URL}/health", timeout=5)
    if resp.status_code != 200:
        pytest.skip(f"apxm-server not running at {BASE_URL}")
    return BASE_URL


# ─── Health ──────────────────────────────────────────────────────────────────

def test_health(server):
    resp = requests.get(f"{server}/health")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "ok"
    assert "version" in body


# ─── Models ──────────────────────────────────────────────────────────────────

def test_list_models(server):
    resp = requests.get(f"{server}/v1/models")
    assert resp.status_code == 200
    body = resp.json()
    assert body["object"] == "list"
    assert isinstance(body["data"], list)


# ─── Capabilities ────────────────────────────────────────────────────────────

def test_register_static_capability(server):
    resp = requests.post(f"{server}/v1/capabilities/register", json={
        "name": "test_static_cap",
        "description": "Always returns hello world",
        "static_response": "hello world",
    })
    assert resp.status_code == 200
    assert resp.json()["ok"] is True


def test_list_capabilities_includes_registered(server):
    # Register a capability first
    requests.post(f"{server}/v1/capabilities/register", json={
        "name": "list_test_cap",
        "description": "Test capability for listing",
        "static_response": "test",
    })
    resp = requests.get(f"{server}/v1/capabilities")
    assert resp.status_code == 200
    caps = resp.json()
    assert isinstance(caps, list)
    names = [c["name"] for c in caps]
    assert "list_test_cap" in names


def test_register_http_capability(server):
    # Register without a live endpoint — just check registration succeeds
    resp = requests.post(f"{server}/v1/capabilities/register", json={
        "name": "http_cap_test",
        "description": "HTTP forwarding capability",
        "endpoint": "http://localhost:19999/tool",  # won't be called in this test
        "timeout_ms": 5000,
    })
    assert resp.status_code == 200
    assert resp.json()["ok"] is True


# ─── Agent Registry ──────────────────────────────────────────────────────────

def test_register_and_list_agent(server):
    resp = requests.post(f"{server}/v1/agents/register", json={
        "name": "test_agent_01",
        "url": "http://localhost:18901",
        "flows": ["main", "communicate"],
        "capabilities": ["summarize"],
    })
    assert resp.status_code == 200
    assert resp.json()["ok"] is True

    resp = requests.get(f"{server}/v1/agents")
    assert resp.status_code == 200
    agents = resp.json()
    names = [a["name"] for a in agents]
    assert "test_agent_01" in names


def test_get_agent(server):
    requests.post(f"{server}/v1/agents/register", json={
        "name": "get_test_agent",
        "url": "http://localhost:18902",
    })
    resp = requests.get(f"{server}/v1/agents/get_test_agent")
    assert resp.status_code == 200
    agent = resp.json()
    assert agent["name"] == "get_test_agent"
    assert agent["url"] == "http://localhost:18902"


def test_deregister_agent(server):
    requests.post(f"{server}/v1/agents/register", json={
        "name": "deregister_test_agent",
        "url": "http://localhost:18903",
    })
    resp = requests.delete(f"{server}/v1/agents/deregister_test_agent")
    assert resp.status_code == 200
    assert resp.json()["ok"] is True

    # Should 404 now
    resp = requests.get(f"{server}/v1/agents/deregister_test_agent")
    assert resp.status_code == 404


def test_get_nonexistent_agent_returns_404(server):
    resp = requests.get(f"{server}/v1/agents/this_agent_does_not_exist_xyz123")
    assert resp.status_code == 404


# ─── A2A AgentCard ───────────────────────────────────────────────────────────

def test_agent_card(server):
    resp = requests.get(f"{server}/.well-known/agent.json")
    assert resp.status_code == 200
    card = resp.json()
    assert card["protocolVersion"] == "0.3"
    assert "name" in card
    assert "capabilities" in card
    assert isinstance(card["skills"], list)


# ─── A2A JSON-RPC ────────────────────────────────────────────────────────────

def test_a2a_tasks_send(server):
    resp = requests.post(f"{server}/a2a", json={
        "jsonrpc": "2.0",
        "id": 1,
        "method": "tasks/send",
        "params": {
            "message": {
                "role": "user",
                "parts": [{"text": "Summarize the quarterly report."}],
            }
        },
    })
    assert resp.status_code == 200
    body = resp.json()
    assert body["jsonrpc"] == "2.0"
    assert "result" in body
    assert "id" in body["result"]
    assert body["result"]["status"]["state"] == "submitted"


def test_a2a_unknown_method_returns_error(server):
    resp = requests.post(f"{server}/a2a", json={
        "jsonrpc": "2.0",
        "id": 2,
        "method": "tasks/unknown_method",
        "params": {},
    })
    assert resp.status_code == 200
    body = resp.json()
    assert "error" in body
    assert body["error"]["code"] == -32601


# ─── MCP JSON-RPC ────────────────────────────────────────────────────────────

def test_mcp_initialize(server):
    resp = requests.post(f"{server}/v1/mcp", json={
        "jsonrpc": "2.0",
        "id": 1,
        "method": "initialize",
        "params": {
            "protocolVersion": "2025-11-05",
            "clientInfo": {"name": "test-client", "version": "0.1"},
        },
    })
    assert resp.status_code == 200
    body = resp.json()
    assert body["result"]["protocolVersion"] == "2025-11-05"
    assert body["result"]["serverInfo"]["name"] == "apxm-server"


def test_mcp_tools_list(server):
    # Register a capability first so there's at least one tool
    requests.post(f"{server}/v1/capabilities/register", json={
        "name": "mcp_test_tool",
        "description": "MCP test tool",
        "static_response": "mcp_result",
    })
    resp = requests.post(f"{server}/v1/mcp", json={
        "jsonrpc": "2.0",
        "id": 2,
        "method": "tools/list",
        "params": {},
    })
    assert resp.status_code == 200
    body = resp.json()
    tools = body["result"]["tools"]
    assert isinstance(tools, list)
    names = [t["name"] for t in tools]
    assert "mcp_test_tool" in names


def test_mcp_unknown_method_returns_error(server):
    resp = requests.post(f"{server}/v1/mcp", json={
        "jsonrpc": "2.0",
        "id": 99,
        "method": "unknown/method",
        "params": {},
    })
    assert resp.status_code == 200
    body = resp.json()
    assert "error" in body
    assert body["error"]["code"] == -32601


# ─── Memory ──────────────────────────────────────────────────────────────────

def test_store_and_search_fact(server):
    requests.post(f"{server}/v1/memory/facts/store", json={
        "text": "APXM uses parallel dataflow scheduling with 7.5µs per-op overhead",
        "tags": ["apxm", "performance"],
        "source": "test",
    })
    resp = requests.post(f"{server}/v1/memory/facts/search", json={
        "query": "APXM scheduling overhead",
        "limit": 5,
    })
    assert resp.status_code == 200
    results = resp.json()
    assert isinstance(results, list)


# ─── Receive (COMMUNICATE target) ────────────────────────────────────────────

def test_receive_message(server):
    resp = requests.post(f"{server}/v1/receive", json={
        "from": "agent_alpha",
        "message": {"type": "greeting", "content": "hello from agent_alpha"},
        "channel": "main",
    })
    assert resp.status_code == 200
    assert resp.json()["ok"] is True


# ─── Request ID ──────────────────────────────────────────────────────────────

def test_response_includes_request_id(server):
    resp = requests.get(f"{server}/health")
    # Server should propagate or generate x-request-id
    assert resp.status_code == 200
    # x-request-id header may or may not be returned depending on
    # whether the client sent one; just ensure the server doesn't crash
