#!/bin/sh
# test_mcp_integration.sh -- End-to-end integration test for the APXM MCP server.
#
# Exercises: initialize handshake, apxm_get_contract, apxm_validate (valid +
# invalid), apxm_merge, apxm_compile.  Skips apxm_execute (requires LLM backend).
#
# Usage:
#   ./tests/test_mcp_integration.sh          # build + test
#   SKIP_BUILD=1 ./tests/test_mcp_integration.sh  # skip cargo build

set -e

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
BINARY="$PROJECT_DIR/target/debug/apxm-mcp-server"
FIFO_IN=""
FIFO_OUT=""
SERVER_PID=""
PASS_COUNT=0
FAIL_COUNT=0
REQ_ID=0

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

cleanup() {
    if [ -n "$SERVER_PID" ] && kill -0 "$SERVER_PID" 2>/dev/null; then
        kill "$SERVER_PID" 2>/dev/null || true
        wait "$SERVER_PID" 2>/dev/null || true
    fi
    [ -n "$FIFO_IN"  ] && rm -f "$FIFO_IN"
    [ -n "$FIFO_OUT" ] && rm -f "$FIFO_OUT"
    rm -f /tmp/test_mcp_*.json /tmp/test_graph_*.apxmobj 2>/dev/null || true
}
trap cleanup EXIT INT TERM

pass() {
    PASS_COUNT=$((PASS_COUNT + 1))
    printf "  PASS: %s\n" "$1"
}

fail() {
    FAIL_COUNT=$((FAIL_COUNT + 1))
    printf "  FAIL: %s\n" "$1" >&2
    if [ -n "$2" ]; then
        printf "        %s\n" "$2" >&2
    fi
}

next_id() {
    REQ_ID=$((REQ_ID + 1))
    echo "$REQ_ID"
}

# Send a JSON-RPC request and capture the response.
# Usage: response=$(rpc_call '{"method":"...","params":{...}}')
rpc_call() {
    local id
    id=$(next_id)
    local payload
    payload=$(printf '%s' "$1" | sed "s/\"id\":0/\"id\":$id/")
    printf '%s\n' "$payload" >&3
    read -r response <&4
    echo "$response"
}

# Extract a field from JSON using jq or grep fallback.
json_field() {
    local json="$1"
    local field="$2"
    if command -v jq >/dev/null 2>&1; then
        echo "$json" | jq -r "$field" 2>/dev/null
    else
        # Primitive grep fallback -- works for simple top-level string/number fields
        echo "$json" | grep -o "\"${field}\"[[:space:]]*:[[:space:]]*\"[^\"]*\"" | head -1 | sed 's/.*: *"\(.*\)"/\1/'
    fi
}

# Check that a string contains a substring.
assert_contains() {
    local haystack="$1"
    local needle="$2"
    local label="$3"
    if echo "$haystack" | grep -q "$needle"; then
        pass "$label"
    else
        fail "$label" "expected to contain '$needle'"
    fi
}

# Check that a JSON field equals an expected value.
assert_json_eq() {
    local json="$1"
    local field="$2"
    local expected="$3"
    local label="$4"
    local actual
    actual=$(json_field "$json" "$field")
    if [ "$actual" = "$expected" ]; then
        pass "$label"
    else
        fail "$label" "expected $field=$expected, got '$actual'"
    fi
}

# ---------------------------------------------------------------------------
# Step 1: Build the MCP server
# ---------------------------------------------------------------------------

printf "=== APXM MCP Integration Tests ===\n\n"

if [ "${SKIP_BUILD:-}" != "1" ]; then
    printf "[1/7] Building apxm-mcp-server...\n"
    (cd "$PROJECT_DIR" && cargo build --bin apxm-mcp-server 2>&1) || {
        echo "FATAL: cargo build failed" >&2
        exit 1
    }
    printf "  Build OK\n\n"
else
    printf "[1/7] Skipping build (SKIP_BUILD=1)\n\n"
fi

if [ ! -x "$BINARY" ]; then
    echo "FATAL: binary not found at $BINARY" >&2
    exit 1
fi

# ---------------------------------------------------------------------------
# Step 2: Start the MCP server via FIFOs
# ---------------------------------------------------------------------------

printf "[2/7] Starting MCP server...\n"

FIFO_IN=$(mktemp -u /tmp/mcp_in.XXXXXX)
FIFO_OUT=$(mktemp -u /tmp/mcp_out.XXXXXX)
mkfifo "$FIFO_IN"
mkfifo "$FIFO_OUT"

# Launch server: reads from FIFO_IN, writes to FIFO_OUT
"$BINARY" < "$FIFO_IN" > "$FIFO_OUT" &
SERVER_PID=$!

# Open persistent file descriptors for writing (fd 3) and reading (fd 4)
exec 3>"$FIFO_IN"
exec 4<"$FIFO_OUT"

# Verify server process is alive
sleep 0.2
if ! kill -0 "$SERVER_PID" 2>/dev/null; then
    echo "FATAL: server process died immediately" >&2
    exit 1
fi
printf "  Server PID=%s\n\n" "$SERVER_PID"

# ---------------------------------------------------------------------------
# Step 3: MCP Initialize Handshake
# ---------------------------------------------------------------------------

printf "[3/7] Testing MCP initialize handshake...\n"

INIT_RESP=$(rpc_call '{"jsonrpc":"2.0","id":0,"method":"initialize","params":{"protocolVersion":"2024-11-05","capabilities":{},"clientInfo":{"name":"test-runner","version":"1.0"}}}')

assert_contains "$INIT_RESP" '"protocolVersion"' "init returns protocolVersion"
assert_contains "$INIT_RESP" '"apxm-mcp-server"' "init returns server name"
assert_contains "$INIT_RESP" '"tools"' "init advertises tools capability"

# Send the initialized notification (no id = notification, no response expected)
printf '{"jsonrpc":"2.0","method":"notifications/initialized"}\n' >&3

printf "\n"

# ---------------------------------------------------------------------------
# Step 4: Test apxm_get_contract
# ---------------------------------------------------------------------------

printf "[4/7] Testing apxm_get_contract...\n"

CONTRACT_RESP=$(rpc_call '{"jsonrpc":"2.0","id":0,"method":"tools/call","params":{"name":"apxm_get_contract","arguments":{}}}')

# The response wraps tool output in content[0].text -- extract it
CONTRACT_TEXT=$(json_field "$CONTRACT_RESP" '.result.content[0].text')

# Check that key operations exist
assert_contains "$CONTRACT_TEXT" '"ASK"' "contract includes ASK"
assert_contains "$CONTRACT_TEXT" '"THINK"' "contract includes THINK"
assert_contains "$CONTRACT_TEXT" '"REASON"' "contract includes REASON"
assert_contains "$CONTRACT_TEXT" '"VERIFY"' "contract includes VERIFY"
assert_contains "$CONTRACT_TEXT" '"PLAN"' "contract includes PLAN"
assert_contains "$CONTRACT_TEXT" '"RETURN"' "contract includes RETURN"
assert_contains "$CONTRACT_TEXT" '"WAIT_ALL"' "contract includes WAIT_ALL"
assert_contains "$CONTRACT_TEXT" '"MERGE"' "contract includes MERGE"
assert_contains "$CONTRACT_TEXT" '"BRANCH_ON_VALUE"' "contract includes BRANCH_ON_VALUE"
assert_contains "$CONTRACT_TEXT" '"COMMUNICATE"' "contract includes COMMUNICATE"
assert_contains "$CONTRACT_TEXT" '"GUARD"' "contract includes GUARD"
assert_contains "$CONTRACT_TEXT" '"PAUSE"' "contract includes PAUSE"

# Count operations (should be 30+)
if command -v jq >/dev/null 2>&1; then
    OP_COUNT=$(echo "$CONTRACT_TEXT" | jq '.operations | keys | length' 2>/dev/null)
    if [ "$OP_COUNT" -ge 30 ] 2>/dev/null; then
        pass "contract has $OP_COUNT operations (>= 30)"
    else
        fail "contract operation count" "expected >= 30, got $OP_COUNT"
    fi
else
    # Fallback: count quoted keys that look like AIS ops (SCREAMING_SNAKE_CASE)
    OP_COUNT=$(echo "$CONTRACT_TEXT" | grep -o '"[A-Z_]*":' | sort -u | wc -l)
    if [ "$OP_COUNT" -ge 25 ]; then
        pass "contract has ~$OP_COUNT operations (grep estimate)"
    else
        fail "contract operation count" "expected >= 25 (grep), got $OP_COUNT"
    fi
fi

assert_contains "$CONTRACT_TEXT" '"dependency_types"' "contract includes dependency_types"
assert_contains "$CONTRACT_TEXT" '"parameter_types"' "contract includes parameter_types"
assert_contains "$CONTRACT_TEXT" '"graph_schema"' "contract includes graph_schema"

printf "\n"

# ---------------------------------------------------------------------------
# Step 5: Test apxm_validate (valid graph)
# ---------------------------------------------------------------------------

printf "[5/7] Testing apxm_validate (valid + invalid)...\n"

# A valid 3-node graph: CONST_STR -> ASK -> RETURN
# Using CONST_STR as the entry because ASK needs a Data input with template_str
VALID_GRAPH='{"name":"test_ask_verify","nodes":[{"id":1,"name":"prompt","op":"CONST_STR","attributes":{"value":"What is 2+2?"}},{"id":2,"name":"ask_llm","op":"ASK","attributes":{"template_str":"{0}"}},{"id":3,"name":"done","op":"RETURN","attributes":{}}],"edges":[{"from":1,"to":2,"dependency":"Data"},{"from":2,"to":3,"dependency":"Data"}],"parameters":[]}'

VALIDATE_VALID_RESP=$(rpc_call "{\"jsonrpc\":\"2.0\",\"id\":0,\"method\":\"tools/call\",\"params\":{\"name\":\"apxm_validate\",\"arguments\":{\"graph_json\":$(printf '%s' "$VALID_GRAPH" | sed 's/"/\\"/g' | sed 's/.*/\"&\"/')}}}")

VALIDATE_VALID_TEXT=$(json_field "$VALIDATE_VALID_RESP" '.result.content[0].text')

assert_contains "$VALIDATE_VALID_TEXT" '"valid": true' "valid graph passes validation"

# Test with an invalid graph: missing required attributes, no name
INVALID_GRAPH='{"name":"","nodes":[{"id":1,"name":"bad","op":"ASK","attributes":{}}],"edges":[],"parameters":[]}'

VALIDATE_INVALID_RESP=$(rpc_call "{\"jsonrpc\":\"2.0\",\"id\":0,\"method\":\"tools/call\",\"params\":{\"name\":\"apxm_validate\",\"arguments\":{\"graph_json\":$(printf '%s' "$INVALID_GRAPH" | sed 's/"/\\"/g' | sed 's/.*/\"&\"/')}}}")

VALIDATE_INVALID_TEXT=$(json_field "$VALIDATE_INVALID_RESP" '.result.content[0].text')

assert_contains "$VALIDATE_INVALID_TEXT" '"valid": false' "invalid graph fails validation"
assert_contains "$VALIDATE_INVALID_TEXT" '"errors"' "invalid graph returns errors array"
assert_contains "$VALIDATE_INVALID_TEXT" "name" "error mentions empty name"

# Test with malformed JSON
MALFORMED_RESP=$(rpc_call '{"jsonrpc":"2.0","id":0,"method":"tools/call","params":{"name":"apxm_validate","arguments":{"graph_json":"this is not json"}}}')

assert_contains "$MALFORMED_RESP" "invalid JSON" "malformed JSON is rejected"

printf "\n"

# ---------------------------------------------------------------------------
# Step 6: Test apxm_merge
# ---------------------------------------------------------------------------

printf "[6/7] Testing apxm_merge...\n"

GRAPH_A='{"name":"graph_a","nodes":[{"id":1,"name":"a_const","op":"CONST_STR","attributes":{"value":"hello"}},{"id":2,"name":"a_ask","op":"ASK","attributes":{"template_str":"{0}"}}],"edges":[{"from":1,"to":2,"dependency":"Data"}],"parameters":[]}'

GRAPH_B='{"name":"graph_b","nodes":[{"id":1,"name":"b_const","op":"CONST_STR","attributes":{"value":"world"}},{"id":2,"name":"b_ask","op":"ASK","attributes":{"template_str":"{0}"}}],"edges":[{"from":1,"to":2,"dependency":"Data"}],"parameters":[]}'

# Build the merge request -- graphs is an array of JSON strings
# We need to double-escape the inner graph JSON since it goes inside a JSON string
GRAPH_A_ESC=$(printf '%s' "$GRAPH_A" | sed 's/\\/\\\\/g; s/"/\\"/g')
GRAPH_B_ESC=$(printf '%s' "$GRAPH_B" | sed 's/\\/\\\\/g; s/"/\\"/g')

MERGE_REQ="{\"jsonrpc\":\"2.0\",\"id\":0,\"method\":\"tools/call\",\"params\":{\"name\":\"apxm_merge\",\"arguments\":{\"name\":\"merged_test\",\"graphs\":[\"$GRAPH_A_ESC\",\"$GRAPH_B_ESC\"]}}}"

MERGE_RESP=$(rpc_call "$MERGE_REQ")

MERGE_TEXT=$(json_field "$MERGE_RESP" '.result.content[0].text')

assert_contains "$MERGE_TEXT" '"merged_graph"' "merge returns merged_graph"
assert_contains "$MERGE_TEXT" "WAIT_ALL" "merged graph has WAIT_ALL sync node"
assert_contains "$MERGE_TEXT" '"input_graphs": 2' "merge stats shows 2 input graphs"

# Verify total nodes: 2 + 2 + 1 (WAIT_ALL) = 5
if command -v jq >/dev/null 2>&1; then
    TOTAL_NODES=$(echo "$MERGE_TEXT" | jq '.stats.total_nodes' 2>/dev/null)
    if [ "$TOTAL_NODES" = "5" ]; then
        pass "merged graph has 5 nodes (2+2+1 sync)"
    else
        fail "merged node count" "expected 5, got $TOTAL_NODES"
    fi
fi

# Verify node IDs are unique (no collisions after remap)
if command -v jq >/dev/null 2>&1; then
    NODE_IDS=$(echo "$MERGE_TEXT" | jq '[.merged_graph.nodes[].id] | unique | length' 2>/dev/null)
    ALL_IDS=$(echo "$MERGE_TEXT" | jq '[.merged_graph.nodes[].id] | length' 2>/dev/null)
    if [ "$NODE_IDS" = "$ALL_IDS" ]; then
        pass "merged graph has unique node IDs after remap"
    else
        fail "merged node ID uniqueness" "unique=$NODE_IDS total=$ALL_IDS"
    fi
fi

printf "\n"

# ---------------------------------------------------------------------------
# Step 7: Test apxm_compile
# ---------------------------------------------------------------------------

printf "[7/7] Testing apxm_compile...\n"

COMPILE_GRAPH_ESC=$(printf '%s' "$VALID_GRAPH" | sed 's/\\/\\\\/g; s/"/\\"/g')

COMPILE_REQ="{\"jsonrpc\":\"2.0\",\"id\":0,\"method\":\"tools/call\",\"params\":{\"name\":\"apxm_compile\",\"arguments\":{\"graph_json\":\"$COMPILE_GRAPH_ESC\"}}}"

COMPILE_RESP=$(rpc_call "$COMPILE_REQ")

COMPILE_TEXT=$(json_field "$COMPILE_RESP" '.result.content[0].text')

# Check if compilation succeeded (isError should be false)
IS_ERROR=$(json_field "$COMPILE_RESP" '.result.isError')
if [ "$IS_ERROR" = "false" ]; then
    pass "compile succeeds without error"

    assert_contains "$COMPILE_TEXT" '"artifact_path"' "compile returns artifact_path"
    assert_contains "$COMPILE_TEXT" '.apxmobj' "artifact path has .apxmobj extension"
    assert_contains "$COMPILE_TEXT" '"stats"' "compile returns stats"
    assert_contains "$COMPILE_TEXT" '"compile_ms"' "compile stats include timing"

    # Verify the artifact file exists
    if command -v jq >/dev/null 2>&1; then
        ARTIFACT_PATH=$(echo "$COMPILE_TEXT" | jq -r '.artifact_path' 2>/dev/null)
        if [ -f "$ARTIFACT_PATH" ]; then
            pass "artifact file exists on disk"
            # Clean up the artifact
            rm -f "$ARTIFACT_PATH" 2>/dev/null || true
        else
            fail "artifact file exists" "file not found: $ARTIFACT_PATH"
        fi
    fi
else
    fail "compile succeeds without error" "isError=$IS_ERROR"
    # Print the error text for debugging
    printf "        Compile error: %s\n" "$COMPILE_TEXT" >&2
fi

printf "\n"

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------

exec 3>&-
exec 4<&-

TOTAL=$((PASS_COUNT + FAIL_COUNT))
printf "=== Results: %d/%d passed" "$PASS_COUNT" "$TOTAL"
if [ "$FAIL_COUNT" -gt 0 ]; then
    printf ", %d FAILED" "$FAIL_COUNT"
fi
printf " ===\n"

if [ "$FAIL_COUNT" -gt 0 ]; then
    exit 1
fi
exit 0
