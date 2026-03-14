//! APXM MCP Server -- exposes the APXM compiler as MCP tools over stdio.
//!
//! Implements JSON-RPC 2.0 over stdin/stdout per the Model Context Protocol
//! (2024-11-05) so that external agents (Claude Code, Codex, etc.) can
//! validate, compile, and execute APXM graphs.
//!
//! # Tools
//!
//! - `apxm_validate`      -- validate an ApxmGraph JSON against the AIS contract
//! - `apxm_compile`       -- compile an ApxmGraph JSON to an optimized artifact
//! - `apxm_execute`       -- compile + execute a graph in one shot
//! - `apxm_merge`         -- merge multiple ApxmGraph sub-graphs into one
//! - `apxm_get_contract`  -- return the full AIS contract (ops, attrs, types)
//!
//! # Running
//!
//! ```bash
//! apxm-mcp-server          # reads JSON-RPC lines from stdin, writes to stdout
//! ```

use std::collections::{HashMap, HashSet, VecDeque};
use std::io::{self, BufRead, Write};
use std::time::Instant;

use apxm_artifact::Artifact;
use apxm_compiler::{Context as CompilerContext, Pipeline as CompilerPipeline};
use apxm_core::types::OptimizationLevel;
use apxm_graph::ApxmGraph;
use serde_json::{json, Value};

const MCP_PROTOCOL_VERSION: &str = "2024-11-05";
const SERVER_NAME: &str = "apxm-mcp-server";
const SERVER_VERSION: &str = env!("CARGO_PKG_VERSION");

// JSON-RPC error codes
const PARSE_ERROR: i64 = -32700;
const METHOD_NOT_FOUND: i64 = -32601;
const INVALID_PARAMS: i64 = -32602;
const _INTERNAL_ERROR: i64 = -32000;

fn main() {
    let stdin = io::stdin();
    let mut stdout = io::stdout();

    for line in stdin.lock().lines() {
        let line = match line {
            Ok(l) => l,
            Err(_) => break,
        };
        let line = line.trim().to_string();
        if line.is_empty() {
            continue;
        }

        let request: Value = match serde_json::from_str(&line) {
            Ok(v) => v,
            Err(e) => {
                let resp = json!({
                    "jsonrpc": "2.0",
                    "id": null,
                    "error": { "code": PARSE_ERROR, "message": format!("invalid JSON: {e}") }
                });
                write_response(&mut stdout, &resp);
                continue;
            }
        };

        let response = handle_request(request);
        if response != Value::Null {
            write_response(&mut stdout, &response);
        }
    }
}

fn write_response(stdout: &mut io::Stdout, response: &Value) {
    let line = serde_json::to_string(response).expect("serialize response");
    let _ = writeln!(stdout, "{line}");
    let _ = stdout.flush();
}

fn handle_request(request: Value) -> Value {
    let id = request.get("id").cloned().unwrap_or(Value::Null);
    let method = request
        .get("method")
        .and_then(Value::as_str)
        .unwrap_or_default();
    let params = request.get("params").cloned().unwrap_or(Value::Null);

    // Notifications (no "id" field) -- handle silently
    if request.get("id").is_none() {
        // notifications/initialized, notifications/cancelled, etc.
        return Value::Null;
    }

    let result = match method {
        "initialize" => handle_initialize(),
        "tools/list" => handle_tools_list(),
        "tools/call" => handle_tools_call(params),
        "ping" => Ok(json!({})),
        "" => Err(rpc_error(PARSE_ERROR, "missing method")),
        _ => Err(rpc_error(METHOD_NOT_FOUND, format!("unknown method: {method}"))),
    };

    match result {
        Ok(result) => json!({
            "jsonrpc": "2.0",
            "id": id,
            "result": result,
        }),
        Err(error) => json!({
            "jsonrpc": "2.0",
            "id": id,
            "error": error,
        }),
    }
}

fn handle_initialize() -> Result<Value, Value> {
    Ok(json!({
        "protocolVersion": MCP_PROTOCOL_VERSION,
        "serverInfo": {
            "name": SERVER_NAME,
            "version": SERVER_VERSION,
        },
        "capabilities": {
            "tools": { "listChanged": false }
        }
    }))
}

fn handle_tools_list() -> Result<Value, Value> {
    let tools = vec![
        json!({
            "name": "apxm_validate",
            "description": "Validate an ApxmGraph JSON against the APXM AIS contract. Checks node ops, edges, parameters, cycles, and required attributes.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "graph_json": {
                        "type": "string",
                        "description": "The ApxmGraph as a JSON string"
                    }
                },
                "required": ["graph_json"]
            }
        }),
        json!({
            "name": "apxm_compile",
            "description": "Compile an ApxmGraph JSON to an optimized APXM artifact (.apxmobj). Returns the artifact path and compilation stats.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "graph_json": {
                        "type": "string",
                        "description": "The ApxmGraph as a JSON string"
                    },
                    "opt_level": {
                        "type": "integer",
                        "description": "Optimization level (0-3). 0=none, 1=basic, 2=standard, 3=aggressive. Default: 2",
                        "minimum": 0,
                        "maximum": 3
                    }
                },
                "required": ["graph_json"]
            }
        }),
        json!({
            "name": "apxm_execute",
            "description": "Compile and execute an ApxmGraph in one shot. Requires APXM runtime environment (LLM backend configured via APXM_BACKEND).",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "graph_json": {
                        "type": "string",
                        "description": "The ApxmGraph as a JSON string"
                    },
                    "parameters": {
                        "type": "object",
                        "description": "Runtime parameters to pass to the graph entry flow",
                        "additionalProperties": { "type": "string" }
                    }
                },
                "required": ["graph_json"]
            }
        }),
        json!({
            "name": "apxm_merge",
            "description": "Merge multiple ApxmGraph sub-graphs into a single graph. Node IDs are remapped to avoid collisions and a WAIT_ALL synchronization node is appended.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "name": {
                        "type": "string",
                        "description": "Name for the merged graph"
                    },
                    "graphs": {
                        "type": "array",
                        "items": { "type": "string" },
                        "description": "Array of ApxmGraph JSON strings to merge"
                    }
                },
                "required": ["name", "graphs"]
            }
        }),
        json!({
            "name": "apxm_get_contract",
            "description": "Return the full AIS contract: all valid operations with required attributes, valid dependency types, parameter types, and graph schema. Use this to discover what operations and attributes are available when building graphs.",
            "inputSchema": {
                "type": "object",
                "properties": {},
                "required": []
            }
        }),
    ];
    Ok(json!({ "tools": tools }))
}

fn handle_tools_call(params: Value) -> Result<Value, Value> {
    let name = params
        .get("name")
        .and_then(Value::as_str)
        .ok_or_else(|| rpc_error(INVALID_PARAMS, "tools/call missing params.name"))?;
    let args = params
        .get("arguments")
        .cloned()
        .unwrap_or_else(|| json!({}));

    let result = match name {
        "apxm_validate" => tool_validate(args),
        "apxm_compile" => tool_compile(args),
        "apxm_execute" => tool_execute(args),
        "apxm_merge" => tool_merge(args),
        "apxm_get_contract" => tool_get_contract(),
        _ => Err(format!("unknown tool: {name}")),
    };

    match result {
        Ok(output) => Ok(json!({
            "content": [{ "type": "text", "text": output }],
            "isError": false,
        })),
        Err(error) => Ok(json!({
            "content": [{ "type": "text", "text": error }],
            "isError": true,
        })),
    }
}

// ---------------------------------------------------------------------------
// Tool: apxm_validate
// ---------------------------------------------------------------------------

fn tool_validate(args: Value) -> Result<String, String> {
    let graph_json = args
        .get("graph_json")
        .and_then(Value::as_str)
        .ok_or("missing required argument: graph_json")?;

    let mut errors: Vec<String> = Vec::new();
    let mut warnings: Vec<String> = Vec::new();

    // Try to parse the JSON first
    let raw: Value = serde_json::from_str(graph_json)
        .map_err(|e| format!("invalid JSON: {e}"))?;

    // Check required top-level fields
    if !raw.get("name").and_then(Value::as_str).is_some_and(|s| !s.is_empty()) {
        errors.push("graph name must not be empty".to_string());
    }

    let nodes = raw.get("nodes").and_then(Value::as_array);
    if nodes.is_none() || nodes.is_some_and(|n| n.is_empty()) {
        errors.push("graph must contain at least one node".to_string());
    }

    // Node-level checks
    let mut node_ids: HashSet<u64> = HashSet::new();
    if let Some(nodes) = nodes {
        for node in nodes {
            let id = node.get("id").and_then(Value::as_u64).unwrap_or(0);
            let name = node.get("name").and_then(Value::as_str).unwrap_or("");
            let op = node.get("op").and_then(Value::as_str).unwrap_or("");

            if id == 0 {
                errors.push(format!("node '{name}' has invalid id (0 or missing)"));
            }
            if !node_ids.insert(id) {
                errors.push(format!("duplicate node id {id}"));
            }
            if name.is_empty() {
                errors.push(format!("node id={id} has empty name"));
            }
            if op.is_empty() {
                errors.push(format!("node '{name}' (id={id}) has empty op"));
            }
        }
    }

    // Edge checks
    if let Some(edges) = raw.get("edges").and_then(Value::as_array) {
        for edge in edges {
            let from = edge.get("from").and_then(Value::as_u64).unwrap_or(0);
            let to = edge.get("to").and_then(Value::as_u64).unwrap_or(0);
            let dep = edge.get("dependency").and_then(Value::as_str).unwrap_or("Data");

            if !matches!(dep, "Data" | "Control" | "Effect") {
                errors.push(format!(
                    "edge {from}->{to} has invalid dependency type '{dep}'"
                ));
            }
            if !node_ids.contains(&from) {
                errors.push(format!("edge references non-existent from_id {from}"));
            }
            if !node_ids.contains(&to) {
                errors.push(format!("edge references non-existent to_id {to}"));
            }
        }

        // DAG cycle check (Kahn's algorithm)
        if !node_ids.is_empty() && !edges.is_empty() {
            let mut in_degree: HashMap<u64, usize> = node_ids.iter().map(|&id| (id, 0)).collect();
            let mut adjacency: HashMap<u64, Vec<u64>> = HashMap::new();

            for edge in edges {
                let from = edge.get("from").and_then(Value::as_u64).unwrap_or(0);
                let to = edge.get("to").and_then(Value::as_u64).unwrap_or(0);
                if node_ids.contains(&from) && node_ids.contains(&to) {
                    adjacency.entry(from).or_default().push(to);
                    *in_degree.entry(to).or_insert(0) += 1;
                }
            }

            let mut queue: VecDeque<u64> = in_degree
                .iter()
                .filter_map(|(&id, &deg)| if deg == 0 { Some(id) } else { None })
                .collect();
            let mut visited = 0usize;
            while let Some(node_id) = queue.pop_front() {
                visited += 1;
                if let Some(neighbors) = adjacency.get(&node_id) {
                    for &neighbor in neighbors {
                        if let Some(current) = in_degree.get_mut(&neighbor) {
                            *current = current.saturating_sub(1);
                            if *current == 0 {
                                queue.push_back(neighbor);
                            }
                        }
                    }
                }
            }

            if visited != node_ids.len() {
                let cycle_nodes = node_ids.len() - visited;
                errors.push(format!(
                    "graph contains a cycle ({cycle_nodes} nodes involved)"
                ));
            }
        }
    }

    // Parameter checks
    if let Some(params) = raw.get("parameters").and_then(Value::as_array) {
        let valid_types: HashSet<&str> = ["str", "int", "float", "bool", "json"].into_iter().collect();
        let mut param_names: HashSet<String> = HashSet::new();
        for param in params {
            let pname = param.get("name").and_then(Value::as_str).unwrap_or("");
            let ptype = param.get("type_name").and_then(Value::as_str).unwrap_or("");
            if pname.is_empty() {
                errors.push("parameter with empty name".to_string());
            }
            if !param_names.insert(pname.to_string()) {
                errors.push(format!("duplicate parameter name '{pname}'"));
            }
            if !valid_types.contains(ptype) {
                warnings.push(format!(
                    "parameter '{pname}' has non-standard type_name '{ptype}'"
                ));
            }
        }
    }

    // Also attempt full Rust-side parse+validate for deeper checks
    match ApxmGraph::from_json(graph_json) {
        Ok(_) => {}
        Err(e) => {
            let msg = e.to_string();
            // Avoid duplicating errors we already caught above
            if !errors.iter().any(|existing| msg.contains(&existing[..existing.len().min(30)])) {
                errors.push(format!("apxm-graph validation: {msg}"));
            }
        }
    }

    let valid = errors.is_empty();
    let result = json!({
        "valid": valid,
        "errors": errors,
        "warnings": warnings,
    });
    Ok(serde_json::to_string_pretty(&result).unwrap())
}

// ---------------------------------------------------------------------------
// Tool: apxm_compile
// ---------------------------------------------------------------------------

fn tool_compile(args: Value) -> Result<String, String> {
    let graph_json = args
        .get("graph_json")
        .and_then(Value::as_str)
        .ok_or("missing required argument: graph_json")?;

    let opt_level_num = args
        .get("opt_level")
        .and_then(Value::as_u64)
        .unwrap_or(2);
    let opt_level = match opt_level_num {
        0 => OptimizationLevel::O0,
        1 => OptimizationLevel::O1,
        2 => OptimizationLevel::O2,
        3 => OptimizationLevel::O3,
        _ => return Err(format!("opt_level must be 0-3, got {opt_level_num}")),
    };

    let graph = ApxmGraph::from_json(graph_json)
        .map_err(|e| format!("invalid graph: {e}"))?;

    let node_count = graph.nodes.len();
    let edge_count = graph.edges.len();
    let graph_name = graph.name.clone();

    let start = Instant::now();

    let context = CompilerContext::new()
        .map_err(|e| format!("compiler context init failed: {e}"))?;
    let pipeline = CompilerPipeline::with_opt_level(&context, opt_level);
    let module = pipeline
        .compile_graph(&graph)
        .map_err(|e| format!("compilation failed: {e}"))?;

    let artifact_bytes = module
        .generate_artifact_bytes()
        .map_err(|e| format!("artifact generation failed: {e}"))?;

    // Validate the artifact is well-formed
    let artifact = Artifact::from_bytes(&artifact_bytes)
        .map_err(|e| format!("artifact decode failed: {e}"))?;

    let compile_ms = start.elapsed().as_millis();

    // Write artifact to a temp file
    let artifact_path = std::env::temp_dir()
        .join(format!("{graph_name}.apxmobj"));
    artifact
        .write_to_path(&artifact_path)
        .map_err(|e| format!("failed to write artifact: {e}"))?;

    let result = json!({
        "artifact_path": artifact_path.display().to_string(),
        "stats": {
            "nodes": node_count,
            "edges": edge_count,
            "artifact_bytes": artifact_bytes.len(),
            "compile_ms": compile_ms,
            "opt_level": opt_level_num,
        }
    });
    Ok(serde_json::to_string_pretty(&result).unwrap())
}

// ---------------------------------------------------------------------------
// Tool: apxm_execute
// ---------------------------------------------------------------------------

fn tool_execute(args: Value) -> Result<String, String> {
    let graph_json = args
        .get("graph_json")
        .and_then(Value::as_str)
        .ok_or("missing required argument: graph_json")?;

    let parameters = args
        .get("parameters")
        .and_then(Value::as_object)
        .cloned()
        .unwrap_or_default();

    let graph = ApxmGraph::from_json(graph_json)
        .map_err(|e| format!("invalid graph: {e}"))?;

    // Compile
    let compile_start = Instant::now();
    let context = CompilerContext::new()
        .map_err(|e| format!("compiler context init failed: {e}"))?;
    let pipeline = CompilerPipeline::with_opt_level(&context, OptimizationLevel::O1);
    let module = pipeline
        .compile_graph(&graph)
        .map_err(|e| format!("compilation failed: {e}"))?;
    let artifact_bytes = module
        .generate_artifact_bytes()
        .map_err(|e| format!("artifact generation failed: {e}"))?;
    let artifact = Artifact::from_bytes(&artifact_bytes)
        .map_err(|e| format!("artifact decode failed: {e}"))?;
    let compile_ms = compile_start.elapsed().as_millis();

    // Build args from parameters map
    let entry_dag = artifact.entry_dag().or_else(|| artifact.dag());
    let args: Vec<String> = if let Some(dag) = entry_dag {
        dag.metadata
            .parameters
            .iter()
            .map(|p| {
                parameters
                    .get(&p.name)
                    .and_then(Value::as_str)
                    .unwrap_or("")
                    .to_string()
            })
            .collect()
    } else {
        Vec::new()
    };

    // Execute via the runtime (requires tokio)
    let rt = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .map_err(|e| format!("tokio runtime init failed: {e}"))?;

    let exec_start = Instant::now();
    let execution = rt.block_on(async {
        use apxm_runtime::{Runtime, RuntimeConfig};
        let runtime = Runtime::new(RuntimeConfig::default())
            .await
            .map_err(|e| format!("runtime init failed: {e}"))?;
        runtime
            .execute_artifact_with_args(artifact, args)
            .await
            .map_err(|e| format!("execution failed: {e}"))
    })?;
    let exec_ms = exec_start.elapsed().as_millis();

    // Format results
    let mut results_map = serde_json::Map::new();
    let mut content: Option<String> = None;
    for (token, value) in &execution.results {
        let json_val = value
            .to_json()
            .unwrap_or_else(|_| Value::String(value.to_string()));
        if content.is_none() {
            if let Some(text) = json_val.as_str() {
                content = Some(text.to_string());
            }
        }
        results_map.insert(token.to_string(), json_val);
    }

    let result = json!({
        "result": content.unwrap_or_default(),
        "results": results_map,
        "stats": {
            "duration_ms": compile_ms as u64 + exec_ms as u64,
            "compile_ms": compile_ms,
            "execute_ms": exec_ms,
            "executed_nodes": execution.stats.executed_nodes,
            "failed_nodes": execution.stats.failed_nodes,
        },
        "llm_usage": {
            "input_tokens": execution.llm_metrics.total_input_tokens,
            "output_tokens": execution.llm_metrics.total_output_tokens,
            "total_requests": execution.llm_metrics.total_requests,
        }
    });
    Ok(serde_json::to_string_pretty(&result).unwrap())
}

// ---------------------------------------------------------------------------
// Tool: apxm_merge
// ---------------------------------------------------------------------------

fn tool_merge(args: Value) -> Result<String, String> {
    let name = args
        .get("name")
        .and_then(Value::as_str)
        .ok_or("missing required argument: name")?;

    let graph_strings = args
        .get("graphs")
        .and_then(Value::as_array)
        .ok_or("missing required argument: graphs")?;

    if graph_strings.is_empty() {
        let merged = ApxmGraph::merge(name, &[]);
        let merged_json = serde_json::to_value(&merged)
            .map_err(|e| format!("failed to serialize merged graph: {e}"))?;
        let result = json!({
            "merged_graph": merged_json,
            "stats": {
                "input_graphs": 0,
                "total_nodes": 0,
                "total_edges": 0,
                "sync_node_id": null,
            }
        });
        return Ok(serde_json::to_string_pretty(&result).unwrap());
    }

    let mut graphs: Vec<ApxmGraph> = Vec::with_capacity(graph_strings.len());
    for (i, entry) in graph_strings.iter().enumerate() {
        let json_str = match entry.as_str() {
            Some(s) => s.to_string(),
            None => {
                // Accept inline JSON objects as well as strings
                serde_json::to_string(entry)
                    .map_err(|e| format!("graphs[{i}]: failed to serialize: {e}"))?
            }
        };
        let graph = ApxmGraph::from_json(&json_str)
            .map_err(|e| format!("graphs[{i}]: invalid graph: {e}"))?;
        graphs.push(graph);
    }

    let input_count = graphs.len();
    let merged = ApxmGraph::merge(name, &graphs);

    let total_nodes = merged.nodes.len();
    let total_edges = merged.edges.len();
    let sync_node_id = merged.nodes.last().map(|n| n.id);

    let merged_json = serde_json::to_value(&merged)
        .map_err(|e| format!("failed to serialize merged graph: {e}"))?;

    let result = json!({
        "merged_graph": merged_json,
        "stats": {
            "input_graphs": input_count,
            "total_nodes": total_nodes,
            "total_edges": total_edges,
            "sync_node_id": sync_node_id,
        }
    });
    Ok(serde_json::to_string_pretty(&result).unwrap())
}

// ---------------------------------------------------------------------------
// Tool: apxm_get_contract
// ---------------------------------------------------------------------------

fn tool_get_contract() -> Result<String, String> {
    let result = json!({
        "operations": {
            "AGENT": { "required_attributes": [], "description": "Agent metadata declaration (memory, beliefs, goals, capabilities)" },
            "QMEM": { "required_attributes": ["query"], "description": "Query memory (read from memory system)" },
            "UMEM": { "required_attributes": ["key", "value"], "description": "Update memory (write to memory system)" },
            "ASK": { "required_attributes": ["template_str"], "description": "Simple Q&A with LLM (no extended thinking)" },
            "THINK": { "required_attributes": ["template_str"], "description": "Extended thinking with token budget" },
            "REASON": { "required_attributes": ["template_str"], "description": "Structured reasoning with beliefs/goals" },
            "PLAN": { "required_attributes": ["goal"], "description": "Generate a plan using LLM" },
            "REFLECT": { "required_attributes": ["trace_id"], "description": "Analyze execution trace" },
            "VERIFY": { "required_attributes": ["condition"], "description": "Fact-check against evidence" },
            "INV": { "required_attributes": ["capability"], "description": "Invoke a capability (tool/function call)" },
            "EXC": { "required_attributes": ["code"], "description": "Execute code in sandbox" },
            "PRINT": { "required_attributes": ["message"], "description": "Print output to stdout" },
            "JUMP": { "required_attributes": ["label"], "description": "Unconditional jump to label" },
            "BRANCH_ON_VALUE": { "required_attributes": ["true_label", "false_label"], "description": "Branch based on value comparison" },
            "LOOP_START": { "required_attributes": ["count"], "description": "Loop start marker" },
            "LOOP_END": { "required_attributes": [], "description": "Loop end marker" },
            "RETURN": { "required_attributes": [], "description": "Return from subgraph with result" },
            "SWITCH": { "required_attributes": ["discriminant", "case_labels"], "description": "Multi-way branch based on string value" },
            "FLOW_CALL": { "required_attributes": ["agent_name", "flow_name"], "description": "Call a flow on another agent" },
            "MERGE": { "required_attributes": [], "description": "Merge multiple tokens into one" },
            "FENCE": { "required_attributes": [], "description": "Memory fence (synchronization barrier)" },
            "WAIT_ALL": { "required_attributes": [], "description": "Wait for all input tokens to be ready" },
            "TRY_CATCH": { "required_attributes": ["try_label", "catch_label"], "description": "Try-catch exception handling" },
            "ERR": { "required_attributes": ["recovery_template"], "description": "Error handler invocation" },
            "COMMUNICATE": { "required_attributes": ["target", "message"], "description": "Communication between agents" },
            "UPDATE_GOAL": { "required_attributes": ["goal_id"], "description": "Update agent goals at runtime (set/remove/clear)" },
            "GUARD": { "required_attributes": ["condition"], "description": "Enforce preconditions before execution continues" },
            "CLAIM": { "required_attributes": ["queue"], "description": "Atomically claim a task from a shared work queue" },
            "PAUSE": { "required_attributes": ["message"], "description": "Suspend execution pending human-in-the-loop review" },
            "RESUME": { "required_attributes": ["checkpoint"], "description": "Resume a suspended execution from a PAUSE checkpoint" },
            "CONST_STR": { "required_attributes": ["value"], "description": "String constant (compiler internal)" },
            "YIELD": { "required_attributes": [], "description": "Yield value from switch case region (compiler internal)" },
        },
        "dependency_types": ["Data", "Control", "Effect"],
        "parameter_types": ["str", "int", "float", "bool", "json"],
        "graph_schema": {
            "required_fields": ["name", "nodes", "edges"],
            "optional_fields": ["parameters", "metadata"],
        }
    });
    Ok(serde_json::to_string_pretty(&result).unwrap())
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn rpc_error(code: i64, message: impl Into<String>) -> Value {
    json!({
        "code": code,
        "message": message.into(),
    })
}
