//! Integration tests for the APXM CLI.
//!
//! Tests the validate, analyze, ops, and template subcommands by invoking the
//! binary and checking stdout/stderr/exit-code.

use std::io::Write;
use std::process::Command;

fn apxm() -> Command {
    let mut cmd = Command::new(env!("CARGO_BIN_EXE_apxm"));
    // Prevent color codes in test output
    cmd.env("NO_COLOR", "1");
    cmd
}

fn write_tmp_graph(content: &str) -> tempfile::NamedTempFile {
    let mut f = tempfile::NamedTempFile::new().unwrap();
    f.write_all(content.as_bytes()).unwrap();
    f.flush().unwrap();
    f
}

// ─── Valid graphs ───────────────────────────────────────────────────────────

const VALID_ASK: &str = r#"{
  "name": "test-ask",
  "nodes": [{"id": 1, "name": "q", "op": "ASK", "attributes": {"template_str": "Hello"}}],
  "edges": [],
  "parameters": [],
  "metadata": {}
}"#;

const VALID_PIPELINE: &str = r#"{
  "name": "test-pipeline",
  "nodes": [
    {"id": 1, "name": "a", "op": "ASK", "attributes": {"template_str": "step 1"}},
    {"id": 2, "name": "b", "op": "ASK", "attributes": {"template_str": "step 2: {{node_1}}"}}
  ],
  "edges": [{"from": 1, "to": 2, "dependency": "Data"}],
  "parameters": [],
  "metadata": {}
}"#;

const VALID_PARALLEL: &str = r#"{
  "name": "test-parallel",
  "nodes": [
    {"id": 1, "name": "a", "op": "ASK", "attributes": {"template_str": "task a"}},
    {"id": 2, "name": "b", "op": "ASK", "attributes": {"template_str": "task b"}},
    {"id": 3, "name": "sync", "op": "WAIT_ALL", "attributes": {"tokens": ["{{node_1}}", "{{node_2}}"]}}
  ],
  "edges": [
    {"from": 1, "to": 3, "dependency": "Data"},
    {"from": 2, "to": 3, "dependency": "Data"}
  ],
  "parameters": [],
  "metadata": {}
}"#;

// ─── validate: valid graphs ─────────────────────────────────────────────────

#[test]
fn validate_valid_ask_json() {
    let f = write_tmp_graph(VALID_ASK);
    let out = apxm().args(["--json", "validate", f.path().to_str().unwrap()]).output().unwrap();
    assert!(out.status.success());
    let v: serde_json::Value = serde_json::from_slice(&out.stdout).unwrap();
    assert_eq!(v["valid"], true);
    assert!(v["errors"].as_array().unwrap().is_empty());
}

#[test]
fn validate_valid_pipeline_json() {
    let f = write_tmp_graph(VALID_PIPELINE);
    let out = apxm().args(["--json", "validate", f.path().to_str().unwrap()]).output().unwrap();
    assert!(out.status.success());
    let v: serde_json::Value = serde_json::from_slice(&out.stdout).unwrap();
    assert_eq!(v["valid"], true);
}

// ─── validate: error cases ──────────────────────────────────────────────────

#[test]
fn validate_empty_name() {
    let f = write_tmp_graph(r#"{"name":"","nodes":[{"id":1,"name":"n","op":"ASK","attributes":{"template_str":"x"}}],"edges":[],"parameters":[]}"#);
    let out = apxm().args(["--json", "validate", f.path().to_str().unwrap()]).output().unwrap();
    assert!(!out.status.success());
    let v: serde_json::Value = serde_json::from_slice(&out.stdout).unwrap();
    assert_eq!(v["valid"], false);
    assert!(v["errors"].as_array().unwrap().iter().any(|e| e.as_str().unwrap().contains("name must not be empty")));
}

#[test]
fn validate_unknown_op() {
    let f = write_tmp_graph(r#"{"name":"t","nodes":[{"id":1,"name":"n","op":"FAKE_OP","attributes":{}}],"edges":[],"parameters":[]}"#);
    let out = apxm().args(["--json", "validate", f.path().to_str().unwrap()]).output().unwrap();
    assert!(!out.status.success());
    let v: serde_json::Value = serde_json::from_slice(&out.stdout).unwrap();
    assert!(v["errors"].as_array().unwrap().iter().any(|e| e.as_str().unwrap().contains("unknown op")));
}

#[test]
fn validate_missing_required_attribute() {
    let f = write_tmp_graph(r#"{"name":"t","nodes":[{"id":1,"name":"n","op":"ASK","attributes":{}}],"edges":[],"parameters":[]}"#);
    let out = apxm().args(["--json", "validate", f.path().to_str().unwrap()]).output().unwrap();
    assert!(!out.status.success());
    let v: serde_json::Value = serde_json::from_slice(&out.stdout).unwrap();
    assert!(v["errors"].as_array().unwrap().iter().any(|e| e.as_str().unwrap().contains("template_str")));
}

#[test]
fn validate_duplicate_node_id() {
    let f = write_tmp_graph(r#"{"name":"t","nodes":[{"id":1,"name":"a","op":"ASK","attributes":{"template_str":"x"}},{"id":1,"name":"b","op":"ASK","attributes":{"template_str":"y"}}],"edges":[],"parameters":[]}"#);
    let out = apxm().args(["--json", "validate", f.path().to_str().unwrap()]).output().unwrap();
    assert!(!out.status.success());
    let v: serde_json::Value = serde_json::from_slice(&out.stdout).unwrap();
    assert!(v["errors"].as_array().unwrap().iter().any(|e| e.as_str().unwrap().contains("duplicate node id")));
}

#[test]
fn validate_self_loop() {
    let f = write_tmp_graph(r#"{"name":"t","nodes":[{"id":1,"name":"n","op":"ASK","attributes":{"template_str":"x"}}],"edges":[{"from":1,"to":1,"dependency":"Data"}],"parameters":[]}"#);
    let out = apxm().args(["--json", "validate", f.path().to_str().unwrap()]).output().unwrap();
    assert!(!out.status.success());
    let v: serde_json::Value = serde_json::from_slice(&out.stdout).unwrap();
    assert!(v["errors"].as_array().unwrap().iter().any(|e| e.as_str().unwrap().contains("self-loop")));
}

#[test]
fn validate_invalid_dependency_type() {
    let f = write_tmp_graph(r#"{"name":"t","nodes":[{"id":1,"name":"a","op":"ASK","attributes":{"template_str":"x"}},{"id":2,"name":"b","op":"ASK","attributes":{"template_str":"y"}}],"edges":[{"from":1,"to":2,"dependency":"Invalid"}],"parameters":[]}"#);
    let out = apxm().args(["--json", "validate", f.path().to_str().unwrap()]).output().unwrap();
    assert!(!out.status.success());
    let v: serde_json::Value = serde_json::from_slice(&out.stdout).unwrap();
    assert!(v["errors"].as_array().unwrap().iter().any(|e| e.as_str().unwrap().contains("invalid dependency type")));
}

#[test]
fn validate_edge_nonexistent_target() {
    let f = write_tmp_graph(r#"{"name":"t","nodes":[{"id":1,"name":"n","op":"ASK","attributes":{"template_str":"x"}}],"edges":[{"from":1,"to":99,"dependency":"Data"}],"parameters":[]}"#);
    let out = apxm().args(["--json", "validate", f.path().to_str().unwrap()]).output().unwrap();
    assert!(!out.status.success());
    let v: serde_json::Value = serde_json::from_slice(&out.stdout).unwrap();
    assert!(v["errors"].as_array().unwrap().iter().any(|e| e.as_str().unwrap().contains("target node")));
}

#[test]
fn validate_cycle_detection() {
    let f = write_tmp_graph(r#"{"name":"t","nodes":[{"id":1,"name":"a","op":"ASK","attributes":{"template_str":"x"}},{"id":2,"name":"b","op":"ASK","attributes":{"template_str":"y"}}],"edges":[{"from":1,"to":2,"dependency":"Data"},{"from":2,"to":1,"dependency":"Data"}],"parameters":[]}"#);
    let out = apxm().args(["--json", "validate", f.path().to_str().unwrap()]).output().unwrap();
    assert!(!out.status.success());
    let v: serde_json::Value = serde_json::from_slice(&out.stdout).unwrap();
    assert!(v["errors"].as_array().unwrap().iter().any(|e| e.as_str().unwrap().contains("cycle")));
}

#[test]
fn validate_file_not_found() {
    let out = apxm().args(["validate", "/nonexistent/path.json"]).output().unwrap();
    assert!(!out.status.success());
}

#[test]
fn validate_invalid_json() {
    let f = write_tmp_graph("{not valid json");
    let out = apxm().args(["validate", f.path().to_str().unwrap()]).output().unwrap();
    assert!(!out.status.success());
}

// ─── analyze ────────────────────────────────────────────────────────────────

#[test]
fn analyze_parallel_graph() {
    let f = write_tmp_graph(VALID_PARALLEL);
    let out = apxm().args(["--json", "analyze", f.path().to_str().unwrap()]).output().unwrap();
    assert!(out.status.success());
    let v: serde_json::Value = serde_json::from_slice(&out.stdout).unwrap();
    assert_eq!(v["max_parallelism"], 2);
    assert_eq!(v["depth"], 2);
    assert_eq!(v["node_count"], 3);
    assert_eq!(v["edge_count"], 2);
}

#[test]
fn analyze_single_node() {
    let f = write_tmp_graph(VALID_ASK);
    let out = apxm().args(["--json", "analyze", f.path().to_str().unwrap()]).output().unwrap();
    assert!(out.status.success());
    let v: serde_json::Value = serde_json::from_slice(&out.stdout).unwrap();
    assert_eq!(v["max_parallelism"], 1);
    assert_eq!(v["depth"], 1);
    assert_eq!(v["speedup"]["estimated_speedup"], "1.00x");
}

#[test]
fn analyze_pipeline_graph() {
    let f = write_tmp_graph(VALID_PIPELINE);
    let out = apxm().args(["--json", "analyze", f.path().to_str().unwrap()]).output().unwrap();
    assert!(out.status.success());
    let v: serde_json::Value = serde_json::from_slice(&out.stdout).unwrap();
    assert_eq!(v["max_parallelism"], 1);
    assert_eq!(v["depth"], 2);
}

// ─── ops ────────────────────────────────────────────────────────────────────

#[test]
fn ops_list_json() {
    let out = apxm().args(["--json", "ops", "list"]).output().unwrap();
    assert!(out.status.success());
    let v: serde_json::Value = serde_json::from_slice(&out.stdout).unwrap();
    let ops = v.as_array().unwrap();
    assert!(ops.len() >= 30, "expected at least 30 ops, got {}", ops.len());
    // Check every op has required fields
    for op in ops {
        assert!(op["op"].is_string());
        assert!(op["category"].is_string());
        assert!(op["description"].is_string());
    }
}

#[test]
fn ops_list_category_filter() {
    let out = apxm().args(["--json", "ops", "list", "--category", "reasoning"]).output().unwrap();
    assert!(out.status.success());
    let v: serde_json::Value = serde_json::from_slice(&out.stdout).unwrap();
    let ops = v.as_array().unwrap();
    for op in ops {
        assert_eq!(op["category"], "reasoning");
    }
    assert!(ops.len() >= 4, "expected at least 4 reasoning ops");
}

#[test]
fn ops_show_ask_json() {
    let out = apxm().args(["--json", "ops", "show", "ASK"]).output().unwrap();
    assert!(out.status.success());
    let v: serde_json::Value = serde_json::from_slice(&out.stdout).unwrap();
    assert_eq!(v["op"], "ASK");
    assert_eq!(v["category"], "reasoning");
    assert!(v["required_fields"].as_array().unwrap().iter().any(|f| f["name"] == "template_str"));
}

#[test]
fn ops_show_unknown() {
    let out = apxm().args(["ops", "show", "NONEXISTENT"]).output().unwrap();
    assert!(!out.status.success());
}

// ─── template ───────────────────────────────────────────────────────────────

#[test]
fn template_list_json() {
    let out = apxm().args(["--json", "template", "list"]).output().unwrap();
    assert!(out.status.success());
    let v: serde_json::Value = serde_json::from_slice(&out.stdout).unwrap();
    let templates = v.as_array().unwrap();
    assert!(templates.len() >= 6);
    for t in templates {
        assert!(t["name"].is_string());
        assert!(t["description"].is_string());
    }
}

#[test]
fn template_show_ask_json() {
    let out = apxm().args(["--json", "template", "show", "ask"]).output().unwrap();
    assert!(out.status.success());
    // Output should be valid JSON graph
    let v: serde_json::Value = serde_json::from_slice(&out.stdout).unwrap();
    assert!(v["name"].is_string());
    assert!(v["nodes"].is_array());
    assert!(v["edges"].is_array());
}

#[test]
fn template_show_unknown() {
    let out = apxm().args(["template", "show", "nonexistent"]).output().unwrap();
    assert!(!out.status.success());
}

#[test]
fn template_roundtrip_validate() {
    // Get template JSON and feed it to validate
    let show_out = apxm().args(["--json", "template", "show", "map-reduce"]).output().unwrap();
    assert!(show_out.status.success());

    let f = write_tmp_graph(std::str::from_utf8(&show_out.stdout).unwrap());
    let val_out = apxm().args(["--json", "validate", f.path().to_str().unwrap()]).output().unwrap();
    assert!(val_out.status.success());
    let v: serde_json::Value = serde_json::from_slice(&val_out.stdout).unwrap();
    assert_eq!(v["valid"], true);
}

// ─── validate: edge cases ───────────────────────────────────────────────────

#[test]
fn validate_no_nodes() {
    let f = write_tmp_graph(r#"{"name":"t","nodes":[],"edges":[],"parameters":[]}"#);
    let out = apxm().args(["--json", "validate", f.path().to_str().unwrap()]).output().unwrap();
    assert!(!out.status.success());
    let v: serde_json::Value = serde_json::from_slice(&out.stdout).unwrap();
    assert_eq!(v["valid"], false);
}

#[test]
fn validate_edge_nonexistent_source() {
    let f = write_tmp_graph(r#"{"name":"t","nodes":[{"id":1,"name":"n","op":"ASK","attributes":{"template_str":"x"}}],"edges":[{"from":99,"to":1,"dependency":"Data"}],"parameters":[]}"#);
    let out = apxm().args(["--json", "validate", f.path().to_str().unwrap()]).output().unwrap();
    assert!(!out.status.success());
    let v: serde_json::Value = serde_json::from_slice(&out.stdout).unwrap();
    assert!(v["errors"].as_array().unwrap().iter().any(|e| e.as_str().unwrap().contains("source")));
}

#[test]
fn validate_duplicate_parameter_name() {
    let f = write_tmp_graph(r#"{"name":"t","nodes":[{"id":1,"name":"n","op":"ASK","attributes":{"template_str":"x"}}],"edges":[],"parameters":[{"name":"p","type_name":"str"},{"name":"p","type_name":"int"}]}"#);
    let out = apxm().args(["--json", "validate", f.path().to_str().unwrap()]).output().unwrap();
    // Should have warning or error about duplicate parameter
    let v: serde_json::Value = serde_json::from_slice(&out.stdout).unwrap();
    let has_dup = v["errors"].as_array().unwrap_or(&vec![]).iter()
        .chain(v["warnings"].as_array().unwrap_or(&vec![]).iter())
        .any(|e| e.as_str().unwrap_or("").contains("duplicate"));
    assert!(has_dup);
}

#[test]
fn validate_disconnected_graph() {
    // Two nodes with no edges — valid but might get a warning
    let f = write_tmp_graph(r#"{"name":"t","nodes":[{"id":1,"name":"a","op":"ASK","attributes":{"template_str":"x"}},{"id":2,"name":"b","op":"ASK","attributes":{"template_str":"y"}}],"edges":[],"parameters":[]}"#);
    let out = apxm().args(["--json", "validate", f.path().to_str().unwrap()]).output().unwrap();
    assert!(out.status.success());
    let v: serde_json::Value = serde_json::from_slice(&out.stdout).unwrap();
    assert_eq!(v["valid"], true);
}

#[test]
fn validate_parameter_invalid_type() {
    let f = write_tmp_graph(r#"{"name":"t","nodes":[{"id":1,"name":"n","op":"ASK","attributes":{"template_str":"x"}}],"edges":[],"parameters":[{"name":"p","type_name":"invalid_type"}]}"#);
    let out = apxm().args(["--json", "validate", f.path().to_str().unwrap()]).output().unwrap();
    // Should succeed but with warning about non-standard type
    let v: serde_json::Value = serde_json::from_slice(&out.stdout).unwrap();
    assert!(v["warnings"].as_array().unwrap().iter().any(|w| w.as_str().unwrap().contains("non-standard")));
}

// ─── analyze: edge cases ────────────────────────────────────────────────────

#[test]
fn analyze_disconnected_components() {
    // Two independent nodes — both should be in phase 1
    let f = write_tmp_graph(r#"{"name":"t","nodes":[{"id":1,"name":"a","op":"ASK","attributes":{"template_str":"x"}},{"id":2,"name":"b","op":"ASK","attributes":{"template_str":"y"}}],"edges":[],"parameters":[]}"#);
    let out = apxm().args(["--json", "analyze", f.path().to_str().unwrap()]).output().unwrap();
    assert!(out.status.success());
    let v: serde_json::Value = serde_json::from_slice(&out.stdout).unwrap();
    assert_eq!(v["max_parallelism"], 2);
    assert_eq!(v["depth"], 1);
}

#[test]
fn analyze_includes_entry_exit_nodes() {
    let f = write_tmp_graph(VALID_PIPELINE);
    let out = apxm().args(["--json", "analyze", f.path().to_str().unwrap()]).output().unwrap();
    assert!(out.status.success());
    let v: serde_json::Value = serde_json::from_slice(&out.stdout).unwrap();
    assert!(v["entry_nodes"].is_array());
    assert!(v["exit_nodes"].is_array());
}

// ─── ops: edge cases ────────────────────────────────────────────────────────

#[test]
fn ops_list_has_coordination_ops() {
    let out = apxm().args(["--json", "ops", "list"]).output().unwrap();
    assert!(out.status.success());
    let v: serde_json::Value = serde_json::from_slice(&out.stdout).unwrap();
    let ops: Vec<String> = v.as_array().unwrap().iter()
        .map(|o| o["op"].as_str().unwrap().to_string())
        .collect();
    assert!(ops.contains(&"UPDATE_GOAL".to_string()));
    assert!(ops.contains(&"GUARD".to_string()));
    assert!(ops.contains(&"CLAIM".to_string()));
    assert!(ops.contains(&"PAUSE".to_string()));
    assert!(ops.contains(&"RESUME".to_string()));
}

#[test]
fn ops_show_has_long_description() {
    let out = apxm().args(["--json", "ops", "show", "THINK"]).output().unwrap();
    assert!(out.status.success());
    let v: serde_json::Value = serde_json::from_slice(&out.stdout).unwrap();
    assert!(v["long_description"].is_string());
    assert!(!v["long_description"].as_str().unwrap().is_empty());
}

// ─── explain ───────────────────────────────────────────────────────────────

#[test]
fn explain_single_node_json() {
    let f = write_tmp_graph(VALID_ASK);
    let out = apxm().args(["--json", "explain", f.path().to_str().unwrap()]).output().unwrap();
    assert!(out.status.success());
    let v: serde_json::Value = serde_json::from_slice(&out.stdout).unwrap();
    assert_eq!(v["graph_name"], "test-ask");
    assert_eq!(v["node_count"], 1);
    assert_eq!(v["edge_count"], 0);
    assert_eq!(v["depth"], 1);
    let flow = v["execution_flow"].as_array().unwrap();
    assert_eq!(flow.len(), 1);
    assert_eq!(flow[0]["phase"], 1);
    let nodes = flow[0]["nodes"].as_array().unwrap();
    assert_eq!(nodes.len(), 1);
    assert_eq!(nodes[0]["op"], "ASK");
}

#[test]
fn explain_pipeline_json() {
    let f = write_tmp_graph(VALID_PIPELINE);
    let out = apxm().args(["--json", "explain", f.path().to_str().unwrap()]).output().unwrap();
    assert!(out.status.success());
    let v: serde_json::Value = serde_json::from_slice(&out.stdout).unwrap();
    assert_eq!(v["depth"], 2);
    let flow = v["execution_flow"].as_array().unwrap();
    assert_eq!(flow.len(), 2);
    // Phase 1: single node, Phase 2: single node
    assert_eq!(flow[0]["nodes"].as_array().unwrap().len(), 1);
    assert_eq!(flow[1]["nodes"].as_array().unwrap().len(), 1);
}

#[test]
fn explain_parallel_json() {
    let f = write_tmp_graph(VALID_PARALLEL);
    let out = apxm().args(["--json", "explain", f.path().to_str().unwrap()]).output().unwrap();
    assert!(out.status.success());
    let v: serde_json::Value = serde_json::from_slice(&out.stdout).unwrap();
    let flow = v["execution_flow"].as_array().unwrap();
    // Phase 1 should have 2 parallel nodes
    assert!(flow[0]["parallel"].as_bool().unwrap());
    assert_eq!(flow[0]["nodes"].as_array().unwrap().len(), 2);
    assert!(v["summary"]["max_parallelism"].as_u64().unwrap() >= 2);
}

#[test]
fn explain_human_readable() {
    let f = write_tmp_graph(VALID_PIPELINE);
    let out = apxm().args(["explain", f.path().to_str().unwrap()]).output().unwrap();
    assert!(out.status.success());
    let stdout = String::from_utf8_lossy(&out.stdout);
    assert!(stdout.contains("Phase 1"));
    assert!(stdout.contains("Phase 2"));
}

#[test]
fn explain_file_not_found() {
    let out = apxm().args(["explain", "/nonexistent/file.json"]).output().unwrap();
    assert!(!out.status.success());
}

#[test]
fn explain_node_metadata() {
    let f = write_tmp_graph(VALID_ASK);
    let out = apxm().args(["--json", "explain", f.path().to_str().unwrap()]).output().unwrap();
    assert!(out.status.success());
    let v: serde_json::Value = serde_json::from_slice(&out.stdout).unwrap();
    let node = &v["execution_flow"][0]["nodes"][0];
    assert!(node["category"].is_string());
    assert!(node["description"].is_string());
    assert!(node["latency"].is_string());
    assert!(node["latency_ms"].is_number());
}

// ─── codelet merge ─────────────────────────────────────────────────────────

#[test]
fn codelet_merge_two_graphs_json() {
    let g1 = write_tmp_graph(VALID_ASK);
    let g2 = write_tmp_graph(VALID_PIPELINE);
    let out = apxm().args([
        "--json", "codelet", "merge",
        g1.path().to_str().unwrap(),
        g2.path().to_str().unwrap(),
        "--name", "merged-test",
    ]).output().unwrap();
    assert!(out.status.success());
    let v: serde_json::Value = serde_json::from_slice(&out.stdout).unwrap();
    assert_eq!(v["stats"]["input_graphs"], 2);
    // 1 node from g1 + 2 from g2 + 1 sync node = 4
    assert!(v["stats"]["total_nodes"].as_u64().unwrap() >= 3);
    assert!(v["merged_graph"]["name"].as_str().unwrap() == "merged-test");
}

#[test]
fn codelet_merge_output_file() {
    let g1 = write_tmp_graph(VALID_ASK);
    let out_dir = tempfile::tempdir().unwrap();
    let out_path = out_dir.path().join("merged.json");
    let out = apxm().args([
        "--json", "codelet", "merge",
        g1.path().to_str().unwrap(),
        "--name", "single-merge",
        "-o", out_path.to_str().unwrap(),
    ]).output().unwrap();
    assert!(out.status.success());
    // File should exist and be valid JSON
    let content = std::fs::read_to_string(&out_path).unwrap();
    let v: serde_json::Value = serde_json::from_str(&content).unwrap();
    assert!(v["merged_graph"].is_object());
}

#[test]
fn codelet_merge_file_not_found() {
    let out = apxm().args([
        "codelet", "merge",
        "/nonexistent/a.json",
        "--name", "fail",
    ]).output().unwrap();
    assert!(!out.status.success());
}

// ─── tools ─────────────────────────────────────────────────────────────────

#[test]
fn tools_list_empty_json() {
    // Use a temp HOME to avoid reading real ~/.apxm/tools.json
    let tmp_home = tempfile::tempdir().unwrap();
    let out = apxm()
        .env("HOME", tmp_home.path())
        .args(["--json", "tools", "list"])
        .output()
        .unwrap();
    assert!(out.status.success());
    let v: serde_json::Value = serde_json::from_slice(&out.stdout).unwrap();
    assert!(v.as_array().unwrap().is_empty());
}

#[test]
fn tools_register_and_list() {
    let tmp_home = tempfile::tempdir().unwrap();
    // Register a tool
    let out = apxm()
        .env("HOME", tmp_home.path())
        .args(["tools", "register", "test-search", "--description", "Test search tool"])
        .output()
        .unwrap();
    assert!(out.status.success());

    // List tools, should contain the registered tool
    let out = apxm()
        .env("HOME", tmp_home.path())
        .args(["--json", "tools", "list"])
        .output()
        .unwrap();
    assert!(out.status.success());
    let v: serde_json::Value = serde_json::from_slice(&out.stdout).unwrap();
    let tools = v.as_array().unwrap();
    assert_eq!(tools.len(), 1);
    assert_eq!(tools[0]["name"], "test-search");
    assert_eq!(tools[0]["description"], "Test search tool");
}

#[test]
fn tools_register_duplicate_fails() {
    let tmp_home = tempfile::tempdir().unwrap();
    // Register once
    apxm()
        .env("HOME", tmp_home.path())
        .args(["tools", "register", "dup-tool", "--description", "First"])
        .output()
        .unwrap();
    // Register same name again
    let out = apxm()
        .env("HOME", tmp_home.path())
        .args(["tools", "register", "dup-tool", "--description", "Second"])
        .output()
        .unwrap();
    assert!(!out.status.success());
}

#[test]
fn tools_remove() {
    let tmp_home = tempfile::tempdir().unwrap();
    // Register then remove
    apxm()
        .env("HOME", tmp_home.path())
        .args(["tools", "register", "rm-tool", "--description", "To be removed"])
        .output()
        .unwrap();
    let out = apxm()
        .env("HOME", tmp_home.path())
        .args(["tools", "remove", "rm-tool"])
        .output()
        .unwrap();
    assert!(out.status.success());

    // Verify it's gone
    let out = apxm()
        .env("HOME", tmp_home.path())
        .args(["--json", "tools", "list"])
        .output()
        .unwrap();
    let v: serde_json::Value = serde_json::from_slice(&out.stdout).unwrap();
    assert!(v.as_array().unwrap().is_empty());
}

#[test]
fn tools_remove_nonexistent_fails() {
    let tmp_home = tempfile::tempdir().unwrap();
    let out = apxm()
        .env("HOME", tmp_home.path())
        .args(["tools", "remove", "no-such-tool"])
        .output()
        .unwrap();
    assert!(!out.status.success());
}

// ─── doctor ────────────────────────────────────────────────────────────────

#[test]
fn doctor_json_output() {
    let out = apxm().args(["--json", "doctor"]).output().unwrap();
    assert!(out.status.success());
    let v: serde_json::Value = serde_json::from_slice(&out.stdout).unwrap();
    assert!(v["mlir"].is_object());
    assert!(v["mlir"]["available"].is_boolean());
    assert!(v["credentials"].is_object());
    assert!(v["credentials"]["count"].is_number());
    assert!(v["environment"].is_object());
}

#[test]
fn doctor_human_readable() {
    let out = apxm().args(["doctor"]).output().unwrap();
    // doctor may exit non-zero when MLIR is missing — that's expected in CI/test
    let stdout = String::from_utf8_lossy(&out.stdout);
    assert!(stdout.contains("MLIR") || stdout.contains("mlir"));
}
