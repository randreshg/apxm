//! SWITCH operation - Region-based multi-way conditional branching
//!
//! The switch operation evaluates a discriminant value against case labels
//! and executes only the matching case region (or default if no match).

use std::collections::HashMap;

use super::{ExecutionContext, Node, Result, Value};
use apxm_core::{
    error::RuntimeError,
    types::{
        DependencyType, TokenId,
        execution::{Edge, ExecutionDag},
        operations::AISOperationType,
    },
};

/// Execute a switch operation with region-based execution
///
/// Matches the discriminant value against case labels and executes
/// only the matching case region's sub-DAG, returning its result (if any).
pub async fn execute(ctx: &ExecutionContext, node: &Node, inputs: Vec<Value>) -> Result<Value> {
    // Get the discriminant value (first input)
    let discriminant_value = if !inputs.is_empty() {
        inputs[0]
            .as_string()
            .map(|s| s.to_string())
            .unwrap_or_default()
    } else {
        String::new()
    };

    tracing::debug!(
        discriminant = %discriminant_value,
        node_id = node.id,
        "Switch: evaluating discriminant"
    );

    // Get case labels array
    let case_labels = node
        .attributes
        .get("case_labels")
        .and_then(|v| v.as_array())
        .map(|arr| {
            arr.iter()
                .filter_map(|v| v.as_string().map(|s| s.to_string()))
                .collect::<Vec<_>>()
        })
        .unwrap_or_default();

    // Find matching case index
    let matched_index = case_labels
        .iter()
        .position(|label| label == &discriminant_value);

    tracing::debug!(
        matched_index = ?matched_index,
        case_labels = ?case_labels,
        "Switch: found match"
    );

    // Get the sub-DAG to execute
    let sub_dag_value = if let Some(idx) = matched_index {
        // Get case_regions array and select the matching one
        node.attributes
            .get("case_regions")
            .and_then(|v| v.as_array())
            .and_then(|arr| arr.get(idx))
            .cloned()
    } else {
        // Use default region
        node.attributes.get("default_region").cloned()
    };

    // If no sub-DAG found, this is a void switch or empty case
    let sub_dag_value = match sub_dag_value {
        Some(v) => v,
        None => {
            tracing::debug!("Switch: no sub-DAG found for case, returning null");
            return Ok(Value::Null);
        }
    };

    // Parse sub-DAG from the Value
    let sub_dag = parse_sub_dag(&sub_dag_value)?;

    if sub_dag.nodes.is_empty() {
        tracing::debug!("Switch: sub-DAG is empty, returning null");
        return Ok(Value::Null);
    }

    // Build token connections: map sub-DAG entry tokens to switch node's input tokens
    let token_connections = build_token_connections(&sub_dag, node);

    tracing::debug!(
        sub_dag_nodes = sub_dag.nodes.len(),
        token_connections = ?token_connections,
        "Switch: splicing sub-DAG"
    );

    // Mark switch's output tokens as delegated BEFORE splicing
    // This prevents the switch handler's Null return from being published to these tokens
    // The sub-DAG will produce the actual values
    if !node.output_tokens.is_empty() {
        ctx.dag_splicer
            .mark_tokens_delegated(node.id, &node.output_tokens);
        tracing::debug!(
            output_tokens = ?node.output_tokens,
            node_id = node.id,
            "Switch: marked output tokens as delegated to sub-DAG"
        );
    }

    // Splice the sub-DAG into execution
    match ctx.dag_splicer.splice_dag(sub_dag, token_connections).await {
        Ok(_token_mapping) => {
            // The spliced DAG will produce the result via the switch node's output token
            // Return null - the actual result flows through the token graph
            Ok(Value::Null)
        }
        Err(e) => {
            // If splicing not supported, return null (void execution)
            if e.to_string().contains("not supported") {
                tracing::warn!("Switch: DAG splicing not supported, returning null");
                Ok(Value::Null)
            } else {
                Err(e)
            }
        }
    }
}

/// Parse a sub-DAG from a Value object
fn parse_sub_dag(value: &Value) -> Result<ExecutionDag> {
    let obj = value.as_object().ok_or_else(|| RuntimeError::Operation {
        op_type: AISOperationType::Switch,
        message: "Sub-DAG must be an object".to_string(),
    })?;

    let mut dag = ExecutionDag::new();

    // Parse nodes
    if let Some(nodes_value) = obj.get("nodes")
        && let Some(nodes_arr) = nodes_value.as_array()
    {
        for node_value in nodes_arr {
            if let Some(parsed_node) = parse_node(node_value)? {
                dag.nodes.push(parsed_node);
            }
        }
    }

    // Parse edges
    if let Some(edges_value) = obj.get("edges")
        && let Some(edges_arr) = edges_value.as_array()
    {
        for edge_value in edges_arr {
            if let Some(parsed_edge) = parse_edge(edge_value)? {
                dag.edges.push(parsed_edge);
            }
        }
    }

    // Parse entry nodes
    if let Some(entry_value) = obj.get("entry_nodes")
        && let Some(entry_arr) = entry_value.as_array()
    {
        dag.entry_nodes = entry_arr
            .iter()
            .filter_map(|v| v.as_i64().map(|i| i as u64))
            .collect();
    }

    // Parse exit nodes
    if let Some(exit_value) = obj.get("exit_nodes")
        && let Some(exit_arr) = exit_value.as_array()
    {
        dag.exit_nodes = exit_arr
            .iter()
            .filter_map(|v| v.as_i64().map(|i| i as u64))
            .collect();
    }

    Ok(dag)
}

/// Parse a single node from a Value object
fn parse_node(value: &Value) -> Result<Option<apxm_core::types::execution::Node>> {
    let obj = match value.as_object() {
        Some(o) => o,
        None => return Ok(None),
    };

    let id = obj
        .get("id")
        .and_then(|v| v.as_i64())
        .map(|i| i as u64)
        .ok_or_else(|| RuntimeError::Operation {
            op_type: AISOperationType::Switch,
            message: "Node missing id".to_string(),
        })?;

    let op_type_num =
        obj.get("op_type")
            .and_then(|v| v.as_i64())
            .ok_or_else(|| RuntimeError::Operation {
                op_type: AISOperationType::Switch,
                message: "Node missing op_type".to_string(),
            })?;

    let op_type = map_op_type(op_type_num as u32).ok_or_else(|| RuntimeError::Operation {
        op_type: AISOperationType::Switch,
        message: format!("Unknown operation type: {}", op_type_num),
    })?;

    let mut node = apxm_core::types::execution::Node::new(id, op_type);

    // Parse attributes
    if let Some(attrs_value) = obj.get("attributes")
        && let Some(attrs_obj) = attrs_value.as_object()
    {
        for (key, val) in attrs_obj {
            node.attributes.insert(key.clone(), val.clone());
        }
    }

    // Parse input tokens
    if let Some(input_value) = obj.get("input_tokens")
        && let Some(input_arr) = input_value.as_array()
    {
        node.input_tokens = input_arr
            .iter()
            .filter_map(|v| v.as_i64().map(|i| i as TokenId))
            .collect();
    }

    // Parse output tokens
    if let Some(output_value) = obj.get("output_tokens")
        && let Some(output_arr) = output_value.as_array()
    {
        node.output_tokens = output_arr
            .iter()
            .filter_map(|v| v.as_i64().map(|i| i as TokenId))
            .collect();
    }

    Ok(Some(node))
}

/// Parse a single edge from a Value object
fn parse_edge(value: &Value) -> Result<Option<Edge>> {
    let obj = match value.as_object() {
        Some(o) => o,
        None => return Ok(None),
    };

    let from = obj
        .get("from")
        .and_then(|v| v.as_i64())
        .map(|i| i as u64)
        .ok_or_else(|| RuntimeError::Operation {
            op_type: AISOperationType::Switch,
            message: "Edge missing from".to_string(),
        })?;

    let to = obj
        .get("to")
        .and_then(|v| v.as_i64())
        .map(|i| i as u64)
        .ok_or_else(|| RuntimeError::Operation {
            op_type: AISOperationType::Switch,
            message: "Edge missing to".to_string(),
        })?;

    let token = obj
        .get("token")
        .and_then(|v| v.as_i64())
        .map(|i| i as TokenId)
        .ok_or_else(|| RuntimeError::Operation {
            op_type: AISOperationType::Switch,
            message: "Edge missing token".to_string(),
        })?;

    let dependency = obj
        .get("dependency")
        .and_then(|v| v.as_i64())
        .map(|i| match i {
            0 => DependencyType::Data,
            1 => DependencyType::Effect,
            2 => DependencyType::Control,
            _ => DependencyType::Data,
        })
        .unwrap_or(DependencyType::Data);

    Ok(Some(Edge::new(from, to, token, dependency)))
}

/// Map operation type number to AISOperationType
/// Must match OP_KIND_MAP in artifact.rs
fn map_op_type(num: u32) -> Option<AISOperationType> {
    match num {
        0 => Some(AISOperationType::Inv),
        1 => Some(AISOperationType::Ask), // was Rsn
        2 => Some(AISOperationType::QMem),
        3 => Some(AISOperationType::UMem),
        4 => Some(AISOperationType::Plan),
        5 => Some(AISOperationType::WaitAll),
        6 => Some(AISOperationType::Merge),
        7 => Some(AISOperationType::Fence),
        8 => Some(AISOperationType::Exc),
        9 => Some(AISOperationType::Communicate),
        10 => Some(AISOperationType::Reflect),
        11 => Some(AISOperationType::Verify),
        12 => Some(AISOperationType::Err),
        13 => Some(AISOperationType::Return),
        14 => Some(AISOperationType::Jump),
        15 => Some(AISOperationType::BranchOnValue),
        16 => Some(AISOperationType::LoopStart),
        17 => Some(AISOperationType::LoopEnd),
        18 => Some(AISOperationType::TryCatch),
        19 => Some(AISOperationType::ConstStr),
        20 => Some(AISOperationType::Switch),
        21 => Some(AISOperationType::FlowCall),
        22 => Some(AISOperationType::Print),
        23 => Some(AISOperationType::Think),
        24 => Some(AISOperationType::Reason),
        _ => None,
    }
}

/// Build token connections from sub-DAG to parent switch node
fn build_token_connections(
    sub_dag: &ExecutionDag,
    switch_node: &Node,
) -> HashMap<TokenId, TokenId> {
    let mut connections = HashMap::new();

    // Find dangling input tokens (not produced by any node in sub-DAG)
    let produced: std::collections::HashSet<TokenId> = sub_dag
        .nodes
        .iter()
        .flat_map(|n| n.output_tokens.iter().copied())
        .collect();

    let mut required_inputs = Vec::new();
    let mut seen = std::collections::HashSet::new();
    for node in &sub_dag.nodes {
        for &token in &node.input_tokens {
            if !produced.contains(&token) && seen.insert(token) {
                required_inputs.push(token);
            }
        }
    }

    // Map sub-DAG's dangling inputs to switch node's inputs
    for (inner, outer) in required_inputs
        .into_iter()
        .zip(switch_node.input_tokens.iter().copied())
    {
        connections.insert(inner, outer);
    }

    // Map sub-DAG's exit tokens to switch node's outputs
    let consumers: HashMap<TokenId, usize> = sub_dag
        .nodes
        .iter()
        .flat_map(|n| n.input_tokens.iter().copied())
        .fold(HashMap::new(), |mut acc, tok| {
            *acc.entry(tok).or_insert(0) += 1;
            acc
        });

    let mut exit_tokens = Vec::new();
    for node in &sub_dag.nodes {
        for &token in &node.output_tokens {
            if consumers.get(&token).copied().unwrap_or(0) == 0 {
                exit_tokens.push(token);
            }
        }
    }

    for (inner, outer) in exit_tokens
        .into_iter()
        .zip(switch_node.output_tokens.iter().copied())
    {
        connections.insert(inner, outer);
    }

    connections
}
