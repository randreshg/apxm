//! LOOP_START operation - Loop initialization

use super::{ExecutionContext, Node, Result, Value};
use apxm_core::constants::graph::attrs as graph_attrs;

pub async fn execute(_ctx: &ExecutionContext, node: &Node, _inputs: Vec<Value>) -> Result<Value> {
    // Initialize loop counter
    let max_iterations = node
        .attributes
        .get(graph_attrs::MAX_ITERATIONS)
        .and_then(|v| v.as_u64())
        .unwrap_or(100);

    Ok(Value::Number(apxm_core::types::values::Number::Integer(
        max_iterations as i64,
    )))
}
