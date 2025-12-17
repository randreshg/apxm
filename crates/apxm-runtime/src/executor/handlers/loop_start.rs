//! LOOP_START operation - Loop initialization

use super::{ExecutionContext, Node, Result, Value};

pub async fn execute(_ctx: &ExecutionContext, node: &Node, _inputs: Vec<Value>) -> Result<Value> {
    // Initialize loop counter
    let max_iterations = node
        .attributes
        .get("max_iterations")
        .and_then(|v| v.as_u64())
        .unwrap_or(100);

    Ok(Value::Number(apxm_core::types::values::Number::Integer(
        max_iterations as i64,
    )))
}
