//! ERR operation - Create error value

use super::{ExecutionContext, Node, Result, Value, get_string_attribute};
use apxm_core::constants::graph::attrs as graph_attrs;

pub async fn execute(_ctx: &ExecutionContext, node: &Node, _inputs: Vec<Value>) -> Result<Value> {
    let message = get_string_attribute(node, graph_attrs::MESSAGE)?;

    // Return error message as string value
    // Actual error propagation handled by scheduler
    Ok(Value::String(format!("Error: {}", message)))
}
