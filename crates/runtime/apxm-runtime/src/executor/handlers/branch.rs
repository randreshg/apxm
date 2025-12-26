//! BRANCH operation - Conditional branching

use super::{ExecutionContext, Node, Result, Value, get_input};

pub async fn execute(_ctx: &ExecutionContext, node: &Node, inputs: Vec<Value>) -> Result<Value> {
    // First input is the condition
    let condition = if !inputs.is_empty() {
        get_input(node, &inputs, 0)?
    } else {
        Value::Bool(false)
    };

    // Evaluate condition as boolean
    let is_true = condition.as_boolean().unwrap_or(false);

    // Return boolean result (actual branching handled by scheduler)
    Ok(Value::Bool(is_true))
}
