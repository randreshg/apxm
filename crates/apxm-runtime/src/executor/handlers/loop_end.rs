//! LOOP_END operation - Loop termination check

use super::{ExecutionContext, Node, Result, Value, get_input};

pub async fn execute(_ctx: &ExecutionContext, node: &Node, inputs: Vec<Value>) -> Result<Value> {
    // Check if we should continue looping
    let counter = get_input(node, &inputs, 0)?;

    if let Some(count) = counter.as_u64() {
        Ok(Value::Bool(count > 0))
    } else {
        Ok(Value::Bool(false))
    }
}
