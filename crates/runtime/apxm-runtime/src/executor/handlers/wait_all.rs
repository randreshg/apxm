//! WAITALL operation - Wait for all dependencies

use super::{ExecutionContext, Node, Result, Value};

pub async fn execute(_ctx: &ExecutionContext, _node: &Node, inputs: Vec<Value>) -> Result<Value> {
    // WAITALL simply passes through all inputs as an array
    // The actual synchronization is handled by the scheduler
    Ok(Value::Array(inputs))
}
