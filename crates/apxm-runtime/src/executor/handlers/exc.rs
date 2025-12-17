//! EXC operation - Raise exception

use super::{ExecutionContext, Node, Result, Value, get_string_attribute};

pub async fn execute(_ctx: &ExecutionContext, node: &Node, _inputs: Vec<Value>) -> Result<Value> {
    let message = get_string_attribute(node, "message")?;

    Err(apxm_core::error::RuntimeError::Operation {
        op_type: node.op_type.clone(),
        message,
    })
}
