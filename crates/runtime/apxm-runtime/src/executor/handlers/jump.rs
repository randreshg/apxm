//! JUMP operation - Unconditional jump

use super::{ExecutionContext, Node, Result, Value};

pub async fn execute(_ctx: &ExecutionContext, _node: &Node, inputs: Vec<Value>) -> Result<Value> {
    // Pass through first input or Null
    inputs
        .first()
        .cloned()
        .ok_or_else(|| apxm_core::error::RuntimeError::Operation {
            op_type: apxm_core::types::operations::AISOperationType::Jump,
            message: "JUMP requires at least one input".to_string(),
        })
}
