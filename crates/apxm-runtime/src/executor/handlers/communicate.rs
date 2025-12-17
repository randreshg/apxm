//! COMMUNICATE operation - Inter-agent communication

use super::{ExecutionContext, Node, Result, Value, get_string_attribute};

pub async fn execute(_ctx: &ExecutionContext, node: &Node, inputs: Vec<Value>) -> Result<Value> {
    // Accept both the historical "target" attribute and the current "recipient"
    // emitted by the compiler.
    let recipient = get_string_attribute(node, "recipient")
        .or_else(|_| get_string_attribute(node, "target"))?;
    let message = inputs.first().cloned().unwrap_or(Value::Null);

    // TODO: Implement actual inter-agent communication
    // For now, return acknowledgment
    Ok(Value::String(format!(
        "Sent message to {}: {:?}",
        recipient, message
    )))
}
