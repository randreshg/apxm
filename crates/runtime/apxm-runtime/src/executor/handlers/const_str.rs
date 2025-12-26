//! CONST_STR operation - String constant

use super::{ExecutionContext, Node, Result, Value, get_string_attribute};

pub async fn execute(_ctx: &ExecutionContext, node: &Node, _inputs: Vec<Value>) -> Result<Value> {
    let value = get_string_attribute(node, "value")?;
    Ok(Value::String(value))
}
