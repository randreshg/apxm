//! CONST_STR operation - String constant

use super::{ExecutionContext, Node, Result, Value, get_string_attribute};
use apxm_core::constants::graph::attrs as graph_attrs;

pub async fn execute(_ctx: &ExecutionContext, node: &Node, _inputs: Vec<Value>) -> Result<Value> {
    let value = get_string_attribute(node, graph_attrs::VALUE)?;
    Ok(Value::String(value))
}
