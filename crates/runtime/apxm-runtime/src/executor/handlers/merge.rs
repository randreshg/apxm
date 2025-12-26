//! MERGE operation - Merge multiple values

use super::{ExecutionContext, Node, Result, Value, get_optional_string_attribute};

pub async fn execute(_ctx: &ExecutionContext, node: &Node, inputs: Vec<Value>) -> Result<Value> {
    let strategy =
        get_optional_string_attribute(node, "strategy")?.unwrap_or_else(|| "array".to_string());

    match strategy.as_str() {
        "array" => Ok(Value::Array(inputs)),
        "concat" => {
            // Concatenate string values
            let concat = inputs
                .iter()
                .filter_map(|v| v.as_string())
                .map(|s| s.as_str())
                .collect::<Vec<_>>()
                .join("");
            Ok(Value::String(concat))
        }
        "sum" => {
            // Sum numeric values
            let sum = inputs
                .iter()
                .filter_map(|v| v.as_number())
                .map(|n| n.as_f64())
                .sum::<f64>();
            Ok(Value::Number(apxm_core::types::values::Number::Float(sum)))
        }
        _ => Ok(Value::Array(inputs)),
    }
}
