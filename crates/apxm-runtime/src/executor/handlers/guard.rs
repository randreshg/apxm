//! GUARD operation — enforce preconditions before execution continues.
//!
//! Evaluates a condition against the first input value. If the condition fails,
//! execution either halts with an error or passes through (depending on `on_fail`).
//!
//! ## Attributes
//! - `condition`     (required): comparison expression, e.g. `"> 0.8"`, `"== true"`, `"!= null"`
//! - `error_message` (optional): message on failure (default: generated from condition)
//! - `on_fail`       (optional): `"halt"` (default) | `"skip"` — skip passes `false` downstream
//!
//! ## AIS usage
//! ```ais
//! guard(condition: "> 0.8", error_message: "Confidence too low", on_fail: "halt") <- confidence
//!
//! guard(condition: "!= null", on_fail: "skip") <- user_input
//! ```
//!
//! ## Condition syntax
//! Simple unary comparisons against the input value:
//! - `"> N"` — input > N (numeric)
//! - `">= N"` — input >= N
//! - `"< N"` — input < N
//! - `"<= N"` — input <= N
//! - `"== X"` — input == X (string or bool)
//! - `"!= X"` — input != X
//! - `"not_empty"` — input is non-empty string/array/object
//! - `"is_null"` — input is null
//! - `"not_null"` — input is not null

use super::{
    ExecutionContext, Node, Result, Value, get_optional_string_attribute, get_string_attribute,
};
use apxm_core::constants::graph::attrs as graph_attrs;
use apxm_core::error::RuntimeError;

pub async fn execute(ctx: &ExecutionContext, node: &Node, inputs: Vec<Value>) -> Result<Value> {
    let condition = get_string_attribute(node, graph_attrs::CONDITION)?;
    let on_fail = get_optional_string_attribute(node, graph_attrs::ON_FAIL)?
        .unwrap_or_else(|| "halt".to_string());
    let error_message = get_optional_string_attribute(node, graph_attrs::ERROR_MESSAGE)?
        .unwrap_or_else(|| format!("GUARD condition failed: '{}'", condition));

    let input = inputs.first().cloned().unwrap_or(Value::Null);

    tracing::debug!(
        execution_id = %ctx.execution_id,
        condition = %condition,
        "Evaluating GUARD condition"
    );

    let passed = evaluate_condition(&condition, &input);

    if passed {
        tracing::debug!(
            execution_id = %ctx.execution_id,
            condition = %condition,
            "GUARD condition passed"
        );
        Ok(Value::Bool(true))
    } else {
        tracing::warn!(
            execution_id = %ctx.execution_id,
            condition = %condition,
            on_fail = %on_fail,
            "GUARD condition failed"
        );
        match on_fail.to_lowercase().as_str() {
            "skip" => Ok(Value::Bool(false)),
            "halt" | _ => Err(RuntimeError::Operation {
                op_type: node.op_type,
                message: error_message,
            }),
        }
    }
}

/// Evaluate a simple condition expression against a value.
///
/// Supports: `"> N"`, `">= N"`, `"< N"`, `"<= N"`, `"== X"`, `"!= X"`,
///           `"not_empty"`, `"is_null"`, `"not_null"`.
fn evaluate_condition(condition: &str, value: &Value) -> bool {
    let cond = condition.trim();

    // Keyword conditions
    match cond {
        "not_empty" => return !is_empty(value),
        "is_empty" => return is_empty(value),
        "is_null" => return matches!(value, Value::Null),
        "not_null" => return !matches!(value, Value::Null),
        "is_true" | "true" => return matches!(value, Value::Bool(true)),
        "is_false" | "false" => return matches!(value, Value::Bool(false)),
        _ => {}
    }

    // Operator expressions: "<op> <rhs>"
    let (op, rhs_str) = if let Some(rest) = cond.strip_prefix(">=") {
        (">=", rest.trim())
    } else if let Some(rest) = cond.strip_prefix(">") {
        (">", rest.trim())
    } else if let Some(rest) = cond.strip_prefix("<=") {
        ("<=", rest.trim())
    } else if let Some(rest) = cond.strip_prefix("<") {
        ("<", rest.trim())
    } else if let Some(rest) = cond.strip_prefix("!=") {
        ("!=", rest.trim())
    } else if let Some(rest) = cond.strip_prefix("==") {
        ("==", rest.trim())
    } else {
        // Unknown condition — default to checking non-null/non-empty
        return !matches!(value, Value::Null);
    };

    // Try numeric comparison
    if let Some(rhs_num) = parse_number(rhs_str) {
        if let Some(lhs_num) = value_as_f64(value) {
            return compare_f64(lhs_num, op, rhs_num);
        }
    }

    // String comparison
    let lhs_str = value_as_str(value);
    let rhs_clean = rhs_str.trim_matches('"').trim_matches('\'');
    match op {
        "==" => lhs_str == rhs_clean,
        "!=" => lhs_str != rhs_clean,
        _ => false,
    }
}

fn is_empty(value: &Value) -> bool {
    match value {
        Value::Null => true,
        Value::String(s) => s.is_empty(),
        Value::Array(a) => a.is_empty(),
        Value::Object(o) => o.is_empty(),
        _ => false,
    }
}

fn parse_number(s: &str) -> Option<f64> {
    s.parse::<f64>().ok()
}

fn value_as_f64(value: &Value) -> Option<f64> {
    use apxm_core::types::values::Number;
    match value {
        Value::Number(Number::Float(f)) => Some(*f),
        Value::Number(Number::Integer(i)) => Some(*i as f64),
        Value::String(s) => s.parse::<f64>().ok(),
        Value::Bool(b) => Some(if *b { 1.0 } else { 0.0 }),
        _ => None,
    }
}

fn value_as_str(value: &Value) -> String {
    match value {
        Value::String(s) => s.clone(),
        Value::Bool(b) => b.to_string(),
        Value::Null => "null".to_string(),
        other => other.to_string(),
    }
}

fn compare_f64(lhs: f64, op: &str, rhs: f64) -> bool {
    match op {
        ">" => lhs > rhs,
        ">=" => lhs >= rhs,
        "<" => lhs < rhs,
        "<=" => lhs <= rhs,
        "==" => (lhs - rhs).abs() < f64::EPSILON,
        "!=" => (lhs - rhs).abs() >= f64::EPSILON,
        _ => false,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use apxm_core::types::values::Number;

    #[test]
    fn test_guard_numeric_gt() {
        assert!(evaluate_condition(
            "> 0.8",
            &Value::Number(Number::Float(0.9))
        ));
        assert!(!evaluate_condition(
            "> 0.8",
            &Value::Number(Number::Float(0.7))
        ));
    }

    #[test]
    fn test_guard_numeric_int() {
        assert!(evaluate_condition(
            ">= 5",
            &Value::Number(Number::Integer(5))
        ));
        assert!(!evaluate_condition(
            "> 5",
            &Value::Number(Number::Integer(5))
        ));
        assert!(evaluate_condition(
            "< 10",
            &Value::Number(Number::Integer(3))
        ));
    }

    #[test]
    fn test_guard_not_null() {
        assert!(evaluate_condition(
            "not_null",
            &Value::String("hello".to_string())
        ));
        assert!(!evaluate_condition("not_null", &Value::Null));
    }

    #[test]
    fn test_guard_not_empty() {
        assert!(evaluate_condition(
            "not_empty",
            &Value::String("hi".to_string())
        ));
        assert!(!evaluate_condition(
            "not_empty",
            &Value::String(String::new())
        ));
        assert!(!evaluate_condition("not_empty", &Value::Null));
    }

    #[test]
    fn test_guard_string_eq() {
        assert!(evaluate_condition(
            "== approved",
            &Value::String("approved".to_string())
        ));
        assert!(!evaluate_condition(
            "== approved",
            &Value::String("rejected".to_string())
        ));
    }

    #[test]
    fn test_guard_is_null() {
        assert!(evaluate_condition("is_null", &Value::Null));
        assert!(!evaluate_condition(
            "is_null",
            &Value::String("x".to_string())
        ));
    }
}
