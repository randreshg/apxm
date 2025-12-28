//! SWITCH operation - Multi-way conditional branching based on string value

use super::{ExecutionContext, Node, Result, Value};

/// Execute a switch operation
///
/// Matches the discriminant value against case labels and returns the matching
/// case's destination label. If no case matches and there's a default, returns that.
pub async fn execute(_ctx: &ExecutionContext, node: &Node, inputs: Vec<Value>) -> Result<Value> {
    // Get the discriminant value (first input or from attribute)
    let discriminant_value = if !inputs.is_empty() {
        inputs[0]
            .as_string()
            .map(|s| s.to_string())
            .unwrap_or_default()
    } else {
        String::new()
    };

    // Get case labels array
    let case_labels = node
        .attributes
        .get("case_labels")
        .and_then(|v| v.as_array())
        .map(|arr| {
            arr.iter()
                .filter_map(|v| v.as_string().map(|s| s.to_string()))
                .collect::<Vec<_>>()
        })
        .unwrap_or_default();

    // Get case destinations array
    let case_destinations = node
        .attributes
        .get("case_destinations")
        .and_then(|v| v.as_array())
        .map(|arr| {
            arr.iter()
                .filter_map(|v| v.as_string().map(|s| s.to_string()))
                .collect::<Vec<_>>()
        })
        .unwrap_or_default();

    // Get optional default destination
    let default_destination = node
        .attributes
        .get("default_destination")
        .and_then(|v| v.as_string())
        .map(|s| s.to_string());

    // Find matching case
    for (i, label) in case_labels.iter().enumerate() {
        if label == &discriminant_value {
            if let Some(dest) = case_destinations.get(i) {
                return Ok(Value::String(dest.clone()));
            }
        }
    }

    // No match - use default if available
    if let Some(default_dest) = default_destination {
        return Ok(Value::String(default_dest));
    }

    // No match and no default - return null
    Ok(Value::Null)
}
