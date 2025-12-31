//! PRINT operation - Print output to stdout with markdown rendering (void operation)

use super::{ExecutionContext, Node, Result, Value, get_string_attribute};
use termimad::MadSkin;

/// Format a Value as a human-readable string (recursive for arrays).
fn format_value(v: &Value) -> String {
    match v {
        Value::String(s) => s.clone(),
        Value::Number(n) => n.to_string(),
        Value::Bool(b) => b.to_string(),
        Value::Null => "null".to_string(),
        Value::Array(arr) => arr.iter().map(format_value).collect::<Vec<_>>().join(""),
        Value::Object(obj) => serde_json::to_string(obj).unwrap_or_else(|_| format!("{:?}", obj)),
        Value::Token(id) => format!("<token:{}>", id),
    }
}

/// Execute a print operation. This is a void operation (no output tokens).
/// Output is rendered as markdown for terminal display.
pub async fn execute(_ctx: &ExecutionContext, node: &Node, inputs: Vec<Value>) -> Result<Value> {
    let message = get_string_attribute(node, "message").unwrap_or_default();

    // Build output: message followed by any input values
    let mut output = message.clone();
    for (i, input) in inputs.iter().enumerate() {
        if i == 0 && !message.is_empty() {
            output.push(' ');
        }
        output.push_str(&format_value(input));
        if i < inputs.len() - 1 {
            output.push(' ');
        }
    }

    // Render markdown to terminal
    let skin = MadSkin::default();
    skin.print_text(&output);

    // Void operation - return Null (no output tokens in artifact)
    Ok(Value::Null)
}
