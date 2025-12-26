//! VERIFY operation - Verification with LLM

use super::{ExecutionContext, Node, Result, Value, execute_llm_request, get_input, get_string_attribute};
use apxm_backends::LLMRequest;

pub async fn execute(ctx: &ExecutionContext, node: &Node, inputs: Vec<Value>) -> Result<Value> {
    let condition = get_string_attribute(node, "condition")?;
    let value_to_verify = if !inputs.is_empty() {
        get_input(node, &inputs, 0)?
    } else {
        Value::Null
    };

    let verification_prompt = format!(
        "Verify if the following condition is met:\n\nCondition: {}\nValue: {:?}\n\nRespond with 'true' or 'false'.",
        condition, value_to_verify
    );

    let request = LLMRequest::new(verification_prompt);
    let response = execute_llm_request(ctx, "VERIFY", &request).await?;

    let is_verified = response.content.to_lowercase().contains("true");
    Ok(Value::Bool(is_verified))
}
