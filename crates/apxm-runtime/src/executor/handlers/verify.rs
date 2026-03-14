//! VERIFY operation - Verification with LLM

use super::{
    ExecutionContext, Node, Result, Value, execute_llm_request, get_input,
    get_optional_string_attribute, get_string_attribute,
};
use apxm_backends::LLMRequest;
use apxm_core::constants::graph::attrs as graph_attrs;

pub async fn execute(ctx: &ExecutionContext, node: &Node, inputs: Vec<Value>) -> Result<Value> {
    let condition = get_string_attribute(node, graph_attrs::CONDITION)?;
    let value_to_verify = if !inputs.is_empty() {
        get_input(node, &inputs, 0)?
    } else {
        Value::Null
    };

    let verification_prompt = format!(
        "Verify if the following condition is met:\n\nCondition: {}\nValue: {:?}\n\nRespond with 'true' or 'false'.",
        condition, value_to_verify
    );

    // Priority: 1) node attribute (agent context), 2) config instruction, 3) template, 4) hardcoded fallback
    let system_prompt = get_optional_string_attribute(node, graph_attrs::SYSTEM_PROMPT)?
        .or_else(|| ctx.instruction_config.verify.clone())
        .or_else(|| apxm_backends::render_prompt("verify_system", &serde_json::json!({})).ok())
        .unwrap_or_else(|| {
            "You are a verification expert. Analyze conditions precisely. Respond with only 'true' or 'false'.".to_string()
        });

    let request = LLMRequest::new(verification_prompt).with_system_prompt(system_prompt);
    let response = execute_llm_request(ctx, "VERIFY", &request).await?;

    let is_verified = response.content.to_lowercase().contains("true");
    Ok(Value::Bool(is_verified))
}
