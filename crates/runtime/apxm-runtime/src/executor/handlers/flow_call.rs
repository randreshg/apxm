//! FLOW_CALL operation - Call a flow on another agent

use super::{get_string_attribute, ExecutionContext, Node, Result, Value};
use crate::aam::TransitionLabel;

/// Execute a flow call operation
///
/// Invokes a flow on another agent, passing arguments and receiving a result.
/// This enables cross-agent communication and coordination with implicit parallelism.
///
/// The flow call records the invocation in the AAM and episodic memory for tracing.
/// In a multi-agent scenario, this operation would coordinate with the scheduler
/// to spawn a sub-execution for the target agent's flow.
pub async fn execute(ctx: &ExecutionContext, node: &Node, inputs: Vec<Value>) -> Result<Value> {
    // Get agent name
    let agent_name = get_string_attribute(node, "agent_name")?;

    // Get flow name
    let flow_name = get_string_attribute(node, "flow_name")?;

    tracing::info!(
        agent = %agent_name,
        flow = %flow_name,
        num_args = inputs.len(),
        "Executing flow call"
    );

    // Create a structured call request
    let call_request = Value::Object(
        vec![
            ("type".to_string(), Value::String("flow_call".to_string())),
            ("agent".to_string(), Value::String(agent_name.clone())),
            ("flow".to_string(), Value::String(flow_name.clone())),
            ("args".to_string(), Value::Array(inputs.clone())),
            (
                "execution_id".to_string(),
                Value::String(ctx.execution_id.clone()),
            ),
        ]
        .into_iter()
        .collect(),
    );

    // Record the flow call in AAM beliefs for tracking
    let label = TransitionLabel::Custom(format!("flow_call:{}:{}", agent_name, flow_name));
    ctx.aam.set_belief(
        format!("_pending_flow_call:{}:{}", agent_name, flow_name),
        call_request.clone(),
        label,
    );

    // Record in episodic memory for tracing
    ctx.memory
        .record_episode(
            format!("flow_call:{}:{}", agent_name, flow_name),
            call_request.clone(),
            ctx.execution_id.clone(),
        )
        .await
        .ok();

    // Check if the target agent has a registered capability for cross-agent calls
    // In a full implementation, this would:
    // 1. Look up the target agent's flow definition from a registry
    // 2. Validate the arguments match the flow's signature
    // 3. Spawn a sub-execution for the target agent's flow
    // 4. Wait for the result or return a future/promise

    // For now, we return a structured result indicating the call is ready to be scheduled
    // The scheduler can then use this information to coordinate multi-agent execution
    let result = Value::Object(
        vec![
            ("status".to_string(), Value::String("invoked".to_string())),
            ("agent".to_string(), Value::String(agent_name.clone())),
            ("flow".to_string(), Value::String(flow_name.clone())),
            (
                "args_count".to_string(),
                Value::Number(apxm_core::types::values::Number::from(inputs.len() as i64)),
            ),
            (
                "execution_id".to_string(),
                Value::String(ctx.execution_id.clone()),
            ),
        ]
        .into_iter()
        .collect(),
    );

    tracing::debug!(
        agent = %agent_name,
        flow = %flow_name,
        "Flow call completed"
    );

    Ok(result)
}
