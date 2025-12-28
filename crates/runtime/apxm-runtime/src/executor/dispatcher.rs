//! Operation dispatcher - Routes operations to appropriate handlers

use super::{Result, context::ExecutionContext, handlers::*};
use apxm_core::types::{execution::Node, operations::AISOperationType, values::Value};

/// Operation dispatcher routes operations to their handlers
pub struct OperationDispatcher;

impl OperationDispatcher {
    /// Dispatch an operation to its handler
    ///
    /// # Arguments
    /// * `ctx` - Execution context
    /// * `node` - Operation node from DAG
    /// * `inputs` - Input values from dependencies
    ///
    /// # Returns
    /// Result value from operation execution
    pub async fn dispatch(
        ctx: &ExecutionContext,
        node: &Node,
        inputs: Vec<Value>,
    ) -> Result<Value> {
        tracing::debug!(
            op_type = ?node.op_type,
            node_id = node.id,
            "Dispatching operation"
        );

        let result = match node.op_type {
            // Memory operations
            AISOperationType::QMem => qmem::execute(ctx, node, inputs).await,
            AISOperationType::UMem => umem::execute(ctx, node, inputs).await,

            // Reasoning operations
            AISOperationType::Rsn => rsn::execute(ctx, node, inputs).await,
            AISOperationType::Plan => plan::execute(ctx, node, inputs).await,
            AISOperationType::Reflect => reflect::execute(ctx, node, inputs).await,
            AISOperationType::Verify => verify::execute(ctx, node, inputs).await,

            // Invocation operations
            AISOperationType::Inv => inv::execute(ctx, node, inputs).await,

            // Synchronization operations
            AISOperationType::WaitAll => wait_all::execute(ctx, node, inputs).await,
            AISOperationType::Merge => merge::execute(ctx, node, inputs).await,
            AISOperationType::Fence => fence::execute(ctx, node, inputs).await,

            // Control flow operations
            AISOperationType::BranchOnValue => branch::execute(ctx, node, inputs).await,
            AISOperationType::Jump => jump::execute(ctx, node, inputs).await,
            AISOperationType::LoopStart => loop_start::execute(ctx, node, inputs).await,
            AISOperationType::LoopEnd => loop_end::execute(ctx, node, inputs).await,
            AISOperationType::Return => return_op::execute(ctx, node, inputs).await,
            AISOperationType::Switch => switch::execute(ctx, node, inputs).await,
            AISOperationType::FlowCall => flow_call::execute(ctx, node, inputs).await,

            // Error handling operations
            AISOperationType::TryCatch => try_catch::execute(ctx, node, inputs).await,
            AISOperationType::Err => err::execute(ctx, node, inputs).await,
            AISOperationType::Exc => exc::execute(ctx, node, inputs).await,

            // Communication operations
            AISOperationType::Communicate => communicate::execute(ctx, node, inputs).await,

            // Literal operations
            AISOperationType::ConstStr => const_str::execute(ctx, node, inputs).await,

            // Metadata operations (Agent is metadata, not executed)
            AISOperationType::Agent => Ok(Value::Null),
        };

        match &result {
            Ok(value) => {
                tracing::debug!(
                    op_type = ?node.op_type,
                    node_id = node.id,
                    "Operation completed successfully"
                );
                // Record in episodic memory
                ctx.memory
                    .record_episode(
                        format!("operation_completed:{:?}", node.op_type),
                        value.clone(),
                        ctx.execution_id.clone(),
                    )
                    .await
                    .ok(); // Ignore episodic recording errors
            }
            Err(e) => {
                tracing::error!(
                    op_type = ?node.op_type,
                    node_id = node.id,
                    error = %e,
                    "Operation failed"
                );
                // Record error in episodic memory
                ctx.memory
                    .record_episode(
                        format!("operation_failed:{:?}", node.op_type),
                        Value::String(e.to_string()),
                        ctx.execution_id.clone(),
                    )
                    .await
                    .ok();
            }
        }

        result
    }
}
