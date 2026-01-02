//! Worker loop for the dataflow scheduler.
//!
//! Workers continuously steal and execute operations until the DAG completes.
//!
//! When the `metrics` feature is enabled, this module instruments key operations
//! to measure scheduler overhead breakdown:
//! - Work stealing
//! - Input collection
//! - Operation dispatch
//! - Token routing

use std::sync::Arc;
use std::sync::atomic::Ordering;
use std::time::{Duration, Instant};

use apxm_core::types::{Node, NodeId, Number, OpStatus, TokenId, Value};
use apxm_core::{apxm_op, apxm_token};
use crossbeam_deque::Worker;

use crate::executor::ExecutionContext;
use crate::executor::ExecutorEngine;
use crate::scheduler::internal_state::{OpState, TokenState};
use crate::scheduler::state::SchedulerState;
use crate::timed;
use apxm_core::error::RuntimeError;

/// Main worker loop.
///
/// Each worker thread runs this loop, stealing work and executing operations
/// until the DAG completes or execution is cancelled.
pub async fn worker_loop(
    worker_id: usize,
    local_queue: Worker<NodeId>,
    state: Arc<SchedulerState>,
    executor: Arc<ExecutorEngine>,
    base_ctx: ExecutionContext,
) {
    loop {
        // Check termination conditions
        if state.remaining.load(Ordering::Relaxed) == 0 || state.is_cancelled() {
            break;
        }

        // Try to steal work (timed when metrics enabled)
        let stolen = timed!(state.metrics, record_work_stealing, {
            state.work_stealing.steal_next(&local_queue, worker_id)
        });
        let Some(node_id) = stolen else {
            // No work available, yield
            tokio::task::yield_now().await;
            continue;
        };

        // Acquire concurrency permit (backpressure)
        let permit = match state.concurrency.acquire().await {
            Ok(p) => p,
            Err(_) => break, // Cancelled
        };

        // Get node
        let Some(node) = state.nodes.get(&node_id).map(|n| n.value().clone()) else {
            drop(permit);
            continue;
        };

        // Mark operation as running
        op_start(&state.op_states, node_id);

        apxm_op!(debug,
            worker = worker_id,
            node_id = node_id,
            op_type = ?node.op_type,
            inputs = node.input_tokens.len(),
            "Dispatching operation"
        );

        // Collect inputs (must all be ready) - timed when metrics enabled
        let collected = timed!(state.metrics, record_input_collection, {
            collect_inputs(&state.tokens, &node)
        });
        let Some(inputs) = collected else {
            // Inputs not ready - requeue at lowest priority and continue
            // This shouldn't normally happen since readiness is tracked,
            // but handle gracefully in case of race conditions
            apxm_op!(trace,
                worker = worker_id,
                node_id = node_id,
                "Inputs not ready, requeuing"
            );
            state
                .queue
                .push(node_id, crate::scheduler::queue::Priority::Low);
            drop(permit);
            tokio::task::yield_now().await;
            continue;
        };

        // Execute operation with retries
        let child_ctx = base_ctx.child();
        let outputs = node.output_tokens.clone();

        let outcome = execute_with_retries(&state, &executor, &node, &inputs, &child_ctx, worker_id).await;

        // Handle outcome
        match outcome {
            ExecutionOutcome::Success {
                value,
                attempts,
                start_time,
            } => {
                let event = WorkerEvent {
                    state: &state,
                    ctx: &child_ctx,
                    node_id,
                    node: &node,
                    outputs: &outputs,
                    start_time,
                };

                handle_success(&event, value, attempts).await;
            }
            ExecutionOutcome::Failed {
                error,
                attempts,
                start_time,
            } => {
                let event = WorkerEvent {
                    state: &state,
                    ctx: &child_ctx,
                    node_id,
                    node: &node,
                    outputs: &outputs,
                    start_time,
                };

                let should_abort = handle_failure(&event, error, attempts).await;

                if should_abort {
                    drop(permit);
                    break;
                }
            }
        }

        drop(permit);
    }
}

/// Collect input values for an operation.
///
/// Returns None if any inputs are not ready (shouldn't happen due to readiness tracking).
fn collect_inputs(
    tokens: &dashmap::DashMap<TokenId, TokenState>,
    node: &Node,
) -> Option<Vec<Value>> {
    let mut values = Vec::with_capacity(node.input_tokens.len());

    for &token_id in &node.input_tokens {
        let token = tokens.get(&token_id)?;
        if !token.ready {
            return None;
        }
        values.push(token.value.as_ref()?.clone());
    }

    Some(values)
}

/// Mark an operation as running.
#[inline]
fn op_start(op_states: &dashmap::DashMap<NodeId, OpState>, node_id: NodeId) {
    if let Some(mut state) = op_states.get_mut(&node_id) {
        state.status = OpStatus::Running;
        if state.started_at.is_none() {
            state.started_at = Some(Instant::now());
        }
    }
}

/// Execution outcome.
enum ExecutionOutcome {
    Success {
        value: Value,
        attempts: u32,
        start_time: Instant,
    },
    Failed {
        error: RuntimeError,
        attempts: u32,
        start_time: Instant,
    },
}

/// Execute an operation with retry logic.
#[allow(unused_variables)] // worker_id only used in tracing
async fn execute_with_retries(
    state: &SchedulerState,
    executor: &ExecutorEngine,
    node: &Node,
    inputs: &[Value],
    ctx: &ExecutionContext,
    worker_id: usize,
) -> ExecutionOutcome {
    let start_time = Instant::now();

    // Record operation start
    record_event(ctx, node.id, node, "op_start", 0, None).await;

    let mut attempt = 0;
    let mut last_error = None;

    while attempt <= state.cfg.max_retries {
        // Record scheduling
        #[cfg(feature = "metrics")]
        state.metrics.record_schedule();

        // Execute operation
        #[cfg(feature = "metrics")]
        let exec_start = Instant::now();

        apxm_op!(trace,
            worker = worker_id,
            node_id = node.id,
            attempt = attempt + 1,
            "Executing operation"
        );

        let result = executor
            .execute_with_context(node, inputs.to_vec(), ctx)
            .await;

        #[cfg(feature = "metrics")]
        let exec_duration = exec_start.elapsed();

        match result {
            Ok(outcome) => {
                #[cfg(feature = "metrics")]
                {
                    state.metrics.record_completion();
                    state.metrics.record_execution_time(exec_duration);
                }
                return ExecutionOutcome::Success {
                    value: outcome.value,
                    attempts: attempt + 1,
                    start_time,
                };
            }
            Err(error) => {
                #[cfg(feature = "metrics")]
                {
                    state.metrics.record_failure();
                    state.metrics.record_execution_time(exec_duration);
                }

                apxm_op!(debug,
                    worker = worker_id,
                    node_id = node.id,
                    attempt = attempt + 1,
                    error = %error,
                    "Operation attempt failed"
                );

                last_error = Some(error);
                attempt += 1;

                // Check if we should retry
                if attempt > state.cfg.max_retries {
                    break;
                }

                // Exponential backoff with cap
                let backoff_ms = calculate_backoff(&state.cfg, attempt);
                apxm_op!(trace,
                    worker = worker_id,
                    node_id = node.id,
                    backoff_ms = backoff_ms,
                    "Retry backoff"
                );
                tokio::time::sleep(Duration::from_millis(backoff_ms)).await;
            }
        }
    }

    ExecutionOutcome::Failed {
        error: last_error.unwrap_or_else(|| RuntimeError::Scheduler {
            message: "Unknown error".to_string(),
        }),
        attempts: attempt,
        start_time,
    }
}

/// Calculate exponential backoff delay in milliseconds.
#[inline]
fn calculate_backoff(cfg: &crate::scheduler::config::SchedulerConfig, attempt: u32) -> u64 {
    let pow = (attempt - 1).min(31); // Prevent overflow
    cfg.retry_backoff_ms
        .saturating_mul(2u64.saturating_pow(pow))
        .min(cfg.retry_backoff_max_ms)
}

struct WorkerEvent<'a> {
    state: &'a SchedulerState,
    ctx: &'a ExecutionContext,
    node_id: NodeId,
    node: &'a Node,
    outputs: &'a [TokenId],
    start_time: Instant,
}

/// Handle successful operation execution.
async fn handle_success(event: &WorkerEvent<'_>, value: Value, attempts: u32) {
    let duration_ms = event.start_time.elapsed().as_millis();

    apxm_op!(debug,
        node_id = event.node_id,
        op_type = ?event.node.op_type,
        duration_ms = duration_ms,
        attempts = attempts,
        output_tokens = event.outputs.len(),
        "Operation completed successfully"
    );

    // Update counters
    event.state.executed.fetch_add(1, Ordering::Relaxed);
    event.state.record_progress();

    // Publish outputs and propagate readiness (timed when metrics enabled)
    #[cfg(feature = "metrics")]
    let routing_start = std::time::Instant::now();

    publish_outputs(event.state, event.node_id, event.outputs, value).await;

    #[cfg(feature = "metrics")]
    event
        .state
        .metrics
        .record_token_routing(routing_start.elapsed());

    // Mark operation as completed
    if let Some(mut op_state) = event.state.op_states.get_mut(&event.node_id) {
        op_state.status = OpStatus::Completed;
        op_state.finished_at = Some(Instant::now());
    }

    // Record success event
    record_event(
        event.ctx,
        event.node_id,
        event.node,
        "op_success",
        attempts,
        Some(duration_ms),
    )
    .await;

    // Decrement remaining count
    finish_one(event.state);
}

/// Handle failed operation execution.
///
/// Returns true if execution should abort.
async fn handle_failure(event: &WorkerEvent<'_>, error: RuntimeError, attempts: u32) -> bool {
    let duration_ms = event.start_time.elapsed().as_millis();

    apxm_op!(error,
        node_id = event.node_id,
        op_type = ?event.node.op_type,
        duration_ms = duration_ms,
        attempts = attempts,
        error = %error,
        "Operation failed"
    );

    // Update counters
    event.state.failed.fetch_add(1, Ordering::Relaxed);

    // Mark operation as failed
    if let Some(mut op_state) = event.state.op_states.get_mut(&event.node_id) {
        op_state.status = OpStatus::Failed;
        op_state.retries = attempts;
        op_state.last_error = Some(error.to_string());
        op_state.finished_at = Some(Instant::now());
    }

    // Record failure event
    record_event(
        event.ctx,
        event.node_id,
        event.node,
        "op_failure",
        attempts,
        Some(duration_ms),
    )
    .await;

    // Set first error
    event
        .state
        .set_first_error(RuntimeError::SchedulerRetryExhausted {
            node_id: event.node_id,
            reason: error.to_string(),
        });

    // Check for fallback value
    if let Some(fallback) = event.node.attributes.get("fallback").cloned() {
        apxm_op!(info,
            node_id = event.node_id,
            "Using fallback value"
        );
        // Use fallback value instead of failing
        publish_outputs(event.state, event.node_id, event.outputs, fallback).await;
        finish_one(event.state);
        false // Don't abort
    } else {
        // No fallback - abort execution
        apxm_op!(warn,
            node_id = event.node_id,
            "No fallback, aborting execution"
        );
        event.state.mark_done();
        true // Abort
    }
}

/// Publish output tokens and propagate readiness to downstream consumers.
///
/// If a token is marked as delegated by this node, we skip publishing
/// (a spliced sub-DAG will produce the actual value).
async fn publish_outputs(state: &SchedulerState, node_id: u64, outputs: &[TokenId], value: Value) {
    for &token_id in outputs {
        // Fast path: check sparse delegation set (almost always empty)
        if state.delegated_tokens.contains(&(node_id, token_id)) {
            apxm_token!(trace,
                token_id = token_id,
                delegator = node_id,
                "Token delegated by this node; skipping publish"
            );
            continue;
        }

        // Mark token as ready with value
        if let Some(mut token) = state.tokens.get_mut(&token_id) {
            if token.ready {
                apxm_token!(trace,
                    token_id = token_id,
                    "Token already ready; skipping duplicate publish"
                );
                continue;
            }
            token.ready = true;
            token.value = Some(value.clone());
        } else {
            let mut token_state = TokenState::new();
            token_state.ready = true;
            token_state.value = Some(value.clone());
            state.tokens.insert(token_id, token_state);
        }

        apxm_token!(trace,
            token_id = token_id,
            "Token produced and ready"
        );

        // Propagate readiness to consumers
        let _ = state.ready_set.on_token_ready(
            token_id,
            &state.tokens,
            &state.priorities,
            &state.op_states,
            &state.queue,
        );
    }
}

/// Decrement remaining count and notify if complete.
///
/// Use the previous atomic value returned by fetch_sub to avoid underflow
/// and reliably detect the transition to zero.
#[inline]
fn finish_one(state: &SchedulerState) {
    let prev = state.remaining.fetch_sub(1, Ordering::Relaxed);
    if prev == 1 {
        state.notify_done.notify_waiters();
    }
}

/// Record an episodic memory event.
async fn record_event(
    ctx: &ExecutionContext,
    node_id: NodeId,
    node: &Node,
    event_type: &str,
    attempts: u32,
    duration_ms: Option<u128>,
) {
    let mut fields = vec![
        (
            "node_id".to_string(),
            Value::Number(Number::from(node_id as i64)),
        ),
        (
            "op".to_string(),
            Value::String(format!("{:?}", node.op_type)),
        ),
        (
            "attempts".to_string(),
            Value::Number(Number::from(attempts as i64)),
        ),
    ];

    if let Some(duration) = duration_ms {
        let clamped = duration.min(i64::MAX as u128) as i64;
        fields.push((
            "duration_ms".to_string(),
            Value::Number(Number::from(clamped)),
        ));
    }

    let _ = ctx
        .memory()
        .record_episodic_event(
            ctx.execution_id.clone(),
            event_type,
            Value::Object(fields.into_iter().collect()),
        )
        .await;
}
