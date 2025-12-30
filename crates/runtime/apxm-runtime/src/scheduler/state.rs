//! Scheduler state management.

use std::collections::HashMap;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::time::Instant;

use apxm_core::types::{ExecutionDag, ExecutionStats, Node, NodeId, NodeStatus, TokenId, Value};
use crossbeam_deque::Worker;
use dashmap::DashMap;
use parking_lot::Mutex;
use tokio::sync::Notify;

use crate::observability::MetricsCollector;
use crate::scheduler::concurrency_control::{ConcurrencyControl, ConcurrencyControlHandle};
use crate::scheduler::config::SchedulerConfig;
use apxm_core::error::RuntimeError;

type RuntimeResult<T> = Result<T, RuntimeError>;
use crate::aam::effects::{OperationEffects, operation_effects};
use crate::scheduler::internal_state::{ExecutionFrame, OpState, PromiseState, TokenState};
use crate::scheduler::queue::{Priority, PriorityQueue};
use crate::scheduler::ready_set::ReadySet;
use crate::scheduler::work_stealing::WorkStealingScheduler;

/// The internal state of the scheduler.
pub struct SchedulerState {
    pub cfg: SchedulerConfig,
    pub metrics: Arc<MetricsCollector>,
    pub start: Instant,

    // Immutable node data
    pub nodes: Arc<DashMap<NodeId, Arc<Node>>>,
    pub priorities: Arc<DashMap<NodeId, Priority>>,

    // Readiness tracking (encapsulated)
    pub(crate) ready_set: ReadySet,
    pub(crate) tokens: Arc<DashMap<TokenId, TokenState>>,
    pub(crate) op_states: Arc<DashMap<NodeId, OpState>>,

    // Work-stealing scheduler (encapsulated)
    pub work_stealing: Arc<WorkStealingScheduler>,
    pub queue: Arc<PriorityQueue>,

    // Concurrency control (encapsulated)
    pub concurrency: ConcurrencyControl,

    // Coordination
    pub executed: Arc<AtomicUsize>,
    pub failed: Arc<AtomicUsize>,
    pub remaining: Arc<AtomicUsize>,
    pub notify_done: Arc<Notify>,
    pub first_error: Arc<Mutex<Option<RuntimeError>>>,
    pub last_progress_ms: Arc<AtomicU64>,
    pub exit_nodes: Vec<NodeId>,

    // Promise tracking for flow calls
    pub pending_promises: Arc<DashMap<TokenId, PromiseState>>,
    pub execution_stack: Arc<Mutex<Vec<ExecutionFrame>>>,
    pub next_promise_token_id: Arc<AtomicU64>,
}

impl SchedulerState {
    pub fn new(
        dag: ExecutionDag,
        cfg: SchedulerConfig,
        metrics: Arc<MetricsCollector>,
        start: Instant,
    ) -> RuntimeResult<(Self, Vec<Worker<NodeId>>)> {
        // Validate configuration
        cfg.validate().map_err(|msg| RuntimeError::Scheduler {
            message: format!("Invalid scheduler config: {}", msg),
        })?;

        // Build node and priority maps
        let nodes: Arc<DashMap<NodeId, Arc<Node>>> = Arc::new(
            dag.nodes
                .iter()
                .map(|n| (n.id, Arc::new(n.clone())))
                .collect(),
        );

        let priorities: Arc<DashMap<NodeId, Priority>> = Arc::new(
            dag.nodes
                .iter()
                .map(|n| (n.id, Priority::from_u8(n.metadata.priority as u8)))
                .collect(),
        );

        // Initialize token and operation state
        let tokens = Arc::new(DashMap::new());
        let op_states = Arc::new(DashMap::new());
        materialize_graph_state(&dag, &tokens, &op_states)?;

        // Create priority queue and work-stealing scheduler
        let queue = Arc::new(PriorityQueue::new());
        let (work_stealing, workers) =
            WorkStealingScheduler::new(cfg.max_concurrency, Arc::clone(&queue));
        let work_stealing = Arc::new(work_stealing);

        // Create readiness tracker
        let ready_set = ReadySet::new();

        // Create concurrency controller
        let concurrency = ConcurrencyControl::new(cfg.max_inflight);

        // Build state
        let state = Self {
            cfg: cfg.clone(),
            metrics,
            start,

            nodes,
            priorities,

            ready_set,
            tokens,
            op_states,

            work_stealing,
            queue: Arc::clone(&queue),

            concurrency,

            executed: Arc::new(AtomicUsize::new(0)),
            failed: Arc::new(AtomicUsize::new(0)),
            remaining: Arc::new(AtomicUsize::new(dag.nodes.len())),
            notify_done: Arc::new(Notify::new()),
            first_error: Arc::new(Mutex::new(None)),
            last_progress_ms: Arc::new(AtomicU64::new(0)),
            exit_nodes: dag.exit_nodes.clone(),

            // Initialize promise tracking
            pending_promises: Arc::new(DashMap::new()),
            execution_stack: Arc::new(Mutex::new(Vec::new())),
            next_promise_token_id: Arc::new(AtomicU64::new(1_000_000)), // Start high to avoid collisions
        };

        // Initialize readiness tracking and seed ready nodes
        let _ = state.ready_set.initialize(
            &dag.nodes,
            &state.tokens,
            &state.priorities,
            &state.op_states,
            &state.queue,
        )?;

        // Initialize last progress timestamp
        state
            .last_progress_ms
            .store(state.elapsed_ms() as u64, Ordering::Relaxed);

        Ok((state, workers))
    }

    #[inline]
    pub fn elapsed_ms(&self) -> u128 {
        self.start.elapsed().as_millis()
    }

    #[inline]
    pub fn mark_done(&self) {
        self.remaining.store(0, Ordering::Relaxed);
        self.concurrency.cancel();
        self.notify_done.notify_waiters();
    }

    /// Get a cloneable handle to the concurrency controller.
    pub fn concurrency_handle(&self) -> ConcurrencyControlHandle {
        self.concurrency.handle()
    }

    /// Check if execution has been cancelled.
    pub fn is_cancelled(&self) -> bool {
        self.concurrency.is_cancelled()
    }

    /// Retrieve the effect metadata for a node, if available.
    pub fn operation_effects(&self, node_id: NodeId) -> Option<OperationEffects> {
        self.op_states
            .get(&node_id)
            .map(|entry| entry.effects().clone())
    }

    /// Set the first error if none has been set yet.
    pub fn set_first_error(&self, error: RuntimeError) {
        let mut guard = self.first_error.lock();
        if guard.is_none() {
            *guard = Some(error);
        }
    }

    /// Record progress to prevent deadlock detection.
    pub fn record_progress(&self) {
        self.last_progress_ms
            .store(self.elapsed_ms() as u64, Ordering::Relaxed);
    }

    pub fn collect_exit_values(&self) -> RuntimeResult<HashMap<TokenId, Value>> {
        let mut results = HashMap::new();

        for &node_id in &self.exit_nodes {
            if let Some(node) = self.nodes.get(&node_id) {
                for token_id in &node.output_tokens {
                    if let Some(value) = self
                        .tokens
                        .get(token_id)
                        .and_then(|state| state.value.clone())
                    {
                        results.insert(*token_id, value);
                    }
                }
            }
        }

        Ok(results)
    }

    /// Create a new promise token for a flow call.
    ///
    /// The token is registered as "not ready" and will be resolved
    /// when the sub-flow completes.
    pub fn create_promise_token(&self, target_agent: String, target_flow: String) -> TokenId {
        let token_id = self.next_promise_token_id.fetch_add(1, Ordering::Relaxed);

        // Register promise state
        self.pending_promises.insert(
            token_id,
            PromiseState::new(target_agent, target_flow),
        );

        // Register token as not ready (consumers will wait)
        self.tokens.insert(token_id, TokenState::new());

        token_id
    }

    /// Resolve a promise token with a value.
    ///
    /// This makes the token ready and propagates readiness to consumers.
    pub fn resolve_promise(&self, token_id: TokenId, value: Value) -> RuntimeResult<()> {
        // Update promise state
        if let Some(mut promise) = self.pending_promises.get_mut(&token_id) {
            promise.resolved = true;
            promise.value = Some(value.clone());
        }

        // Update token state (makes downstream nodes ready)
        if let Some(mut token) = self.tokens.get_mut(&token_id) {
            token.ready = true;
            token.value = Some(value);
        }

        // Propagate readiness to consumers
        self.ready_set.on_token_ready(
            token_id,
            &self.tokens,
            &self.priorities,
            &self.op_states,
            &self.queue,
        )?;

        self.record_progress();
        Ok(())
    }

    /// Push an execution frame onto the stack (for sub-flow execution).
    pub fn push_execution_frame(&self, frame: ExecutionFrame) {
        let mut stack = self.execution_stack.lock();
        stack.push(frame);
    }

    /// Pop an execution frame from the stack.
    pub fn pop_execution_frame(&self) -> Option<ExecutionFrame> {
        let mut stack = self.execution_stack.lock();
        stack.pop()
    }

    /// Get the current execution stack depth.
    pub fn execution_stack_depth(&self) -> usize {
        let stack = self.execution_stack.lock();
        stack.len()
    }

    pub fn build_stats(&self) -> ExecutionStats {
        let duration_ms = self.elapsed_ms();

        let node_statuses = self
            .op_states
            .iter()
            .map(|entry| {
                let v = entry.value();
                let dur = match (v.started_at, v.finished_at) {
                    (Some(s), Some(f)) => Some((f - s).as_millis()),
                    _ => None,
                };
                NodeStatus {
                    node_id: *entry.key(),
                    status: v.status,
                    retries: v.retries,
                    last_error: v.last_error.clone(),
                    started_at_ms: v
                        .started_at
                        .map(|t| t.duration_since(self.start).as_millis()),
                    finished_at_ms: v
                        .finished_at
                        .map(|t| t.duration_since(self.start).as_millis()),
                    duration_ms: dur,
                }
            })
            .collect();

        ExecutionStats {
            executed_nodes: self.executed.load(Ordering::Relaxed),
            failed_nodes: self.failed.load(Ordering::Relaxed),
            duration_ms,
            node_statuses,
        }
    }
}

/// Initialize token and operation state from the DAG.
///
/// This validates that each token has exactly one producer and registers
/// consumers for each token.
fn materialize_graph_state(
    dag: &ExecutionDag,
    tokens: &DashMap<TokenId, TokenState>,
    op_states: &DashMap<NodeId, OpState>,
) -> RuntimeResult<()> {
    for node in dag.nodes.iter() {
        op_states.insert(
            node.id,
            OpState::new_with_effects(operation_effects(&node.op_type)),
        );

        // Outputs: must have a single producer.
        for &token_id in &node.output_tokens {
            if tokens.contains_key(&token_id) {
                return Err(RuntimeError::SchedulerDuplicateProducer { token_id });
            }
            tokens.insert(token_id, TokenState::new());
        }

        // Inputs: if missing, treat as pre-ready null, and register consumer.
        for &token_id in &node.input_tokens {
            tokens
                .entry(token_id)
                .or_insert({
                    let mut ts = TokenState::new();
                    ts.ready = true;
                    ts.value = Some(Value::Null);
                    ts
                })
                .consumers
                .push(node.id);
        }
    }
    Ok(())
}
