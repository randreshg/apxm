//! APXM Server — HTTP gateway for the APXM agent runtime.
//!
//! Exposes the runtime's capabilities over a REST+SSE API with support for:
//! - **Graph execution**: `POST /v1/execute`, `POST /v1/execute/stream`
//! - **Memory**: `GET|POST /v1/memory`
//! - **Model discovery**: `GET /v1/models`
//! - **Agent registry**: `POST /v1/capabilities/register`
//! - **Task queue (CLAIM op)**: `POST /v1/tasks`, `GET /v1/tasks/:queue`,
//!   `POST /v1/tasks/:queue/claim`, `POST /v1/tasks/:id/complete`
//! - **HITL checkpoints (PAUSE/RESUME)**: `POST /v1/checkpoints`,
//!   `GET /v1/checkpoints/:id`, `POST /v1/checkpoints/:id/resume`
//! - **MCP 2025-11-05** (JSON-RPC 2.0): `POST /v1/mcp`
//! - **A2A v0.3** (REST): `POST /a2a/tasks/send`, `GET /a2a/tasks/:id`,
//!   `GET /.well-known/agent.json`
//!
//! # Running
//! ```bash
//! APXM_BACKEND=openai OPENAI_API_KEY=sk-... apxm-server --port 18800
//! ```
//!
//! # Architecture
//! Each request handler is a thin Axum layer over [`Runtime`]; no business
//! logic lives in this file — heavy lifting is in `apxm-runtime` and
//! `apxm-compiler`.

use std::collections::{HashMap, VecDeque};
use std::net::SocketAddr;
use std::sync::Arc;
use std::time::{SystemTime, UNIX_EPOCH};

use apxm_artifact::Artifact;
use apxm_compiler::{Context as CompilerContext, Pipeline as CompilerPipeline};
use apxm_core::constants::graph::attrs as graph_attrs;
use apxm_core::error::RuntimeError;
use apxm_core::types::values::Value;
use apxm_core::types::{AISOperationType, DependencyType};
use apxm_graph::{ApxmGraph, GraphEdge, GraphNode};
use apxm_runtime::capability::executor::{CapabilityExecutor, CapabilityResult};
use apxm_runtime::capability::metadata::CapabilityMetadata;
use apxm_runtime::executor::ExecutionEventEmitter;
use apxm_runtime::{Runtime, RuntimeConfig};
use async_trait::async_trait;
use axum::extract::{Path, State};
use axum::response::sse::{Event, KeepAlive, Sse};
use axum::{Json, Router, response::IntoResponse, routing::delete, routing::get, routing::post};
use dashmap::DashMap;
use futures::Stream;
use serde::{Deserialize, Serialize};
use serde_json::Value as JsonValue;
use tokio::sync::{Mutex, mpsc};
use tower_http::cors::CorsLayer;
use tower_http::request_id::{MakeRequestUuid, PropagateRequestIdLayer, SetRequestIdLayer};
use tower_http::trace::TraceLayer;
use tracing::{error, info};

fn now_ms() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis() as u64
}

// ─── Agent Registry ─────────────────────────────────────────────────────────

/// A remote agent registered with this server.
///
/// Agents register themselves so that COMMUNICATE operations (and the A2A agent
/// card) can resolve a name → URL mapping at runtime.
#[derive(Clone, Debug, Serialize, Deserialize)]
struct AgentRegistration {
    /// Unique agent name (matches the name used in AIS `communicate "AgentName"`)
    name: String,
    /// Base URL of the agent's apxm-server (e.g. "http://localhost:18801")
    url: String,
    /// Flow names this agent exposes
    #[serde(default)]
    flows: Vec<String>,
    /// Capability names this agent advertises
    #[serde(default)]
    capabilities: Vec<String>,
    /// Unix millisecond timestamp of when the agent registered
    registered_at: u64,
}

// ─── Task Queue ───────────────────────────────────────────────────────────────

/// Status of a queued task.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
enum TaskStatus {
    Pending,
    Claimed,
    Completed,
    Failed,
}

/// A task in the queue.
#[derive(Debug, Clone, Serialize, Deserialize)]
struct QueuedTask {
    id: String,
    queue: String,
    data: JsonValue,
    status: TaskStatus,
    #[serde(default)]
    claimed_by: Option<String>,
    #[serde(default)]
    claim_token: Option<String>,
    #[serde(default)]
    lease_expires_ms: Option<u64>,
    #[serde(default)]
    result: Option<JsonValue>,
    created_at_ms: u64,
    #[serde(default)]
    completed_at_ms: Option<u64>,
}

/// In-memory task queue manager.
///
/// Uses a `DashMap<queue_name, Mutex<VecDeque<QueuedTask>>>` for O(1) queue
/// lookup and ordered insertion.
#[derive(Clone)]
struct TaskQueueManager {
    inner: Arc<DashMap<String, Arc<Mutex<VecDeque<QueuedTask>>>>>,
    all_tasks: Arc<DashMap<String, QueuedTask>>,
}

impl TaskQueueManager {
    fn new() -> Self {
        Self {
            inner: Arc::new(DashMap::new()),
            all_tasks: Arc::new(DashMap::new()),
        }
    }

    async fn enqueue(&self, task: QueuedTask) {
        let queue: Arc<Mutex<VecDeque<QueuedTask>>> = {
            let entry = self
                .inner
                .entry(task.queue.clone())
                .or_insert_with(|| Arc::new(Mutex::new(VecDeque::new())));
            entry.value().clone()
        };
        self.all_tasks.insert(task.id.clone(), task.clone());
        queue.lock().await.push_back(task);
    }

    /// Atomically claim the next pending task from `queue_name`.
    async fn claim(&self, queue_name: &str, agent_id: &str, lease_ms: u64) -> Option<QueuedTask> {
        let queue: Arc<Mutex<VecDeque<QueuedTask>>> = self.inner.get(queue_name)?.value().clone();
        let mut guard = queue.lock().await;

        // Expire stale claims so their tasks become available again.
        let now = now_ms();
        for task in guard.iter_mut() {
            if task.status == TaskStatus::Claimed {
                if let Some(expires) = task.lease_expires_ms {
                    if now > expires {
                        task.status = TaskStatus::Pending;
                        task.claim_token = None;
                        task.claimed_by = None;
                        task.lease_expires_ms = None;
                        // Mirror into all_tasks index.
                        if let Some(mut indexed) = self.all_tasks.get_mut(&task.id) {
                            indexed.status = TaskStatus::Pending;
                            indexed.claim_token = None;
                            indexed.claimed_by = None;
                            indexed.lease_expires_ms = None;
                        }
                    }
                }
            }
        }

        let pos = guard.iter().position(|t| t.status == TaskStatus::Pending)?;
        let task = guard.get_mut(pos)?;
        let claim_token = uuid::Uuid::new_v4().to_string();
        let deadline = now_ms() + lease_ms;
        task.status = TaskStatus::Claimed;
        task.claimed_by = Some(agent_id.to_string());
        task.claim_token = Some(claim_token.clone());
        task.lease_expires_ms = Some(deadline);
        let claimed = task.clone();
        self.all_tasks.insert(claimed.id.clone(), claimed.clone());
        Some(claimed)
    }

    async fn complete(
        &self,
        task_id: &str,
        claim_token: &str,
        result: JsonValue,
    ) -> Result<(), String> {
        let mut task = self
            .all_tasks
            .get_mut(task_id)
            .ok_or_else(|| format!("Task '{}' not found", task_id))?;
        if task.claim_token.as_deref() != Some(claim_token) {
            return Err("Invalid claim token".to_string());
        }
        if let Some(expires) = task.lease_expires_ms {
            if now_ms() > expires {
                return Err("lease_expired: Task lease has expired. Task may have been reclaimed by another worker.".to_string());
            }
        }
        task.status = TaskStatus::Completed;
        task.result = Some(result.clone());
        task.completed_at_ms = Some(now_ms());
        let queue_name = task.queue.clone();
        let tid = task.id.clone();
        drop(task);
        if let Some(queue_ref) = self.inner.get(&queue_name) {
            let queue: Arc<Mutex<VecDeque<QueuedTask>>> = queue_ref.value().clone();
            drop(queue_ref);
            let mut guard = queue.lock().await;
            if let Some(t) = guard.iter_mut().find(|t| t.id == tid) {
                t.status = TaskStatus::Completed;
                t.result = Some(result);
                t.completed_at_ms = Some(now_ms());
            }
        }
        Ok(())
    }

    fn list_queue(&self, queue_name: &str) -> Vec<QueuedTask> {
        match self.inner.get(queue_name) {
            Some(q) => {
                let arc = q.value().clone();
                drop(q);
                match arc.try_lock() {
                    Ok(guard) => guard.iter().cloned().collect(),
                    Err(_) => vec![],
                }
            }
            None => vec![],
        }
    }
}

// ─── Checkpoint Store ─────────────────────────────────────────────────────────

/// Status of a PAUSE/RESUME checkpoint.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
enum CheckpointStatus {
    Pending,
    Resumed,
    Expired,
}

/// A PAUSE checkpoint waiting for human input.
#[derive(Debug, Clone, Serialize, Deserialize)]
struct Checkpoint {
    id: String,
    message: String,
    display_data: JsonValue,
    status: CheckpointStatus,
    #[serde(default)]
    human_input: Option<JsonValue>,
    #[serde(default)]
    notification_url: Option<String>,
    created_at_ms: u64,
    #[serde(default)]
    resumed_at_ms: Option<u64>,
}

/// In-memory checkpoint store for PAUSE/RESUME HITL workflow.
#[derive(Clone)]
struct CheckpointStore {
    inner: Arc<DashMap<String, Checkpoint>>,
}

impl CheckpointStore {
    fn new() -> Self {
        Self {
            inner: Arc::new(DashMap::new()),
        }
    }

    fn create(&self, checkpoint: Checkpoint) {
        self.inner.insert(checkpoint.id.clone(), checkpoint);
    }

    fn get(&self, id: &str) -> Option<Checkpoint> {
        self.inner.get(id).map(|c| c.clone())
    }

    fn resume(&self, id: &str, human_input: JsonValue) -> Result<Checkpoint, String> {
        let mut entry = self
            .inner
            .get_mut(id)
            .ok_or_else(|| format!("Checkpoint '{}' not found", id))?;
        if entry.status != CheckpointStatus::Pending {
            return Err(format!("Checkpoint '{}' is not in pending state", id));
        }
        entry.status = CheckpointStatus::Resumed;
        entry.human_input = Some(human_input);
        entry.resumed_at_ms = Some(now_ms());
        Ok(entry.clone())
    }
}

// ─── HTTP Capability ────────────────────────────────────────────────────────
/// A capability that forwards invocations to an external HTTP endpoint.
///
/// The endpoint receives the capability arguments as a JSON object via POST and
/// must return a JSON object with a `"result"` key (or any parseable JSON value).
#[derive(Clone)]
struct HttpCapability {
    metadata: CapabilityMetadata,
    endpoint: String,
    timeout_ms: u64,
    client: reqwest::Client,
}

#[async_trait]
impl CapabilityExecutor for HttpCapability {
    async fn execute(&self, args: HashMap<String, Value>) -> CapabilityResult<Value> {
        // Serialize arguments to JSON
        let body: serde_json::Map<String, JsonValue> = args
            .iter()
            .map(|(k, v)| {
                (
                    k.clone(),
                    v.to_json()
                        .unwrap_or_else(|_| JsonValue::String(v.to_string())),
                )
            })
            .collect();

        let cap_err = |msg: String| RuntimeError::Capability {
            capability: self.metadata.name.clone(),
            message: msg,
        };

        let resp = self
            .client
            .post(&self.endpoint)
            .timeout(std::time::Duration::from_millis(self.timeout_ms))
            .json(&body)
            .send()
            .await
            .map_err(|e| cap_err(format!("request failed: {e}")))?;

        if !resp.status().is_success() {
            let status = resp.status();
            let text = resp.text().await.unwrap_or_default();
            return Err(cap_err(format!("endpoint returned {status}: {text}")));
        }

        let result: JsonValue = resp
            .json()
            .await
            .map_err(|e| cap_err(format!("response parse failed: {e}")))?;

        // Unwrap a `{"result": ...}` envelope if present, otherwise use the raw response
        let inner = if let Some(r) = result.get("result") {
            r.clone()
        } else {
            result
        };

        Value::try_from(inner).map_err(|e| cap_err(format!("value conversion failed: {e}")))
    }

    fn metadata(&self) -> &CapabilityMetadata {
        &self.metadata
    }
}

#[derive(Clone)]
struct AppState {
    runtime: Arc<Runtime>,
    /// In-memory agent registry: name → registration record
    agent_registry: Arc<DashMap<String, AgentRegistration>>,
    /// Task queue manager backing the CLAIM op.
    task_manager: TaskQueueManager,
    /// Checkpoint store for PAUSE/RESUME HITL workflow.
    checkpoint_store: CheckpointStore,
    /// Server start time for uptime reporting.
    start_time: SystemTime,
    /// In-flight A2A task records (task_id → record).
    a2a_tasks: Arc<DashMap<String, A2aTaskRecord>>,
}

// ─── A2A v0.3 Types ──────────────────────────────────────────────────────────

/// Lifecycle state of an A2A task.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
enum A2aState {
    Submitted,
    Working,
    Completed,
    Failed,
    Canceled,
}

/// In-memory record for a running or completed A2A task.
#[derive(Debug, Clone, Serialize, Deserialize)]
struct A2aTaskRecord {
    id: String,
    state: A2aState,
    #[serde(default)]
    output_text: Option<String>,
    #[serde(default)]
    error_message: Option<String>,
    created_at_ms: u64,
    #[serde(default)]
    completed_at_ms: Option<u64>,
}

/// Inbound A2A message part — text or opaque data.
#[derive(Debug, Deserialize)]
#[serde(tag = "type", rename_all = "lowercase")]
enum A2aPart {
    Text { text: String },
    Data {
        #[allow(dead_code)]
        data: JsonValue,
    },
}

/// Inbound A2A message envelope.
#[derive(Debug, Deserialize)]
struct A2aMessage {
    #[allow(dead_code)]
    role: String,
    parts: Vec<A2aPart>,
}

/// Body for `POST /a2a/tasks/send`.
#[derive(Debug, Deserialize)]
struct A2aSendTaskRequest {
    id: String,
    message: A2aMessage,
    #[serde(default)]
    #[allow(dead_code)]
    metadata: Option<JsonValue>,
}

// ─── Execute Request ─────────────────────────────────────────────────────────

#[derive(Debug, Deserialize)]
struct ExecuteRequest {
    graph: JsonValue,
    #[serde(default)]
    args: Vec<String>,
    #[serde(default)]
    session_id: Option<String>,
    #[serde(default)]
    token_budget: Option<u64>,
    #[serde(default)]
    output_schema: Option<JsonValue>,
    #[serde(default)]
    max_schema_retries: Option<u32>,
}

#[derive(Debug, Serialize)]
struct ExecuteResponse {
    results: HashMap<String, JsonValue>,
    content: Option<String>,
    stats: JsonValue,
    llm_usage: JsonValue,
}

#[derive(Debug, Deserialize)]
struct StoreFactRequest {
    text: String,
    #[serde(default)]
    tags: Vec<String>,
    #[serde(default)]
    source: String,
    #[serde(default)]
    session_id: Option<String>,
}

#[derive(Debug, Serialize)]
struct StoreFactResponse {
    id: String,
}

#[derive(Debug, Deserialize)]
struct SearchFactsRequest {
    query: String,
    #[serde(default = "default_limit")]
    limit: usize,
}

fn default_limit() -> usize {
    5
}

#[derive(Debug, Deserialize)]
struct DeleteFactRequest {
    id: String,
}

#[derive(Debug, Deserialize)]
struct RegisterCapabilityRequest {
    name: String,
    description: String,
    #[serde(default)]
    parameters_schema: JsonValue,
    /// If set, an `HttpCapability` is created that POSTs to this URL.
    /// Takes priority over `static_response`.
    #[serde(default)]
    endpoint: Option<String>,
    /// Timeout in milliseconds for HTTP capability calls (default: 30 000).
    #[serde(default)]
    timeout_ms: Option<u64>,
    /// Fallback: return a fixed static value (used when `endpoint` is absent).
    #[serde(default)]
    static_response: JsonValue,
}

#[derive(Clone)]
struct StaticCapability {
    metadata: CapabilityMetadata,
    static_response: Value,
}

#[async_trait]
impl CapabilityExecutor for StaticCapability {
    async fn execute(&self, _args: HashMap<String, Value>) -> CapabilityResult<Value> {
        Ok(self.static_response.clone())
    }

    fn metadata(&self) -> &CapabilityMetadata {
        &self.metadata
    }
}

#[derive(Clone)]
struct ChannelEventEmitter {
    tx: mpsc::Sender<JsonValue>,
}

impl ExecutionEventEmitter for ChannelEventEmitter {
    fn emit_llm_token(&self, content: &str) {
        let _ = self
            .tx
            .try_send(serde_json::json!({ "type": "llm_token", "content": content }));
    }

    fn emit_tool_start(&self, name: &str, args: &HashMap<String, Value>) {
        let args = args
            .iter()
            .map(|(k, v)| {
                (
                    k.clone(),
                    v.to_json()
                        .unwrap_or_else(|_| JsonValue::String(v.to_string())),
                )
            })
            .collect::<serde_json::Map<String, JsonValue>>();
        let _ = self.tx.try_send(serde_json::json!({
            "type": "tool_start",
            "name": name,
            "args": args
        }));
    }

    fn emit_tool_end(&self, name: &str, result: &Value) {
        let result_json = result
            .to_json()
            .unwrap_or_else(|_| JsonValue::String(result.to_string()));
        let _ = self.tx.try_send(serde_json::json!({
            "type": "tool_end",
            "name": name,
            "result": result_json
        }));
    }
}

/// Build the Axum router for the APXM server.
///
/// Extracted from `main()` so that integration tests can call it directly
/// without binding to a TCP port.
fn build_app(state: AppState) -> Router {
    let req_id_header = axum::http::HeaderName::from_static("x-request-id");
    Router::new()
        // Health + meta
        .route("/health", get(health))
        .route("/v1/models", get(list_models))
        // Execution
        .route("/v1/execute", post(execute))
        .route("/v1/execute/stream", post(execute_stream))
        // Memory
        .route("/v1/memory/facts/store", post(store_fact))
        .route("/v1/memory/facts/search", post(search_facts))
        .route("/v1/memory/facts/delete", post(delete_fact))
        // Capabilities
        .route("/v1/capabilities", get(list_capabilities))
        .route("/v1/capabilities/register", post(register_capability))
        // COMMUNICATE receive target
        .route("/v1/receive", post(receive_message))
        // Agent registry
        .route("/v1/agents", get(list_agents))
        .route("/v1/agents/register", post(register_agent))
        .route("/v1/agents/:name", get(get_agent))
        .route("/v1/agents/:name", delete(deregister_agent))
        // Task queue (Plan 07 — CLAIM op backend)
        .route("/v1/tasks", post(create_task))
        .route("/v1/tasks/:queue", get(list_tasks))
        .route("/v1/tasks/:queue/claim", post(claim_task))
        .route("/v1/tasks/:id/complete", post(complete_task))
        // Checkpoints (Plan 07 — PAUSE/RESUME HITL)
        .route("/v1/checkpoints", post(create_checkpoint))
        .route("/v1/checkpoints/:id", get(get_checkpoint))
        .route("/v1/checkpoints/:id/resume", post(resume_checkpoint))
        // A2A v0.3 — AgentCard discovery + REST task lifecycle
        .route("/.well-known/agent.json", get(agent_card))
        .route("/a2a", post(a2a_jsonrpc)) // legacy JSON-RPC compat
        .route("/a2a/tasks/send", post(a2a_send_task)) // REST: submit task + execute
        .route("/a2a/tasks/:id", get(a2a_get_task)) // REST: poll task result
        // MCP 2025-11-05 — JSON-RPC tools endpoint
        .route("/v1/mcp", post(mcp_jsonrpc))
        .with_state(state)
        .layer(CorsLayer::permissive())
        .layer(TraceLayer::new_for_http())
        // Propagate X-Request-Id from clients; generate one when absent
        .layer(PropagateRequestIdLayer::new(req_id_header.clone()))
        .layer(SetRequestIdLayer::new(
            req_id_header,
            MakeRequestUuid::default(),
        ))
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(
            std::env::var("RUST_LOG").unwrap_or_else(|_| "info,apxm_server=debug".to_string()),
        )
        .init();

    let runtime = Arc::new(Runtime::new(RuntimeConfig::default()).await?);
    let state = AppState {
        runtime,
        agent_registry: Arc::new(DashMap::new()),
        task_manager: TaskQueueManager::new(),
        checkpoint_store: CheckpointStore::new(),
        start_time: SystemTime::now(),
        a2a_tasks: Arc::new(DashMap::new()),
    };

    let app = build_app(state);

    let addr = std::env::var("APXM_SERVER_ADDR")
        .ok()
        .and_then(|s| s.parse::<SocketAddr>().ok())
        .unwrap_or_else(|| "127.0.0.1:18800".parse().expect("valid default addr"));
    info!(%addr, "starting apxm-server");
    let listener = tokio::net::TcpListener::bind(addr).await?;

    // Graceful shutdown on Ctrl+C / SIGTERM
    let shutdown = async {
        #[cfg(unix)]
        {
            use tokio::signal::unix::{SignalKind, signal};
            let mut sigterm = signal(SignalKind::terminate()).expect("SIGTERM handler");
            tokio::select! {
                _ = tokio::signal::ctrl_c() => {},
                _ = sigterm.recv() => {},
            }
        }
        #[cfg(not(unix))]
        {
            tokio::signal::ctrl_c().await.expect("Ctrl+C handler");
        }
        info!("shutdown signal received");
    };

    axum::serve(listener, app)
        .with_graceful_shutdown(shutdown)
        .await?;
    Ok(())
}

async fn health(State(state): State<AppState>) -> Json<JsonValue> {
    let uptime_secs = state.start_time.elapsed().map(|d| d.as_secs()).unwrap_or(0);
    Json(serde_json::json!({
        "status": "ok",
        "version": env!("CARGO_PKG_VERSION"),
        "uptime_secs": uptime_secs,
    }))
}

// ─── Model Discovery (/v1/models) ────────────────────────────────────────────

/// OpenAI-compatible `/v1/models` endpoint.
///
/// Returns a list of LLM backends registered with the runtime so clients
/// (AgentMate, OpenAI SDK, etc.) can discover which models are available.
async fn list_models(State(state): State<AppState>) -> Json<JsonValue> {
    // The APXM runtime exposes registered backend names via the LLM registry.
    // We surface them in the OpenAI models format for compatibility.
    let models: Vec<JsonValue> = state
        .runtime
        .llm_registry()
        .backend_names()
        .into_iter()
        .map(|name| {
            serde_json::json!({
                "id": name,
                "object": "model",
                "created": 0,
                "owned_by": "apxm",
            })
        })
        .collect();
    Json(serde_json::json!({
        "object": "list",
        "data": models,
    }))
}

// ─── MCP 2025-11-05 JSON-RPC endpoint (/v1/mcp) ─────────────────────────────

/// MCP 2025-11-05 compatible JSON-RPC handler.
///
/// Supports:
/// - `tools/list`  — enumerate APXM capabilities as MCP tools
/// - `tools/call`  — invoke an APXM capability by name
///
/// Wire format: `{"jsonrpc":"2.0","id":1,"method":"tools/list","params":{}}`
#[derive(Debug, Deserialize)]
struct McpRequest {
    #[serde(default)]
    id: JsonValue,
    method: String,
    #[serde(default)]
    params: JsonValue,
}

async fn mcp_jsonrpc(
    State(state): State<AppState>,
    Json(req): Json<McpRequest>,
) -> Json<JsonValue> {
    let id = req.id.clone();
    match req.method.as_str() {
        "tools/list" => {
            let tools: Vec<JsonValue> = state
                .runtime
                .capability_system()
                .list_capabilities()
                .iter()
                .map(|m| {
                    serde_json::json!({
                        "name": m.name,
                        "description": m.description,
                        "inputSchema": m.parameters_schema,
                    })
                })
                .collect();
            Json(serde_json::json!({
                "jsonrpc": "2.0",
                "id": id,
                "result": { "tools": tools },
            }))
        }
        "tools/call" => {
            let tool_name = req
                .params
                .get("name")
                .and_then(|v| v.as_str())
                .unwrap_or("");
            let tool_args = req
                .params
                .get("arguments")
                .cloned()
                .unwrap_or(JsonValue::Object(Default::default()));

            // Convert JSON args to Value map
            let args: HashMap<String, Value> = if let JsonValue::Object(map) = &tool_args {
                map.iter()
                    .filter_map(|(k, v)| {
                        Value::try_from(v.clone()).ok().map(|val| (k.clone(), val))
                    })
                    .collect()
            } else {
                HashMap::new()
            };

            let cap_sys = state.runtime.capability_system();
            let executor = cap_sys.registry().get(tool_name);
            match executor {
                None => Json(serde_json::json!({
                    "jsonrpc": "2.0",
                    "id": id,
                    "result": {
                        "content": [{ "type": "text", "text": format!("Tool '{}' not found", tool_name) }],
                        "isError": true,
                    },
                })),
                Some(exec) => match exec.execute(args).await {
                    Ok(result) => {
                        let result_json = result
                            .to_json()
                            .unwrap_or_else(|_| JsonValue::String(result.to_string()));
                        Json(serde_json::json!({
                            "jsonrpc": "2.0",
                            "id": id,
                            "result": {
                                "content": [{ "type": "text", "text": result_json.to_string() }],
                                "isError": false,
                            },
                        }))
                    }
                    Err(e) => Json(serde_json::json!({
                        "jsonrpc": "2.0",
                        "id": id,
                        "result": {
                            "content": [{ "type": "text", "text": e.to_string() }],
                            "isError": true,
                        },
                    })),
                },
            }
        }
        "initialize" => {
            // MCP protocol handshake
            Json(serde_json::json!({
                "jsonrpc": "2.0",
                "id": id,
                "result": {
                    "protocolVersion": "2025-11-05",
                    "serverInfo": {
                        "name": "apxm-server",
                        "version": env!("CARGO_PKG_VERSION"),
                    },
                    "capabilities": {
                        "tools": { "listChanged": false },
                    },
                },
            }))
        }
        unknown => Json(serde_json::json!({
            "jsonrpc": "2.0",
            "id": id,
            "error": {
                "code": -32601,
                "message": format!("Method not found: {}", unknown),
            },
        })),
    }
}

// ─── A2A v0.3 JSON-RPC endpoint (/a2a) ───────────────────────────────────────

/// A2A v0.3 JSON-RPC handler.
///
/// Supports:
/// - `tasks/send`    — submit a task for execution
/// - `tasks/get`     — retrieve task status (stored in memory as fact)
/// - `tasks/cancel`  — cancel a pending task (best-effort)
///
/// Wire format: `{"jsonrpc":"2.0","id":1,"method":"tasks/send","params":{...}}`
async fn a2a_jsonrpc(
    State(state): State<AppState>,
    Json(req): Json<McpRequest>,
) -> Json<JsonValue> {
    let id = req.id.clone();
    match req.method.as_str() {
        "tasks/send" => {
            // Extract task from params
            let task_id = uuid::Uuid::new_v4().to_string();
            let message = req
                .params
                .get("message")
                .cloned()
                .unwrap_or(req.params.clone());
            let text = message.to_string();
            let source = "a2a".to_string();

            // Store the incoming task in memory so agents can retrieve it
            match state
                .runtime
                .memory()
                .store_fact(&text, &["a2a:task".to_string()], &source, None)
                .await
            {
                Ok(fact_id) => Json(serde_json::json!({
                    "jsonrpc": "2.0",
                    "id": id,
                    "result": {
                        "id": task_id,
                        "factId": fact_id,
                        "status": { "state": "submitted" },
                        "message": message,
                    },
                })),
                Err(e) => Json(serde_json::json!({
                    "jsonrpc": "2.0",
                    "id": id,
                    "error": {
                        "code": -32000,
                        "message": e.to_string(),
                    },
                })),
            }
        }
        "tasks/get" => {
            let task_id = req.params.get("id").and_then(|v| v.as_str()).unwrap_or("");
            // Tasks are stored as memory facts — search by task ID
            match state.runtime.memory().search_facts(task_id, 1).await {
                Ok(facts) if !facts.is_empty() => Json(serde_json::json!({
                    "jsonrpc": "2.0",
                    "id": id,
                    "result": {
                        "id": task_id,
                        "status": { "state": "completed" },
                        "facts": serde_json::to_value(&facts).unwrap_or(JsonValue::Null),
                    },
                })),
                Ok(_) => Json(serde_json::json!({
                    "jsonrpc": "2.0",
                    "id": id,
                    "error": {
                        "code": -32001,
                        "message": format!("Task '{}' not found", task_id),
                    },
                })),
                Err(e) => Json(serde_json::json!({
                    "jsonrpc": "2.0",
                    "id": id,
                    "error": { "code": -32000, "message": e.to_string() },
                })),
            }
        }
        "tasks/cancel" => {
            // Best-effort cancellation — APXM runtime doesn't currently support mid-flight cancel
            Json(serde_json::json!({
                "jsonrpc": "2.0",
                "id": id,
                "result": { "cancelled": true },
            }))
        }
        unknown => Json(serde_json::json!({
            "jsonrpc": "2.0",
            "id": id,
            "error": {
                "code": -32601,
                "message": format!("A2A method not found: {}", unknown),
            },
        })),
    }
}

async fn execute(
    State(state): State<AppState>,
    Json(req): Json<ExecuteRequest>,
) -> Result<Json<ExecuteResponse>, ApiError> {
    let (graph, args, session_id) = prepare_request(req)?;
    let artifact = graph_to_artifact(graph)?;
    let execution = state
        .runtime
        .execute_artifact_with_session(artifact, args, session_id)
        .await
        .map_err(ApiError::runtime)?;
    Ok(Json(to_execute_response(execution)))
}

async fn execute_stream(
    State(state): State<AppState>,
    Json(req): Json<ExecuteRequest>,
) -> Result<Sse<impl Stream<Item = Result<Event, std::convert::Infallible>>>, ApiError> {
    let (graph, args, session_id) = prepare_request(req)?;
    let artifact = graph_to_artifact(graph)?;
    let (tx, mut rx) = mpsc::channel::<JsonValue>(128);
    let runtime = Arc::clone(&state.runtime);
    tokio::spawn(async move {
        let emitter = Arc::new(ChannelEventEmitter { tx: tx.clone() });
        match runtime
            .execute_artifact_with_session_and_emitter(artifact, args, session_id, Some(emitter))
            .await
        {
            Ok(result) => {
                let _ = tx
                    .send(serde_json::json!({
                        "type": "complete",
                        "result": to_execute_response(result)
                    }))
                    .await;
            }
            Err(err) => {
                let _ = tx
                    .send(serde_json::json!({
                        "type": "error",
                        "message": err.to_string()
                    }))
                    .await;
            }
        }
    });

    let stream = async_stream::stream! {
        while let Some(item) = rx.recv().await {
            let data = item.to_string();
            yield Ok(Event::default().data(data));
        }
    };
    Ok(Sse::new(stream).keep_alive(KeepAlive::default()))
}

async fn store_fact(
    State(state): State<AppState>,
    Json(req): Json<StoreFactRequest>,
) -> Result<Json<StoreFactResponse>, ApiError> {
    let id = state
        .runtime
        .memory()
        .store_fact(&req.text, &req.tags, &req.source, req.session_id)
        .await
        .map_err(ApiError::runtime)?;
    Ok(Json(StoreFactResponse { id }))
}

async fn search_facts(
    State(state): State<AppState>,
    Json(req): Json<SearchFactsRequest>,
) -> Result<Json<JsonValue>, ApiError> {
    let results = state
        .runtime
        .memory()
        .search_facts(&req.query, req.limit)
        .await
        .map_err(ApiError::runtime)?;
    Ok(Json(
        serde_json::to_value(results).map_err(|e| ApiError::internal_message(e.to_string()))?,
    ))
}

async fn delete_fact(
    State(state): State<AppState>,
    Json(req): Json<DeleteFactRequest>,
) -> Result<Json<JsonValue>, ApiError> {
    state
        .runtime
        .memory()
        .delete_fact(&req.id)
        .await
        .map_err(ApiError::runtime)?;
    Ok(Json(serde_json::json!({ "ok": true })))
}

async fn list_capabilities(State(state): State<AppState>) -> Result<Json<JsonValue>, ApiError> {
    let caps = state
        .runtime
        .capability_system()
        .list_capabilities()
        .iter()
        .map(|m| {
            serde_json::json!({
                "name": m.name,
                "description": m.description,
                "parameters_schema": m.parameters_schema,
            })
        })
        .collect::<Vec<_>>();
    Ok(Json(JsonValue::Array(caps)))
}

async fn register_capability(
    State(state): State<AppState>,
    Json(req): Json<RegisterCapabilityRequest>,
) -> Result<Json<JsonValue>, ApiError> {
    let metadata = CapabilityMetadata::new(
        req.name.clone(),
        req.description.clone(),
        req.parameters_schema,
    );

    let capability: Arc<dyn CapabilityExecutor> = if let Some(endpoint) = req.endpoint {
        // HTTP capability — forwards invocations to external server
        info!(name = %req.name, endpoint = %endpoint, "registering HTTP capability");
        Arc::new(HttpCapability {
            metadata,
            endpoint,
            timeout_ms: req.timeout_ms.unwrap_or(30_000),
            client: reqwest::Client::new(),
        })
    } else {
        // Static capability — always returns the same value (legacy behaviour)
        let response_value = Value::try_from(req.static_response)
            .map_err(|e| ApiError::bad_request(e.to_string()))?;
        info!(name = %req.name, "registering static capability");
        Arc::new(StaticCapability {
            metadata,
            static_response: response_value,
        })
    };

    state
        .runtime
        .capability_system()
        .register(capability)
        .map_err(ApiError::runtime)?;
    Ok(Json(serde_json::json!({ "ok": true, "name": req.name })))
}

fn prepare_request(
    mut req: ExecuteRequest,
) -> Result<(ApxmGraph, Vec<String>, Option<String>), ApiError> {
    let mut graph = ApxmGraph::from_json(
        &serde_json::to_string(&req.graph).map_err(|e| ApiError::bad_request(format!("invalid json: {e}")))?,
    )
    .map_err(|e| ApiError::bad_request(format!("invalid graph: {e}")))?;
    apply_runtime_attributes(
        &mut graph,
        req.token_budget.take(),
        req.output_schema.take(),
        req.max_schema_retries,
    )
    .map_err(ApiError::bad_request)?;
    Ok((graph, req.args, req.session_id))
}

fn apply_runtime_attributes(
    graph: &mut ApxmGraph,
    token_budget: Option<u64>,
    output_schema: Option<JsonValue>,
    max_schema_retries: Option<u32>,
) -> Result<(), String> {
    for node in &mut graph.nodes {
        if node.op != AISOperationType::Ask {
            continue;
        }
        if let Some(budget) = token_budget {
            let budget = i64::try_from(budget).unwrap_or(i64::MAX);
            node.attributes.insert(
                graph_attrs::TOKEN_BUDGET.to_string(),
                Value::Number(budget.into()),
            );
        }
        if let Some(schema) = output_schema.as_ref() {
            let schema_value = Value::try_from(schema.clone())
                .map_err(|e| format!("invalid output_schema: {e}"))?;
            node.attributes
                .insert(graph_attrs::OUTPUT_SCHEMA.to_string(), schema_value);
        }
        if let Some(retries) = max_schema_retries {
            let retries = i64::from(retries);
            node.attributes.insert(
                graph_attrs::MAX_SCHEMA_RETRIES.to_string(),
                Value::Number(retries.into()),
            );
        }
    }
    Ok(())
}

fn graph_to_artifact(graph: ApxmGraph) -> Result<Artifact, ApiError> {
    let context = CompilerContext::new().map_err(|error| {
        ApiError::internal_message(format!(
            "failed to initialize APXM compiler context: {error}"
        ))
    })?;
    let pipeline =
        CompilerPipeline::with_opt_level(&context, apxm_core::types::OptimizationLevel::O1);
    let module = pipeline.compile_graph(&graph).map_err(|error| {
        ApiError::bad_request(format!("failed to compile graph '{}': {error}", graph.name))
    })?;
    let artifact_bytes = module
        .generate_artifact_bytes()
        .map_err(|error| ApiError::internal_message(format!("failed to emit artifact: {error}")))?;
    Artifact::from_bytes(&artifact_bytes)
        .map_err(|error| ApiError::internal_message(format!("failed to decode artifact: {error}")))
}

fn to_execute_response(result: apxm_runtime::RuntimeExecutionResult) -> ExecuteResponse {
    let mut mapped = HashMap::new();
    let mut content = None;
    for (token, value) in &result.results {
        let json = value
            .to_json()
            .unwrap_or_else(|_| JsonValue::String(value.to_string()));
        if content.is_none()
            && let Some(text) = json.as_str()
        {
            content = Some(text.to_string());
        }
        mapped.insert(token.to_string(), json);
    }

    ExecuteResponse {
        results: mapped,
        content,
        stats: serde_json::json!({
            "executed_nodes": result.stats.executed_nodes,
            "failed_nodes": result.stats.failed_nodes,
            "duration_ms": result.stats.duration_ms
        }),
        llm_usage: serde_json::json!({
            "input_tokens": result.llm_metrics.total_input_tokens,
            "output_tokens": result.llm_metrics.total_output_tokens,
            "total_requests": result.llm_metrics.total_requests
        }),
    }
}

// ─── Receive Message (HTTP COMMUNICATE target) ─────────────────────────────

#[derive(Debug, Deserialize)]
struct ReceiveMessageRequest {
    from: String,
    message: JsonValue,
    #[serde(default)]
    channel: Option<String>,
}

async fn receive_message(
    State(state): State<AppState>,
    Json(req): Json<ReceiveMessageRequest>,
) -> Result<Json<JsonValue>, ApiError> {
    let text = req.message.to_string();
    let source = req.from.clone();
    let tags = req
        .channel
        .as_deref()
        .map(|c| vec![format!("channel:{}", c)])
        .unwrap_or_default();
    let id = state
        .runtime
        .memory()
        .store_fact(&text, &tags, &source, None)
        .await
        .map_err(ApiError::runtime)?;
    info!(from = %req.from, id = %id, "Received inter-agent message");
    Ok(Json(serde_json::json!({ "ok": true, "id": id })))
}

// ─── Agent Registry Handlers ─────────────────────────────────────────────────

#[derive(Debug, Deserialize)]
struct RegisterAgentRequest {
    name: String,
    url: String,
    #[serde(default)]
    flows: Vec<String>,
    #[serde(default)]
    capabilities: Vec<String>,
}

async fn register_agent(
    State(state): State<AppState>,
    Json(req): Json<RegisterAgentRequest>,
) -> Result<Json<JsonValue>, ApiError> {
    let reg = AgentRegistration {
        name: req.name.clone(),
        url: req.url,
        flows: req.flows,
        capabilities: req.capabilities,
        registered_at: now_ms(),
    };
    info!(name = %req.name, "Registering agent");
    state.agent_registry.insert(req.name.clone(), reg);
    Ok(Json(serde_json::json!({ "ok": true, "name": req.name })))
}

async fn list_agents(State(state): State<AppState>) -> Json<JsonValue> {
    let agents: Vec<JsonValue> = state
        .agent_registry
        .iter()
        .map(|e| serde_json::to_value(e.value()).unwrap_or(JsonValue::Null))
        .collect();
    Json(JsonValue::Array(agents))
}

async fn get_agent(
    State(state): State<AppState>,
    Path(name): Path<String>,
) -> Result<Json<JsonValue>, ApiError> {
    match state.agent_registry.get(&name) {
        Some(entry) => Ok(Json(
            serde_json::to_value(entry.value())
                .map_err(|e| ApiError::internal_message(e.to_string()))?,
        )),
        None => Err(ApiError::not_found(format!(
            "Agent '{}' not registered",
            name
        ))),
    }
}

async fn deregister_agent(
    State(state): State<AppState>,
    Path(name): Path<String>,
) -> Result<Json<JsonValue>, ApiError> {
    if state.agent_registry.remove(&name).is_some() {
        info!(%name, "Deregistered agent");
        Ok(Json(serde_json::json!({ "ok": true })))
    } else {
        Err(ApiError::not_found(format!(
            "Agent '{}' not registered",
            name
        )))
    }
}

// ─── A2A AgentCard ───────────────────────────────────────────────────────────

async fn agent_card(State(state): State<AppState>) -> Json<JsonValue> {
    let base_url =
        std::env::var("APXM_PUBLIC_URL").unwrap_or_else(|_| "http://localhost:18800".to_string());

    let skills: Vec<JsonValue> = state
        .agent_registry
        .iter()
        .flat_map(|entry| {
            let agent = entry.value();
            agent
                .flows
                .iter()
                .map(|flow| {
                    serde_json::json!({
                        "id": format!("{}.{}", agent.name, flow),
                        "name": format!("{}/{}", agent.name, flow),
                        "description": format!("Execute {} flow on agent {}", flow, agent.name),
                        "inputModes": ["text"],
                        "outputModes": ["text"],
                    })
                })
                .collect::<Vec<_>>()
        })
        .collect();

    Json(serde_json::json!({
        "protocolVersion": "0.3",
        "name": "APXM Agent Runtime",
        "description": "Program Execution Model for AI agents — parallel dataflow, multi-model councils, formal agent programs.",
        "version": env!("CARGO_PKG_VERSION"),
        "url": base_url,
        "capabilities": {
            "streaming": true,
            "pushNotifications": false,
            "stateTransitionHistory": true
        },
        "defaultInputModes": ["text"],
        "defaultOutputModes": ["text"],
        "skills": skills,
        "authentication": {
            "schemes": ["bearer"],
            "credentials": null
        }
    }))
}

// ─── A2A v0.3 REST Handlers ───────────────────────────────────────────────────

/// `POST /a2a/tasks/send`
///
/// Accepts an A2A v0.3 task, constructs a minimal CONST_STR → ASK graph from
/// the text parts, executes it immediately via the APXM runtime, and returns
/// the result in A2A response format.  The task record is stored in
/// `AppState.a2a_tasks` for subsequent `GET /a2a/tasks/:id` polling.
async fn a2a_send_task(
    State(state): State<AppState>,
    Json(req): Json<A2aSendTaskRequest>,
) -> impl IntoResponse {
    let now = now_ms();
    state.a2a_tasks.insert(
        req.id.clone(),
        A2aTaskRecord {
            id: req.id.clone(),
            state: A2aState::Working,
            output_text: None,
            error_message: None,
            created_at_ms: now,
            completed_at_ms: None,
        },
    );

    // Concatenate all text parts into a single prompt.
    let user_text: String = req
        .message
        .parts
        .iter()
        .filter_map(|p| match p {
            A2aPart::Text { text } => Some(text.clone()),
            A2aPart::Data { .. } => None,
        })
        .collect::<Vec<_>>()
        .join("\n");

    if user_text.is_empty() {
        if let Some(mut record) = state.a2a_tasks.get_mut(&req.id) {
            record.state = A2aState::Failed;
            record.error_message = Some("No text content in message".to_string());
            record.completed_at_ms = Some(now_ms());
        }
        return (
            axum::http::StatusCode::BAD_REQUEST,
            Json(serde_json::json!({
                "id": req.id,
                "status": {"state": "failed"},
                "error": {"message": "No text content in A2A message parts"}
            })),
        )
            .into_response();
    }

    // Build a minimal CONST_STR → ASK ApxmGraph from the inbound text.
    let graph = ApxmGraph {
        name: format!("a2a_{}", req.id),
        nodes: vec![
            GraphNode {
                id: 1,
                name: "input".to_string(),
                op: AISOperationType::ConstStr,
                attributes: HashMap::from([("value".to_string(), Value::String(user_text))]),
            },
            GraphNode {
                id: 2,
                name: "response".to_string(),
                op: AISOperationType::Ask,
                attributes: HashMap::from([(
                    graph_attrs::TEMPLATE_STR.to_string(),
                    Value::String("Complete the following task:\n{0}".to_string()),
                )]),
            },
        ],
        edges: vec![GraphEdge {
            from: 1,
            to: 2,
            dependency: DependencyType::Data,
        }],
        parameters: vec![],
        metadata: HashMap::new(),
    };

    let artifact = match graph_to_artifact(graph) {
        Ok(a) => a,
        Err(e) => {
            if let Some(mut record) = state.a2a_tasks.get_mut(&req.id) {
                record.state = A2aState::Failed;
                record.error_message = Some(e.message.clone());
                record.completed_at_ms = Some(now_ms());
            }
            return (
                axum::http::StatusCode::INTERNAL_SERVER_ERROR,
                Json(serde_json::json!({"id": req.id, "status": {"state": "failed"}, "error": {"message": e.message}})),
            ).into_response();
        }
    };

    match state
        .runtime
        .execute_artifact_with_session(artifact, vec![], None)
        .await
    {
        Ok(result) => {
            let resp = to_execute_response(result);
            let output = resp
                .content
                .clone()
                .unwrap_or_else(|| serde_json::to_string(&resp.results).unwrap_or_default());
            if let Some(mut record) = state.a2a_tasks.get_mut(&req.id) {
                record.state = A2aState::Completed;
                record.output_text = Some(output.clone());
                record.completed_at_ms = Some(now_ms());
            }
            Json(serde_json::json!({
                "id": req.id,
                "status": {"state": "completed"},
                "result": {"message": {"role": "agent", "parts": [{"type": "text", "text": output}]}}
            })).into_response()
        }
        Err(e) => {
            if let Some(mut record) = state.a2a_tasks.get_mut(&req.id) {
                record.state = A2aState::Failed;
                record.error_message = Some(e.to_string());
                record.completed_at_ms = Some(now_ms());
            }
            (
                axum::http::StatusCode::INTERNAL_SERVER_ERROR,
                Json(serde_json::json!({"id": req.id, "status": {"state": "failed"}, "error": {"message": e.to_string()}})),
            ).into_response()
        }
    }
}

/// `GET /a2a/tasks/:id` — return status and result of a previously submitted A2A task.
async fn a2a_get_task(State(state): State<AppState>, Path(id): Path<String>) -> impl IntoResponse {
    match state.a2a_tasks.get(&id) {
        Some(record) => {
            let mut body = serde_json::json!({"id": record.id, "status": {"state": record.state}});
            if let Some(ref text) = record.output_text {
                body["result"] = serde_json::json!({"message": {"role": "agent", "parts": [{"type": "text", "text": text}]}});
            }
            if let Some(ref err) = record.error_message {
                body["error"] = serde_json::json!({"message": err});
            }
            Json(body).into_response()
        }
        None => (
            axum::http::StatusCode::NOT_FOUND,
            Json(serde_json::json!({"error": format!("Task '{}' not found", id)})),
        )
            .into_response(),
    }
}

// ─── Error Types ─────────────────────────────────────────────────────────────

#[derive(Debug)]
struct ApiError {
    status: axum::http::StatusCode,
    message: String,
}

impl ApiError {
    fn bad_request(message: impl Into<String>) -> Self {
        Self {
            status: axum::http::StatusCode::BAD_REQUEST,
            message: message.into(),
        }
    }

    fn not_found(message: impl Into<String>) -> Self {
        Self {
            status: axum::http::StatusCode::NOT_FOUND,
            message: message.into(),
        }
    }

    fn runtime(error: RuntimeError) -> Self {
        error!(error = %error, "runtime error");
        Self {
            status: axum::http::StatusCode::INTERNAL_SERVER_ERROR,
            message: error.to_string(),
        }
    }

    fn internal_message(message: impl Into<String>) -> Self {
        Self {
            status: axum::http::StatusCode::INTERNAL_SERVER_ERROR,
            message: message.into(),
        }
    }
}

impl IntoResponse for ApiError {
    fn into_response(self) -> axum::response::Response {
        let body = Json(serde_json::json!({
            "error": self.message
        }));
        (self.status, body).into_response()
    }
}

// ─── Task Queue Handlers (Plan 07 — CLAIM op backend) ───────────────────────

#[derive(Debug, Deserialize)]
struct CreateTaskRequest {
    queue: String,
    data: JsonValue,
    #[serde(default)]
    id: Option<String>,
}

#[derive(Debug, Deserialize)]
struct ClaimTaskRequest {
    agent_id: String,
    #[serde(default = "default_lease_ms")]
    lease_ms: u64,
    /// Maximum milliseconds to long-poll for a task. 0 = return immediately.
    #[serde(default)]
    max_wait_ms: u64,
}

fn default_lease_ms() -> u64 {
    60_000
}

#[derive(Debug, Deserialize)]
struct CompleteTaskRequest {
    claim_token: String,
    #[serde(default)]
    result: JsonValue,
    #[serde(default)]
    success: bool,
}

async fn create_task(
    State(state): State<AppState>,
    Json(req): Json<CreateTaskRequest>,
) -> Json<JsonValue> {
    let id = req.id.unwrap_or_else(|| uuid::Uuid::new_v4().to_string());
    let task = QueuedTask {
        id: id.clone(),
        queue: req.queue.clone(),
        data: req.data,
        status: TaskStatus::Pending,
        claimed_by: None,
        claim_token: None,
        lease_expires_ms: None,
        result: None,
        created_at_ms: now_ms(),
        completed_at_ms: None,
    };
    state.task_manager.enqueue(task).await;
    info!(id = %id, queue = %req.queue, "Task enqueued");
    Json(serde_json::json!({ "ok": true, "id": id, "queue": req.queue }))
}

async fn list_tasks(State(state): State<AppState>, Path(queue): Path<String>) -> Json<JsonValue> {
    let tasks = state.task_manager.list_queue(&queue);
    Json(serde_json::json!({
        "queue": queue,
        "count": tasks.len(),
        "tasks": tasks
    }))
}

async fn claim_task(
    State(state): State<AppState>,
    Path(queue): Path<String>,
    Json(req): Json<ClaimTaskRequest>,
) -> Result<Json<JsonValue>, ApiError> {
    let deadline_ms = now_ms() + req.max_wait_ms;
    let mut task = state
        .task_manager
        .claim(&queue, &req.agent_id, req.lease_ms)
        .await;
    if task.is_none() && req.max_wait_ms > 0 {
        let mut wait_ms = 100u64;
        loop {
            let now = now_ms();
            if now >= deadline_ms {
                break;
            }
            let sleep_ms = wait_ms.min(deadline_ms - now);
            tokio::time::sleep(std::time::Duration::from_millis(sleep_ms)).await;
            task = state
                .task_manager
                .claim(&queue, &req.agent_id, req.lease_ms)
                .await;
            if task.is_some() {
                break;
            }
            wait_ms = (wait_ms * 2).min(1000);
        }
    }
    match task {
        Some(t) => {
            info!(id = %t.id, queue = %queue, agent_id = %req.agent_id, "Task claimed");
            Ok(Json(serde_json::json!({
                "task_id": t.id,
                "queue": t.queue,
                "data": t.data,
                "claim_token": t.claim_token,
                "expires_at_ms": t.lease_expires_ms
            })))
        }
        None => Err(ApiError {
            status: axum::http::StatusCode::NOT_FOUND,
            message: format!("No pending tasks in queue '{}'", queue),
        }),
    }
}

async fn complete_task(
    State(state): State<AppState>,
    Path(id): Path<String>,
    Json(req): Json<CompleteTaskRequest>,
) -> Result<Json<JsonValue>, ApiError> {
    state
        .task_manager
        .complete(&id, &req.claim_token, req.result)
        .await
        .map_err(|e| {
            let status = if e.starts_with("lease_expired:") {
                axum::http::StatusCode::CONFLICT
            } else {
                axum::http::StatusCode::BAD_REQUEST
            };
            ApiError { status, message: e }
        })?;
    info!(id = %id, success = %req.success, "Task completed");
    Ok(Json(serde_json::json!({ "ok": true, "id": id })))
}

// ─── Checkpoint Handlers (Plan 07 — PAUSE/RESUME HITL) ───────────────────────

#[derive(Debug, Deserialize)]
struct CreateCheckpointRequest {
    checkpoint_id: String,
    message: String,
    #[serde(default)]
    display_data: JsonValue,
    /// Optional webhook URL — server POSTs a notification when checkpoint is created.
    #[serde(default)]
    notification_url: Option<String>,
}

#[derive(Debug, Deserialize)]
struct ResumeCheckpointRequest {
    human_input: JsonValue,
}

async fn create_checkpoint(
    State(state): State<AppState>,
    Json(req): Json<CreateCheckpointRequest>,
) -> Json<JsonValue> {
    let id = req.checkpoint_id.clone();
    let notification_url = req.notification_url.clone();
    let checkpoint = Checkpoint {
        id: id.clone(),
        message: req.message.clone(),
        display_data: req.display_data,
        status: CheckpointStatus::Pending,
        human_input: None,
        notification_url: notification_url.clone(),
        created_at_ms: now_ms(),
        resumed_at_ms: None,
    };
    state.checkpoint_store.create(checkpoint);
    info!(id = %id, "Checkpoint created — awaiting human input");

    // Fire-and-forget webhook notification
    if let Some(notify_url) = notification_url {
        let payload = serde_json::json!({
            "type": "checkpoint_created",
            "checkpoint_id": id,
            "message": req.message,
            "review_url": format!("/v1/checkpoints/{}", id)
        });
        tokio::spawn(async move {
            if let Ok(client) = reqwest::Client::builder()
                .timeout(std::time::Duration::from_secs(5))
                .build()
            {
                let _ = client.post(&notify_url).json(&payload).send().await;
            }
        });
    }

    Json(serde_json::json!({
        "ok": true,
        "checkpoint_id": id,
        "status": "pending",
        "resume_url": format!("/v1/checkpoints/{}/resume", id)
    }))
}

async fn get_checkpoint(
    State(state): State<AppState>,
    Path(id): Path<String>,
) -> Result<Json<JsonValue>, ApiError> {
    match state.checkpoint_store.get(&id) {
        Some(cp) => Ok(Json(serde_json::to_value(cp).unwrap_or_default())),
        None => Err(ApiError {
            status: axum::http::StatusCode::NOT_FOUND,
            message: format!("Checkpoint '{}' not found", id),
        }),
    }
}

async fn resume_checkpoint(
    State(state): State<AppState>,
    Path(id): Path<String>,
    Json(req): Json<ResumeCheckpointRequest>,
) -> Result<Json<JsonValue>, ApiError> {
    let cp = state
        .checkpoint_store
        .resume(&id, req.human_input)
        .map_err(|e| ApiError {
            status: axum::http::StatusCode::BAD_REQUEST,
            message: e,
        })?;
    info!(id = %id, "Checkpoint resumed with human input");
    Ok(Json(serde_json::json!({
        "ok": true,
        "checkpoint_id": cp.id,
        "status": "resumed",
        "human_input": cp.human_input,
        "resumed_at_ms": cp.resumed_at_ms
    })))
}

// ─── Integration Tests ────────────────────────────────────────────────────────
//
// These tests exercise the HTTP API layer without binding to a TCP port.
// They use `tower::ServiceExt::oneshot` + `axum::body::Body` to send requests
// directly to the Axum router and collect responses via `http_body_util::BodyExt`.
//
// No LLM API key is required — all LLM-touching tests are gated behind
// `#[cfg(feature = "integration")]` and use stub responses.

#[cfg(test)]
mod tests {
    use super::*;

    use axum::body::Body;
    use axum::http::{Request, StatusCode};
    use http_body_util::BodyExt;
    use tower::ServiceExt;

    // ── Test helpers ──────────────────────────────────────────────────────────

    /// Build a test AppState backed by a real (but unconfigured) Runtime.
    ///
    /// The runtime has no LLM backends registered, so any graph that calls
    /// ASK/THINK/REASON will error.  Tests that need execution should use
    /// graphs composed entirely of CONST_STR and synchronisation ops.
    async fn test_state() -> AppState {
        // Use in-memory LTM to avoid SQLite file-locking across parallel tests.
        let runtime = Arc::new(
            Runtime::new(RuntimeConfig::in_memory())
                .await
                .expect("test runtime"),
        );
        AppState {
            runtime,
            agent_registry: Arc::new(DashMap::new()),
            task_manager: TaskQueueManager::new(),
            checkpoint_store: CheckpointStore::new(),
            start_time: SystemTime::now(),
            a2a_tasks: Arc::new(DashMap::new()),
        }
    }

    /// POST a JSON body to `path` and return `(StatusCode, serde_json::Value)`.
    async fn post_json(
        app: Router,
        path: &str,
        body: serde_json::Value,
    ) -> (StatusCode, serde_json::Value) {
        let req = Request::builder()
            .method("POST")
            .uri(path)
            .header("Content-Type", "application/json")
            .body(Body::from(body.to_string()))
            .unwrap();
        let resp = app.oneshot(req).await.unwrap();
        let status = resp.status();
        let bytes = resp.into_body().collect().await.unwrap().to_bytes();
        let json: serde_json::Value =
            serde_json::from_slice(&bytes).unwrap_or(serde_json::Value::Null);
        (status, json)
    }

    /// GET `path` and return `(StatusCode, serde_json::Value)`.
    async fn get_json(app: Router, path: &str) -> (StatusCode, serde_json::Value) {
        let req = Request::builder()
            .method("GET")
            .uri(path)
            .body(Body::empty())
            .unwrap();
        let resp = app.oneshot(req).await.unwrap();
        let status = resp.status();
        let bytes = resp.into_body().collect().await.unwrap().to_bytes();
        let json: serde_json::Value =
            serde_json::from_slice(&bytes).unwrap_or(serde_json::Value::Null);
        (status, json)
    }

    // ── Health ────────────────────────────────────────────────────────────────

    #[tokio::test]
    async fn health_returns_ok() {
        let app = build_app(test_state().await);
        let (status, body) = get_json(app, "/health").await;
        assert_eq!(status, StatusCode::OK);
        assert_eq!(body["status"], "ok");
        assert!(body["version"].is_string());
    }

    // ── /v1/models ────────────────────────────────────────────────────────────

    #[tokio::test]
    async fn models_returns_json_array() {
        let app = build_app(test_state().await);
        let (status, body) = get_json(app, "/v1/models").await;
        assert_eq!(status, StatusCode::OK);
        // With no backends configured the array may be empty, but must be an array.
        assert!(
            body["data"].is_array() || body["models"].is_array(),
            "expected 'data' or 'models' array, got: {body}"
        );
    }

    // ── Agent Registry ────────────────────────────────────────────────────────

    #[tokio::test]
    async fn agent_registry_register_returns_ok() {
        let app = build_app(test_state().await);

        let (status, body) = post_json(
            app,
            "/v1/agents/register",
            serde_json::json!({
                "name": "test-agent",
                "url": "http://localhost:19999",
                "flows": ["research"],
                "capabilities": ["web-search"]
            }),
        )
        .await;
        assert_eq!(status, StatusCode::OK, "register failed: {body}");
        assert_eq!(body["ok"], true);
        assert_eq!(body["name"], "test-agent");
    }

    #[tokio::test]
    async fn agent_registry_list_returns_array() {
        let app = build_app(test_state().await);
        let (status, body) = get_json(app, "/v1/agents").await;
        assert_eq!(status, StatusCode::OK);
        assert!(body.is_array(), "expected array: {body}");
    }

    #[tokio::test]
    async fn agent_registry_missing_name_returns_400() {
        let app = build_app(test_state().await);
        let (status, body) = post_json(
            app,
            "/v1/agents/register",
            serde_json::json!({ "url": "http://localhost:19999" }),
        )
        .await;
        // Missing `name` field → deserialization error → 400 or 422
        assert!(
            status == StatusCode::BAD_REQUEST || status == StatusCode::UNPROCESSABLE_ENTITY,
            "expected 400/422, got {status}: {body}"
        );
    }

    // ── /.well-known/agent.json (A2A Agent Card) ──────────────────────────────

    #[tokio::test]
    async fn a2a_agent_card_has_required_fields() {
        let app = build_app(test_state().await);
        let (status, body) = get_json(app, "/.well-known/agent.json").await;
        assert_eq!(status, StatusCode::OK, "agent card failed: {body}");
        assert!(body["name"].is_string(), "missing 'name': {body}");
        assert!(body["url"].is_string(), "missing 'url': {body}");
        assert!(body["version"].is_string(), "missing 'version': {body}");
        assert!(
            body["capabilities"].is_object(),
            "missing 'capabilities': {body}"
        );
    }

    // ── /v1/mcp (MCP JSON-RPC) ────────────────────────────────────────────────

    #[tokio::test]
    async fn mcp_initialize_returns_protocol_version() {
        let app = build_app(test_state().await);
        let (status, body) = post_json(
            app,
            "/v1/mcp",
            serde_json::json!({
                "jsonrpc": "2.0",
                "id": 1,
                "method": "initialize",
                "params": { "protocolVersion": "2024-11-05" }
            }),
        )
        .await;
        assert_eq!(status, StatusCode::OK, "mcp initialize failed: {body}");
        assert_eq!(body["jsonrpc"], "2.0");
        assert_eq!(body["id"], 1);
        assert!(body["result"].is_object(), "expected result object: {body}");
        let version = body["result"]["protocolVersion"].as_str().unwrap_or("");
        assert!(!version.is_empty(), "protocolVersion missing: {body}");
    }

    #[tokio::test]
    async fn mcp_tools_list_returns_array() {
        let app = build_app(test_state().await);
        let (status, body) = post_json(
            app,
            "/v1/mcp",
            serde_json::json!({
                "jsonrpc": "2.0",
                "id": 2,
                "method": "tools/list",
                "params": {}
            }),
        )
        .await;
        assert_eq!(status, StatusCode::OK);
        let tools = &body["result"]["tools"];
        assert!(tools.is_array(), "expected tools array: {body}");
    }

    #[tokio::test]
    async fn mcp_unknown_method_returns_error_code() {
        let app = build_app(test_state().await);
        let (status, body) = post_json(
            app,
            "/v1/mcp",
            serde_json::json!({
                "jsonrpc": "2.0",
                "id": 99,
                "method": "totally/unknown",
                "params": {}
            }),
        )
        .await;
        assert_eq!(
            status,
            StatusCode::OK,
            "should always return 200 JSON-RPC: {body}"
        );
        assert!(body["error"].is_object(), "expected error object: {body}");
        let code = body["error"]["code"].as_i64().unwrap_or(0);
        assert_eq!(code, -32601, "expected method-not-found code: {body}");
    }

    // ── /v1/execute (graph execution) ─────────────────────────────────────────

    #[tokio::test]
    async fn execute_invalid_graph_returns_400() {
        let app = build_app(test_state().await);
        let (status, body) = post_json(
            app,
            "/v1/execute",
            serde_json::json!({
                "graph": { "not": "a valid graph" }
            }),
        )
        .await;
        // Invalid graph → compiler rejects → 400 Bad Request
        assert_eq!(status, StatusCode::BAD_REQUEST, "expected 400: {body}");
        assert!(body["error"].is_string(), "expected error message: {body}");
    }

    #[tokio::test]
    async fn execute_empty_graph_returns_error() {
        let app = build_app(test_state().await);
        let (status, _body) = post_json(app, "/v1/execute", serde_json::json!({})).await;
        // Missing graph field → 400 or 422
        assert!(
            status == StatusCode::BAD_REQUEST || status == StatusCode::UNPROCESSABLE_ENTITY,
            "expected 400/422 for empty request, got {status}"
        );
    }

    // ── /v1/memory (LTM facts) ────────────────────────────────────────────────

    #[tokio::test]
    async fn memory_store_and_search_roundtrip() {
        let state = test_state().await;
        let app = build_app(state);

        // Store a fact
        let (store_status, store_body) = post_json(
            app.clone(),
            "/v1/memory/facts/store",
            serde_json::json!({
                "text": "RDNA 4 uses a unified compute architecture",
                "tags": ["gpu", "rdna4"],
                "source": "test"
            }),
        )
        .await;
        assert_eq!(store_status, StatusCode::OK, "store failed: {store_body}");
        assert!(
            store_body["id"].is_string(),
            "expected fact id: {store_body}"
        );
    }

    #[tokio::test]
    async fn memory_search_returns_array() {
        let app = build_app(test_state().await);
        let (status, body) = post_json(
            app,
            "/v1/memory/facts/search",
            serde_json::json!({ "query": "GPU architecture", "limit": 5 }),
        )
        .await;
        assert_eq!(status, StatusCode::OK, "search failed: {body}");
        // May return empty array if nothing stored, but must be an array
        assert!(body.is_array(), "expected array response: {body}");
    }

    // ── /a2a (A2A JSON-RPC) ───────────────────────────────────────────────────

    #[tokio::test]
    async fn a2a_jsonrpc_unknown_method_returns_error() {
        let app = build_app(test_state().await);
        let (status, body) = post_json(
            app,
            "/a2a",
            serde_json::json!({
                "jsonrpc": "2.0",
                "id": "t1",
                "method": "tasks/reopen",
                "params": {}
            }),
        )
        .await;
        assert_eq!(status, StatusCode::OK, "a2a should return 200: {body}");
        // Unrecognised method should produce an error response
        assert!(body["error"].is_object(), "expected error: {body}");
    }

    // ── 404 for unknown routes ────────────────────────────────────────────────

    #[tokio::test]
    async fn unknown_route_returns_404() {
        let app = build_app(test_state().await);
        let req = Request::builder()
            .method("GET")
            .uri("/does/not/exist")
            .body(Body::empty())
            .unwrap();
        let resp = app.oneshot(req).await.unwrap();
        assert_eq!(resp.status(), StatusCode::NOT_FOUND);
    }

    // ── /v1/tasks (task queue) ────────────────────────────────────────────────

    #[tokio::test]
    async fn task_queue_create_returns_id() {
        let app = build_app(test_state().await);
        let (status, body) = post_json(
            app,
            "/v1/tasks",
            serde_json::json!({
                "queue": "test-queue",
                "data": { "work": "process this" }
            }),
        )
        .await;
        assert_eq!(status, StatusCode::OK, "create task failed: {body}");
        assert_eq!(body["ok"], true);
        assert!(body["id"].is_string(), "expected task id: {body}");
        assert_eq!(body["queue"], "test-queue");
    }

    #[tokio::test]
    async fn task_queue_list_returns_tasks() {
        let state = test_state().await;
        let app = build_app(state);

        // Create a task first
        post_json(
            app.clone(),
            "/v1/tasks",
            serde_json::json!({
                "queue": "list-test",
                "data": { "item": 1 }
            }),
        )
        .await;

        let (status, body) = get_json(app, "/v1/tasks/list-test").await;
        assert_eq!(status, StatusCode::OK, "list tasks failed: {body}");
        assert_eq!(body["queue"], "list-test");
        assert!(
            body["count"].as_u64().unwrap_or(0) >= 1,
            "expected count >= 1: {body}"
        );
        assert!(body["tasks"].is_array(), "expected tasks array: {body}");
    }

    #[tokio::test]
    async fn task_queue_claim_and_complete() {
        let state = test_state().await;
        let app = build_app(state);

        // 1. Create task
        let (_, create_body) = post_json(
            app.clone(),
            "/v1/tasks",
            serde_json::json!({ "queue": "work", "data": { "job": "test" } }),
        )
        .await;
        let task_id = create_body["id"].as_str().unwrap().to_string();

        // 2. Claim task
        let (claim_status, claim_body) = post_json(
            app.clone(),
            "/v1/tasks/work/claim",
            serde_json::json!({ "agent_id": "test-agent", "lease_ms": 30000 }),
        )
        .await;
        assert_eq!(claim_status, StatusCode::OK, "claim failed: {claim_body}");
        assert_eq!(claim_body["task_id"], task_id);
        let claim_token = claim_body["claim_token"].as_str().unwrap().to_string();

        // 3. Complete task
        let complete_path = format!("/v1/tasks/{}/complete", task_id);
        let (complete_status, complete_body) = post_json(
            app,
            &complete_path,
            serde_json::json!({
                "claim_token": claim_token,
                "result": { "output": "done" },
                "success": true
            }),
        )
        .await;
        assert_eq!(
            complete_status,
            StatusCode::OK,
            "complete failed: {complete_body}"
        );
        assert_eq!(complete_body["ok"], true);
    }

    #[tokio::test]
    async fn task_queue_claim_empty_queue_returns_404() {
        let app = build_app(test_state().await);
        let (status, body) = post_json(
            app,
            "/v1/tasks/empty-queue/claim",
            serde_json::json!({ "agent_id": "agent-1" }),
        )
        .await;
        assert_eq!(status, StatusCode::NOT_FOUND, "expected 404: {body}");
    }

    // ── /v1/checkpoints (PAUSE/RESUME HITL) ───────────────────────────────────

    #[tokio::test]
    async fn checkpoint_create_returns_pending() {
        let app = build_app(test_state().await);
        let (status, body) = post_json(
            app,
            "/v1/checkpoints",
            serde_json::json!({
                "checkpoint_id": "cp-test-001",
                "message": "Please review the generated plan",
                "display_data": { "plan": "step 1, step 2, step 3" }
            }),
        )
        .await;
        assert_eq!(status, StatusCode::OK, "create checkpoint failed: {body}");
        assert_eq!(body["ok"], true);
        assert_eq!(body["checkpoint_id"], "cp-test-001");
        assert_eq!(body["status"], "pending");
        assert!(
            body["resume_url"].is_string(),
            "expected resume_url: {body}"
        );
    }

    #[tokio::test]
    async fn checkpoint_get_returns_checkpoint() {
        let state = test_state().await;
        let app = build_app(state);

        // Create first
        post_json(
            app.clone(),
            "/v1/checkpoints",
            serde_json::json!({
                "checkpoint_id": "cp-get-001",
                "message": "Review needed"
            }),
        )
        .await;

        // Get it
        let (status, body) = get_json(app, "/v1/checkpoints/cp-get-001").await;
        assert_eq!(status, StatusCode::OK, "get checkpoint failed: {body}");
        assert_eq!(body["id"], "cp-get-001");
        assert_eq!(body["status"], "pending");
        assert!(body["message"].is_string());
    }

    #[tokio::test]
    async fn checkpoint_resume_workflow() {
        let state = test_state().await;
        let app = build_app(state);

        // 1. Create checkpoint
        post_json(
            app.clone(),
            "/v1/checkpoints",
            serde_json::json!({
                "checkpoint_id": "cp-resume-001",
                "message": "Human decision required"
            }),
        )
        .await;

        // 2. Resume with human input
        let (status, body) = post_json(
            app,
            "/v1/checkpoints/cp-resume-001/resume",
            serde_json::json!({ "human_input": { "decision": "approved", "notes": "LGTM" } }),
        )
        .await;
        assert_eq!(status, StatusCode::OK, "resume failed: {body}");
        assert_eq!(body["ok"], true);
        assert_eq!(body["checkpoint_id"], "cp-resume-001");
        assert_eq!(body["status"], "resumed");
        assert_eq!(body["human_input"]["decision"], "approved");
        assert!(body["resumed_at_ms"].is_number());
    }

    #[tokio::test]
    async fn checkpoint_get_missing_returns_404() {
        let app = build_app(test_state().await);
        let (status, body) = get_json(app, "/v1/checkpoints/does-not-exist").await;
        assert_eq!(status, StatusCode::NOT_FOUND, "expected 404: {body}");
    }

    #[tokio::test]
    async fn checkpoint_resume_already_resumed_returns_400() {
        let state = test_state().await;
        let app = build_app(state);

        // Create + resume once
        post_json(
            app.clone(),
            "/v1/checkpoints",
            serde_json::json!({
                "checkpoint_id": "cp-double-001",
                "message": "Once only"
            }),
        )
        .await;
        post_json(
            app.clone(),
            "/v1/checkpoints/cp-double-001/resume",
            serde_json::json!({ "human_input": { "ok": true } }),
        )
        .await;

        // Attempt to resume again
        let (status, body) = post_json(
            app,
            "/v1/checkpoints/cp-double-001/resume",
            serde_json::json!({ "human_input": { "ok": false } }),
        )
        .await;
        assert_eq!(
            status,
            StatusCode::BAD_REQUEST,
            "expected 400 on double-resume: {body}"
        );
    }
}
