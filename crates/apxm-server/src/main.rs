use std::collections::HashMap;
use std::net::SocketAddr;
use std::sync::Arc;

use apxm_artifact::{Artifact, ArtifactMetadata};
use apxm_core::error::RuntimeError;
use apxm_core::types::values::Value;
use apxm_graph::{ApxmGraph, OperationType};
use apxm_runtime::capability::executor::{CapabilityExecutor, CapabilityResult};
use apxm_runtime::capability::metadata::CapabilityMetadata;
use apxm_runtime::executor::ExecutionEventEmitter;
use apxm_runtime::{Runtime, RuntimeConfig};
use async_trait::async_trait;
use axum::extract::State;
use axum::response::sse::{Event, KeepAlive, Sse};
use axum::{Json, Router, response::IntoResponse, routing::get, routing::post};
use futures::Stream;
use serde::{Deserialize, Serialize};
use serde_json::Value as JsonValue;
use tokio::sync::mpsc;
use tower_http::cors::CorsLayer;
use tower_http::trace::TraceLayer;
use tracing::{error, info};

#[derive(Clone)]
struct AppState {
    runtime: Arc<Runtime>,
}

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

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(
            std::env::var("RUST_LOG").unwrap_or_else(|_| "info,apxm_server=debug".to_string()),
        )
        .init();

    let runtime = Arc::new(Runtime::new(RuntimeConfig::default()).await?);
    let state = AppState { runtime };

    let app = Router::new()
        .route("/health", get(health))
        .route("/v1/execute", post(execute))
        .route("/v1/execute/stream", post(execute_stream))
        .route("/v1/memory/facts/store", post(store_fact))
        .route("/v1/memory/facts/search", post(search_facts))
        .route("/v1/memory/facts/delete", post(delete_fact))
        .route("/v1/capabilities/register", post(register_capability))
        .with_state(state)
        .layer(CorsLayer::permissive())
        .layer(TraceLayer::new_for_http());

    let addr = std::env::var("APXM_SERVER_ADDR")
        .ok()
        .and_then(|s| s.parse::<SocketAddr>().ok())
        .unwrap_or_else(|| "127.0.0.1:18800".parse().expect("valid default addr"));
    info!(%addr, "starting apxm-server");
    let listener = tokio::net::TcpListener::bind(addr).await?;
    axum::serve(listener, app).await?;
    Ok(())
}

async fn health() -> Json<JsonValue> {
    Json(serde_json::json!({
        "status": "ok"
    }))
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
        serde_json::to_value(results).map_err(ApiError::internal)?,
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

async fn register_capability(
    State(state): State<AppState>,
    Json(req): Json<RegisterCapabilityRequest>,
) -> Result<Json<JsonValue>, ApiError> {
    let response_value =
        Value::try_from(req.static_response).map_err(|e| ApiError::bad_request(e.to_string()))?;
    let metadata = CapabilityMetadata::new(
        req.name.clone(),
        req.description.clone(),
        req.parameters_schema,
    );
    let capability = StaticCapability {
        metadata,
        static_response: response_value,
    };
    state
        .runtime
        .capability_system()
        .register(Arc::new(capability))
        .map_err(ApiError::runtime)?;
    Ok(Json(serde_json::json!({ "ok": true, "name": req.name })))
}

fn prepare_request(
    mut req: ExecuteRequest,
) -> Result<(ApxmGraph, Vec<String>, Option<String>), ApiError> {
    let mut graph = ApxmGraph::from_json(
        &serde_json::to_string(&req.graph).map_err(ApiError::bad_request_json)?,
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
        if node.op != OperationType::Ask {
            continue;
        }
        if let Some(budget) = token_budget {
            let budget = i64::try_from(budget).unwrap_or(i64::MAX);
            node.attributes
                .insert("token_budget".to_string(), Value::Number(budget.into()));
        }
        if let Some(schema) = output_schema.as_ref() {
            let schema_value = Value::try_from(schema.clone())
                .map_err(|e| format!("invalid output_schema: {e}"))?;
            node.attributes
                .insert("output_schema".to_string(), schema_value);
        }
        if let Some(retries) = max_schema_retries {
            let retries = i64::from(retries);
            node.attributes.insert(
                "max_schema_retries".to_string(),
                Value::Number(retries.into()),
            );
        }
    }
    Ok(())
}

fn graph_to_artifact(graph: ApxmGraph) -> Result<Artifact, ApiError> {
    let dag = graph
        .to_execution_dag()
        .map_err(ApiError::bad_request_graph)?;
    Ok(Artifact::new(
        ArtifactMetadata::new(Some(graph.name), "apxm-server"),
        vec![dag],
    ))
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

    fn bad_request_json(error: serde_json::Error) -> Self {
        Self::bad_request(format!("invalid json: {error}"))
    }

    fn bad_request_graph(error: apxm_graph::GraphError) -> Self {
        Self::bad_request(format!("{error}"))
    }

    fn runtime(error: RuntimeError) -> Self {
        error!(error = %error, "runtime error");
        Self {
            status: axum::http::StatusCode::INTERNAL_SERVER_ERROR,
            message: error.to_string(),
        }
    }

    fn internal(error: serde_json::Error) -> Self {
        Self {
            status: axum::http::StatusCode::INTERNAL_SERVER_ERROR,
            message: error.to_string(),
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
