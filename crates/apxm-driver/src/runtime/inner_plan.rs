//! Compiler-backed implementation of the runtime linker trait.

use std::sync::Arc;

use apxm_artifact::Artifact;
use apxm_compiler::{self, Pipeline};
use apxm_core::constants::graph::{attrs as graph_attrs, metadata as graph_meta};
use apxm_core::types::OptimizationLevel;
use apxm_core::types::execution::{CodeletDag, ExecutionDag};
use apxm_core::types::{AISOperationType, DependencyType, Value};
use apxm_core::{log_debug, log_info};
use apxm_graph::{ApxmGraph, GraphEdge, GraphNode, Parameter};
use apxm_runtime::{InnerPlanLinker, RuntimeError};
use async_trait::async_trait;
use std::collections::{HashMap, HashSet};

/// Compiler-backed implementation of the runtime linker trait.
pub struct CompilerInnerPlanLinker {
    context: Arc<parking_lot::Mutex<apxm_compiler::Context>>,
}

impl CompilerInnerPlanLinker {
    pub fn new() -> Result<Self, RuntimeError> {
        let context = apxm_compiler::Context::new().map_err(|e| {
            RuntimeError::State(format!("Failed to initialize compiler context: {}", e))
        })?;

        Ok(Self {
            context: Arc::new(parking_lot::Mutex::new(context)),
        })
    }
}

#[async_trait]
impl InnerPlanLinker for CompilerInnerPlanLinker {
    async fn link_inner_plan(
        &self,
        graph_payload: &str,
        source_name: &str,
    ) -> Result<ExecutionDag, RuntimeError> {
        log_debug!(
            "driver::inner_plan",
            source = %source_name,
            payload_length = graph_payload.len(),
            "Linking inner plan graph JSON"
        );

        let graph = ApxmGraph::from_json(graph_payload).map_err(|e| {
            RuntimeError::State(format!("Inner plan graph JSON parsing failed: {}", e))
        })?;

        let context = self.context.lock();
        let pipeline = Pipeline::with_opt_level(&context, OptimizationLevel::O1);
        let module = pipeline.compile_graph(&graph).map_err(|e| {
            RuntimeError::State(format!(
                "Inner plan graph compilation failed for '{}': {}",
                source_name, e
            ))
        })?;

        let artifact_bytes = module.generate_artifact_bytes().map_err(|e| {
            RuntimeError::State(format!("Inner plan artifact generation failed: {}", e))
        })?;

        let artifact = Artifact::from_bytes(&artifact_bytes).map_err(|e| {
            RuntimeError::State(format!("Inner plan artifact parsing failed: {}", e))
        })?;

        let dag = artifact.into_dag();

        // Validate the inner DAG before returning it to the runtime.
        // This ensures we catch cycles or inconsistent token/node references
        // early and return a clear error instead of allowing the scheduler
        // to detect a deadlock at runtime.
        if let Err(e) = dag.validate() {
            return Err(RuntimeError::State(format!(
                "Inner plan DAG validation failed: {}",
                e
            )));
        }

        log_info!(
            "driver::inner_plan",
            source = %source_name,
            nodes = dag.nodes.len(),
            edges = dag.edges.len(),
            "Inner plan linked successfully"
        );

        Ok(dag)
    }

    async fn link_codelet_dag(&self, dag: CodeletDag) -> Result<ExecutionDag, RuntimeError> {
        log_debug!(
            "driver::inner_plan",
            dag_name = %dag.name,
            codelets = dag.codelets.len(),
            "Linking inner plan codelet DAG via graph canonicalization"
        );

        let graph = codelet_dag_to_graph(&dag)?;
        let context = self.context.lock();
        let pipeline = Pipeline::with_opt_level(&context, OptimizationLevel::O1);
        let module = pipeline.compile_graph(&graph).map_err(|e| {
            RuntimeError::State(format!(
                "Inner plan codelet graph compilation failed: {}",
                e
            ))
        })?;

        let artifact_bytes = module.generate_artifact_bytes().map_err(|e| {
            RuntimeError::State(format!(
                "Inner plan codelet artifact generation failed: {}",
                e
            ))
        })?;
        let artifact = Artifact::from_bytes(&artifact_bytes).map_err(|e| {
            RuntimeError::State(format!("Inner plan codelet artifact parsing failed: {}", e))
        })?;
        let execution_dag = artifact.into_dag();
        execution_dag.validate()?;

        log_info!(
            "driver::inner_plan",
            nodes = execution_dag.nodes.len(),
            edges = execution_dag.edges.len(),
            "Inner plan codelet DAG linked successfully via graph path"
        );

        Ok(execution_dag)
    }
}

fn codelet_dag_to_graph(dag: &CodeletDag) -> Result<ApxmGraph, RuntimeError> {
    dag.validate()?;

    let mut nodes = Vec::new();
    let mut edges = Vec::new();
    let mut seen_node_ids = HashSet::new();
    let mut first_node_by_codelet = HashMap::new();
    let mut last_node_by_codelet = HashMap::new();

    for codelet in &dag.codelets {
        let realized_nodes: Vec<u64> = if codelet.nodes.is_empty() {
            vec![codelet.id * 1000]
        } else {
            codelet.nodes.clone()
        };

        for (index, node_id) in realized_nodes.iter().copied().enumerate() {
            if !seen_node_ids.insert(node_id) {
                return Err(RuntimeError::State(format!(
                    "CodeletDag '{}' maps to duplicate node id {}",
                    dag.name, node_id
                )));
            }

            let node_name = if realized_nodes.len() == 1 {
                codelet.name.clone()
            } else {
                format!("{}_{index}", codelet.name)
            };

            let mut attributes = HashMap::new();
            attributes.insert(
                graph_attrs::TEMPLATE_STR.to_string(),
                Value::String(codelet.description.clone()),
            );

            nodes.push(GraphNode {
                id: node_id,
                name: node_name,
                op: AISOperationType::Ask,
                attributes,
            });
        }

        let first = *realized_nodes.first().expect("realized_nodes is non-empty");
        let last = *realized_nodes.last().expect("realized_nodes is non-empty");
        first_node_by_codelet.insert(codelet.id, first);
        last_node_by_codelet.insert(codelet.id, last);

        for pair in realized_nodes.windows(2) {
            edges.push(GraphEdge {
                from: pair[0],
                to: pair[1],
                dependency: DependencyType::Data,
            });
        }
    }

    for codelet in &dag.codelets {
        let to = *first_node_by_codelet.get(&codelet.id).ok_or_else(|| {
            RuntimeError::State(format!(
                "Missing first node mapping for codelet {}",
                codelet.id
            ))
        })?;
        for dep in &codelet.depends_on {
            let from = *last_node_by_codelet.get(dep).ok_or_else(|| {
                RuntimeError::State(format!("Missing dependency mapping for codelet {}", dep))
            })?;
            edges.push(GraphEdge {
                from,
                to,
                dependency: DependencyType::Data,
            });
        }
    }

    let mut metadata = HashMap::new();
    if dag.metadata.is_entry {
        metadata.insert(graph_meta::IS_ENTRY.to_string(), Value::Bool(true));
    }

    let graph = ApxmGraph {
        name: dag
            .metadata
            .name
            .clone()
            .unwrap_or_else(|| dag.name.clone()),
        nodes,
        edges,
        parameters: dag
            .metadata
            .parameters
            .iter()
            .map(|p| Parameter {
                name: p.name.clone(),
                type_name: p.type_name.clone(),
            })
            .collect(),
        metadata,
    };

    graph
        .validate()
        .map_err(|e| RuntimeError::State(format!("CodeletDag graph validation failed: {}", e)))?;

    Ok(graph)
}
