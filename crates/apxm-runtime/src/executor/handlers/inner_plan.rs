//! Shared helpers for compiling and splicing inner plans emitted at runtime.

use super::{ExecutionContext, Result};
use apxm_core::{
    InnerPlanPayload,
    error::RuntimeError,
    types::{
        TokenId,
        execution::{ExecutionDag, Node},
    },
};
use std::collections::{HashMap, HashSet};

// Note: InnerPlanPayload is now imported from apxm_core
// This ensures consistency across the entire system

/// Options controlling how the inner plan should be spliced into the outer DAG.
#[derive(Debug, Clone, Copy)]
pub struct InnerPlanOptions {
    /// Whether the outer node's outputs should be satisfied by the inner plan outputs.
    pub bind_outer_outputs: bool,
}

impl Default for InnerPlanOptions {
    fn default() -> Self {
        Self {
            bind_outer_outputs: true,
        }
    }
}

/// Compile and splice an inner plan graph payload into the currently executing DAG.
pub async fn execute_inner_plan(
    ctx: &ExecutionContext,
    node: &Node,
    inner_plan: &InnerPlanPayload,
    options: InnerPlanOptions,
) -> Result<usize> {
    let dag = if let Some(codelet_dag) = inner_plan.codelet_dag.clone() {
        match ctx.inner_plan_linker.link_codelet_dag(codelet_dag).await {
            Ok(dag) => dag,
            Err(err) => {
                if linker_not_supported(&err) {
                    tracing::warn!(
                        execution_id = %ctx.execution_id,
                        "Inner plan linking is not available in this context"
                    );
                    return Ok(0);
                }
                return Err(err);
            }
        }
    } else {
        let trimmed = inner_plan.graph.as_deref().unwrap_or("").trim();
        if trimmed.is_empty() {
            return Err(RuntimeError::State(
                "Inner plan must include non-empty graph or codelet_dag".to_string(),
            ));
        }

        let source_name = format!("inner_plan_{}.json", ctx.execution_id);
        match ctx
            .inner_plan_linker
            .link_inner_plan(trimmed, &source_name)
            .await
        {
            Ok(dag) => dag,
            Err(err) => {
                if linker_not_supported(&err) {
                    tracing::warn!(
                        execution_id = %ctx.execution_id,
                        "Inner plan linking is not available in this context"
                    );
                    return Ok(0);
                }
                return Err(err);
            }
        }
    };

    dag.validate()?;

    let token_connections = build_token_connections(&dag, node, options);
    let node_count = dag.nodes.len();

    match ctx.dag_splicer.splice_dag(dag, token_connections).await {
        Ok(_) => Ok(node_count),
        Err(err) => {
            if splicing_not_supported(&err) {
                tracing::warn!(
                    execution_id = %ctx.execution_id,
                    "Dynamic DAG splicing is not available in this context"
                );
                Ok(0)
            } else {
                Err(err)
            }
        }
    }
}

fn linker_not_supported(err: &RuntimeError) -> bool {
    matches!(err, RuntimeError::State(msg) if msg.contains("Inner plan linking not supported"))
}

fn splicing_not_supported(err: &RuntimeError) -> bool {
    matches!(err, RuntimeError::State(msg) if msg.contains("Dynamic DAG splicing not supported"))
}

fn build_token_connections(
    dag: &ExecutionDag,
    node: &Node,
    options: InnerPlanOptions,
) -> HashMap<TokenId, TokenId> {
    let mut connections = HashMap::new();

    // Map dangling inner inputs to the outer node inputs in order of appearance.
    let produced: HashSet<TokenId> = dag
        .nodes
        .iter()
        .flat_map(|n| n.output_tokens.iter().copied())
        .collect();

    let mut seen_inputs = HashSet::new();
    let mut required_inputs = Vec::new();
    for inner_node in &dag.nodes {
        for &token in &inner_node.input_tokens {
            if !produced.contains(&token) && seen_inputs.insert(token) {
                required_inputs.push(token);
            }
        }
    }

    for (inner, outer) in required_inputs
        .into_iter()
        .zip(node.input_tokens.iter().copied())
    {
        connections.insert(inner, outer);
    }

    if options.bind_outer_outputs {
        let consumers: HashMap<TokenId, usize> = dag
            .nodes
            .iter()
            .flat_map(|n| n.input_tokens.iter().copied())
            .fold(HashMap::new(), |mut acc, tok| {
                *acc.entry(tok).or_insert(0) += 1;
                acc
            });

        let mut exit_tokens = Vec::new();
        for node in &dag.nodes {
            for &token in &node.output_tokens {
                if consumers.get(&token).copied().unwrap_or(0) == 0 {
                    exit_tokens.push(token);
                }
            }
        }

        for (inner, outer) in exit_tokens
            .into_iter()
            .zip(node.output_tokens.iter().copied())
        {
            connections.insert(inner, outer);
        }
    }

    connections
}

#[cfg(test)]
mod tests {
    use super::*;
    use apxm_core::types::operations::AISOperationType;

    #[test]
    fn map_inputs_prefers_outer_node_inputs() {
        let mut dag = ExecutionDag::new();

        let mut dangling = Node::new(1, AISOperationType::Ask);
        dangling.input_tokens = vec![5, 6];
        dangling.output_tokens = vec![7];

        let mut producer = Node::new(2, AISOperationType::ConstStr);
        producer.output_tokens = vec![6];

        dag.nodes.push(dangling);
        dag.nodes.push(producer);

        let mut outer = Node::new(3, AISOperationType::Plan);
        outer.input_tokens = vec![100, 200];

        let mapping = build_token_connections(&dag, &outer, InnerPlanOptions::default());
        assert_eq!(mapping.get(&5), Some(&100));
        assert!(!mapping.contains_key(&6));
    }

    #[test]
    fn map_outputs_aligns_with_outer_outputs() {
        let mut dag = ExecutionDag::new();

        let mut node = Node::new(1, AISOperationType::Inv);
        node.output_tokens = vec![10, 11];
        dag.nodes.push(node);

        let mut outer = Node::new(2, AISOperationType::Plan);
        outer.output_tokens = vec![300, 400];

        let mapping = build_token_connections(&dag, &outer, InnerPlanOptions::default());
        assert_eq!(mapping.get(&10), Some(&300));
        assert_eq!(mapping.get(&11), Some(&400));
    }

    #[test]
    fn disable_output_binding_leaves_exit_tokens_unmapped() {
        let mut dag = ExecutionDag::new();
        let mut node = Node::new(1, AISOperationType::Inv);
        node.output_tokens = vec![10];
        dag.nodes.push(node);

        let outer = Node::new(2, AISOperationType::Plan);
        let mapping = build_token_connections(
            &dag,
            &outer,
            InnerPlanOptions {
                bind_outer_outputs: false,
            },
        );
        assert!(!mapping.contains_key(&10));
    }
}
