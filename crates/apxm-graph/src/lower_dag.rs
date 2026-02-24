use crate::{ApxmGraph, GraphError};
use apxm_core::types::{
    DependencyType, Edge, Node,
    execution::{DagMetadata, ExecutionDag, FlowParameter},
};
use std::collections::HashMap;

pub fn lower_to_execution_dag(graph: &ApxmGraph) -> Result<ExecutionDag, GraphError> {
    graph.validate()?;

    let mut nodes = graph
        .nodes
        .iter()
        .map(|graph_node| {
            let mut node = Node::new(graph_node.id, graph_node.op);
            node.attributes = graph_node.attributes.clone();
            node
        })
        .collect::<Vec<_>>();

    let index_by_id = nodes
        .iter()
        .enumerate()
        .map(|(index, node)| (node.id, index))
        .collect::<HashMap<_, _>>();

    let mut edges = Vec::with_capacity(graph.edges.len());
    for (index, graph_edge) in graph.edges.iter().enumerate() {
        let token_id = index as u64 + 1;

        if let Some(from_idx) = index_by_id.get(&graph_edge.from) {
            nodes[*from_idx].output_tokens.push(token_id);
        }
        if let Some(to_idx) = index_by_id.get(&graph_edge.to) {
            nodes[*to_idx].input_tokens.push(token_id);
        }

        edges.push(Edge::new(
            graph_edge.from,
            graph_edge.to,
            token_id,
            map_dependency(&graph_edge.dependency),
        ));
    }

    let entry_nodes = nodes
        .iter()
        .filter(|node| node.input_tokens.is_empty())
        .map(|node| node.id)
        .collect::<Vec<_>>();

    // Assign synthetic output tokens to exit nodes (nodes with no outgoing edges)
    // so the scheduler can capture their results via the standard token mechanism.
    let next_token_id = edges.len() as u64 + 1;
    let mut synthetic_token = next_token_id;

    let exit_nodes: Vec<u64> = nodes
        .iter_mut()
        .filter(|node| node.output_tokens.is_empty())
        .map(|node| {
            let token_id = synthetic_token;
            synthetic_token += 1;
            node.output_tokens.push(token_id);
            node.id
        })
        .collect();

    let is_entry = graph
        .metadata
        .get("is_entry")
        .and_then(|value| value.as_boolean())
        .unwrap_or(true);

    Ok(ExecutionDag {
        nodes,
        edges,
        entry_nodes,
        exit_nodes,
        metadata: DagMetadata {
            name: Some(graph.name.clone()),
            is_entry,
            parameters: graph
                .parameters
                .iter()
                .map(|parameter| FlowParameter {
                    name: parameter.name.clone(),
                    type_name: parameter.type_name.clone(),
                })
                .collect(),
        },
    })
}

fn map_dependency(dependency: &DependencyType) -> DependencyType {
    dependency.clone()
}
