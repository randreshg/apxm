use crate::{ApxmGraph, GraphError, OperationType};
use apxm_core::types::{
    AISOperationType, DependencyType, Edge, Node,
    execution::{DagMetadata, ExecutionDag, FlowParameter},
};
use std::collections::HashMap;

pub fn lower_to_execution_dag(graph: &ApxmGraph) -> Result<ExecutionDag, GraphError> {
    graph.validate()?;

    let mut nodes = graph
        .nodes
        .iter()
        .map(|graph_node| {
            let mut node = Node::new(graph_node.id, map_operation(&graph_node.op));
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

    let exit_nodes = nodes
        .iter()
        .filter(|node| node.output_tokens.is_empty())
        .map(|node| node.id)
        .collect::<Vec<_>>();

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

fn map_operation(op: &OperationType) -> AISOperationType {
    match op {
        OperationType::Ask => AISOperationType::Ask,
        OperationType::Think => AISOperationType::Think,
        OperationType::Reason => AISOperationType::Reason,
        OperationType::QueryMemory => AISOperationType::QMem,
        OperationType::UpdateMemory => AISOperationType::UMem,
        OperationType::Invoke => AISOperationType::Inv,
        OperationType::Branch => AISOperationType::BranchOnValue,
        OperationType::Switch => AISOperationType::Switch,
        OperationType::WaitAll => AISOperationType::WaitAll,
        OperationType::Merge => AISOperationType::Merge,
        OperationType::Fence => AISOperationType::Fence,
        OperationType::Plan => AISOperationType::Plan,
        OperationType::Reflect => AISOperationType::Reflect,
        OperationType::Verify => AISOperationType::Verify,
        OperationType::Const => AISOperationType::ConstStr,
    }
}

fn map_dependency(dependency: &DependencyType) -> DependencyType {
    dependency.clone()
}
