use crate::{ApxmGraph, GraphError};
use std::collections::{HashMap, HashSet, VecDeque};

pub fn validate_graph(graph: &ApxmGraph) -> Result<(), GraphError> {
    if graph.name.trim().is_empty() {
        return Err(GraphError::Validation(
            "graph name cannot be empty".to_string(),
        ));
    }

    if graph.nodes.is_empty() {
        return Err(GraphError::Validation(
            "graph must contain at least one node".to_string(),
        ));
    }

    let mut node_ids = HashSet::with_capacity(graph.nodes.len());
    for node in &graph.nodes {
        if !node_ids.insert(node.id) {
            return Err(GraphError::Validation(format!(
                "duplicate node id detected: {}",
                node.id
            )));
        }

        if node.name.trim().is_empty() {
            return Err(GraphError::Validation(format!(
                "node {} has empty name",
                node.id
            )));
        }
    }

    for edge in &graph.edges {
        if !node_ids.contains(&edge.from) {
            return Err(GraphError::Validation(format!(
                "edge references unknown source node {}",
                edge.from
            )));
        }
        if !node_ids.contains(&edge.to) {
            return Err(GraphError::Validation(format!(
                "edge references unknown destination node {}",
                edge.to
            )));
        }
    }

    validate_acyclic(graph)?;
    validate_parameters(graph)?;

    Ok(())
}

fn validate_acyclic(graph: &ApxmGraph) -> Result<(), GraphError> {
    let mut in_degree: HashMap<u64, usize> = graph.nodes.iter().map(|node| (node.id, 0)).collect();
    let mut adjacency: HashMap<u64, Vec<u64>> = HashMap::new();

    for edge in &graph.edges {
        adjacency.entry(edge.from).or_default().push(edge.to);
        *in_degree.entry(edge.to).or_insert(0) += 1;
    }

    let mut queue: VecDeque<u64> = in_degree
        .iter()
        .filter_map(|(id, degree)| if *degree == 0 { Some(*id) } else { None })
        .collect();
    let mut visited = 0usize;

    while let Some(node_id) = queue.pop_front() {
        visited += 1;
        if let Some(neighbors) = adjacency.get(&node_id) {
            for neighbor in neighbors {
                if let Some(current) = in_degree.get_mut(neighbor) {
                    *current = current.saturating_sub(1);
                    if *current == 0 {
                        queue.push_back(*neighbor);
                    }
                }
            }
        }
    }

    if visited != graph.nodes.len() {
        return Err(GraphError::Validation(
            "graph must be acyclic (DAG validation failed)".to_string(),
        ));
    }

    Ok(())
}

fn validate_parameters(graph: &ApxmGraph) -> Result<(), GraphError> {
    let mut names = HashSet::new();
    for parameter in &graph.parameters {
        if parameter.name.trim().is_empty() {
            return Err(GraphError::Validation(
                "parameter name cannot be empty".to_string(),
            ));
        }
        if parameter.type_name.trim().is_empty() {
            return Err(GraphError::Validation(format!(
                "parameter '{}' has empty type",
                parameter.name
            )));
        }
        if !names.insert(parameter.name.clone()) {
            return Err(GraphError::Validation(format!(
                "duplicate parameter name '{}'",
                parameter.name
            )));
        }
    }
    Ok(())
}
