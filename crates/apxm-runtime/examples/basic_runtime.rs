//! Basic runtime usage example.
//!
//! Demonstrates core A-PXM runtime features:
//! - Agent Abstract Machine (AAM) state management
//! - DAG-based execution
//!
//! Run: `cargo run --example basic_runtime -p apxm-runtime`

use apxm_core::types::{execution::NodeMetadata, operations::AISOperationType};
use apxm_runtime::{
    ExecutionDag, Node, Runtime, RuntimeConfig, Value,
    aam::{Goal, GoalId, GoalStatus, TransitionLabel},
};
use std::collections::HashMap;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    // Initialize runtime
    let runtime = Runtime::new(RuntimeConfig::in_memory()).await?;

    // --- AAM State Management ---
    let aam = runtime.aam();

    // Set beliefs
    aam.set_belief(
        "user_name".to_string(),
        Value::String("Alice".to_string()),
        TransitionLabel::custom("init"),
    );

    // Create a goal
    let goal = Goal {
        id: GoalId::new(),
        description: "Complete task".to_string(),
        priority: 1,
        status: GoalStatus::Active,
    };
    let goal_id = goal.id.clone();
    aam.add_goal(goal, TransitionLabel::custom("goal_created"));

    // Inspect state
    println!("Beliefs: {:?}", aam.beliefs());
    println!("Goals: {:?}", aam.goals().len());

    // --- DAG Execution ---
    // Create a simple DAG: ConstStr -> output
    let node = Node {
        id: 0,
        op_type: AISOperationType::ConstStr,
        attributes: HashMap::from([(
            "value".to_string(),
            Value::String("Hello, A-PXM!".to_string()),
        )]),
        input_tokens: vec![],
        output_tokens: vec![1],
        metadata: NodeMetadata::default(),
    };

    let dag = ExecutionDag {
        nodes: vec![node],
        edges: vec![],
        entry_nodes: vec![0],
        exit_nodes: vec![0],
        metadata: Default::default(),
    };

    // Execute
    let result = runtime.execute(dag).await?;
    println!("Execution result: {:?}", result);

    // Mark goal complete
    aam.update_goal_status(
        goal_id,
        GoalStatus::Completed,
        TransitionLabel::custom("done"),
    );

    Ok(())
}
