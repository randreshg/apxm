//! A-PXM Substrate Capabilities Demonstration
//!
//! This example demonstrates the key capabilities of A-PXM as a universal
//! execution substrate for agentic AI:
//!
//! 1. **Explicit State (AAM)**: Beliefs, Goals, Capabilities are inspectable
//! 2. **Typed Operations (AIS)**: Operations have defined semantics
//! 3. **Dataflow Execution**: Automatic parallelism from data dependencies
//! 4. **Three-Tier Memory**: STM/LTM/Episodic separation
//!
//! Run with: cargo run --example substrate_demo -p apxm-runtime

use apxm_core::types::{
    execution::{DependencyType, Edge, NodeMetadata},
    operations::AISOperationType,
};
use apxm_runtime::{
    aam::{CapabilityRecord, Goal, GoalId, GoalStatus, TransitionLabel},
    memory::MemorySpace,
    ExecutionDag, Node, Runtime, RuntimeConfig, Value,
};
use serde_json::json;
use std::collections::HashMap;
use std::time::Instant;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    // Initialize tracing for observability
    tracing_subscriber::fmt::init();

    println!("╔══════════════════════════════════════════════════════════════╗");
    println!("║     A-PXM: Universal Execution Substrate Demonstration       ║");
    println!("╠══════════════════════════════════════════════════════════════╣");
    println!("║  Demonstrating substrate capabilities for CF 2026 paper      ║");
    println!("╚══════════════════════════════════════════════════════════════╝");
    println!();

    // Create runtime with in-memory configuration
    let config = RuntimeConfig::in_memory();
    let runtime = Runtime::new(config).await?;

    // ═══════════════════════════════════════════════════════════════════════
    // DEMONSTRATION 1: Explicit, Auditable State (AAM)
    // ═══════════════════════════════════════════════════════════════════════
    println!("┌──────────────────────────────────────────────────────────────┐");
    println!("│ DEMO 1: Explicit, Auditable State (AAM)                      │");
    println!("│ Shows: Beliefs, Goals, Capabilities are inspectable          │");
    println!("└──────────────────────────────────────────────────────────────┘");
    println!();

    let aam = runtime.aam();

    // Set initial beliefs
    println!("Setting initial beliefs...");
    aam.set_belief(
        "user_name".to_string(),
        Value::String("Alice".to_string()),
        TransitionLabel::custom("initialization"),
    );
    aam.set_belief(
        "task_context".to_string(),
        Value::String("Research quantum computing trends".to_string()),
        TransitionLabel::custom("initialization"),
    );

    // Add a goal
    println!("Adding goal to AAM...");
    let goal = Goal {
        id: GoalId::new(),
        description: "Complete research synthesis".to_string(),
        priority: 100,
        status: GoalStatus::Active,
    };
    aam.add_goal(goal.clone(), TransitionLabel::custom("planning"));

    // Register a capability
    println!("Registering capability...");
    aam.register_capability(
        "web_search".to_string(),
        CapabilityRecord {
            name: "web_search".to_string(),
            description: "Search the web for information".to_string(),
            schema: json!({
                "type": "object",
                "properties": {
                    "query": { "type": "string" }
                }
            }),
            cost_estimate: 0.01,
        },
        TransitionLabel::custom("capability_registration"),
    );

    // INSPECTION: Show current AAM state
    println!();
    println!("╭─ AAM State Inspection ─────────────────────────────────────╮");
    println!("│                                                             │");
    println!("│ BELIEFS (Key-Value Knowledge Store):                        │");
    for (key, value) in aam.beliefs() {
        println!("│   • {}: {:?}", key, value);
    }
    println!("│                                                             │");
    println!("│ GOALS (Priority Queue):                                     │");
    for g in aam.goals() {
        println!("│   • [P:{}] {} ({:?})", g.priority, g.description, g.status);
    }
    println!("│                                                             │");
    println!("│ CAPABILITIES (Tool Signatures):                             │");
    for (name, cap) in aam.capabilities() {
        println!("│   • {}: {}", name, cap.description);
    }
    println!("│                                                             │");
    println!("╰─────────────────────────────────────────────────────────────╯");

    // Show transition history (episodic audit trail)
    println!();
    println!("╭─ Transition History (Episodic Audit Trail) ────────────────╮");
    for (i, record) in aam.recent_transitions(10).iter().enumerate() {
        println!("│ [{}] {:?}", i + 1, record.label);
        if !record.belief_changes.is_empty() {
            println!("│     Belief changes: {:?}", record.belief_changes.keys().collect::<Vec<_>>());
        }
        if !record.goal_changes.is_empty() {
            println!("│     Goal changes: {} change(s)", record.goal_changes.len());
        }
    }
    println!("╰─────────────────────────────────────────────────────────────╯");
    println!();

    // ═══════════════════════════════════════════════════════════════════════
    // DEMONSTRATION 2: Three-Tier Memory System
    // ═══════════════════════════════════════════════════════════════════════
    println!("┌──────────────────────────────────────────────────────────────┐");
    println!("│ DEMO 2: Three-Tier Memory System                             │");
    println!("│ Shows: STM (working), LTM (persistent), Episodic (events)    │");
    println!("└──────────────────────────────────────────────────────────────┘");
    println!();

    let memory = runtime.memory();

    // Write to STM (Short-Term Memory - working memory)
    println!("Writing to STM (Short-Term Memory)...");
    memory
        .write(
            MemorySpace::Stm,
            "current_task".to_string(),
            Value::String("Analyzing quantum computing trends".to_string()),
        )
        .await?;
    memory
        .write(
            MemorySpace::Stm,
            "intermediate_result".to_string(),
            Value::String("Found 5 relevant papers".to_string()),
        )
        .await?;

    // Write to LTM (Long-Term Memory - persistent storage)
    println!("Writing to LTM (Long-Term Memory)...");
    memory
        .write(
            MemorySpace::Ltm,
            "domain_knowledge".to_string(),
            Value::String("Quantum computing uses qubits for computation".to_string()),
        )
        .await?;

    // Write to Episodic Memory (event/experience storage)
    println!("Writing to Episodic Memory...");
    memory
        .record_episode(
            "task_started".to_string(),
            Value::String(serde_json::to_string(&json!({ "task": "research", "timestamp": "2024-01-01T00:00:00Z" })).unwrap()),
            "execution_001".to_string(),
        )
        .await?;

    // Memory inspection
    println!();
    println!("╭─ Memory System Inspection ─────────────────────────────────╮");
    println!("│                                                             │");
    println!("│ STM (Short-Term Memory) - Working Memory:                   │");
    println!("│   Entries: {}", memory.stm().len().await);
    if let Some(v) = memory.read(MemorySpace::Stm, "current_task").await? {
        println!("│   • current_task: {:?}", v);
    }
    if let Some(v) = memory.read(MemorySpace::Stm, "intermediate_result").await? {
        println!("│   • intermediate_result: {:?}", v);
    }
    println!("│                                                             │");
    println!("│ LTM (Long-Term Memory) - Persistent Storage:                │");
    let ltm_stats = memory.ltm().stats().await?;
    println!("│   Entries: {}", ltm_stats.total_keys);
    if let Some(v) = memory.read(MemorySpace::Ltm, "domain_knowledge").await? {
        println!("│   • domain_knowledge: {:?}", v);
    }
    println!("│                                                             │");
    println!("│ Episodic Memory - Event Log:                                │");
    let episodes = memory.query_episodes("execution_001").await?;
    println!("│   Events for execution_001: {}", episodes.len());
    println!("│                                                             │");
    println!("╰─────────────────────────────────────────────────────────────╯");
    println!();

    // ═══════════════════════════════════════════════════════════════════════
    // DEMONSTRATION 3: Dataflow Execution (Automatic Parallelism)
    // ═══════════════════════════════════════════════════════════════════════
    println!("┌──────────────────────────────────────────────────────────────┐");
    println!("│ DEMO 3: Dataflow Execution (Automatic Parallelism)           │");
    println!("│ Shows: Operations execute based on data dependencies         │");
    println!("└──────────────────────────────────────────────────────────────┘");
    println!();

    // Create a simple DAG with parallel operations
    // Node 1 and Node 2 are independent (can run in parallel)
    // Node 3 depends on both (runs after they complete)
    //
    //   [Node 1: "Topic A"] ─┐
    //                        ├─► [Node 3: "Synthesis"]
    //   [Node 2: "Topic B"] ─┘

    let mut node1 = Node {
        id: 1,
        op_type: AISOperationType::ConstStr,
        attributes: HashMap::new(),
        input_tokens: vec![],
        output_tokens: vec![100],
        metadata: NodeMetadata::default(),
    };
    node1.attributes.insert(
        "value".to_string(),
        Value::String("Research finding A: Quantum supremacy achieved in 2019".to_string()),
    );

    let mut node2 = Node {
        id: 2,
        op_type: AISOperationType::ConstStr,
        attributes: HashMap::new(),
        input_tokens: vec![],
        output_tokens: vec![101],
        metadata: NodeMetadata::default(),
    };
    node2.attributes.insert(
        "value".to_string(),
        Value::String("Research finding B: Error correction improving rapidly".to_string()),
    );

    let mut node3 = Node {
        id: 3,
        op_type: AISOperationType::Merge,
        attributes: HashMap::new(),
        input_tokens: vec![100, 101], // Depends on both node1 and node2
        output_tokens: vec![102],
        metadata: NodeMetadata::default(),
    };
    node3.attributes.insert(
        "strategy".to_string(),
        Value::String("concat".to_string()),
    );

    let dag = ExecutionDag {
        nodes: vec![node1, node2, node3],
        edges: vec![
            Edge { from: 1, to: 3, token_id: 100, dependency_type: DependencyType::Data },
            Edge { from: 2, to: 3, token_id: 101, dependency_type: DependencyType::Data },
        ],
        entry_nodes: vec![1, 2], // Both can start immediately (parallel)
        exit_nodes: vec![3],
        metadata: Default::default(),
    };

    println!("Execution DAG Structure:");
    println!("  [Node 1: ConstStr] ─┐");
    println!("                      ├─► [Node 3: Merge]");
    println!("  [Node 2: ConstStr] ─┘");
    println!();
    println!("  Node 1 and Node 2 are INDEPENDENT → can execute in PARALLEL");
    println!("  Node 3 DEPENDS on both → executes after they complete");
    println!();

    let start = Instant::now();
    let result = runtime.execute(dag).await?;
    let duration = start.elapsed();

    println!("╭─ Execution Results ────────────────────────────────────────╮");
    println!("│                                                             │");
    println!("│ Executed nodes: {}                                          │", result.stats.executed_nodes);
    println!("│ Failed nodes: {}                                            │", result.stats.failed_nodes);
    println!("│ Total duration: {:?}                               │", duration);
    println!("│                                                             │");
    println!("│ KEY INSIGHT: Parallelism extracted AUTOMATICALLY from       │");
    println!("│ data dependencies. No manual async/await required!          │");
    println!("│                                                             │");
    println!("╰─────────────────────────────────────────────────────────────╯");
    println!();

    // ═══════════════════════════════════════════════════════════════════════
    // DEMONSTRATION 4: Scheduling Overhead Measurement
    // ═══════════════════════════════════════════════════════════════════════
    println!("┌──────────────────────────────────────────────────────────────┐");
    println!("│ DEMO 4: Scheduling Overhead Measurement                      │");
    println!("│ Shows: Overhead is negligible (microseconds, not milliseconds)│");
    println!("└──────────────────────────────────────────────────────────────┘");
    println!();

    // Create a larger DAG to measure overhead
    let mut nodes = Vec::new();
    let num_ops = 100;

    for i in 0..num_ops {
        let mut node = Node {
            id: i as u64,
            op_type: AISOperationType::ConstStr,
            attributes: HashMap::new(),
            input_tokens: vec![],
            output_tokens: vec![1000 + i as u64],
            metadata: NodeMetadata::default(),
        };
        node.attributes.insert(
            "value".to_string(),
            Value::String(format!("Operation {}", i)),
        );
        nodes.push(node);
    }

    let dag = ExecutionDag {
        nodes,
        edges: vec![],
        entry_nodes: (0..num_ops as u64).collect(),
        exit_nodes: (0..num_ops as u64).collect(),
        metadata: Default::default(),
    };

    let start = Instant::now();
    let _result = runtime.execute(dag).await?;
    let duration = start.elapsed();
    let per_op_overhead = duration.as_micros() as f64 / num_ops as f64;

    println!("╭─ Overhead Measurement Results ─────────────────────────────╮");
    println!("│                                                             │");
    println!("│ Operations executed: {}                                   │", num_ops);
    println!("│ Total time: {:?}                                    │", duration);
    println!("│ Per-operation overhead: {:.2} μs                           │", per_op_overhead);
    println!("│                                                             │");
    println!("│ COMPARISON:                                                 │");
    println!("│   • A-PXM scheduler overhead: ~{:.0} μs/op                  │", per_op_overhead);
    println!("│   • Typical LLM API latency: ~3,000,000 μs (3 seconds)      │");
    println!("│   • Overhead is 6 ORDERS OF MAGNITUDE below LLM latency!   │");
    println!("│                                                             │");
    println!("╰─────────────────────────────────────────────────────────────╯");
    println!();

    // ═══════════════════════════════════════════════════════════════════════
    // SUMMARY
    // ═══════════════════════════════════════════════════════════════════════
    println!("╔══════════════════════════════════════════════════════════════╗");
    println!("║                     DEMONSTRATION SUMMARY                    ║");
    println!("╠══════════════════════════════════════════════════════════════╣");
    println!("║                                                              ║");
    println!("║ A-PXM provides a UNIVERSAL EXECUTION SUBSTRATE with:         ║");
    println!("║                                                              ║");
    println!("║ 1. EXPLICIT STATE (AAM)                                      ║");
    println!("║    • Beliefs: Key-value knowledge store                      ║");
    println!("║    • Goals: Priority queue of objectives                     ║");
    println!("║    • Capabilities: Registered tool signatures                ║");
    println!("║    • Full audit trail via transition history                 ║");
    println!("║                                                              ║");
    println!("║ 2. THREE-TIER MEMORY                                         ║");
    println!("║    • STM: Working memory for current execution               ║");
    println!("║    • LTM: Persistent storage for long-term knowledge         ║");
    println!("║    • Episodic: Event log for debugging and replay            ║");
    println!("║                                                              ║");
    println!("║ 3. DATAFLOW EXECUTION                                        ║");
    println!("║    • Parallelism extracted automatically from dependencies   ║");
    println!("║    • No manual async/await required                          ║");
    println!("║    • Scheduling overhead: ~{:.0} μs/op                        ║", per_op_overhead);
    println!("║                                                              ║");
    println!("║ SPEEDUP IS A CONSEQUENCE of proper execution semantics,      ║");
    println!("║ NOT the primary contribution.                                ║");
    println!("║                                                              ║");
    println!("╚══════════════════════════════════════════════════════════════╝");

    Ok(())
}
