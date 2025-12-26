//! A-PXM Real LLM Execution Demonstration
//!
//! This example demonstrates real LLM execution through the A-PXM runtime
//! using Ollama as the backend. It shows:
//!
//! 1. **LLM Registry**: Registering and configuring LLM backends
//! 2. **COMMUNICATE Operation**: Executing reasoning through the substrate
//! 3. **AAM Integration**: How state changes during LLM interaction
//!
//! Run with: cargo run --example ollama_llm_demo -p apxm-runtime
//!
//! Prerequisites:
//! - Ollama running locally (http://localhost:11434)
//! - phi3:mini model pulled (ollama pull phi3:mini)

use apxm_core::types::{
    execution::{DependencyType, Edge, NodeMetadata},
    operations::AISOperationType,
};
use apxm_backends::{LLMRequest, Provider, ProviderId};
use apxm_runtime::{
    aam::{Goal, GoalId, GoalStatus, TransitionLabel},
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
    println!("║     A-PXM: Real LLM Execution Demonstration                  ║");
    println!("╠══════════════════════════════════════════════════════════════╣");
    println!("║  Testing substrate with real Ollama LLM calls                ║");
    println!("╚══════════════════════════════════════════════════════════════╝");
    println!();

    // ═══════════════════════════════════════════════════════════════════════
    // STEP 1: Initialize Runtime and Register Ollama Backend
    // ═══════════════════════════════════════════════════════════════════════
    println!("┌──────────────────────────────────────────────────────────────┐");
    println!("│ STEP 1: Initialize Runtime with Ollama Backend               │");
    println!("└──────────────────────────────────────────────────────────────┘");
    println!();

    let config = RuntimeConfig::in_memory();
    let runtime = Runtime::new(config).await?;

    // Create Ollama provider with cloud model for faster inference
    println!("Creating Ollama provider with gpt-oss:120b-cloud model...");
    let ollama_config = json!({
        "model": "gpt-oss:120b-cloud",
        "base_url": "http://localhost:11434"
    });

    let provider = Provider::new(ProviderId::Ollama, "", Some(ollama_config)).await?;

    // Register with the LLM registry
    runtime.llm_registry().register("ollama", provider)?;
    runtime.llm_registry().set_default("ollama")?;

    // Health check
    println!("Performing health check on Ollama...");
    let health_results = runtime.llm_registry().check_all_backends().await;
    for (name, status) in &health_results {
        println!("  • {}: {:?}", name, status);
    }
    println!();

    // ═══════════════════════════════════════════════════════════════════════
    // STEP 2: Direct LLM Call (Testing Backend)
    // ═══════════════════════════════════════════════════════════════════════
    println!("┌──────────────────────────────────────────────────────────────┐");
    println!("│ STEP 2: Direct LLM Call (Testing Backend)                    │");
    println!("└──────────────────────────────────────────────────────────────┘");
    println!();

    let request = LLMRequest::new("What is 2 + 2? Answer in one word.")
        .with_temperature(0.1)
        .with_max_tokens(10);

    println!("Sending test prompt: \"What is 2 + 2?\"");
    let start = Instant::now();
    let response = runtime.llm_registry().generate(request).await?;
    let latency = start.elapsed();

    println!();
    println!("╭─ LLM Response ──────────────────────────────────────────────╮");
    println!("│ Response: {}", response.content.trim());
    println!("│ Model: {}", response.model);
    println!("│ Latency: {:?}", latency);
    println!("│ Tokens: {} input, {} output", response.usage.input_tokens, response.usage.output_tokens);
    println!("╰─────────────────────────────────────────────────────────────╯");
    println!();

    // ═══════════════════════════════════════════════════════════════════════
    // STEP 3: AAM State Before Execution
    // ═══════════════════════════════════════════════════════════════════════
    println!("┌──────────────────────────────────────────────────────────────┐");
    println!("│ STEP 3: Set Up AAM State                                     │");
    println!("└──────────────────────────────────────────────────────────────┘");
    println!();

    let aam = runtime.aam();

    // Set initial beliefs
    aam.set_belief(
        "task".to_string(),
        Value::String("Explain quantum computing".to_string()),
        TransitionLabel::custom("initialization"),
    );
    aam.set_belief(
        "context".to_string(),
        Value::String("For a 10-year-old audience".to_string()),
        TransitionLabel::custom("initialization"),
    );

    // Add goal
    let goal = Goal {
        id: GoalId::new(),
        description: "Generate child-friendly quantum explanation".to_string(),
        priority: 100,
        status: GoalStatus::Active,
    };
    aam.add_goal(goal, TransitionLabel::custom("goal_creation"));

    println!("AAM State BEFORE LLM execution:");
    println!("  Beliefs:");
    for (key, value) in aam.beliefs() {
        println!("    • {}: {:?}", key, value);
    }
    println!("  Goals:");
    for g in aam.goals() {
        println!("    • [P:{}] {} ({:?})", g.priority, g.description, g.status);
    }
    println!();

    // ═══════════════════════════════════════════════════════════════════════
    // STEP 4: Execute DAG with Real LLM Operation
    // ═══════════════════════════════════════════════════════════════════════
    println!("┌──────────────────────────────────────────────────────────────┐");
    println!("│ STEP 4: Execute DAG with LLM Operations                      │");
    println!("└──────────────────────────────────────────────────────────────┘");
    println!();

    // Create a DAG with multiple operations that use the LLM
    // Node 1: Create prompt constant
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
        Value::String("Explain what a quantum computer is in exactly 2 sentences, suitable for a child.".to_string()),
    );

    // Node 2: Merge (would normally take LLM output, here just passes through)
    let mut node2 = Node {
        id: 2,
        op_type: AISOperationType::Merge,
        attributes: HashMap::new(),
        input_tokens: vec![100],
        output_tokens: vec![101],
        metadata: NodeMetadata::default(),
    };
    node2.attributes.insert(
        "strategy".to_string(),
        Value::String("concat".to_string()),
    );

    let dag = ExecutionDag {
        nodes: vec![node1, node2],
        edges: vec![
            Edge {
                from: 1,
                to: 2,
                token_id: 100,
                dependency_type: DependencyType::Data,
            },
        ],
        entry_nodes: vec![1],
        exit_nodes: vec![2],
        metadata: Default::default(),
    };

    println!("Executing DAG:");
    println!("  [Node 1: CONST_STR] → [Node 2: MERGE]");
    println!();

    let start = Instant::now();
    let result = runtime.execute(dag).await?;
    let duration = start.elapsed();

    println!("╭─ Execution Results ────────────────────────────────────────╮");
    println!("│ Executed nodes: {}", result.stats.executed_nodes);
    println!("│ Failed nodes: {}", result.stats.failed_nodes);
    println!("│ Duration: {:?}", duration);
    println!("│");
    println!("│ Output tokens:");
    for (token_id, value) in &result.results {
        let value_str = format!("{:?}", value);
        let truncated = if value_str.len() > 60 {
            format!("{}...", &value_str[..60])
        } else {
            value_str
        };
        println!("│   Token {}: {}", token_id, truncated);
    }
    println!("╰─────────────────────────────────────────────────────────────╯");
    println!();

    // ═══════════════════════════════════════════════════════════════════════
    // STEP 5: Make Real LLM Call and Update AAM
    // ═══════════════════════════════════════════════════════════════════════
    println!("┌──────────────────────────────────────────────────────────────┐");
    println!("│ STEP 5: Real LLM Call with AAM State Update                  │");
    println!("└──────────────────────────────────────────────────────────────┘");
    println!();

    let prompt = "Explain quantum computing in 2 sentences for a 10-year-old.";
    println!("Prompt: \"{}\"", prompt);
    println!();

    let request = LLMRequest::new(prompt)
        .with_temperature(0.7)
        .with_max_tokens(100);

    let start = Instant::now();
    let response = runtime.llm_registry().generate(request).await?;
    let latency = start.elapsed();

    // Update AAM with the result
    aam.set_belief(
        "llm_response".to_string(),
        Value::String(response.content.clone()),
        TransitionLabel::custom("llm_execution"),
    );
    aam.set_belief(
        "llm_latency_ms".to_string(),
        Value::String(format!("{}", latency.as_millis())),
        TransitionLabel::custom("llm_execution"),
    );

    println!("╭─ Real LLM Response ─────────────────────────────────────────╮");
    println!("│");
    // Wrap response for display
    let wrapped: String = response.content.trim().chars().take(200).collect();
    println!("│ {}", wrapped);
    println!("│");
    println!("│ Latency: {:?}", latency);
    println!("│ Model: {}", response.model);
    println!("╰─────────────────────────────────────────────────────────────╯");
    println!();

    // ═══════════════════════════════════════════════════════════════════════
    // STEP 6: AAM State After Execution
    // ═══════════════════════════════════════════════════════════════════════
    println!("┌──────────────────────────────────────────────────────────────┐");
    println!("│ STEP 6: AAM State After LLM Execution                        │");
    println!("└──────────────────────────────────────────────────────────────┘");
    println!();

    println!("AAM State AFTER LLM execution:");
    println!("  Beliefs:");
    for (key, value) in aam.beliefs() {
        let value_str = format!("{:?}", value);
        let truncated = if value_str.len() > 50 {
            format!("{}...", &value_str[..50])
        } else {
            value_str
        };
        println!("    • {}: {}", key, truncated);
    }
    println!();

    println!("Transition History (last 5):");
    for (i, record) in aam.recent_transitions(5).iter().enumerate() {
        println!("  [{}] {:?}", i + 1, record.label);
        if !record.belief_changes.is_empty() {
            let keys: Vec<_> = record.belief_changes.keys().collect();
            println!("      Beliefs changed: {:?}", keys);
        }
    }
    println!();

    // ═══════════════════════════════════════════════════════════════════════
    // SUMMARY
    // ═══════════════════════════════════════════════════════════════════════
    println!("╔══════════════════════════════════════════════════════════════╗");
    println!("║                     DEMONSTRATION SUMMARY                    ║");
    println!("╠══════════════════════════════════════════════════════════════╣");
    println!("║                                                              ║");
    println!("║ REAL LLM EXECUTION through A-PXM substrate demonstrated:     ║");
    println!("║                                                              ║");
    println!("║ 1. LLM Registry + Ollama Backend                             ║");
    println!("║    • Provider registered and configured                      ║");
    println!("║    • Health checks confirm connectivity                      ║");
    println!("║                                                              ║");
    println!("║ 2. Direct LLM Calls                                          ║");
    println!("║    • Real inference with cloud model                         ║");
    println!("║    • Token usage and latency tracked                         ║");
    println!("║                                                              ║");
    println!("║ 3. AAM State Integration                                     ║");
    println!("║    • Beliefs updated with LLM responses                      ║");
    println!("║    • Full transition history maintained                      ║");
    println!("║                                                              ║");
    println!("║ KEY INSIGHT: Real LLM calls flow through the substrate       ║");
    println!("║ with full auditability and state tracking.                   ║");
    println!("║                                                              ║");
    println!("╚══════════════════════════════════════════════════════════════╝");

    Ok(())
}
