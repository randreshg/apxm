//! Three-tier memory system example.
//!
//! Demonstrates the A-PXM memory model:
//! - STM (Short-Term Memory): Fast working memory for current task
//! - LTM (Long-Term Memory): Persistent knowledge across sessions
//! - Episodic: Execution traces for auditability
//!
//! Run: `cargo run --example memory_tiers -p apxm-runtime`

use apxm_runtime::{Runtime, RuntimeConfig, Value, memory::MemorySpace};

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let runtime = Runtime::new(RuntimeConfig::in_memory()).await?;
    let memory = runtime.memory();

    // --- STM: Short-Term Memory ---
    println!("STM (Short-Term Memory):");
    memory
        .write(
            MemorySpace::Stm,
            "current_task".to_string(),
            Value::String("Process user request".to_string()),
        )
        .await?;

    let task = memory.read(MemorySpace::Stm, "current_task").await?;
    println!("  current_task: {:?}", task);

    // --- LTM: Long-Term Memory ---
    println!("\nLTM (Long-Term Memory):");
    memory
        .write(
            MemorySpace::Ltm,
            "user_preference".to_string(),
            Value::String("dark_mode".to_string()),
        )
        .await?;

    let pref = memory.read(MemorySpace::Ltm, "user_preference").await?;
    println!("  user_preference: {:?}", pref);

    // --- Episodic Memory ---
    println!("\nEpisodic Memory:");
    memory
        .record_episode(
            "user_login".to_string(),
            Value::String("User logged in at 10:30".to_string()),
            "exec_001".to_string(),
        )
        .await?;

    let events = memory.query_episodes("exec_001").await?;
    println!("  execution trace: {} events", events.len());

    // Access individual tiers directly
    println!("\nDirect tier access:");
    println!("  STM keys: {:?}", memory.stm().list_keys().await?);
    println!("  LTM keys: {:?}", memory.ltm().list_keys().await?);

    Ok(())
}
