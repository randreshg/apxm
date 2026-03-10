//! Basic LLM integration example.
//!
//! Demonstrates connecting to Ollama and executing an LLM request.
//!
//! Prerequisites:
//! - Ollama running locally: `ollama serve`
//! - Model available: `ollama pull phi3:mini`
//!
//! Run: `cargo run --example simple_llm -p apxm-runtime`

use apxm_backends::{LLMRequest, Provider, ProviderId};
use apxm_runtime::{Runtime, RuntimeConfig};
use serde_json::json;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    // Initialize runtime
    let runtime = Runtime::new(RuntimeConfig::in_memory()).await?;

    // Configure Ollama backend
    let config = json!({
        "model": "phi3:mini",
        "base_url": "http://localhost:11434"
    });

    let provider = Provider::new(ProviderId::Ollama, "", Some(config)).await?;
    runtime.llm_registry().register("ollama", provider)?;
    runtime.llm_registry().set_default("ollama")?;

    // Health check
    println!("Checking LLM backend...");
    let health = runtime.llm_registry().check_all_backends().await;
    for (name, status) in &health {
        println!("  {}: {:?}", name, status);
    }

    // Execute a simple request
    println!("\nSending request to LLM...");
    let request = LLMRequest::new("What is 2 + 2? Answer in one word.");
    let response = runtime.llm_registry().generate(request).await?;

    println!("Response: {}", response.content);

    Ok(())
}
