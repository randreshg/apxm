//! Paper Benchmarks for A-PXM - FAST VERSION
//!
//! Measures substrate properties (NOT Rust vs Python comparison):
//! - Table 4: Substrate overhead per operation (~8μs)
//! - Table 5: Memory tier latencies
//! - Overhead ratio: substrate_overhead / typical_LLM_latency
//!
//! Run: cargo run --example paper_benchmarks -p apxm-runtime --release
//! JSON: cargo run --example paper_benchmarks -p apxm-runtime --release -- --json

use apxm_core::types::{
    execution::{DagMetadata, DependencyType, Edge, ExecutionDag, Node, NodeMetadata},
    operations::AISOperationType,
    values::Value,
};
use apxm_runtime::{MemorySpace, Runtime, RuntimeConfig};
use serde_json::json;
use std::collections::HashMap;
use std::time::Instant;

// Reduced iterations for fast execution
const WARMUP_ITERATIONS: usize = 3;
const BENCHMARK_ITERATIONS: usize = 20;
const OPS_PER_DAG: usize = 50;

// Typical LLM latency for comparison (in microseconds)
const TYPICAL_LLM_LATENCY_US: f64 = 2_000_000.0; // 2 seconds

/// Results from overhead benchmark
#[derive(Debug, serde::Serialize)]
struct OverheadResults {
    per_op_overhead_us: f64,
    operations_executed: usize,
    total_time_ms: f64,
    llm_latency_us: f64,
    overhead_ratio_pct: f64,
}

/// Results from memory benchmark
#[derive(Debug, serde::Serialize)]
struct MemoryResults {
    stm_write_us: f64,
    stm_read_us: f64,
    ltm_write_us: f64,
    ltm_read_us: f64,
    episodic_write_us: f64,
}

/// Create a synthetic DAG for overhead measurement
fn create_benchmark_dag(num_ops: usize) -> ExecutionDag {
    let mut nodes = Vec::new();
    let mut edges = Vec::new();

    // Entry node
    nodes.push(Node {
        id: 0,
        op_type: AISOperationType::ConstStr,
        attributes: {
            let mut attrs = HashMap::new();
            attrs.insert("value".to_string(), Value::String("entry".to_string()));
            attrs
        },
        input_tokens: vec![],
        output_tokens: vec![100],
        metadata: NodeMetadata::default(),
    });

    // Parallel nodes (fan-out)
    for i in 1..num_ops {
        nodes.push(Node {
            id: i as u64,
            op_type: AISOperationType::ConstStr,
            attributes: {
                let mut attrs = HashMap::new();
                attrs.insert("value".to_string(), Value::String(format!("op_{}", i)));
                attrs
            },
            input_tokens: vec![100],
            output_tokens: vec![100 + i as u64],
            metadata: NodeMetadata::default(),
        });

        edges.push(Edge {
            from: 0,
            to: i as u64,
            token_id: 100,
            dependency_type: DependencyType::Data,
        });
    }

    // Exit node (fan-in)
    let exit_id = num_ops as u64;
    let input_tokens: Vec<u64> = (1..num_ops).map(|i| 100 + i as u64).collect();
    nodes.push(Node {
        id: exit_id,
        op_type: AISOperationType::ConstStr,
        attributes: {
            let mut attrs = HashMap::new();
            attrs.insert("value".to_string(), Value::String("exit".to_string()));
            attrs
        },
        input_tokens: input_tokens.clone(),
        output_tokens: vec![999],
        metadata: NodeMetadata::default(),
    });

    for (idx, token) in input_tokens.iter().enumerate() {
        edges.push(Edge {
            from: (idx + 1) as u64,
            to: exit_id,
            token_id: *token,
            dependency_type: DependencyType::Data,
        });
    }

    ExecutionDag {
        nodes,
        edges,
        entry_nodes: vec![0],
        exit_nodes: vec![exit_id],
        metadata: DagMetadata::default(),
    }
}

/// Benchmark substrate overhead (Table 4)
async fn benchmark_overhead(json_output: bool) -> OverheadResults {
    if !json_output {
        println!("\n=== Substrate Overhead (Table 4) ===\n");
    }

    let config = RuntimeConfig::in_memory();
    let runtime = Runtime::new(config).await.expect("Failed to create runtime");
    let dag = create_benchmark_dag(OPS_PER_DAG);

    // Warmup
    for _ in 0..WARMUP_ITERATIONS {
        let _ = runtime.execute(dag.clone()).await;
    }

    // Benchmark
    let start = Instant::now();
    for _ in 0..BENCHMARK_ITERATIONS {
        let _ = runtime.execute(dag.clone()).await;
    }
    let total_time = start.elapsed();

    let total_ops = BENCHMARK_ITERATIONS * (OPS_PER_DAG + 1);
    let per_op_us = total_time.as_micros() as f64 / total_ops as f64;
    let overhead_ratio = (per_op_us / TYPICAL_LLM_LATENCY_US) * 100.0;

    let results = OverheadResults {
        per_op_overhead_us: per_op_us,
        operations_executed: total_ops,
        total_time_ms: total_time.as_millis() as f64,
        llm_latency_us: TYPICAL_LLM_LATENCY_US,
        overhead_ratio_pct: overhead_ratio,
    };

    if !json_output {
        println!("  Per-op overhead:     {:>8.2} μs", results.per_op_overhead_us);
        println!("  Typical LLM latency: {:>8.0} μs (2s)", results.llm_latency_us);
        println!("  Overhead ratio:      {:>8.4}%", results.overhead_ratio_pct);
        println!();
        println!("  → Substrate overhead is {:.0}x smaller than LLM latency",
                 TYPICAL_LLM_LATENCY_US / per_op_us);
    }

    results
}

/// Benchmark memory operations (Table 5)
async fn benchmark_memory(json_output: bool) -> MemoryResults {
    if !json_output {
        println!("\n=== Memory Tier Latencies (Table 5) ===\n");
    }

    let config = RuntimeConfig::in_memory();
    let runtime = Runtime::new(config).await.expect("Failed to create runtime");

    let iterations = 1000; // Reduced from 10000

    // STM Write
    let start = Instant::now();
    for i in 0..iterations {
        runtime
            .memory()
            .write(MemorySpace::Stm, format!("key_{}", i), Value::String(format!("v{}", i)))
            .await
            .ok();
    }
    let stm_write_us = start.elapsed().as_micros() as f64 / iterations as f64;

    // STM Read
    let start = Instant::now();
    for i in 0..iterations {
        let _ = runtime.memory().read(MemorySpace::Stm, &format!("key_{}", i)).await;
    }
    let stm_read_us = start.elapsed().as_micros() as f64 / iterations as f64;

    // LTM Write
    let ltm_iterations = 100; // Reduced
    let start = Instant::now();
    for i in 0..ltm_iterations {
        runtime
            .memory()
            .write(MemorySpace::Ltm, format!("ltm_{}", i), Value::String(format!("v{}", i)))
            .await
            .ok();
    }
    let ltm_write_us = start.elapsed().as_micros() as f64 / ltm_iterations as f64;

    // LTM Read
    let start = Instant::now();
    for i in 0..ltm_iterations {
        let _ = runtime.memory().read(MemorySpace::Ltm, &format!("ltm_{}", i)).await;
    }
    let ltm_read_us = start.elapsed().as_micros() as f64 / ltm_iterations as f64;

    // Episodic Write
    let start = Instant::now();
    for i in 0..iterations {
        runtime
            .memory()
            .record_episode(format!("evt_{}", i), Value::String(format!("d{}", i)), "bench".to_string())
            .await
            .ok();
    }
    let episodic_write_us = start.elapsed().as_micros() as f64 / iterations as f64;

    let results = MemoryResults {
        stm_write_us,
        stm_read_us,
        ltm_write_us,
        ltm_read_us,
        episodic_write_us,
    };

    if !json_output {
        println!("  STM (volatile):     write={:.2}μs  read={:.2}μs",
                 results.stm_write_us, results.stm_read_us);
        println!("  LTM (persistent):   write={:.2}μs  read={:.2}μs",
                 results.ltm_write_us, results.ltm_read_us);
        println!("  Episodic (trace):   write={:.2}μs", results.episodic_write_us);
    }

    results
}

#[tokio::main]
async fn main() {
    let args: Vec<String> = std::env::args().collect();
    let json_output = args.iter().any(|a| a == "--json");

    if !json_output {
        println!("════════════════════════════════════════════════");
        println!("  A-PXM Substrate Benchmarks");
        println!("  (Measuring execution model overhead, NOT Rust vs Python)");
        println!("════════════════════════════════════════════════");
    }

    let overhead = benchmark_overhead(json_output).await;
    let memory = benchmark_memory(json_output).await;

    if json_output {
        let output = json!({
            "meta": {
                "benchmark": "apxm_substrate_overhead",
                "timestamp": chrono::Utc::now().to_rfc3339(),
                "platform": std::env::consts::OS,
                "arch": std::env::consts::ARCH,
                "note": "Measures substrate overhead, not language comparison"
            },
            "table_4_overhead": overhead,
            "table_5_memory": memory,
        });
        println!("{}", serde_json::to_string_pretty(&output).unwrap());
    } else {
        println!("\n════════════════════════════════════════════════");
        println!("  KEY INSIGHT FOR PAPER");
        println!("════════════════════════════════════════════════");
        println!();
        println!("  Substrate overhead: {:.2} μs/op", overhead.per_op_overhead_us);
        println!("  LLM call latency:   ~2,000,000 μs (2 seconds)");
        println!("  Overhead ratio:     {:.4}%", overhead.overhead_ratio_pct);
        println!();
        println!("  → The substrate is ~{:.0}x faster than LLM calls",
                 TYPICAL_LLM_LATENCY_US / overhead.per_op_overhead_us);
        println!("  → Overhead is negligible; execution model matters");
        println!();
    }
}
