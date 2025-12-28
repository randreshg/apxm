//! A-PXM Scheduler Overhead Benchmark
//!
//! Measures pure scheduler overhead with synthetic operations (no LLM calls).
//! Outputs JSON for reproducible analysis.
//!
//! Run with:
//!   cargo run --example benchmark_overhead -p apxm-runtime --features metrics --release
//!   cargo run --example benchmark_overhead -p apxm-runtime --features metrics --release -- --json

use apxm_core::types::{
    execution::{DependencyType, Edge, NodeMetadata},
    operations::AISOperationType,
};
use apxm_runtime::{
    memory::MemorySpace,
    ExecutionDag, Node, Runtime, RuntimeConfig, Value,
};
use std::collections::HashMap;
use std::time::Instant;

const WARMUP_ITERATIONS: usize = 5;
const BENCHMARK_ITERATIONS: usize = 100;
const SYNTHETIC_OPS: usize = 500;

fn std_dev(samples: &[f64], mean: f64) -> f64 {
    if samples.len() <= 1 {
        return 0.0;
    }
    let variance = samples.iter()
        .map(|x| (x - mean).powi(2))
        .sum::<f64>() / (samples.len() - 1) as f64;
    variance.sqrt()
}

fn ci_95(std_dev: f64, n: usize) -> f64 {
    1.96 * std_dev / (n as f64).sqrt()
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let args: Vec<String> = std::env::args().collect();
    let json_output = args.iter().any(|a| a == "--json");

    let config = RuntimeConfig::in_memory();
    let runtime = Runtime::new(config).await?;

    // Warmup
    for _ in 0..WARMUP_ITERATIONS {
        let dag = create_synthetic_dag(100);
        let _ = runtime.execute(dag).await?;
    }

    // Benchmark scheduler overhead
    let mut overhead_samples = Vec::new();
    for _ in 0..BENCHMARK_ITERATIONS {
        let dag = create_synthetic_dag(SYNTHETIC_OPS);
        let start = Instant::now();
        let _ = runtime.execute(dag).await?;
        let duration = start.elapsed();
        let per_op_us = duration.as_micros() as f64 / SYNTHETIC_OPS as f64;
        overhead_samples.push(per_op_us);
    }

    let avg_overhead = overhead_samples.iter().sum::<f64>() / overhead_samples.len() as f64;
    let overhead_std = std_dev(&overhead_samples, avg_overhead);
    let overhead_ci = ci_95(overhead_std, overhead_samples.len());
    let min_overhead = overhead_samples.iter().cloned().fold(f64::MAX, f64::min);
    let max_overhead = overhead_samples.iter().cloned().fold(f64::MIN, f64::max);

    overhead_samples.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let p50 = overhead_samples[overhead_samples.len() / 2];
    let p99_idx = ((overhead_samples.len() as f64 * 0.99) as usize).min(overhead_samples.len() - 1);
    let p99 = overhead_samples[p99_idx];

    // State inspection benchmark
    let aam = runtime.aam();
    for i in 0..100 {
        aam.set_belief(
            format!("belief_{}", i),
            Value::String(format!("value_{}", i)),
            apxm_runtime::aam::TransitionLabel::custom("benchmark"),
        );
    }

    let mut inspection_times = Vec::new();
    for _ in 0..100 {
        let start = Instant::now();
        let beliefs = aam.beliefs();
        let goals = aam.goals();
        let caps = aam.capabilities();
        let transitions = aam.recent_transitions(10);
        let _ = (beliefs.len(), goals.len(), caps.len(), transitions.len());
        inspection_times.push(start.elapsed());
    }

    let avg_inspection_us = inspection_times.iter().map(|d| d.as_micros()).sum::<u128>() as f64 / 100.0;

    // Memory access benchmark
    let memory = runtime.memory();

    let stm_start = Instant::now();
    for i in 0..100 {
        memory.write(
            MemorySpace::Stm,
            format!("key_{}", i),
            Value::String(format!("value_{}", i)),
        ).await?;
    }
    let stm_write_us = stm_start.elapsed().as_micros() as f64 / 100.0;

    let stm_read_start = Instant::now();
    for i in 0..100 {
        let _ = memory.read(MemorySpace::Stm, &format!("key_{}", i)).await?;
    }
    let stm_read_us = stm_read_start.elapsed().as_micros() as f64 / 100.0;

    let ltm_start = Instant::now();
    for i in 0..100 {
        memory.write(
            MemorySpace::Ltm,
            format!("ltm_key_{}", i),
            Value::String(format!("ltm_value_{}", i)),
        ).await?;
    }
    let ltm_write_us = ltm_start.elapsed().as_micros() as f64 / 100.0;

    let ltm_read_start = Instant::now();
    for i in 0..100 {
        let _ = memory.read(MemorySpace::Ltm, &format!("ltm_key_{}", i)).await?;
    }
    let ltm_read_us = ltm_read_start.elapsed().as_micros() as f64 / 100.0;

    if json_output {
        let json = serde_json::json!({
            "meta": {
                "benchmark": "scheduler_overhead",
                "timestamp": chrono::Utc::now().to_rfc3339(),
                "os": std::env::consts::OS,
            },
            "config": {
                "iterations": BENCHMARK_ITERATIONS,
                "warmup": WARMUP_ITERATIONS,
                "operations_per_iteration": SYNTHETIC_OPS,
            },
            "results": {
                "scheduler_overhead": {
                    "mean_us": avg_overhead,
                    "std_us": overhead_std,
                    "ci_95_us": overhead_ci,
                    "min_us": min_overhead,
                    "max_us": max_overhead,
                    "p50_us": p50,
                    "p99_us": p99,
                },
                "state_inspection": {
                    "avg_us": avg_inspection_us,
                },
                "memory_access": {
                    "stm_read_us": stm_read_us,
                    "stm_write_us": stm_write_us,
                    "ltm_read_us": ltm_read_us,
                    "ltm_write_us": ltm_write_us,
                },
            },
        });
        println!("{}", serde_json::to_string_pretty(&json)?);
    } else {
        println!("A-PXM Scheduler Overhead Benchmark");
        println!("===================================");
        println!();
        println!("Config: {} iterations, {} ops each", BENCHMARK_ITERATIONS, SYNTHETIC_OPS);
        println!();
        println!("Scheduler Overhead (per operation):");
        println!("  Mean:  {:.2} ± {:.2} μs (95% CI)", avg_overhead, overhead_ci);
        println!("  Std:   {:.2} μs", overhead_std);
        println!("  P50:   {:.2} μs", p50);
        println!("  P99:   {:.2} μs", p99);
        println!();
        println!("State Inspection: {:.2} μs", avg_inspection_us);
        println!();
        println!("Memory Access:");
        println!("  STM read:  {:.2} μs", stm_read_us);
        println!("  LTM read:  {:.2} μs", ltm_read_us);
    }

    Ok(())
}

fn create_synthetic_dag(n: usize) -> ExecutionDag {
    let mut nodes = Vec::new();
    for i in 0..n {
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
            Value::String(format!("op_{}", i)),
        );
        nodes.push(node);
    }

    ExecutionDag {
        nodes,
        edges: vec![],
        entry_nodes: (0..n as u64).collect(),
        exit_nodes: (0..n as u64).collect(),
        metadata: Default::default(),
    }
}
