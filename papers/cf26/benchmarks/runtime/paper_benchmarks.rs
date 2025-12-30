//! Paper Benchmarks for A-PXM - FAST VERSION
//!
//! Measures substrate properties (NOT Rust vs Python comparison):
//! - Table 4: Substrate overhead per operation (~8μs)
//! - Table 5: Memory tier latencies
//! - Overhead ratio: substrate_overhead / typical_LLM_latency
//!
//! Run: cargo run --example paper_benchmarks -p apxm-runtime --release
//! JSON: cargo run --example paper_benchmarks -p apxm-runtime --release -- --json
//! METRICS: cargo run --example paper_benchmarks -p apxm-runtime --release --features metrics -- --json

use apxm_core::types::{
    execution::{DagMetadata, DependencyType, Edge, ExecutionDag, Node, NodeMetadata},
    operations::AISOperationType,
    values::Value,
};
use apxm_runtime::{MemorySpace, Runtime, RuntimeConfig, SchedulerMetrics};
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
    /// Detailed overhead breakdown (when metrics feature is enabled)
    overhead_breakdown: Option<OverheadBreakdown>,
    /// Parallelism metrics
    max_parallelism: usize,
    avg_parallelism: f64,
}

/// Detailed overhead breakdown by phase
#[derive(Debug, serde::Serialize)]
struct OverheadBreakdown {
    ready_set_update_us: f64,
    work_stealing_us: f64,
    input_collection_us: f64,
    operation_dispatch_us: f64,
    token_routing_us: f64,
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
/// Uses built-in MetricsCollector for detailed phase breakdown
async fn benchmark_overhead(json_output: bool) -> OverheadResults {
    if !json_output {
        println!("\n=== Substrate Overhead (Table 4) ===\n");
    }

    let config = RuntimeConfig::in_memory();
    let runtime = Runtime::new(config)
        .await
        .expect("Failed to create runtime");
    let dag = create_benchmark_dag(OPS_PER_DAG);

    // Warmup
    for _ in 0..WARMUP_ITERATIONS {
        let _ = runtime.execute(dag.clone()).await;
    }

    // Benchmark - collect metrics from each run
    let start = Instant::now();
    let mut all_metrics: Vec<SchedulerMetrics> = Vec::with_capacity(BENCHMARK_ITERATIONS);

    for _ in 0..BENCHMARK_ITERATIONS {
        let result = runtime
            .execute(dag.clone())
            .await
            .expect("Execution failed");
        all_metrics.push(result.scheduler_metrics);
    }
    let total_time = start.elapsed();

    // Aggregate metrics from built-in collector
    let total_ops: usize = all_metrics.iter().map(|m| m.operations_executed).sum();

    // Use averaged per-op overhead from the collector (more accurate than wall-clock)
    let avg_per_op_us = if !all_metrics.is_empty() {
        all_metrics
            .iter()
            .map(|m| m.per_op_overhead_us)
            .sum::<f64>()
            / all_metrics.len() as f64
    } else {
        total_time.as_micros() as f64 / total_ops as f64
    };

    let overhead_ratio = (avg_per_op_us / TYPICAL_LLM_LATENCY_US) * 100.0;

    // Aggregate parallelism metrics
    let max_parallelism = all_metrics
        .iter()
        .map(|m| m.max_parallelism)
        .max()
        .unwrap_or(1);
    let avg_parallelism = if !all_metrics.is_empty() {
        all_metrics.iter().map(|m| m.avg_parallelism).sum::<f64>() / all_metrics.len() as f64
    } else {
        1.0
    };

    // Extract detailed breakdown (averaged across all runs)
    let overhead_breakdown = if !all_metrics.is_empty() {
        let n = all_metrics.len() as f64;
        Some(OverheadBreakdown {
            ready_set_update_us: all_metrics
                .iter()
                .map(|m| m.overhead_breakdown.ready_set_update_us)
                .sum::<f64>()
                / n,
            work_stealing_us: all_metrics
                .iter()
                .map(|m| m.overhead_breakdown.work_stealing_us)
                .sum::<f64>()
                / n,
            input_collection_us: all_metrics
                .iter()
                .map(|m| m.overhead_breakdown.input_collection_us)
                .sum::<f64>()
                / n,
            operation_dispatch_us: all_metrics
                .iter()
                .map(|m| m.overhead_breakdown.operation_dispatch_us)
                .sum::<f64>()
                / n,
            token_routing_us: all_metrics
                .iter()
                .map(|m| m.overhead_breakdown.token_routing_us)
                .sum::<f64>()
                / n,
        })
    } else {
        None
    };

    let results = OverheadResults {
        per_op_overhead_us: avg_per_op_us,
        operations_executed: total_ops,
        total_time_ms: total_time.as_millis() as f64,
        llm_latency_us: TYPICAL_LLM_LATENCY_US,
        overhead_ratio_pct: overhead_ratio,
        overhead_breakdown,
        max_parallelism,
        avg_parallelism,
    };

    if !json_output {
        println!(
            "  Per-op overhead:     {:>8.2} μs",
            results.per_op_overhead_us
        );
        println!(
            "  Typical LLM latency: {:>8.0} μs (2s)",
            results.llm_latency_us
        );
        println!(
            "  Overhead ratio:      {:>8.4}%",
            results.overhead_ratio_pct
        );
        println!("  Max parallelism:     {:>8}", results.max_parallelism);
        println!("  Avg parallelism:     {:>8.2}", results.avg_parallelism);

        if let Some(ref breakdown) = results.overhead_breakdown {
            println!();
            println!("  Overhead breakdown:");
            println!(
                "    Ready-set update:    {:>6.2} μs",
                breakdown.ready_set_update_us
            );
            println!(
                "    Work stealing:       {:>6.2} μs",
                breakdown.work_stealing_us
            );
            println!(
                "    Input collection:    {:>6.2} μs",
                breakdown.input_collection_us
            );
            println!(
                "    Operation dispatch:  {:>6.2} μs",
                breakdown.operation_dispatch_us
            );
            println!(
                "    Token routing:       {:>6.2} μs",
                breakdown.token_routing_us
            );
        }

        println!();
        println!(
            "  → Substrate overhead is {:.0}x smaller than LLM latency",
            TYPICAL_LLM_LATENCY_US / avg_per_op_us
        );
    }

    results
}

/// Benchmark memory operations (Table 5)
async fn benchmark_memory(json_output: bool) -> MemoryResults {
    if !json_output {
        println!("\n=== Memory Tier Latencies (Table 5) ===\n");
    }

    let config = RuntimeConfig::in_memory();
    let runtime = Runtime::new(config)
        .await
        .expect("Failed to create runtime");

    let iterations = 1000; // Reduced from 10000

    // STM Write
    let start = Instant::now();
    for i in 0..iterations {
        runtime
            .memory()
            .write(
                MemorySpace::Stm,
                format!("key_{}", i),
                Value::String(format!("v{}", i)),
            )
            .await
            .ok();
    }
    let stm_write_us = start.elapsed().as_micros() as f64 / iterations as f64;

    // STM Read
    let start = Instant::now();
    for i in 0..iterations {
        let _ = runtime
            .memory()
            .read(MemorySpace::Stm, &format!("key_{}", i))
            .await;
    }
    let stm_read_us = start.elapsed().as_micros() as f64 / iterations as f64;

    // LTM Write
    let ltm_iterations = 100; // Reduced
    let start = Instant::now();
    for i in 0..ltm_iterations {
        runtime
            .memory()
            .write(
                MemorySpace::Ltm,
                format!("ltm_{}", i),
                Value::String(format!("v{}", i)),
            )
            .await
            .ok();
    }
    let ltm_write_us = start.elapsed().as_micros() as f64 / ltm_iterations as f64;

    // LTM Read
    let start = Instant::now();
    for i in 0..ltm_iterations {
        let _ = runtime
            .memory()
            .read(MemorySpace::Ltm, &format!("ltm_{}", i))
            .await;
    }
    let ltm_read_us = start.elapsed().as_micros() as f64 / ltm_iterations as f64;

    // Episodic Write
    let start = Instant::now();
    for i in 0..iterations {
        runtime
            .memory()
            .record_episode(
                format!("evt_{}", i),
                Value::String(format!("d{}", i)),
                "bench".to_string(),
            )
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
        println!(
            "  STM (volatile):     write={:.2}μs  read={:.2}μs",
            results.stm_write_us, results.stm_read_us
        );
        println!(
            "  LTM (persistent):   write={:.2}μs  read={:.2}μs",
            results.ltm_write_us, results.ltm_read_us
        );
        println!(
            "  Episodic (trace):   write={:.2}μs",
            results.episodic_write_us
        );
    }

    results
}

/// Results from parallelism scaling benchmark (for fig/speedup-plot.tex)
#[derive(Debug, serde::Serialize)]
struct ParallelismScalingResults {
    n: usize,
    baseline_time_ms: f64, // Time for N=1 sequential
    parallel_time_ms: f64, // Time for N parallel ops
    speedup: f64,          // Theoretical N / actual
    efficiency_pct: f64,   // speedup / N * 100
    max_observed_parallelism: usize,
}

/// Create a DAG with N fully parallel operations
fn create_parallel_dag(n: usize) -> ExecutionDag {
    let mut nodes = Vec::new();
    let mut edges = Vec::new();

    // Entry node (broadcasts to all parallel ops)
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

    // N parallel nodes
    for i in 1..=n {
        nodes.push(Node {
            id: i as u64,
            op_type: AISOperationType::ConstStr,
            attributes: {
                let mut attrs = HashMap::new();
                attrs.insert(
                    "value".to_string(),
                    Value::String(format!("parallel_{}", i)),
                );
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

    // Exit node (collects all parallel results)
    let exit_id = (n + 1) as u64;
    let input_tokens: Vec<u64> = (1..=n).map(|i| 100 + i as u64).collect();
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

    let entry_nodes = vec![0];
    let exit_nodes = vec![exit_id];

    ExecutionDag {
        nodes,
        edges,
        entry_nodes,
        exit_nodes,
        metadata: DagMetadata::default(),
    }
}

/// Benchmark parallelism scaling for fig/speedup-plot.tex
async fn benchmark_parallelism_scaling(json_output: bool) -> Vec<ParallelismScalingResults> {
    if !json_output {
        println!("\n=== Parallelism Scaling (fig/speedup-plot) ===\n");
    }

    let config = RuntimeConfig::in_memory();
    let runtime = Runtime::new(config)
        .await
        .expect("Failed to create runtime");

    // Test with N = 1, 2, 4, 8, 16, 32
    let parallel_levels = [1, 2, 4, 8, 16, 32];
    let mut results = Vec::new();

    // First, establish baseline (N=1)
    let baseline_dag = create_parallel_dag(1);

    // Warmup
    for _ in 0..WARMUP_ITERATIONS {
        let _ = runtime.execute(baseline_dag.clone()).await;
    }

    // Measure baseline
    let start = Instant::now();
    for _ in 0..BENCHMARK_ITERATIONS {
        let _ = runtime.execute(baseline_dag.clone()).await;
    }
    let baseline_time_per_dag =
        start.elapsed().as_secs_f64() * 1000.0 / BENCHMARK_ITERATIONS as f64;

    for &n in &parallel_levels {
        let dag = create_parallel_dag(n);

        // Warmup
        for _ in 0..WARMUP_ITERATIONS {
            let _ = runtime.execute(dag.clone()).await;
        }

        // Benchmark
        let start = Instant::now();
        let mut max_parallelism = 0usize;

        for _ in 0..BENCHMARK_ITERATIONS {
            let result = runtime
                .execute(dag.clone())
                .await
                .expect("Execution failed");
            max_parallelism = max_parallelism.max(result.scheduler_metrics.max_parallelism);
        }
        let parallel_time_ms = start.elapsed().as_secs_f64() * 1000.0 / BENCHMARK_ITERATIONS as f64;

        // Calculate speedup relative to running N sequential (baseline * N)
        let sequential_equivalent = baseline_time_per_dag * n as f64;
        let speedup = sequential_equivalent / parallel_time_ms;
        let efficiency = (speedup / n as f64) * 100.0;

        results.push(ParallelismScalingResults {
            n,
            baseline_time_ms: sequential_equivalent,
            parallel_time_ms,
            speedup,
            efficiency_pct: efficiency,
            max_observed_parallelism: max_parallelism,
        });
    }

    if !json_output {
        println!(
            "{:>4} | {:>12} | {:>12} | {:>8} | {:>8} | {:>12}",
            "N", "Seq. (ms)", "Par. (ms)", "Speedup", "Eff. %", "Max Par."
        );
        println!(
            "{:-<4}-+-{:-<12}-+-{:-<12}-+-{:-<8}-+-{:-<8}-+-{:-<12}",
            "", "", "", "", "", ""
        );

        for r in &results {
            println!(
                "{:>4} | {:>12.3} | {:>12.3} | {:>7.2}x | {:>7.1}% | {:>12}",
                r.n,
                r.baseline_time_ms,
                r.parallel_time_ms,
                r.speedup,
                r.efficiency_pct,
                r.max_observed_parallelism
            );
        }
    }

    results
}

#[tokio::main]
async fn main() {
    let args: Vec<String> = std::env::args().collect();
    let json_output = args.iter().any(|a| a == "--json");
    let run_scaling = args.iter().any(|a| a == "--scaling");
    let run_all = !run_scaling; // By default run overhead/memory, or all if --scaling specified

    if !json_output {
        println!("════════════════════════════════════════════════");
        println!("  A-PXM Substrate Benchmarks");
        println!("  (Measuring execution model overhead, NOT Rust vs Python)");
        println!("════════════════════════════════════════════════");
    }

    let overhead = if run_all {
        Some(benchmark_overhead(json_output).await)
    } else {
        None
    };

    let memory = if run_all {
        Some(benchmark_memory(json_output).await)
    } else {
        None
    };

    let parallelism = if run_scaling || run_all {
        Some(benchmark_parallelism_scaling(json_output).await)
    } else {
        None
    };

    if json_output {
        let mut output = json!({
            "meta": {
                "benchmark": "apxm_substrate_overhead",
                "timestamp": chrono::Utc::now().to_rfc3339(),
                "platform": std::env::consts::OS,
                "arch": std::env::consts::ARCH,
                "note": "Measures substrate overhead, not language comparison"
            }
        });

        if let Some(ref o) = overhead {
            output["table_4_overhead"] = json!(o);
        }
        if let Some(ref m) = memory {
            output["table_5_memory"] = json!(m);
        }
        if let Some(ref p) = parallelism {
            output["fig_speedup_plot"] = json!(p);
        }

        println!("{}", serde_json::to_string_pretty(&output).unwrap());
    } else {
        if let Some(ref o) = overhead {
            println!("\n════════════════════════════════════════════════");
            println!("  KEY INSIGHT FOR PAPER");
            println!("════════════════════════════════════════════════");
            println!();
            println!("  Substrate overhead: {:.2} μs/op", o.per_op_overhead_us);
            println!("  LLM call latency:   ~2,000,000 μs (2 seconds)");
            println!("  Overhead ratio:     {:.4}%", o.overhead_ratio_pct);
            println!();
            println!(
                "  → The substrate is ~{:.0}x faster than LLM calls",
                TYPICAL_LLM_LATENCY_US / o.per_op_overhead_us
            );
            println!("  → Overhead is negligible; execution model matters");
        }

        if parallelism.is_some() {
            println!();
            println!("════════════════════════════════════════════════");
            println!("  Use --json to export data for fig/speedup-plot.tex");
            println!("════════════════════════════════════════════════");
        }
        println!();
    }
}
