#!/bin/bash
# FuseReasoning Benchmark
#
# This benchmark measures the actual speedup from the FuseReasoning compiler pass
# by running the same AIS workflow with and without the optimization.
#
# Usage:
#   cd /path/to/apxm
#   ./papers/cf26/benchmarks/workloads/2_chain_fusion/run_benchmark.sh
#
# Prerequisites:
#   - Ollama running with gpt-oss:120b-cloud model
#   - CONDA_PREFIX set to the apxm conda environment

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
APXM_ROOT="$(cd "$SCRIPT_DIR/../../../../.." && pwd)"
WORKFLOW="$SCRIPT_DIR/workflow.ais"
TRIALS=3

echo "════════════════════════════════════════════════════════════════"
echo "  FuseReasoning Benchmark"
echo "  (Actual A-PXM Compiler + Runtime with Real LLM Calls)"
echo "════════════════════════════════════════════════════════════════"
echo ""
echo "Workflow: $WORKFLOW"
echo "Trials per configuration: $TRIALS"
echo ""

# Check prerequisites
if [ -z "$CONDA_PREFIX" ]; then
    echo "ERROR: CONDA_PREFIX not set. Run: conda activate apxm"
    exit 1
fi

echo "Checking Ollama..."
if ! ollama list | grep -q "gpt-oss:120b-cloud"; then
    echo "WARNING: gpt-oss:120b-cloud not found. Using default model."
fi
echo ""

# Build CLI if needed
echo "Building CLI..."
cd "$APXM_ROOT"
cargo build -p apxm-cli --features driver --release --quiet
CLI="$APXM_ROOT/target/release/apxm"

echo ""
echo "════════════════════════════════════════════════════════════════"
echo "  Running O0 (Unfused - No FuseReasoning)"
echo "════════════════════════════════════════════════════════════════"

O0_TIMES=()
for i in $(seq 1 $TRIALS); do
    echo -n "  Trial $i/$TRIALS... "
    START=$(python3 -c 'import time; print(time.time())')

    # Run with -O0 (no FuseReasoning)
    OUTPUT=$($CLI run "$WORKFLOW" -O0 2>&1)

    END=$(python3 -c 'import time; print(time.time())')
    DURATION=$(python3 -c "print(int(($END - $START) * 1000))")

    # Extract node count from output
    NODES=$(echo "$OUTPUT" | grep -o "[0-9]* nodes" | grep -o "[0-9]*" || echo "?")

    echo "${DURATION}ms (${NODES} nodes)"
    O0_TIMES+=($DURATION)
done

echo ""
echo "════════════════════════════════════════════════════════════════"
echo "  Running O1 (Fused - FuseReasoning Enabled)"
echo "════════════════════════════════════════════════════════════════"

O1_TIMES=()
for i in $(seq 1 $TRIALS); do
    echo -n "  Trial $i/$TRIALS... "
    START=$(python3 -c 'import time; print(time.time())')

    # Run with -O1 (FuseReasoning enabled)
    OUTPUT=$($CLI run "$WORKFLOW" -O1 2>&1)

    END=$(python3 -c 'import time; print(time.time())')
    DURATION=$(python3 -c "print(int(($END - $START) * 1000))")

    # Extract node count from output
    NODES=$(echo "$OUTPUT" | grep -o "[0-9]* nodes" | grep -o "[0-9]*" || echo "?")

    echo "${DURATION}ms (${NODES} nodes)"
    O1_TIMES+=($DURATION)
done

echo ""
echo "════════════════════════════════════════════════════════════════"
echo "  RESULTS"
echo "════════════════════════════════════════════════════════════════"

# Calculate means using Python
O0_MEAN=$(python3 -c "times = [${O0_TIMES[*]}]; print(sum(times)/len(times) if times else 0)")
O1_MEAN=$(python3 -c "times = [${O1_TIMES[*]}]; print(sum(times)/len(times) if times else 0)")
SPEEDUP=$(python3 -c "print(f'{$O0_MEAN / $O1_MEAN:.2f}' if $O1_MEAN > 0 else 'N/A')")

echo ""
echo "  O0 (Unfused) mean latency: ${O0_MEAN}ms"
echo "  O1 (Fused)   mean latency: ${O1_MEAN}ms"
echo ""
echo "  Speedup: ${SPEEDUP}x"
echo ""

# Expected speedup analysis
echo "  Expected speedup: 5.0x (5 RSN ops → 1 batched call)"
EFFICIENCY=$(python3 -c "print(f'{($SPEEDUP / 5.0) * 100:.1f}' if '$SPEEDUP' != 'N/A' else 'N/A')")
echo "  Efficiency: ${EFFICIENCY}%"

echo ""
echo "════════════════════════════════════════════════════════════════"
echo "  FOR PAPER (tex/05_evaluation.tex)"
echo "════════════════════════════════════════════════════════════════"
echo ""
echo "  FuseReasoning achieves ${SPEEDUP}x speedup on 5-operation chain"
echo "  Unfused (O0): ${O0_MEAN}ms (5 sequential LLM calls)"
echo "  Fused (O1):   ${O1_MEAN}ms (1 batched LLM call)"

# Save results to JSON
RESULTS_FILE="$SCRIPT_DIR/benchmark_results.json"
cat > "$RESULTS_FILE" << EOF
{
  "experiment": "fuse_reasoning",
  "workflow": "workflow.ais",
  "model": "gpt-oss:120b-cloud",
  "trials": $TRIALS,
  "o0_times_ms": [${O0_TIMES[*]}],
  "o1_times_ms": [${O1_TIMES[*]}],
  "o0_mean_ms": $O0_MEAN,
  "o1_mean_ms": $O1_MEAN,
  "speedup": $SPEEDUP,
  "expected_speedup": 5.0,
  "rsn_operations": 5
}
EOF

echo ""
echo "Results saved to: $RESULTS_FILE"
