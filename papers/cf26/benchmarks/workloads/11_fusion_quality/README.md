# Workload 13: Fusion Quality

## Purpose

Analyze FuseReasoning effectiveness across different task types. This informs when fusion is beneficial vs. harmful.

## What We're Demonstrating

**A-PXM Property**: FuseReasoning task-type analysis

The FuseReasoning compiler pass batches dependent RSN chains into single prompts. However, effectiveness varies by task type. This workload measures fusion quality for:
- Classification tasks
- Extraction tasks
- Reasoning chains
- Creative tasks

```
Task Type Analysis:
+----------------+------------------+------------------+
| Task Type      | O0 (Unfused)     | O1 (Fused)       |
+----------------+------------------+------------------+
| Classification | 3 calls, ~6s     | 1 call, ~3s      | <-- Good fusion
| Extraction     | 3 calls, ~6s     | 1 call, ~2.5s    | <-- Good fusion
| Reasoning      | 5 calls, ~10s    | 1 call, ~15s     | <-- Fusion hurts!
| Creative       | 3 calls, ~6s     | 1 call, ~4s      | <-- Mixed results
+----------------+------------------+------------------+
```

### Task Type Workflows

**Classification (classification.ais)**:
```
agent ClassificationTest {
    @entry flow main() -> str {
        // Parallel classification (no dependencies)
        rsn("Classify this sentiment: 'I love this product!'") -> sent1
        rsn("Classify this sentiment: 'This is terrible'") -> sent2
        rsn("Classify this sentiment: 'It works okay'") -> sent3
        merge(sent1, sent2, sent3) -> results
        return results
    }
}
```

**Reasoning (reasoning.ais)**: Sequential dependent chain

**Extraction (extraction.ais)**: Parallel extraction tasks

**Creative (creative.ais)**: Creative generation tasks

---

## How to Run

### Prerequisites

```bash
# Start Ollama (local LLM backend)
ollama serve
ollama pull gpt-oss:20b-cloud

# Build A-PXM compiler (from repo root)
apxm compiler build
```

### Run Fusion Quality Benchmark

```bash
cd papers/CF26/benchmarks/workloads/13_fusion_quality

# Run all task types
python run.py

# With JSON output
python run.py --json
```

### Run via CLI

```bash
# From repo root
apxm workloads run 13_fusion_quality --json
```

---

## Results

*To be filled after benchmark execution*

| Task Type | O0 Time | O1 Time | Speedup | Quality |
|-----------|---------|---------|---------|---------|
| Classification | - | - | - | - |
| Extraction | - | - | - | - |
| Reasoning | - | - | - | - |
| Creative | - | - | - | - |

---

## Analysis

*To be filled after benchmark execution*

### Expected Observations

1. **Classification benefits from fusion**: Independent classification tasks batch well.

2. **Deep reasoning may degrade**: Long dependent chains create complex prompts that LLMs process slower.

3. **Task-aware heuristics needed**: Optimal fusion depends on task type.

### Research Insights

From EVALUATION_DISCUSSION.md:
- Fusion mechanism works (the IR enables the transformation)
- Naive fusion isn't always optimal
- Opens research into cost-aware fusion heuristics

### Key Insight

This workload demonstrates that A-PXM's IR enables fusion research impossible in opaque frameworks. The results inform when FuseReasoning should be applied, paving the way for learned cost models.
