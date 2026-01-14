# Workload 11: HotpotQA Multi-hop Question Answering

## Purpose

Evaluate A-PXM on real-world multi-hop question answering using the HotpotQA dataset. This benchmark measures:
- **Accuracy**: Exact match and F1 score against ground truth
- **Latency**: End-to-end execution time per question
- **Reasoning quality**: Ability to handle sequential dependencies

## Dataset

Uses `papers/cf26/benchmarks/datasets/hotpotqa_comparison.json` (subset of HotpotQA dev set).

Each example contains:
- `id`: Unique identifier
- `question`: Multi-hop question requiring reasoning over multiple facts
- `answer`: Ground truth answer (often yes/no or entity name)

## Workflow

The workflow performs multi-hop reasoning:
1. Analyze question to identify entities and reasoning steps
2. Gather information (simulated via ask operations; full implementation would use INV("search"))
3. Synthesize final answer

## Metrics

- **Accuracy**: Fraction of exact matches (normalized string comparison)
- **F1 Score**: Token-level F1 between prediction and ground truth
- **Latency**: Mean/std execution time per question

## How to Run

### Quick Run

```bash
cd papers/cf26/benchmarks/workloads/11_hotpotqa
apxm execute workflow.ais "Were Scott Derrickson and Ed Wood of the same nationality?"
```

### Full Benchmark

```bash
# From repo root
python papers/cf26/benchmarks/run_all.py --workload 11

# With specific sample size
python papers/cf26/benchmarks/workloads/runner.py --workload 11 --iterations 50
```

## Expected Results

For a well-performing system:
- **Accuracy**: >0.6 (60% exact match)
- **F1 Score**: >0.7
- **Latency**: <5s per question (depends on LLM)

## Notes

This is a simplified implementation. A production system would:
- Use actual Wikipedia search via INV("search", entity)
- Implement proper entity extraction
- Use structured reasoning chains
