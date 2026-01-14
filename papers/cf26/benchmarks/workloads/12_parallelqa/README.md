# Workload 12: ParallelQA Parallel Question Answering

## Purpose

Evaluate A-PXM on embarrassingly parallel question answering using the ParallelQA dataset. This benchmark measures:
- **Accuracy**: Exact match against ground truth
- **Parallelism efficiency**: Ability to identify and execute independent sub-questions in parallel
- **Latency**: End-to-end execution time per question

## Dataset

Uses `papers/cf26/benchmarks/datasets/parallelqa_dataset.json`.

Each example contains:
- `id`: Unique identifier
- `question`: Question that can be decomposed into parallel sub-questions
- `answer`: Ground truth answer
- `branch`: Number of parallel branches (typically 2)

## Workflow

The workflow demonstrates parallel sub-question handling:
1. Decompose question into independent sub-questions
2. Answer sub-questions (can be parallelized)
3. Combine sub-answers
4. Synthesize final answer

## Metrics

- **Accuracy**: Fraction of exact matches (normalized string comparison)
- **F1 Score**: Token-level F1 between prediction and ground truth
- **Latency**: Mean/std execution time per question
- **Parallelism**: Number of parallel operations identified

## How to Run

### Quick Run

```bash
cd papers/cf26/benchmarks/workloads/12_parallelqa
apxm execute workflow.ais "If Mariana Trench was 20% shallower and the Puerto Rico Trench was 20% deeper, which one would be shallower?"
```

### Full Benchmark

```bash
# From repo root
python papers/cf26/benchmarks/run_all.py --workload 12

# With specific sample size
python papers/cf26/benchmarks/workloads/runner.py --workload 12 --iterations 50
```

## Expected Results

For a well-performing system:
- **Accuracy**: >0.7 (70% exact match)
- **F1 Score**: >0.75
- **Latency**: <4s per question (depends on LLM and parallelism)

## Notes

This workload is designed to test A-PXM's ability to:
- Identify parallelizable sub-tasks
- Execute independent operations concurrently
- Combine results efficiently

A full implementation would use actual parallel execution of sub-questions.
