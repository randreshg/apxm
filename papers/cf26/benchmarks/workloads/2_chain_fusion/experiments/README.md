# FuseReasoning Experiments

These experiments validate the FuseReasoning optimization claims with proper tradeoff analysis.

## Experiments

### 1. Speedup vs Fusion Count (`speedup_vs_count.py`)
Measures latency for 1, 2, 3, 5, 8 RSN operations (fused vs unfused).
- **Expected**: ~N× speedup for N operations (I/O bound)
- **Output**: speedup curve with variance bars

### 2. Quality Preservation (`quality_preservation.py`)
Tests whether fused prompts produce equivalent quality to unfused prompts.
- **Expected**: <5% accuracy difference for short contexts
- **Output**: accuracy delta with confidence interval

### 3. Context Length Impact (`context_length_impact.py`)
Varies combined context from 500 to 8000 tokens and measures quality degradation.
- **Tests**: "Lost in the middle" phenomenon (Liu et al., 2024)
- **Expected**: degradation after ~3-5K tokens
- **Output**: quality vs context length curve

### 4. Task Type Impact (`task_type_impact.py`)
Compares fusion effectiveness across different task types:
- Classification (should help)
- Extraction (should help)
- Multi-step reasoning (might hurt)
- Creative generation (might hurt)

## Running

```bash
# Requires Ollama running with the cloud model used in the paper
ollama serve
# Ensure `gpt-oss:120b-cloud` (or fallback `gpt-oss:20b-cloud`) is available via `ollama list`

# Run individual experiments
python experiments/speedup_vs_count.py
python experiments/quality_preservation.py
python experiments/context_length_impact.py
python experiments/task_type_impact.py

# Run all
for f in experiments/*.py; do python "$f"; done
```

## Dependencies

```bash
pip install aiohttp
```

## Expected Results for Paper

| Experiment | Metric | Expected Value |
|------------|--------|----------------|
| Speedup vs Count | N ops speedup | ~N× (I/O bound) |
| Quality Preservation | Accuracy delta | <5% for short contexts |
| Context Length | Degradation threshold | ~3-5K tokens |
| Task Type | Classification/Extraction | Fusion recommended |
| Task Type | Reasoning/Creative | Fusion NOT recommended |

## Key Findings (TODO: Fill after running)

1. **Speedup is real**: \TODO{Confirm N× speedup for N operations}
2. **Quality tradeoffs exist**: \TODO{Document quality delta by task type}
3. **Context length matters**: \TODO{Identify degradation threshold}
4. **Task type matters**: \TODO{Classification vs reasoning results}

## Paper Integration

Results should update:
- `tex/05_evaluation.tex` - Fusion applicability subsection
- `tab/fuse-speedup.tex` - Speedup measurements table
- `tab/fuse-quality.tex` - Quality preservation table
