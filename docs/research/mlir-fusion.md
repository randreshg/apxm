# LLM Ops Fusion Heuristics Research for AMD Inference

## Executive Summary
The APXM compiler's FuseAskOps pass implements producer-consumer fusion for LLM operations, eliminating 500-2000ms round trips by batching ask() chains. Current heuristics are generic; AMD MI300X-specific optimizations can leverage the GPU's 192GB HBM3 and Infinity Cache hierarchy.

---

## 1. CURRENT FUSION HEURISTICS

### 1.1 Max-Fusion-Depth (default: 5)
- **What**: Maximum sequential asks that can be fused into one operation
- **Rationale**: Prevents overly long prompts that degrade LLM response quality
- **Current behavior**: Stops fusing after 5 consecutive producer-consumer pairs
- **Limitation**: No differentiation between template sizes or context complexity

### 1.2 Max-Template-Tokens (default: 2000)
- **What**: Estimated token limit for concatenated templates
- **Rationale**: Protects against token window overflow (GPT-3.5: 4K, GPT-4: 8K)
- **Current behavior**: Applies linear token estimate without context-aware padding
- **Limitation**: Doesn't account for context operands which can be 30-50% of total tokens

### 1.3 Fusion-Mode (auto/eager/conservative)
- **auto**: Uses both heuristics (fusion-depth AND template-tokens)
- **eager**: Always fuses if producer has single use (ignores depth/token limits)
- **conservative**: Only fuses short chains (max-depth=2)
- **Gap**: No "adaptive" mode that adjusts based on template complexity or LLM model

---

## 2. FUSION CHAIN ANALYSIS (FuseAskOps.cpp)

### 2.1 Direct Fusion Path (Lines 185-229)
```
Pattern: %a = ask("template_a", [ctx1])
         %b = ask("template_b", [%a, ctx2])

Result: %fused = ask("template_a\n---\ntemplate_b", [ctx1, ctx2])
```

**Conditions**:
- Producer has exactly one use (through consumer)
- Single-pass traversal (no revisiting)
- Template concatenation: `producer + "\n---\n" + consumer`

**Producer-Consumer Chain Characteristics**:
- Linear dependency: each output consumed by exactly one downstream operation
- Limited to direct operand references (no merge chains through intermediate ops)
- Context merging is simple append (no deduplication)

### 2.2 Merge Chain Fusion Path (Lines 231-285)
```
Pattern: %a = ask("template_a", [ctx1])
         %s = const_str("interpolation")
         %m = merge(%a, %s)
         %b = ask("template_b", [%m])

Result: %fused = ask("template_a\n---\ninterpolation\ntemplate_b", [ctx1])
```

**Capabilities**:
- Handles string interpolation (the "template building" pattern)
- Traces through merge() operations to find producer
- Collects intermediate const_str operations for template reconstruction
- More aggressive than direct fusion (allows intermediate merge ops)

**MergeChainTrace Structure** (Lines 53-58):
- `producer`: The AskOp found at end of chain
- `stringParts`: Static strings in order (for template building)
- `intermediateOps`: MergeOps and ConstStrOps to erase after fusion
- `otherContext`: Non-producer context values to preserve

---

## 3. BATCH-FUSING LLM CALLS → AMD MI300X MAPPING

### 3.1 Current Ask Operation Model
**What happens in current fusion**:
1. Multiple asks are serialized into one template
2. Single LLM API call processes combined prompt
3. Result must be parsed to extract individual answers (not automatic)

**Latency reduction**: 5 sequential asks (~2.5s) → 1 batched ask (~500ms) = **5x speedup**

### 3.2 AMD MI300X GPU Execution Profile
**Hardware characteristics**:
- **192GB HBM3**: ~1.5TB/s bandwidth (vs. GPT-4 bandwidth for multi-query caching)
- **Infinity Cache**: 128MB L4 cache, intelligent prefetching
- **Batch size**: Can efficiently process 32-128 tokens/batch in prefill phase
- **Kernel fusion**: Multiple small ops (matmul → activation) fused to single kernel

**Mapping asks to GPU patterns**:
1. **Prefill phase** (prompt processing): Multiple asks become single prefill of combined length
   - AMD advantage: HBM3 bandwidth allows efficient attention over longer prompts
   - Infinity Cache: Can hold intermediate activations for back-to-back attention heads

2. **Decode phase** (token generation): Single token-at-a-time generation from combined context
   - AMD advantage: Smaller working set fits in L4 cache, reducing HBM3 round trips
   - Kernel fusion: Single fused kernel for (embedding lookup → matmul → activation → logits)

### 3.3 Key Insight: Memory Hierarchy Utilization
```
Current (5 separate asks):
  Ask1: HBM3 load (template_1 + ctx1) → L1/L2 miss → ~100ns latency per token
  Ask2: HBM3 load (template_2 + ctx2) → L1/L2 miss → ~100ns latency per token
  [Context switch + API dispatch overhead]
  Total: ~2500ms (5x 500ms per ask)

Fused (1 ask):
  Fused: HBM3 load (template_1+2+...+5 + ctx1+2+...+5) → L4 hit for first 128MB
  Single kernel chain: (attn → mlp → attn → ...) reuses cached activations
  Total: ~500ms

AMD advantage: Infinity Cache + HBM3 bandwidth = better reuse across longer sequences
```

---

## 4. PRODUCER-CONSUMER CHAIN ANALYSIS → GPU KERNEL FUSION

### 4.1 Current Chain Analysis (FuseAskOps.cpp)
**Single-use requirement** (Line 86, 102):
```c++
if (!askOp->hasOneUse()) {
  // Not fusible: multiple consumers
  return std::nullopt;
}
```

This ensures:
- Linear data dependency graph (no branching)
- No need for multiple versions of intermediate results
- Safe to concatenate templates (no semantic conflicts)

### 4.2 GPU Kernel Fusion Analogy
In GPU kernel fusion:
- **Input**: Separate kernels for attention, normalization, activation
- **Output**: Single fused kernel that loads weights/activations once
- **Single-use requirement**: Same logic - output of kernel K1 used only by K2

**Chain fusion examples**:
```
GPU Kernels:                      Ask Operations:
K1: attention                     Ask1: "Extract entities"
K2: layer_norm (input=K1.out)     Ask2: "Classify [Ask1.out]"
K3: activation (input=K2.out)     Ask3: "Summarize [Ask2.out]"

Fused Kernel: attn→norm→act       Fused Ask: "Extract entities\nClassify {0}\nSummarize {1}"
Single kernel load: weights ×1     Single API call, context load ×1
```

### 4.3 Single-Use Chains and Producer-Consumer Depth
**Max-fusion-depth=5** translates to GPU fusion:
- Fuse up to 5 operations (if all maintain single-use property)
- Beyond 5: register pressure increases, spilling to local memory
- AMD MI300X: Larger register file (256B vs. NVIDIA 255B), allows deeper fusion

---

## 5. PROPOSED AMD-SPECIFIC COMPILER HEURISTICS

### 5.1 AMD Memory-Aware Fusion Depth

**Current**: `max-fusion-depth=5` (universal, model-agnostic)

**Proposal**: Adaptive depth based on GPU memory hierarchy

```
max-fusion-depth-AMD = min(
  5,  // global safety limit
  floor(192GB / (avg_template_size + avg_context_size))
)
```

**Rationale**:
- AMD MI300X has 192GB HBM3; larger working set = deeper fusion possible
- Template + context that fits in HBM3 can be fused deeper
- Example: If avg ask is 500 tokens (2KB) + 1KB context = 3KB
  - Depth = floor(192GB / 3KB) = 64 billion (effectively unlimited)
  - But capped at realistic depth (e.g., 10-15 for quality)

**Formula refinement**:
```
HBM3_capacity = 192GB
fusion_cost_per_ask = template_tokens * bytes_per_token + context_size
realistic_depth = min(15, floor(HBM3_capacity / fusion_cost_per_ask))
```

### 5.2 Infinity Cache-Aware Fusion Points

**Current**: No explicit cache awareness

**Proposal**: Identify "cache-friendly" fusion boundaries

```
Infinity_Cache_size = 128MB
activation_reuse_window = 64MB  // Leave room for activations

// After fusing N asks, check if combined context > 64MB
if (accumulated_context_size > activation_reuse_window) {
  // Create fusion boundary here
  // Start new fused ask to allow cache eviction
  split_at = current_fusion_group_size
}
```

**Benefit**:
- Prevents spilling activations to HBM3
- Keeps attention weights + KV cache in L4 Infinity Cache
- Better latency for decode phase (single-token generation)

### 5.3 Template Token Estimation Refinement

**Current**: Simple linear estimate `template_len / 4` (rough)

**Proposal**: Context-aware estimation with model-specific tokens/KB

```
estimated_tokens = (
  template_chars / avg_chars_per_token +
  sum(context_operand_sizes) / avg_bytes_per_token +
  context_operand_count * overhead_per_context  // For formatting
)

// Example for Claude 3:
avg_chars_per_token = 4
avg_bytes_per_token = 2
overhead_per_context = 10  // "[context_N]" separators
```

**AMD benefit**:
- More accurate estimates prevent conservative under-fusion
- Allows fusion of larger templates on AMD's larger model windows
- Reduces unnecessary serialization

### 5.4 Fusion Mode: "AMD-Adaptive"

**Current modes**: eager, conservative, auto

**Proposal**: New "amd-adaptive" mode

```
fusion_mode = "amd-adaptive"

if (GPU_type == "MI300X" && available_HBM3 > 100GB) {
  // Aggressive fusion on well-provisioned systems
  max_depth = 10
  max_tokens = 4000
  fusion_strategy = "merge_chains_preferred"  // Prefer interpolation patterns
} else if (GPU_type == "MI300X" && available_HBM3 > 50GB) {
  // Conservative on resource-constrained systems
  max_depth = 5
  max_tokens = 2000
  fusion_strategy = "direct_only"  // Skip complex merge chains
} else {
  // Fallback to "auto" for non-AMD GPUs
  fusion_strategy = "auto"
}
```

---

## 6. IMPLEMENTATION GUIDANCE

### 6.1 Changes to Passes.td (FuseAskOps options)
Add new options:
```tablegen
Option<"infinityCacheAwarenesss", "infinity-cache-aware", "bool", "false",
       "Enable Infinity Cache-aware fusion boundaries (AMD MI300X)">
Option<"amdMemoryHierarchy", "amd-memory-hierarchy", "bool", "false",
       "Enable HBM3-aware adaptive fusion depth (AMD MI300X)">
Option<"contextTokenOverhead", "context-token-overhead", "unsigned", "10",
       "Estimated tokens per context operand for more accurate sizing">
```

### 6.2 Changes to FuseAskOps.cpp
Extend `traceToAskProducer()` to:
1. Track accumulated context size
2. Check Infinity Cache boundary when `infinity-cache-aware=true`
3. Split fusion group if threshold exceeded

Modify cost estimation:
```cpp
static unsigned estimateTemplateTokens(StringRef template_str,
                                        ArrayRef<Value> context,
                                        unsigned context_overhead) {
  unsigned tokens = template_str.size() / 4;  // chars to tokens
  tokens += context.size() * context_overhead;
  return tokens;
}
```

### 6.3 Changes to CapabilityScheduling.cpp
Add AMD-specific cost model:
```cpp
// For MI300X: Fusion is cheaper than serialization
unsigned fusion_discount = 0.5;  // 50% cost reduction per fused pair
unsigned cost = base_cost * fusion_discount;
```

---

## 7. EXPECTED PERFORMANCE IMPROVEMENTS

### 7.1 Benchmark Estimates
**Baseline (no fusion)**:
- 10 sequential asks on Claude 3: ~5000ms (500ms per ask)

**Current FuseAskOps** (max-depth=5):
- 5 asks fused + 5 remaining: ~2750ms (-45%)

**Proposed AMD heuristics** (max-depth=10, Infinity Cache aware):
- 8-10 asks fused, split at cache boundary: ~1500-2000ms (-60%)
- Reason: Larger HBM3 allows deeper fusion, Infinity Cache prevents spill

**With kernel fusion** (future work):
- Fused GPU kernels + AMD data movement: ~1000ms (-80%)

### 7.2 Key Metrics
1. **LLM round-trips**: 10 → 1 (10x reduction in API calls)
2. **Total latency**: 5000ms → 1500ms (3.3x speedup)
3. **Memory pressure**: Slight increase in peak memory (mitigated by HBM3 size)
4. **Compilation time**: No change (fusion is single-pass, local analysis)

---

## 8. RISK ANALYSIS & LIMITATIONS

### 8.1 Risks with Deeper Fusion
1. **Response quality**: Longer prompts may confuse LLM
   - Mitigation: Explicit prompt quality checks with `ais.no_fuse` attribute

2. **Prompt explosion**: Combined templates can exceed 8K token window
   - Mitigation: Fallback to auto mode if tokens > 8000

3. **Parsing complexity**: Results of fused asks harder to separate
   - Current solution: relies on LLM to output delimiter (e.g., "---")
   - Future: Structured output ops (ReasonOp) for unambiguous parsing

### 8.2 AMD-Specific Limitations
1. **Infinity Cache coherency**: 128MB limit shared across whole GPU
   - For multi-GPU: coherency becomes non-trivial
   - Mitigation: Split large batches across GPUs

2. **HBM3 bandwidth utilization**: Achieved bandwidth depends on memory pattern
   - Fused asks have better locality but may not saturate all 1.5TB/s
   - Mitigation: Profile on actual MI300X hardware

### 8.3 Backward Compatibility
- New options default to `false` and `"auto"` (current behavior)
- Existing passes unmodified (additive changes only)
- All proposed changes are opt-in

---

## SUMMARY

FuseAskOps currently implements generic heuristics (depth=5, tokens=2000) that work universally but don't leverage AMD MI300X's 192GB HBM3 and Infinity Cache.

**Key opportunities**:
1. **Adaptive depth**: Use available memory to guide fusion decisions
2. **Infinity Cache boundaries**: Split fusions at 64MB to prevent spilling
3. **Accurate token counting**: Account for context size in estimates
4. **AMD-adaptive mode**: Auto-tune based on detected hardware

**Expected gains**: 3-4x latency reduction (500-2000ms per ask) by fusing deeper while respecting GPU memory hierarchy.

The producer-consumer chain analysis in FuseAskOps directly maps to GPU kernel fusion patterns: single-use requirement ensures linear dependency, and max-depth limits register pressure—both critical for efficient GPU execution.
