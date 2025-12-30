# Enhance A-PXM Testing Plan for CF'26 Submission

## Context

You are helping enhance a testing plan for a research paper submission to **ACM Computing Frontiers 2026** (CF'26). The paper presents **A-PXM: A Program Execution Model for Agentic AI**, which provides a formal execution substrate for agentic AI systems with three pillars:

1. **Agent Abstract Machine (AAM)**: Explicit state model (Beliefs, Goals, Capabilities) with 3-tier memory (STM, LTM, Episodic)
2. **Agent Instruction Set (AIS)**: 19 typed operations implemented as an MLIR dialect with compile-time verification
3. **Dataflow Runtime**: Token-based scheduling that extracts parallelism automatically from data dependencies

## Your Task

Review and enhance the testing plan located at:
- **Testing Plan**: `/Users/randreshg/Documents/apxm/papers/cf26/TESTING_PLAN.md`

Ensure the plan:
1. **Validates all paper claims** against actual implementation
2. **Aligns with CF'26 requirements** (8 pages max, artifact evaluation encouraged)
3. **Covers all crates** in the project architecture
4. **Verifies consistency** between paper text, testing plan, and codebase
5. **Addresses all TODO/VERIFY markers** in the paper
6. **Provides actionable execution steps** with clear success criteria

## Key Documents to Reference

### 1. Paper Content
- **Main Paper**: `/Users/randreshg/Documents/-CFP-A-PXM-for-Unlocking-Parallelism-in-Agentic-AI/main.tex`
- **Evaluation Section**: `/Users/randreshg/Documents/-CFP-A-PXM-for-Unlocking-Parallelism-in-Agentic-AI/tex/05_evaluation.tex`
- **Key Claims**: See Table in TESTING_PLAN.md (C1-C7)

### 2. Project Architecture
- **Crates Documentation**: `/Users/randreshg/Documents/apxm/docs/CRATES.md`
- **Architecture Overview**: `/Users/randreshg/Documents/apxm/docs/architecture.md`
- **Crate Structure**: `/Users/randreshg/Documents/apxm/crates/`

**Key Crates to Understand:**
- `apxm-ais`: Defines 19 AIS operations (single source of truth)
- `apxm-compiler`: MLIR-based compiler (5-stage pipeline)
- `apxm-runtime`: Dataflow execution engine with scheduler
- `apxm-backends`: LLM and storage backends
- `apxm-core`: Shared types and error handling
- `apxm-artifact`: Binary artifact format
- `apxm-driver`: Orchestrates compiler + runtime

### 3. Benchmarks
- **Benchmark Suite**: `/Users/randreshg/Documents/apxm/papers/cf26/benchmarks/workloads/`
- **10 Workloads**: Each has `.ais` (A-PXM) and `.py` (LangGraph) implementations
- **Verification Catalog**: `/Users/randreshg/Documents/apxm/papers/cf26/benchmarks/workloads/3_type_verification/VERIFICATION_CATALOG.md`

### 4. CF'26 Requirements
- **Conference**: ACM Computing Frontiers 2026 (CF'26)
- **Deadline**: Paper submission 19 January 2026 (AoE)
- **Format**: Double-column ACM format, double-blind
- **Page Limit**: 8 pages (excluding references), can buy 2 extra pages
- **Artifact Evaluation**: Strongly encouraged (submission 16 March 2026)
- **Topics**: Hardware/Software frontiers, AI for Systems, Systems for AI

## Paper Claims to Validate

The testing plan identifies 7 key claims (C1-C7). Verify each has:
- **Clear benchmark(s)** that demonstrate the claim
- **Measurable metrics** with target values
- **Statistical rigor** (10+ iterations, confidence intervals)
- **Comparison baseline** (LangGraph where applicable)

### Critical Claims:

**C1: AAM State Model**
- Validate: Beliefs (B), Goals (G), Capabilities (C) are inspectable
- Validate: 3-tier memory (STM/LTM/Episodic) works correctly
- Benchmark: `5_memory_augmented`

**C2: AIS Operations (19 typed operations)**
- Validate: All 19 operations are implemented and type-checked
- Validate: Compile-time verification catches errors
- Benchmark: `3_type_verification` (52+ checks documented)
- Reference: `apxm-ais/src/operations/definitions.rs`

**C3: Scheduler Overhead**
- Target: < 10μs per operation (0.0004% of LLM latency)
- Measured: 7.5μs (already done)
- Benchmark: Runtime microbenchmark (`paper_benchmarks.rs`)

**C4: Automatic Parallelism**
- Target: 4x+ speedup, efficiency > 70% at N=4
- Benchmarks: `1_parallel_research`, `4_scalability`
- Validate: Dataflow extracts parallelism without manual async/await

**C5: FuseReasoning (5x fewer LLM calls)**
- Target: 5x speedup with quality preservation
- Benchmark: `2_chain_fusion`
- Validate: 5 RSN operations → 1 fused call

**C6: Lines of Code (12.6x fewer)**
- Target: >10x reduction on average
- Benchmark: All 10 workloads
- Compare: A-PXM `.ais` vs LangGraph `.py`

**C7: Error Detection (64x faster)**
- Target: >50x faster (compile-time vs runtime)
- Measured: 50ms vs 3200ms (64x)
- Benchmark: `3_type_verification`

## TODO/VERIFY Markers in Paper

The paper contains TODO and VERIFY markers that need data. Check:

1. **Evaluation Section TODOs:**
   - `\TODO{Run Ollama benchmarks with gpt-oss:120b-cloud and gpt-oss:20b-cloud}` → Table `tab/real-llm.tex`
   - `\TODO{Profile compilation pipeline phases}` → Table `tab/compilation-scaling.tex`
   - `\TODO{Run synthetic DAG experiments}` → Parallelism efficiency data
   - `\TODO{Execute materials discovery workflow}` → Multi-agent speedup (2.67x claimed)
   - `\TODO{This achieves 5x latency reduction}` → FuseReasoning results
   - `\TODO{Classification and extraction tasks preserve quality}` → Table `tab/fusion-applicability.tex`
   - `\TODO{A-PXM requires fewer lines of code}` → Table `tab/apxm-vs-langgraph.tex`

2. **VERIFY Markers (need actual data):**
   - `\VERIFY{2.67$\times$}` speedup (abstract, conclusions)
   - `\VERIFY{4$\times$}` speedup at 32 parallel ops
   - `\VERIFY{5$\times$}` FuseReasoning speedup
   - `\VERIFY{12.6$\times$}` fewer lines of code
   - `\VERIFY{64$\times$}` faster error detection
   - All table values in `tab/*.tex` files

## Specific Enhancement Tasks

### 1. Verify Crate Coverage

Ensure the testing plan covers all major crates:

- [ ] **apxm-ais**: Operation definitions, type system, validation
- [ ] **apxm-compiler**: 5-stage pipeline (parse → MLIR → optimize → lower → artifact)
- [ ] **apxm-runtime**: Scheduler, executor, memory tiers, AAM state
- [ ] **apxm-backends**: LLM backends (Ollama), storage backends
- [ ] **apxm-artifact**: Binary format serialization/deserialization
- [ ] **apxm-driver**: End-to-end compile/run workflows

**Action**: Add crate-specific test coverage section if missing.

### 2. Validate AIS Operations

The paper claims "19 typed operations" but code shows 21 total (19 public + 1 metadata + 1 internal).

- [ ] Verify paper text matches implementation
- [ ] Ensure all 19 public operations are tested
- [ ] Document which operations are used in each benchmark

**Reference**: `apxm-ais/src/operations/definitions.rs` shows:
- Memory: QMem, UMem (2)
- Reasoning: Rsn, Plan, Reflect, Verify (4)
- Tools: Inv, Exc (2)
- Control Flow: Jump, BranchOnValue, LoopStart, LoopEnd, Return, Switch, FlowCall (7)
- Synchronization: Merge, Fence, WaitAll (3)
- Error Handling: TryCatch, Err (2)
- Communication: Communicate (1)
- **Total: 19 public operations**

### 3. Complete Benchmark Execution Plan

The testing plan lists 10 benchmarks. Verify:

- [ ] Each benchmark has clear success criteria
- [ ] All benchmarks can be executed with `apxm_runner.py`
- [ ] JSON output format is standardized
- [ ] Comparison with LangGraph is fair and reproducible

**Benchmarks to verify:**
1. `1_parallel_research` - Auto-parallelism
2. `2_chain_fusion` - FuseReasoning optimization
3. `3_type_verification` - Compile-time errors (52+ checks)
4. `4_scalability` - N-way parallelism efficiency
5. `5_memory_augmented` - 3-tier memory
6. `6_tool_invocation` - INV operation
7. `7_reflection` - REFLECT operation
8. `8_planning` - PLAN operation
9. `9_conditional_routing` - Dataflow routing
10. `10_multi_agent` - COMMUNICATE operation

### 4. Statistical Rigor

CF'26 expects rigorous evaluation. Ensure:

- [ ] **Minimum 10 iterations** per measurement (warmup: 3)
- [ ] **Report**: mean, std, p50, min, max, 95% CI
- [ ] **Effect size**: Cohen's d for speedup claims
- [ ] **Variance**: CV < 10% for stable measurements
- [ ] **System configuration**: Document CPU, RAM, LLM backend versions

**Current Status**: Testing plan mentions these but verify they're implemented in benchmark scripts.

### 5. Artifact Evaluation Readiness

CF'26 strongly encourages artifact evaluation. Ensure:

- [ ] **Reproducibility**: All benchmarks can be run from scratch
- [ ] **Documentation**: Clear setup instructions
- [ ] **Dependencies**: All requirements listed (Ollama, Rust, Python packages)
- [ ] **Data**: Results can be regenerated
- [ ] **Code**: All benchmark code is included

**Reference**: CF'26 artifact evaluation uses ACM Artifact Review and Badging (Version 1.1).

### 6. Table and Figure Data

The paper has 7+ tables and 6+ figures. Verify:

- [ ] All table values have corresponding benchmarks
- [ ] All figures can be generated from collected data
- [ ] Placeholder values (`\VERIFY{}`) are replaced with actual measurements
- [ ] Data collection scripts exist for each table/figure

**Tables to populate:**
- `tab/aam-inspection.tex` - AAM state sample
- `tab/type-errors.tex` - Compile-time error examples
- `tab/runtime-overhead.tex` - Scheduler overhead breakdown
- `tab/memory-ops.tex` - Memory tier latencies
- `tab/real-llm.tex` - Ollama LLM latencies
- `tab/compilation-scaling.tex` - Compilation time scaling
- `tab/fusion-applicability.tex` - FuseReasoning by task type
- `tab/apxm-vs-langgraph.tex` - Head-to-head comparison

**Figures to generate:**
- `fig/dataflow-extraction.tex` - Parallelism extraction
- `fig/fuse-reasoning-demo.tex` - FuseReasoning transformation
- `fig/speedup-plot.tex` - Parallelism efficiency curve
- `fig/latency-breakdown.tex` - Overhead breakdown
- (Others as needed)

### 7. Consistency Checks

Verify consistency across:

- [ ] **Paper text** vs **Testing plan claims** vs **Implementation**
- [ ] **Abstract claims** match **evaluation results**
- [ ] **Figure captions** match **actual data**
- [ ] **Table values** match **benchmark results**
- [ ] **Operation counts** (19 vs 21) are explained correctly

### 8. CF'26 Topic Alignment

Ensure evaluation addresses CF'26 topics:

- [x] **AI for Systems**: A-PXM optimizes agent execution
- [x] **Systems for AI**: Formal substrate for agentic AI
- [x] **Compilers**: MLIR-based compilation pipeline
- [x] **Runtime Environments**: Dataflow execution engine
- [ ] **Energy Efficiency**: (Not currently measured - consider adding)
- [ ] **Scalability**: (Partially covered - expand if needed)

## Output Format

Provide an enhanced testing plan that:

1. **Maintains existing structure** (Research-Informed Methodology, Claims, Experiments, etc.)
2. **Adds missing coverage** for crates, operations, or benchmarks
3. **Clarifies execution steps** with specific commands and expected outputs
4. **Adds validation checkpoints** to ensure data quality
5. **Includes artifact preparation** checklist for CF'26 submission
6. **Documents any discrepancies** between paper claims and implementation
7. **Provides timeline** for completing remaining measurements

## Success Criteria

The enhanced plan should enable:

- ✅ Complete execution of all benchmarks
- ✅ Generation of all tables and figures
- ✅ Resolution of all TODO/VERIFY markers
- ✅ Artifact evaluation submission readiness
- ✅ Paper acceptance at CF'26

## Questions to Answer

While enhancing the plan, address:

1. Are all 19 AIS operations covered by benchmarks?
2. Can all paper claims be validated with the current benchmark suite?
3. Are there gaps in coverage (e.g., error handling, multi-agent coordination)?
4. Is the statistical methodology sufficient for CF'26 standards?
5. Are there any inconsistencies between paper text and implementation?
6. What measurements are still pending, and what's the plan to complete them?

## Additional Resources

- **CF'26 Website**: https://www.computingfrontiers.org/2026/
- **ACM Artifact Evaluation Guide**: https://ctuning.org/ae/checklist.html
- **MLIR Documentation**: (for compiler understanding)
- **LangGraph Documentation**: (for baseline comparison)

---

**Start by reading the testing plan, then systematically work through each section above to enhance it.**
