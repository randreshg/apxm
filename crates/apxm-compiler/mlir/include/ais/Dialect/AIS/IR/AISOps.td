/**
 * @file  AISOps.td
 * @brief Operation definitions for the Agent Instruction Set (AIS) dialect.
 *
 * This file declares every AIS operation, grouped by functional area:
 *   - Agent metadata
 *   - Memory (qmem/umem)
 *   - Capabilities (inv)
 *   - LLM Operations (ask/think/reason - three distinct ops for analysis)
 *   - Reasoning (reflect, verify)
 *   - Planning (plan)
 *   - Execution (exc)
 *   - Synchronization (wait_all, merge, fence)
 *   - Control flow (jump, branch_on_value, loop_*, try_catch, return)
 *   - Error handling (err)
 *   - Communication (communicate)
 *
 * LLM Operations Architecture:
 *   Three separate ops (ask/think/reason) enable compile-time analysis:
 *   - ask:    LOW latency marker  - simple Q&A
 *   - think:  HIGH latency marker - extended thinking
 *   - reason: MEDIUM latency marker - structured reasoning
 *
 *   At artifact emission, these lower to runtime markers with config lookup.
 *   Runtime config (passed at execution) maps markers to LLM parameters.
 *
 * All operations inherit from `AIS_Op<>` defined in AISDialect.td and
 * use the resource-based side-effect interface introduced in AISOps.h.
 * That allows fine-grained dependence analysis without falling back to
 * the generic "memory" effect.
 *
 * Canonicalisers and folders are declared only where they are already
 * implemented; other operations carry `hasVerifier = 1` so that
 * invariants can be checked early.
 */

#ifndef APXM_AIS_OPS
#define APXM_AIS_OPS

include "ais/Dialect/AIS/IR/AISDialect.td"
include "ais/Dialect/AIS/IR/AISTypes.td"
include "mlir/IR/CommonTypeConstraints.td"
include "mlir/Interfaces/ControlFlowInterfaces.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

//===----------------------------------------------------------------------===//
// Agent Metadata
//===----------------------------------------------------------------------===//

def AIS_AgentOp : AIS_Op<"agent", [HasParent<"mlir::ModuleOp">]> {
  let summary = "Declarative agent metadata";
  let description = [{
    Captures the structural declarations associated with an agent (memory
    stores, beliefs, goals, capabilities). This keeps the DSL information
    available to later compiler stages and the runtime even though flows are
    lowered to `func.func`.
  }];
  let arguments = (ins StrAttr:$name);
  let assemblyFormat = "$name attr-dict";
}

//===----------------------------------------------------------------------===//
// Memory Operations
//===----------------------------------------------------------------------===//

def AIS_ConstStrOp : AIS_Op<"const_str", [Pure]> {
  let summary = "String constant";
  let description = [{
    Defines a string constant value that can be used as input to other operations.
    The string value is stored as an attribute and a token handle is returned.

    Example:
    ```mlir
    %str = ais.const_str "Hello, world!" : !ais.token<i64>
    ```
  }];
  let arguments = (ins StrAttr:$value);
  let results = (outs AIS_TokenType:$result);
  let assemblyFormat = "$value attr-dict `:` type($result)";

  // Enable constant folding and canonicalization for string deduplication
  let hasFolder = 1;
}

def AIS_QMemOp : AIS_Op<"qmem",
    [MemoryEffects<[MemRead<AIS_BeliefResource>]>]> {
  let summary = "Query memory (AAM)";
  let description = [{
    Queries the Agent Abstract Machine memory at the specified tier
    and returns a handle to the retrieved content.

    Example:
    ```mlir
    %handle = ais.qmem "find facts about topic X" stage "belief_db" in "ltm" : !ais.handle
    ```
  }];
  let arguments = (ins
    StrAttr:$query,
    StrAttr:$sid,
    StrAttr:$space,
    OptionalAttr<I64Attr>:$limit
  );
  let results = (outs AIS_HandleType:$result);
  let assemblyFormat = [{
    $query `stage` $sid `in` $space (`limit` $limit^)? attr-dict `:` type($result)
  }];
  let hasVerifier = 1;
}

def AIS_UMemOp : AIS_Op<"umem",
    [MemoryEffects<[MemWrite<AIS_BeliefResource>]>]> {
  let summary = "Update memory (AAM)";
  let description = [{
    Updates the Agent Abstract Machine memory with a new value.

    Example:
    ```mlir
    ais.umem %token into "ltm" : !ais.token
    ```
  }];
  let arguments = (ins
    AIS_TokenType:$value,
    StrAttr:$space
  );
  let assemblyFormat = [{
    $value `into` $space attr-dict `:` type($value)
  }];
  let hasVerifier = 1;
}

//===----------------------------------------------------------------------===//
// Capability Operations
//===----------------------------------------------------------------------===//

def AIS_InvOp : AIS_Op<"inv",
    [MemoryEffects<[MemRead<AIS_CapabilityResource>]>]> {
  let summary = "Invoke capability (tool/API)";
  let description = [{
    Invokes a registered capability (tool, API, or function) with parameters.

    Example:
    ```mlir
    %result = ais.inv "web_search" ("query": "MLIR compiler") : !ais.token
    ```
  }];
  let arguments = (ins
    StrAttr:$capability,
    StrAttr:$params_json
  );
  let results = (outs AIS_TokenType:$result);
  let assemblyFormat = [{
    $capability `(` $params_json `)` attr-dict `:` type($result)
  }];
  let hasVerifier = 1;
}

//===----------------------------------------------------------------------===//
// LLM Operations (Three Distinct Ops for Critical Path Analysis)
//===----------------------------------------------------------------------===//
//
// These operations enable compile-time analysis:
// - Critical path detection (think ops have HIGH latency)
// - Cost estimation (each op has distinct latency profile)
// - Optimization passes (fuse adjacent ask ops, batch think ops)
// - Trait-based analysis (different memory effects per op type)
//
// At artifact emission, these lower to runtime markers with config lookup.
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// AskOp - Simple Q&A LLM Call (LOW Latency)
//===----------------------------------------------------------------------===//

def AIS_AskOp : AIS_Op<"ask",
    [MemoryEffects<[MemRead<AIS_BeliefResource>]>]> {
  let summary = "Simple Q&A with LLM (no extended thinking)";
  let description = [{
    Performs a quick LLM call for simple questions. No extended thinking,
    returns plain text. Lowest latency of the LLM operations.

    At compile time: Enables critical path analysis (LOW latency marker).
    At runtime: Looks up "ask" config for model/temperature/etc.

    Example:
    ```mlir
    %result = ais.ask "What is 2+2?" : !ais.token
    %result = ais.ask "Summarize: {{context}}" [%ctx : !ais.token] : !ais.token
    ```
  }];
  let arguments = (ins
    StrAttr:$template_str,
    Variadic<AIS_AnyTokenHandleOrGoal>:$context
  );
  let results = (outs AIS_TokenType:$result);
  let assemblyFormat = [{
    $template_str (`[` $context^ `:` type($context) `]`)? attr-dict `:` type($result)
  }];
  let hasVerifier = 1;
}

//===----------------------------------------------------------------------===//
// ThinkOp - Extended Thinking LLM Call (HIGH Latency)
//===----------------------------------------------------------------------===//

def AIS_ThinkOp : AIS_Op<"think",
    [MemoryEffects<[MemRead<AIS_BeliefResource>, MemWrite<AIS_BeliefResource>]>]> {
  let summary = "Extended thinking with budget_tokens";
  let description = [{
    Enables Claude's extended thinking mode with configurable budget.
    High latency but deep reasoning capabilities. Use for complex problems.

    At compile time: Enables critical path analysis (HIGH latency marker).
    At runtime: Looks up "think" config and enables extended thinking.

    Example:
    ```mlir
    %result = ais.think "Solve this complex problem" : !ais.token
    %result = ais.think "Analyze deeply: {{data}}" [%data : !ais.token] : !ais.token
    ```
  }];
  let arguments = (ins
    StrAttr:$template_str,
    Variadic<AIS_AnyTokenHandleOrGoal>:$context
  );
  let results = (outs AIS_TokenType:$result);
  let assemblyFormat = [{
    $template_str (`[` $context^ `:` type($context) `]`)? attr-dict `:` type($result)
  }];
  let hasVerifier = 1;
}

//===----------------------------------------------------------------------===//
// ReasonOp - Structured Reasoning LLM Call (MEDIUM Latency)
//===----------------------------------------------------------------------===//

def AIS_ReasonOp : AIS_Op<"reason",
    [MemoryEffects<[MemRead<AIS_BeliefResource>, MemWrite<AIS_BeliefResource>,
                    MemWrite<AIS_GoalResource>]>]> {
  let summary = "Structured reasoning with belief/goal updates";
  let description = [{
    Full structured reasoning that can update beliefs and goals.
    Returns structured output (JSON) with optional inner_plan support.
    Medium latency, suitable for analysis tasks.

    At compile time: Enables critical path analysis (MEDIUM latency marker).
    At runtime: Looks up "reason" config, enables structured output parsing.

    Example:
    ```mlir
    %result = ais.reason "Analyze and plan: {{context}}" [%ctx : !ais.token] : !ais.token
    ```
  }];
  let arguments = (ins
    StrAttr:$template_str,
    Variadic<AIS_AnyTokenHandleOrGoal>:$context
  );
  let results = (outs AIS_TokenType:$result);
  let assemblyFormat = [{
    $template_str (`[` $context^ `:` type($context) `]`)? attr-dict `:` type($result)
  }];
  let hasVerifier = 1;
  let hasCanonicalizer = 1;
}

def AIS_ReflectOp : AIS_Op<"reflect",
    [MemoryEffects<[MemRead<AIS_EpisodicResource>, MemWrite<AIS_BeliefResource>]>]> {
  let summary = "Reflect over episodic traces";
  let description = [{
    Analyzes past execution traces to extract insights or patterns.

    Example:
    ```mlir
    %insight = ais.reflect "trace_123" : !ais.token
    ```
  }];
  let arguments = (ins
    StrAttr:$trace_id,
    Variadic<AIS_AnyTokenHandleOrGoal>:$context
  );
  let results = (outs AIS_TokenType:$result);
  let assemblyFormat = [{
    $trace_id (`(` $context^ `:` type($context) `)`)? attr-dict `:` type($result)
  }];
  let hasVerifier = 1;
}

def AIS_VerifyOp : AIS_Op<"verify",
    [MemoryEffects<[MemRead<AIS_BeliefResource>]>]> {
  let summary = "Verify claim against evidence";
  let description = [{
    Verifies a claim using provided evidence and a verification template.

    Example:
    ```mlir
    %verified = ais.verify %claim : !ais.token vs %evidence : !ais.handle
                with "Check if claim is supported" : !ais.token
    ```
  }];
  let arguments = (ins
    AIS_AnyTokenHandleOrGoal:$claim,
    AIS_AnyTokenHandleOrGoal:$evidence,
    StrAttr:$template_str
  );
  let results = (outs AIS_TokenType:$result);
  let assemblyFormat = [{
    $claim `:` type($claim) `vs` $evidence `:` type($evidence)
    `with` $template_str attr-dict `:` type($result)
  }];
  let hasVerifier = 1;
}

//===----------------------------------------------------------------------===//
// Planning Operations
//===----------------------------------------------------------------------===//

def AIS_PlanOp : AIS_Op<"plan",
    [MemoryEffects<[MemWrite<AIS_GoalResource>]>]> {
  let summary = "Plan new subgraph";
  let description = [{
    Creates a planning goal that can be refined into executable operations.

    Example:
    ```mlir
    %goal = ais.plan "Solve task X" (%ctx : !ais.token) : !ais.goal
    ```
  }];
  let arguments = (ins
    StrAttr:$goal,
    Variadic<AIS_AnyTokenHandleOrGoal>:$context
  );
  let results = (outs AIS_GoalType:$result);
  let assemblyFormat = [{
    $goal (`(` $context^ `:` type($context) `)`)? attr-dict `:` type($result)
  }];
  let hasVerifier = 1;
}

//===----------------------------------------------------------------------===//
// Execution Operations
//===----------------------------------------------------------------------===//

def AIS_ExcOp : AIS_Op<"exc"> {
  let summary = "Execute sandboxed code";
  let description = [{
    Executes code in a sandboxed environment with provided context.

    Example:
    ```mlir
    %result = ais.exc "print('hello')" : !ais.token
    ```
  }];
  let arguments = (ins
    StrAttr:$code,
    Variadic<AIS_AnyTokenHandleOrGoal>:$context
  );
  let results = (outs AIS_TokenType:$result);
  let assemblyFormat = [{
    $code (`(` $context^ `:` type($context) `)`)? attr-dict `:` type($result)
  }];
  let hasVerifier = 1;
}

def AIS_PrintOp : AIS_Op<"print",
    [MemoryEffects<[MemWrite<AIS_EpisodicResource>]>]> {
  let summary = "Print output to stdout";
  let description = [{
    Outputs a message to standard output with optional context values.
    This is a side-effect operation with no return value.

    Example:
    ```mlir
    ais.print "Debug: value is" [%data : !ais.token]
    ```
  }];
  let arguments = (ins
    StrAttr:$message,
    Variadic<AIS_AnyTokenHandleOrGoal>:$context
  );
  let assemblyFormat = [{
    $message (`[` $context^ `:` type($context) `]`)? attr-dict
  }];
}

//===----------------------------------------------------------------------===//
// Synchronization Operations
//===----------------------------------------------------------------------===//

def AIS_WaitAllOp : AIS_Op<"wait_all", [Pure]> {
  let summary = "Barrier waiting for tokens";
  let description = [{
    Waits for all input tokens to be ready before producing the result.

    Example:
    ```mlir
    %synced = ais.wait_all %tok1, %tok2, %tok3 : !ais.token
    ```
  }];
  let arguments = (ins Variadic<AIS_TokenType>:$tokens);
  let results = (outs AIS_TokenType:$result);
  let assemblyFormat = [{
    ($tokens^ `:` type($tokens))? attr-dict `->` type($result)
  }];
  let hasVerifier = 1;
  let hasCanonicalizer = 1;
}

def AIS_MergeOp : AIS_Op<"merge", [Pure]> {
  let summary = "Merge tokens";
  let description = [{
    Merges multiple tokens into a single token.

    Example:
    ```mlir
    %merged = ais.merge %tok1, %tok2 : !ais.token
    ```
  }];
  let arguments = (ins Variadic<AIS_TokenType>:$tokens);
  let results = (outs AIS_TokenType:$result);
  let assemblyFormat = [{
    ($tokens^ `:` type($tokens))? attr-dict `->` type($result)
  }];
  let hasVerifier = 1;
  let hasCanonicalizer = 1;
}

def AIS_FenceOp : AIS_Op<"fence"> {
  let summary = "Memory fence";
  let description = [{
    Ensures memory operations are ordered across the fence.
  }];
  let hasVerifier = 1;
}

//===----------------------------------------------------------------------===//
// Control Flow Operations
//===----------------------------------------------------------------------===//

def AIS_JumpOp : AIS_Op<"jump"> {
  let summary = "Unconditional jump";
  let arguments = (ins StrAttr:$label);
  let assemblyFormat = "$label attr-dict";
  let hasVerifier = 1;
}

def AIS_BranchOnValueOp : AIS_Op<"branch_on_value"> {
  let summary = "Conditional branch";
  let arguments = (ins
    AIS_AnyTokenOrHandle:$condition,
    StrAttr:$true_label,
    StrAttr:$false_label
  );
  let assemblyFormat = [{
    $condition `,` $true_label `,` $false_label attr-dict `:` type($condition)
  }];
  let hasVerifier = 1;
}

def AIS_LoopStartOp : AIS_Op<"loop_start"> {
  let summary = "Begin bounded loop";
  let arguments = (ins
    AIS_AnyTokenOrHandle:$count,
    StrAttr:$label
  );
  let results = (outs AIS_TokenType:$state);
  let assemblyFormat = [{
    $count `as` $label attr-dict `:` type($count) `->` type($state)
  }];
  let hasVerifier = 1;
}

def AIS_LoopEndOp : AIS_Op<"loop_end"> {
  let summary = "End bounded loop";
  let arguments = (ins AIS_TokenType:$state);
  let results = (outs AIS_TokenType:$result);
  let assemblyFormat = [{
    $state attr-dict `:` type($state) `->` type($result)
  }];
  let hasVerifier = 1;
}

def AIS_TryCatchOp : AIS_Op<"try_catch"> {
  let summary = "Try / catch region markers";
  let arguments = (ins
    StrAttr:$try_label,
    StrAttr:$catch_label
  );
  let assemblyFormat = "$try_label `->` $catch_label attr-dict";
  let hasVerifier = 1;
}

def AIS_YieldOp : AIS_Op<"yield", [Pure, Terminator]> {
  let summary = "Yield value from switch case region";
  let description = [{
    Terminates a switch case region and optionally yields a value as the result
    of the switch operation when this case is selected. If no value is provided,
    the switch produces no result (void).

    Example:
    ```mlir
    // With value
    ais.yield %result : !ais.token

    // Without value (void)
    ais.yield
    ```
  }];
  let arguments = (ins Optional<AIS_TokenType>:$value);
  let assemblyFormat = "($value^ `:` type($value))? attr-dict";
}

def AIS_SwitchOp : AIS_Op<"switch", [
    RecursiveMemoryEffects,
    SingleBlockImplicitTerminator<"YieldOp">
]> {
  let summary = "Multi-way branch with case regions";
  let description = [{
    Evaluates discriminant and executes exactly one case region based on
    matching the discriminant value against case labels. Returns the value
    yielded by the executed region (if any). Only the matching branch runs.

    The regions array contains N+1 regions: N case regions followed by the
    default region (always the last entry). The case_labels array has N entries
    corresponding to the first N regions.

    Example with result:
    ```mlir
    %result = ais.switch %category : !ais.token
        case "technical" {
            %t = ais.rsn "Technical explanation..." : !ais.token
            ais.yield %t : !ais.token
        }
        case "creative" {
            %t = ais.rsn "Creative response..." : !ais.token
            ais.yield %t : !ais.token
        }
        default {
            %t = ais.rsn "Default response..." : !ais.token
            ais.yield %t : !ais.token
        } -> !ais.token
    ```

    Example without result (void):
    ```mlir
    ais.switch %action : !ais.token
        case "log" {
            ais.umem %data into "ltm" : !ais.token
            ais.yield
        }
        default {
            ais.yield
        }
    ```
  }];
  let arguments = (ins
    AIS_TokenType:$discriminant,
    ArrayAttr:$case_labels
  );
  // N+1 regions: N case regions + 1 default region (last)
  // Variadic must be the last (and only) region field per TableGen constraints
  let regions = (region
    VariadicRegion<SizedRegion<1>>:$regions
  );
  let results = (outs Optional<AIS_TokenType>:$result);
  let hasVerifier = 1;
  let hasCustomAssemblyFormat = 1;

  let extraClassDeclaration = [{
    /// Get the number of case regions (excludes default)
    size_t getNumCases() { return getCaseLabels().size(); }

    /// Get case regions (all regions except the last one)
    auto getCaseRegions() {
      auto allRegions = getRegions();
      return llvm::make_range(allRegions.begin(), allRegions.end() - 1);
    }

    /// Get the default region (the last region)
    mlir::Region &getDefaultRegion() { return getRegions().back(); }
  }];
}

def AIS_FlowCallOp : AIS_Op<"flow_call",
    [MemoryEffects<[MemRead<AIS_BeliefResource>, MemWrite<AIS_BeliefResource>]>]> {
  let summary = "Call a flow on another agent";
  let description = [{
    Invokes a flow on a specified agent, passing arguments and receiving a result.
    This enables cross-agent communication and coordination with implicit parallelism.

    Example:
    ```mlir
    %result = ais.flow_call "Researcher" "research" (%topic : !ais.token) : !ais.token
    ```
  }];
  let arguments = (ins
    StrAttr:$agent_name,
    StrAttr:$flow_name,
    Variadic<AIS_AnyTokenHandleOrGoal>:$args
  );
  let results = (outs AIS_TokenType:$result);
  let assemblyFormat = [{
    $agent_name $flow_name (`(` $args^ `:` type($args) `)`)? attr-dict `:` type($result)
  }];
  let hasVerifier = 1;
}

def AIS_ReturnOp : AIS_Op<"return", [Terminator]> {
  let summary = "Return from subgraph";
  let arguments = (ins AIS_TokenType:$value);
  let assemblyFormat = "$value attr-dict `:` type($value)";
  let hasVerifier = 1;
}

//===----------------------------------------------------------------------===//
// Error Handling Operations
//===----------------------------------------------------------------------===//

def AIS_ErrOp : AIS_Op<"err"> {
  let summary = "Error handling";
  let description = [{
    Handles errors with optional recovery logic.

    Example:
    ```mlir
    %recovered = ais.err %failed with "Retry with fallback" : !ais.token
    ```
  }];
  let arguments = (ins
    Optional<AIS_TokenType>:$input,
    StrAttr:$recovery_template
  );
  let results = (outs AIS_TokenType:$result);
  let assemblyFormat = [{
    ($input^ `:` type($input))? `with` $recovery_template attr-dict
      `->` type($result)
  }];
  let hasVerifier = 1;
}

//===----------------------------------------------------------------------===//
// Communication Operations
//===----------------------------------------------------------------------===//

def AIS_CommunicateOp : AIS_Op<"communicate"> {
  let summary = "Multi-agent communication";
  let description = [{
    Sends a message to another agent with optional attachments.

    Example:
    ```mlir
    %ack = ais.communicate "agent_bob" (%data : !ais.token) : !ais.token
    ```
  }];
  let arguments = (ins
    StrAttr:$recipient,
    OptionalAttr<StrAttr>:$protocol,
    OptionalAttr<StrAttr>:$payload,
    Variadic<AIS_AnyTokenHandleOrGoal>:$attachments
  );
  let results = (outs AIS_TokenType:$result);
  let assemblyFormat = [{
    $recipient (`(` $attachments^ `:` type($attachments) `)`)? attr-dict `->` type($result)
  }];
  let hasVerifier = 1;
}

#endif // APXM_AIS_OPS
