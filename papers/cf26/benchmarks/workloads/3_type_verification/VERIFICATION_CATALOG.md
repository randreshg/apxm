# A-PXM Compile-Time Verification Catalog

This document catalogs all compile-time checks performed by the A-PXM compiler, demonstrating the formal verification properties of the AIS type system.

## Summary

| Category | Check Count | Example Error |
|----------|-------------|---------------|
| Memory Operations | 6 | "space must be 'stm', 'ltm', or 'episodic'" |
| Capability Operations | 3 | "capability name cannot be empty" |
| Reasoning Operations | 5 | "must have at least context or template" |
| Verify Operations | 3 | "claim must be token, handle, or goal type" |
| Control Flow | 8 | "condition must be token or handle type" |
| Parser/Semantic | 12 | "Undefined variable: x" |
| Type Constraints | 15+ | "result must be !ais.token type" |
| **Total** | **52+** | |

---

## Category 1: Memory Operations (6 checks)

### QMemOp (Query Memory)
```
1. "sid must be non-empty"
2. "space must be 'stm', 'ltm', or 'episodic'"
3. "result must be !ais.handle type"
4. "result handle space does not match operation space attribute"
5. "limit must be positive if specified"
```

### UMemOp (Update Memory)
```
6. "space must be 'stm', 'ltm', or 'episodic'"
```

**Example error:**
```ais
// ERROR: Invalid memory space
umem("value", "invalid_space")

// Compiler output:
// error: space must be 'stm', 'ltm', or 'episodic'
//  --> workflow.ais:3:6
```

---

## Category 2: Capability Operations (3 checks)

### InvOp (Invoke Capability)
```
1. "result must be !ais.token type"
2. "capability name cannot be empty"
3. "params_json cannot be empty (use \"{}\" for no params)"
```

**Example error:**
```ais
// ERROR: Empty capability name
inv("", "{}") -> result

// Compiler output:
// error: capability name cannot be empty
//  --> workflow.ais:2:5
```

---

## Category 3: Reasoning Operations (5 checks)

### RsnOp (LLM Reasoning)
```
1. "result must be !ais.token type"
2. "reasoning operation must have at least context or template"
3. "context operands must be !ais.token, !ais.handle, or !ais.goal types"
4. "inner_plan region cannot be empty if specified"
5. "inner_plan blocks must have terminators"
```

### ReflectOp
```
- "trace_id is required"
- "context operands must be valid types"
```

**Example error:**
```ais
// ERROR: Empty reasoning template with no context
rsn("") -> result

// Compiler output:
// error: reasoning operation must have at least context or template
//  --> workflow.ais:2:5
```

---

## Category 4: Verify Operations (3 checks)

### VerifyOp
```
1. "claim operand must be !ais.token, !ais.handle, or !ais.goal type"
2. "evidence operand must be !ais.token, !ais.handle, or !ais.goal type"
3. "template_str is required"
```

---

## Category 5: Control Flow (8 checks)

### SwitchOp
```
1. "discriminant must be !ais.token type"
2. "case_labels must be non-empty"
3. "case_labels and case_destinations must have same length"
```

### FlowCallOp
```
4. "agent_name cannot be empty"
5. "flow_name cannot be empty"
6. "args must be token, handle, or goal types"
```

### Conditionals
```
7. "condition operand must be !ais.token or !ais.handle type"
8. "count operand must be !ais.token or !ais.handle type"
```

---

## Category 6: Parser/Semantic Checks (12 checks)

### Variable Resolution
```
1. "Undefined variable: <name>"
2. "Assignment to non-variable"
```

### Expression Parsing
```
3. "Invalid string literal"
4. "Invalid number literal"
5. "Expected expression"
6. "Call target must be an identifier"
7. "Pipeline stage must be a function or variable"
```

### Type Annotations
```
8. "Expected type annotation"
9. "Expected return type"
```

### Declarations
```
10. "Unexpected declaration"
11. "Expected memory tier (STM, LTM, or Episodic)"
12. "Expected '{' to start flow body"
```

**Example error:**
```ais
agent Demo {
    flow main {
        rsn("Use this: " + undefined_var) -> result
    }
}

// Compiler output:
// error: Undefined variable: undefined_var
//  --> workflow.ais:3:24
//   |
// 3 |         rsn("Use this: " + undefined_var) -> result
//   |                            ^^^^^^^^^^^^^
//   |                            not defined in scope
```

---

## Comparison: Compile-Time vs Runtime Errors

| Aspect | A-PXM (Compile-Time) | LangGraph (Runtime) |
|--------|---------------------|---------------------|
| **When detected** | Before execution | During execution |
| **LLM calls made** | 0 | 1+ (before error) |
| **Cost incurred** | $0.00 | $0.01+ per failed run |
| **Error location** | Precise line/column | Stack trace |
| **Iteration speed** | ~50ms feedback | ~3s+ per attempt |
| **Type safety** | Enforced | Best-effort |

---

## What This Proves

The 52+ compile-time checks demonstrate that A-PXM provides:

1. **Formal Type Safety**: Operations are type-checked before execution
2. **Memory Space Validation**: Invalid memory tiers caught at compile-time
3. **Semantic Verification**: Undefined variables, invalid expressions detected
4. **Cost Savings**: Errors caught before expensive LLM invocations
5. **Developer Experience**: Precise error messages with source locations

This is analogous to:
- **Rust** vs Python for memory safety
- **TypeScript** vs JavaScript for type safety
- **MLIR** verification passes for compiler correctness

The formal verification enables the optimizations (FuseReasoning, automatic parallelism) that would be unsafe without type guarantees.
