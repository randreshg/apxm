# Compile-Time Error Detection Examples

This directory contains intentionally invalid `.ais` files that demonstrate A-PXM's compile-time error detection capabilities.

## Why Compile-Time Errors Matter

Unlike Python-based agent frameworks (LangChain, CrewAI, AutoGen) that crash at runtime when errors occur, A-PXM's MLIR-based compiler catches many errors **before execution**:

| Error Type | Python Frameworks | A-PXM |
|------------|-------------------|-------|
| Type mismatches | Runtime crash | **Compile-time error** |
| Undefined variables | Runtime crash | **Compile-time error** |
| Invalid memory spaces | Runtime crash | **Compile-time error** |
| Missing capabilities | Runtime crash | **Compile-time error** |

## Examples

### 1. `invalid_type.ais`
Demonstrates type mismatch detection. Passing a string where a number is expected.

```
Error: Type mismatch at line 7: expected 'number', got 'string'
```

### 2. `undefined_variable.ais`
Demonstrates undefined variable detection. Using a variable that was never defined.

```
Error: Undefined variable 'unknown_var' at line 6
```

### 3. `invalid_memory_space.ais`
Demonstrates invalid memory space detection. Using a memory space that doesn't exist.

```
Error: Invalid memory space 'invalid_space' at line 5. Valid spaces: stm, ltm, episodic
```

### 4. `missing_capability.ais`
Demonstrates missing capability detection. Invoking a tool that isn't registered.

```
Error: Undefined capability 'unregistered_tool' at line 6
```

## Running the Examples

To see the compile errors:

```bash
# Each of these should produce a compile-time error
cargo run -p apxm-cli -- compile examples/error_handling/invalid_type.ais
cargo run -p apxm-cli -- compile examples/error_handling/undefined_variable.ais
cargo run -p apxm-cli -- compile examples/error_handling/invalid_memory_space.ais
cargo run -p apxm-cli -- compile examples/error_handling/missing_capability.ais
```

## Benefits

A-PXM's Agent Instruction Set (AIS) provides static verification through MLIR's type system. Errors that would crash Python frameworks at runtime are caught during compilation, enabling fail-fast development and safer agent deployments.
