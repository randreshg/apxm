# AIS DSL Front End

This folder documents the AIS DSL front end in the MLIR compiler.

## Responsibilities

- Lex and parse AIS DSL into an AST
- Lower the AST into AIS MLIR via MLIRGen

## Flow

```
AIS DSL → Lexer → Parser → AST → MLIRGen → AIS MLIR Module
```

## Key Entry Points

- DSL C API: `crates/apxm-compiler/mlir/lib/CAPI/DSL.cpp`
- DSL parse hook: `apxm_parse_dsl*` (FFI entry used by `Module::parse_dsl`)

## Components

- **Lexer**: tokenizes DSL source
  - `crates/apxm-compiler/mlir/include/ais/Parser/Lexer/Lexer.h`
  - `crates/apxm-compiler/mlir/include/ais/Parser/Utils/Token.h`
- **Parser**: recursive-descent parsing into AST nodes
  - `crates/apxm-compiler/mlir/lib/Parser/Parsers/`
  - AST types: `crates/apxm-compiler/mlir/include/ais/Parser/AST/`
- **MLIRGen**: lowers AST into the AIS MLIR dialect
  - `crates/apxm-compiler/mlir/lib/Parser/MLIR/MLIRGen*.cpp`
  - Header: `crates/apxm-compiler/mlir/include/ais/Parser/MLIR/MLIRGen.h`

## How It Fits

- Operation metadata is defined in Rust (`apxm-ais`), exported to TableGen,
  and used by the MLIR dialect and runtime validators.
- The front end produces MLIR modules that then flow through the pass pipeline
  and code generation to artifacts or Rust output.
