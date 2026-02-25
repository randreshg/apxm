# AIS DSL Front End

This folder documents the AIS DSL front end in the MLIR compiler.

## Responsibilities

- Lex and parse AIS DSL into an AST
- Lower the AST into canonical `ApxmGraph` JSON
- Lower canonical graph into AIS MLIR via `apxm-graph`

## Flow

```
AIS DSL → Lexer → Parser → AST → GraphGen → ApxmGraph → AIS MLIR Module
```

## Key Entry Points

- DSL C API: `crates/apxm-compiler/mlir/lib/CAPI/DSL.cpp`
- DSL graph hook: `apxm_parse_dsl*_to_graph_json` (FFI entry used by `Module::parse_dsl`)

## Components

- **Lexer**: tokenizes DSL source
  - `crates/apxm-compiler/mlir/include/ais/Parser/Lexer/Lexer.h`
  - `crates/apxm-compiler/mlir/include/ais/Parser/Utils/Token.h`
- **Parser**: recursive-descent parsing into AST nodes
  - `crates/apxm-compiler/mlir/lib/Parser/Parsers/`
  - AST types: `crates/apxm-compiler/mlir/include/ais/Parser/AST/`
- **GraphGen**: lowers AST into canonical `ApxmGraph` JSON
  - `crates/apxm-compiler/mlir/lib/Parser/Graph/GraphGen.cpp`
  - Header: `crates/apxm-compiler/mlir/include/ais/Parser/Graph/GraphGen.h`
- **MLIRGen**: legacy direct AST-to-MLIR path kept for C API compatibility

## How It Fits

- Operation metadata is defined in Rust (`apxm-ais`), exported to TableGen,
  and used by the MLIR dialect and runtime validators.
- The frontend canonical form is always `ApxmGraph`; MLIR lowering and pass
  optimization run after graph normalization.
