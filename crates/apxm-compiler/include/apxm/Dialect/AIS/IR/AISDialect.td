//===- AISDialect.td - AIS Dialect Definition --------------*- tablegen -*-===//
//
// Part of the A-PXM project, under the Apache License v2.0.
// See LICENSE for license information.
//
//===----------------------------------------------------------------------===//
//
// This file defines the AIS (Agent Instruction Set) dialect.
//
//===----------------------------------------------------------------------===//

#ifndef APXM_AIS_DIALECT
#define APXM_AIS_DIALECT

include "mlir/IR/OpBase.td"

//===----------------------------------------------------------------------===//
// AIS Dialect Definition
//===----------------------------------------------------------------------===//

def AIS_Dialect : Dialect {
  let name = "ais";
  let cppNamespace = "::mlir::ais";
  let summary = "Agent Instruction Set (AIS) dialect for A-PXM";
  let description = [{
    The AIS dialect provides operations for agent programming with explicit
    state management and data-driven execution.
    
    Key features:
    - Token-based dataflow for dependency tracking
    - Memory operations (query/update) for AAM (Agent Abstract Machine)
    - LLM integration via reasoning operations
    - Sandboxed code execution
    - Multi-agent communication primitives
  }];

  let extraClassDeclaration = [{
    /// Type parsing and printing for custom AIS types
    ::mlir::Type parseType(::mlir::DialectAsmParser &parser) const override;
    void printType(::mlir::Type type, ::mlir::DialectAsmPrinter &printer) const override;
  }];
  
  let useDefaultTypePrinterParser = 0;
}

/// Base class for AIS operations
class AIS_Op<string mnemonic, list<Trait> traits = []> :
    Op<AIS_Dialect, mnemonic, traits>;

#endif // APXM_AIS_DIALECT
