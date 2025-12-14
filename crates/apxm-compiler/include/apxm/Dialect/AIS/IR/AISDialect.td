/**
 * @file  AISDialect.td
 * @brief Dialect skeleton and base operation class for AIS.
 *
 * The dialect is intentionally light-weight: only the name, C++ namespace,
 * summary/description, and a hook for custom type parser/printer are
 * specified here.  The actual operations are defined in AISOps.td so that
 * dialect-level options (e.g. new interfaces) can be added without
 * recompiling every operation.
 *
 * `AIS_Op<>` is the conventional base class used by *all* AIS operations;
 * it wires the mnemonic and trait list into the dialect automatically.
 */

#ifndef APXM_AIS_DIALECT
#define APXM_AIS_DIALECT

include "mlir/IR/OpBase.td"

// AIS Dialect definition
def AIS_Dialect : Dialect {
  let name = "ais";
  let cppNamespace = "::mlir::ais";
  let summary = "Agent Instruction Set (AIS) dialect for A-PXM";
  let description = [{
    The AIS dialect provides operations for agent programming with explicit
    state management and data-driven execution.
  }];

  let extraClassDeclaration = [{
    /// Type parsing and printing for custom AIS types
    ::mlir::Type parseType(::mlir::DialectAsmParser &parser) const override;
    void printType(::mlir::Type type, ::mlir::DialectAsmPrinter &printer) const override;
  }];

  let useDefaultTypePrinterParser = 0;
  let useDefaultAttributePrinterParser = 1;
}

/// Base class for AIS operations
class AIS_Op<string mnemonic, list<Trait> traits = []> :
    Op<AIS_Dialect, mnemonic, traits>;

#endif // APXM_AIS_DIALECT
