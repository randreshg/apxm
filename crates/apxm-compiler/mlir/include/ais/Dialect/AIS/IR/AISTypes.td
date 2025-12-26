/**
 * @file  AISTypes.td
 * @brief Type definitions and type constraints for the AIS dialect.
 *
 * Three first-class types are exposed:
 *   - !ais.token<innerType>   – data-flow token carrying a typed value
 *   - !ais.handle<space,payload> – reference to content in an AAM memory space
 *   - !ais.goal<priority>    – planning goal that can be refined into a subgraph
 *
 * Type constraints `AIS_AnyTokenOrHandle` and `AIS_AnyTokenHandleOrGoal` are
 * used throughout AISOps.td to accept the common super-sets without resorting
 * to `AnyType`.
 *
 * Memory resources (`AIS_BeliefResource`, ...) are also declared here so that
 * both ODS and C++ code use the same symbolic names.
 */

#ifndef APXM_AIS_TYPES
#define APXM_AIS_TYPES

include "ais/Dialect/AIS/IR/AISDialect.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

//===----------------------------------------------------------------------===//
// AIS Type Definitions
//===----------------------------------------------------------------------===//

/// TokenType - Represents a dataflow token carrying a value
def AIS_TokenType : DialectType<AIS_Dialect,
    CPred<"::llvm::isa<::mlir::ais::TokenType>($_self)">,
    "AIS token type", "::mlir::ais::TokenType"> {
  let summary = "Token carrying a typed value";
  let description = [{
    TokenType represents a dataflow token in the AIS execution model.
    Tokens carry typed values and establish data dependencies between operations.
  }];
  let builderCall =
      "::mlir::ais::TokenType::get($_builder.getContext(), $_builder.getNoneType())";
}

/// HandleType - Represents a handle to memory content
def AIS_HandleType : DialectType<AIS_Dialect,
    CPred<"::llvm::isa<::mlir::ais::HandleType>($_self)">,
    "AIS handle type", "::mlir::ais::HandleType"> {
  let summary = "Handle to memory space content";
  let description = [{
    HandleType represents a reference to content in one of the AAM memory spaces:
    - STM (Short-Term Memory)
    - LTM (Long-Term Memory)
    - Episodic (Episodic Memory)
  }];
  let builderCall =
      "::mlir::ais::HandleType::get($_builder.getContext(), "
      "::mlir::ais::MemorySpace::STM, $_builder.getNoneType())";
}

/// GoalType - Represents a planning goal
def AIS_GoalType : DialectType<AIS_Dialect,
    CPred<"::llvm::isa<::mlir::ais::GoalType>($_self)">,
    "AIS goal type", "::mlir::ais::GoalType"> {
  let summary = "Goal for planning operations";
  let description = [{
    GoalType represents a planning goal that can be refined into
    executable subgraphs.
  }];
  let builderCall = "::mlir::ais::GoalType::get($_builder.getContext(), 0)";
}

/// Type constraint: Accept either Token or Handle
def AIS_AnyTokenOrHandle : AnyTypeOf<[AIS_TokenType, AIS_HandleType],
    "token or handle">;

/// Type constraint: Accept Token, Handle, or Goal
def AIS_AnyTokenHandleOrGoal : AnyTypeOf<[AIS_TokenType, AIS_HandleType, AIS_GoalType],
    "token, handle, or goal">;

//===----------------------------------------------------------------------===//
// Memory Resources (for Side Effect Interface)
//===----------------------------------------------------------------------===//

def AIS_BeliefResource : Resource<"::mlir::ais::BeliefResource">;
def AIS_GoalResource : Resource<"::mlir::ais::GoalResource">;
def AIS_CapabilityResource : Resource<"::mlir::ais::CapabilityResource">;
def AIS_EpisodicResource : Resource<"::mlir::ais::EpisodicResource">;

#endif // APXM_AIS_TYPES
