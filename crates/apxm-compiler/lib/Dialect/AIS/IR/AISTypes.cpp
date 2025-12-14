/**
 * @file  AISTypes.cpp
 * @brief Runtime implementation of the AIS type system.
 *
 * Provides the body of every out-of-line method declared in AISTypes.h:
 *   - `get` constructors that uniquify the type in the MLIRContext
 *   - String ↔ enum conversion helpers for MemorySpace
 *   - `getDefaultPayloadType` helper shared by both Token and Handle
 *
 * The file is intentionally free of dialect or operation logic; it only
 * realises the low-level type storage manipulation required by the context.
 */

#include "apxm/Dialect/AIS/IR/AISTypes.h"
#include "apxm/Dialect/AIS/IR/AISTypesImpl.h"
#include "mlir/IR/BuiltinTypes.h"
#include "llvm/ADT/StringSwitch.h"
#include <utility>

using namespace mlir;
using namespace mlir::ais;

Type mlir::ais::getDefaultPayloadType(MLIRContext *context) {
  return NoneType::get(context);
}

std::optional<MemorySpace> mlir::ais::symbolizeMemorySpace(StringRef value) {
  return llvm::StringSwitch<std::optional<MemorySpace>>(value)
      .CaseLower("stm", MemorySpace::STM)
      .CaseLower("ltm", MemorySpace::LTM)
      .CaseLower("episodic", MemorySpace::Episodic)
      .Default(std::nullopt);
}

StringRef mlir::ais::stringifyMemorySpace(MemorySpace space) {
  switch (space) {
  case MemorySpace::STM:
    return "stm";
  case MemorySpace::LTM:
    return "ltm";
  case MemorySpace::Episodic:
    return "episodic";
  }
  llvm_unreachable("Unknown memory space");
}

TokenType TokenType::get(MLIRContext *context, Type innerType) {
  if (!innerType)
    innerType = getDefaultPayloadType(context);
  return Base::get(context, innerType);
}

Type TokenType::getInnerType() const {
  return getImpl()->innerType;
}

HandleType HandleType::get(MLIRContext *context, MemorySpace space, Type payload) {
  if (!payload)
    payload = getDefaultPayloadType(context);
  return Base::get(context, std::make_pair(space, payload));
}

MemorySpace HandleType::getSpace() const {
  return getImpl()->space;
}

Type HandleType::getPayload() const {
  return getImpl()->payload;
}

GoalType GoalType::get(MLIRContext *context, unsigned priority) {
  return Base::get(context, priority);
}

unsigned GoalType::getPriority() const {
  return getImpl()->priority;
}
