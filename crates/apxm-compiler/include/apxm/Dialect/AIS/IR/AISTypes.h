//===- AISTypes.h ---------------------------------------------------------===//
//
// Part of the A-PXM project, under the Apache License v2.0.
// See LICENSE for license information.
//
//===----------------------------------------------------------------------===//

#ifndef APXM_AIS_TYPES_H
#define APXM_AIS_TYPES_H

#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/TypeSupport.h"
#include "mlir/IR/Types.h"
#include "llvm/ADT/StringRef.h"
#include <cstdint>
#include <optional>

namespace mlir::ais {

// Forward declarations of storage classes
namespace detail {
struct TokenTypeStorage;
struct HandleTypeStorage;
struct GoalTypeStorage;
}  // namespace detail

enum class MemorySpace : uint8_t { STM = 0, LTM, Episodic };

std::optional<MemorySpace> symbolizeMemorySpace(StringRef);
StringRef stringifyMemorySpace(MemorySpace space);
Type getDefaultPayloadType(MLIRContext *context);

class TokenType : public Type::TypeBase<TokenType, Type, detail::TokenTypeStorage> {
public:
  using Base::Base;

  static constexpr StringLiteral name = "ais.token";

  static TokenType get(MLIRContext *context, Type innerType);

  Type getInnerType() const;
};

class HandleType : public Type::TypeBase<HandleType, Type, detail::HandleTypeStorage> {
public:
  using Base::Base;

  static constexpr StringLiteral name = "ais.handle";

  static HandleType get(MLIRContext *context, MemorySpace space, Type payload = Type());

  MemorySpace getSpace() const;
  Type getPayload() const;
};

class GoalType : public Type::TypeBase<GoalType, Type, detail::GoalTypeStorage> {
public:
  using Base::Base;

  static constexpr StringLiteral name = "ais.goal";

  static GoalType get(MLIRContext *context, unsigned priority);

  unsigned getPriority() const;
};

}  // namespace mlir::ais

#endif  // APXM_AIS_TYPES_H
