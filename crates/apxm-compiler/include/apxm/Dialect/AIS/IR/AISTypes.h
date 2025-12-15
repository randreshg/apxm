/**
 * @file  AISTypes.h
 * @brief Type interface and uniqued storage for the AIS dialect.
 *
 * Defines the three first-class types exported by the dialect:
 *   - TokenType  – data-flow token carrying a typed payload
 *   - HandleType – reference to content in an AAM memory space
 *   - GoalType   – planning goal annotated with a priority
 *
 * Storage classes live in the private `detail` namespace and conform to
 * MLIR's TypeStorage contract so that identical types are uniqued in the
 * context.  Public classes expose only immutable accessors; construction
 * is handled by the static `get` helpers.
 */

#ifndef APXM_AIS_TYPES_H
#define APXM_AIS_TYPES_H

#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/TypeSupport.h"
#include "mlir/IR/Types.h"
#include "llvm/ADT/StringRef.h"
#include <cstdint>
#include <optional>
#include <utility>

namespace mlir::ais {

//===----------------------------------------------------------------------===//
// Public enums and utilities
//===----------------------------------------------------------------------===//

enum class MemorySpace : uint8_t { STM = 0, LTM, Episodic };

std::optional<MemorySpace> symbolizeMemorySpace(StringRef str);
StringRef stringifyMemorySpace(MemorySpace space);
Type getDefaultPayloadType(MLIRContext *ctx);

//===----------------------------------------------------------------------===//
// Forward declarations (public)
//===----------------------------------------------------------------------===//

class TokenType;
class HandleType;
class GoalType;

//===----------------------------------------------------------------------===//
// Storage implementation (private to the dialect)
//===----------------------------------------------------------------------===//

namespace detail {

struct TokenTypeStorage final : public TypeStorage {
  using KeyTy = Type;

  explicit TokenTypeStorage(Type inner) : innerType(inner) {}

  bool operator==(const KeyTy &key) const { return key == innerType; }

  static TokenTypeStorage *construct(TypeStorageAllocator &alloc, const KeyTy &key) {
    return new (alloc.allocate<TokenTypeStorage>()) TokenTypeStorage(key);
  }

  Type innerType;
};

struct HandleTypeStorage final : public TypeStorage {
  using KeyTy = std::pair<MemorySpace, Type>;

  HandleTypeStorage(MemorySpace space, Type payload) : space(space), payload(payload) {}

  bool operator==(const KeyTy &key) const {
    return key.first == space && key.second == payload;
  }

  static HandleTypeStorage *construct(TypeStorageAllocator &alloc, const KeyTy &key) {
    return new (alloc.allocate<HandleTypeStorage>()) HandleTypeStorage(key.first, key.second);
  }

  MemorySpace space;
  Type payload;
};

struct GoalTypeStorage final : public TypeStorage {
  using KeyTy = unsigned;

  explicit GoalTypeStorage(unsigned prio) : priority(prio) {}

  bool operator==(const KeyTy &key) const { return key == priority; }

  static GoalTypeStorage *construct(TypeStorageAllocator &alloc, const KeyTy &key) {
    return new (alloc.allocate<GoalTypeStorage>()) GoalTypeStorage(key);
  }

  unsigned priority;
};

} // namespace detail

//===----------------------------------------------------------------------===//
// Public type classes
//===----------------------------------------------------------------------===//

class TokenType : public Type::TypeBase<TokenType, Type, detail::TokenTypeStorage> {
public:
  using Base::Base;

  static constexpr StringLiteral name = "ais.token";

  static TokenType get(MLIRContext *ctx, Type innerType);
  Type getInnerType() const;
};

class HandleType : public Type::TypeBase<HandleType, Type, detail::HandleTypeStorage> {
public:
  using Base::Base;

  static constexpr StringLiteral name = "ais.handle";

  static HandleType get(MLIRContext *ctx, MemorySpace space, Type payload = Type());

  MemorySpace getSpace() const;
  Type getPayload() const;
};

class GoalType : public Type::TypeBase<GoalType, Type, detail::GoalTypeStorage> {
public:
  using Base::Base;

  static constexpr StringLiteral name = "ais.goal";

  static GoalType get(MLIRContext *ctx, unsigned priority);
  unsigned getPriority() const;
};

} // namespace mlir::ais

#endif // APXM_AIS_TYPES_H
