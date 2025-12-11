//===- AISTypesImpl.h -----------------------------------------------------===//
//
// Part of the A-PXM project, under the Apache License v2.0.
// See LICENSE for license information.
//
//===----------------------------------------------------------------------===//

#ifndef APXM_AIS_TYPES_IMPL_H
#define APXM_AIS_TYPES_IMPL_H

#include "apxm/Dialect/AIS/IR/AISTypes.h"
#include "mlir/IR/TypeSupport.h"
#include <utility>

namespace mlir::ais::detail {

struct TokenTypeStorage : public TypeStorage {
  using KeyTy = Type;

  explicit TokenTypeStorage(Type inner) : innerType(inner) {}

  bool operator==(const KeyTy &key) const {
    return key == innerType;
  }

  static TokenTypeStorage *construct(TypeStorageAllocator &allocator, const KeyTy &key) {
    return new (allocator.allocate<TokenTypeStorage>()) TokenTypeStorage(key);
  }

  Type innerType;
};

struct HandleTypeStorage : public TypeStorage {
  using KeyTy = std::pair<MemorySpace, Type>;

  HandleTypeStorage(MemorySpace space, Type payload) : space(space), payload(payload) {}

  bool operator==(const KeyTy &key) const {
    return key.first == space && key.second == payload;
  }

  static HandleTypeStorage *construct(TypeStorageAllocator &allocator, const KeyTy &key) {
    return new (allocator.allocate<HandleTypeStorage>()) HandleTypeStorage(key.first, key.second);
  }

  MemorySpace space;
  Type payload;
};

struct GoalTypeStorage : public TypeStorage {
  using KeyTy = unsigned;

  explicit GoalTypeStorage(unsigned priority) : priority(priority) {}

  bool operator==(const KeyTy &key) const {
    return key == priority;
  }

  static GoalTypeStorage *construct(TypeStorageAllocator &allocator, const KeyTy &key) {
    return new (allocator.allocate<GoalTypeStorage>()) GoalTypeStorage(key);
  }

  unsigned priority;
};

}  // namespace mlir::ais::detail

#endif  // APXM_AIS_TYPES_IMPL_H
