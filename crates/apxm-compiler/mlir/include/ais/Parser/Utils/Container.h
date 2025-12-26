/*
 * @file Container.h
 * @brief Utilities for managing AST node containers
 *
 * An AST node container is a collection of unique pointers to AST nodes.
 * This file provides utilities for managing these containers.
 */

#ifndef APXM_PARSER_PARSERS_CONTAINERUTILS_H
#define APXM_PARSER_PARSERS_CONTAINERUTILS_H

#include "llvm/ADT/SmallVector.h"
#include <memory>
#include <utility>

namespace apxm::parser {

/// Deduplicate and validate container elements
template <typename T>
void deduplicateAndValidate(llvm::SmallVectorImpl<std::unique_ptr<T>> &container) {
  // Simple deduplication for now - could be enhanced with equality comparison
  llvm::SmallVector<std::unique_ptr<T>, 8> unique;
  unique.reserve(container.size());

  for (auto &item : container) {
    if (item) {
      unique.push_back(std::move(item));
    }
  }

  container = std::move(unique);
}

/// Validate that all elements in container are non-null
template <typename T>
bool validateContainer(const llvm::SmallVectorImpl<std::unique_ptr<T>> &container) noexcept {
  for (const auto &item : container) {
    if (!item) {
      return false;
    }
  }
  return true;
}

} // namespace apxm::parser

#endif // APXM_PARSER_PARSERS_CONTAINERUTILS_H
