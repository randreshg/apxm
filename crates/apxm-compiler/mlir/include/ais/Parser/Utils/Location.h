/*
 * @file Location.h
 * @brief Source location utilities for the APXM DSL parser
 *
 * This file provides utilities for tracking source locations within the APXM DSL parser.
 * The location is represented by a start and end position within a source file.
 */

#ifndef APXM_PARSER_LOCATION_H
#define APXM_PARSER_LOCATION_H

#include "llvm/Support/SMLoc.h"
#include "llvm/Support/SourceMgr.h"
#include <string>

namespace apxm::parser {

/// Source location range
class Location {
  llvm::SMLoc start_;
  llvm::SMLoc end_;

public:
  Location() = default;
  explicit Location(llvm::SMLoc loc) : start_(loc), end_(loc) {}
  Location(llvm::SMLoc start, llvm::SMLoc end) : start_(start), end_(end) {}

  bool isValid() const noexcept { return start_.isValid(); }
  llvm::SMLoc getStart() const noexcept { return start_; }
  llvm::SMLoc getEnd() const noexcept { return end_; }

  /// Get string representation with full file:line:col information
  std::string toString(const llvm::SourceMgr &srcMgr) const;
};

} // namespace apxm::parser

#endif // APXM_PARSER_LOCATION_H
