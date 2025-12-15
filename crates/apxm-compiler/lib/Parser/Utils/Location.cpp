/*
 * @file Location.cpp
 * @brief This file contains the implementation of the Location class.
 *
 * Location contains information about the source location
 * of a token or a range of tokens.
 */

#include "apxm/Parser/Utils/Location.h"
#include "llvm/Support/Format.h"
#include "llvm/Support/raw_ostream.h"

namespace apxm::parser {

std::string Location::toString(const llvm::SourceMgr &srcMgr) const {
  if (!isValid()) {
    return "<unknown location>";
  }

  // Get line and column information
  auto startInfo = srcMgr.getLineAndColumn(start_);
  auto endInfo = srcMgr.getLineAndColumn(end_);

  // Get filename
  unsigned bufId = srcMgr.FindBufferContainingLoc(start_);
  std::string filename = "<input>";
  if (bufId != 0 && bufId <= srcMgr.getNumBuffers()) {
    filename = srcMgr.getMemoryBuffer(bufId)->getBufferIdentifier().str();
  }

  // Format location string
  std::string result;
  llvm::raw_string_ostream os(result);

  os << filename << ":" << startInfo.first << ":" << startInfo.second;

  // Add end position if different from start
  if (start_ != end_ && (startInfo.first != endInfo.first || startInfo.second != endInfo.second)) {
    os << "-" << endInfo.first << ":" << endInfo.second;
  }

  return os.str();
}

} // namespace apxm::parser
