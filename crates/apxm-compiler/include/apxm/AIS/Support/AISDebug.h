///==========================================================================///
/// File: AISDebug.h
///
/// Centralized debug utilities for AIS compiler passes and components.
/// Based on the A-PXM execution model for Agentic AI.
///==========================================================================///

#ifndef APXM_AIS_SUPPORT_AISDEBUG_H
#define APXM_AIS_SUPPORT_AISDEBUG_H

#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"

namespace mlir {
namespace ais {

#define APXM_AIS_DEBUG_COLORS true

/// Returns the AIS debug output stream. Intended as the single entrypoint
/// for emitting debug logs so callers don't use llvm::dbgs() directly.
static inline llvm::raw_ostream &debugStream() {
  static llvm::raw_ostream &stream = llvm::errs();
  // Force color support if we're connected to a terminal
  if (stream.is_displayed())
    stream.enable_colors(APXM_AIS_DEBUG_COLORS);
  return stream;
}

/// Common line separator used in debug headers/footers
#ifndef APXM_AIS_LINE
#define APXM_AIS_LINE "-----------------------------------------\n"
#endif

#ifndef APXM_AIS_SEPARATOR
#define APXM_AIS_SEPARATOR "===================================\n"
#endif

/// Macro to set up debug infrastructure for a pass/component
/// Usage: AIS_DEBUG_SETUP(normalize) at the top of your file
#define APXM_AIS_DEBUG_SETUP(pass_name)                                             \
  static constexpr const char *AIS_DEBUG_TYPE_STR = #pass_name;

/// AIS debug stream accessor. Use this instead of llvm::dbgs()
#define APXM_AIS_DBGS() ::mlir::ais::debugStream()

#define APXM_AIS_DEBUG_TYPE(x)                                                      \
  DEBUG_WITH_TYPE(AIS_DEBUG_TYPE_STR, {                                        \
    APXM_AIS_DBGS() << "[" << AIS_DEBUG_TYPE_STR << "] " << x << "\n";              \
  })

#define APXM_AIS_DEBUG_MSG(msg)                                                     \
  DEBUG_WITH_TYPE(AIS_DEBUG_TYPE_STR, { APXM_AIS_DBGS() << msg << "\n"; })

/// Multi-statement debug region. Wraps an entire block under LLVM_DEBUG.
/// Example:
///   AIS_DEBUG_REGION(
///     AIS_DBGS() << "Operations:\n";
///     for (auto &op : ops)
///       AIS_DBGS() << "  " << op << "\n";
///   );
#define APXM_AIS_DEBUG_REGION(...)                                                  \
  DEBUG_WITH_TYPE(AIS_DEBUG_TYPE_STR, { __VA_ARGS__ })

/// Debug section with automatic header/footer around a region.
/// Example:
///   AIS_DEBUG_SECTION("MemoryAnalysis",
///     AIS_DBGS() << "Analyzing memory operations\n";
///     // ... analysis code
///   );
#define APXM_AIS_DEBUG_SECTION(title, ...)                                          \
  DEBUG_WITH_TYPE(AIS_DEBUG_TYPE_STR, {                                        \
    APXM_AIS_DBGS() << "\n" << APXM_AIS_LINE << title << " STARTED\n" << APXM_AIS_LINE;       \
    __VA_ARGS__;                                                               \
    APXM_AIS_DBGS() << "\n" << APXM_AIS_LINE << title << " FINISHED\n" << APXM_AIS_LINE;      \
  })

/// Standard header/footer patterns for passes
#define APXM_AIS_DEBUG_HEADER(x)                                                    \
  DEBUG_WITH_TYPE(AIS_DEBUG_TYPE_STR, {                                        \
    auto &__os = APXM_AIS_DBGS();                                                   \
    __os.changeColor(llvm::raw_ostream::CYAN, /*bold=*/true);                  \
    __os << "\n" << #x " STARTED\n" << APXM_AIS_SEPARATOR;                          \
    __os.resetColor();                                                         \
  })

#define APXM_AIS_DEBUG_FOOTER(x)                                                    \
  DEBUG_WITH_TYPE(AIS_DEBUG_TYPE_STR, {                                        \
    auto &__os = APXM_AIS_DBGS();                                                   \
    __os.changeColor(llvm::raw_ostream::CYAN, /*bold=*/true);                  \
    __os << "\n" << #x " FINISHED\n" << APXM_AIS_SEPARATOR << "\n";                 \
    __os.resetColor();                                                         \
  })

/// Colored log levels

/// INFO - Blue, for high-level progress information
#define APXM_AIS_INFO(x)                                                            \
  DEBUG_WITH_TYPE(AIS_DEBUG_TYPE_STR, {                                        \
    auto &__os = APXM_AIS_DBGS();                                                   \
    __os.changeColor(llvm::raw_ostream::BLUE, /*bold=*/true);                  \
    __os << "[INFO] [" << AIS_DEBUG_TYPE_STR << "]";                           \
    __os.resetColor();                                                         \
    __os << " " << x << "\n";                                                  \
  })

/// DEBUG - Magenta, for detailed debugging information
#define APXM_AIS_DEBUG(x)                                                           \
  DEBUG_WITH_TYPE(AIS_DEBUG_TYPE_STR, {                                        \
    auto &__os = APXM_AIS_DBGS();                                                   \
    __os.changeColor(llvm::raw_ostream::MAGENTA, /*bold=*/true);               \
    __os << "[DEBUG] [" << AIS_DEBUG_TYPE_STR << "]";                          \
    __os.resetColor();                                                         \
    __os << " " << x << "\n";                                                  \
  })

/// WARN - Yellow, for warnings that don't stop execution
#define APXM_AIS_WARN(x)                                                            \
  DEBUG_WITH_TYPE(AIS_DEBUG_TYPE_STR, {                                        \
    auto &__os = APXM_AIS_DBGS();                                                   \
    __os.changeColor(llvm::raw_ostream::YELLOW, /*bold=*/true);                \
    __os << "[WARN] [" << AIS_DEBUG_TYPE_STR << "]";                           \
    __os.resetColor();                                                         \
    __os << " " << x << "\n";                                                  \
  })

/// ERROR - Red, always prints (not guarded by DEBUG_WITH_TYPE)
#define APXM_AIS_ERROR(x)                                                           \
  {                                                                            \
    auto &__os = llvm::errs();                                                 \
    __os.changeColor(llvm::raw_ostream::RED, /*bold=*/true);                   \
    __os << "[ERROR] [" << AIS_DEBUG_TYPE_STR << "]";                          \
    __os.resetColor();                                                         \
    __os << " " << x << "\n";                                                  \
  }

/// AAM State logging - for Agent Abstract Machine state transitions
#define APXM_AIS_AAM_STATE(x)                                                       \
  DEBUG_WITH_TYPE(AIS_DEBUG_TYPE_STR, {                                        \
    auto &__os = APXM_AIS_DBGS();                                                   \
    __os.changeColor(llvm::raw_ostream::GREEN, /*bold=*/true);                 \
    __os << "[AAM] [" << AIS_DEBUG_TYPE_STR << "]";                            \
    __os.resetColor();                                                         \
    __os << " " << x << "\n";                                                  \
  })

/// Dataflow logging - for dependency and scheduling information
#define APXM_AIS_DATAFLOW(x)                                                        \
  DEBUG_WITH_TYPE(AIS_DEBUG_TYPE_STR, {                                        \
    auto &__os = APXM_AIS_DBGS();                                                   \
    __os.changeColor(llvm::raw_ostream::WHITE, /*bold=*/true);                 \
    __os << "[FLOW] [" << AIS_DEBUG_TYPE_STR << "]";                           \
    __os.resetColor();                                                         \
    __os << " " << x << "\n";                                                  \
  })

} // namespace ais
} // namespace mlir

#endif // APXM_AIS_SUPPORT_AISDEBUG_H
