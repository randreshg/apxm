/**
 * @file  NormalizeAgentGraph.cpp
 * @brief Puts the AIS IR into canonical form so later passes can rely on
 *        predictable attribute values and operand order.
 *
 * The pass performs two local transformations:
 *   1. Deduplicate the context operand list of every `ais.rsn`
 *   2. Lower-case the string attributes `space` and `capability` on every op
 *
 * Both changes are semantics-preserving and idempotent, so the pass can be
 * run repeatedly or inserted anywhere in the pipeline.  The total number of
 * edits is recorded in the module attribute `ais.graph_normalized`.
 */

#include "apxm/Dialect/AIS/Transforms/Passes.h"

#include "apxm/Dialect/AIS/IR/AISAttributes.h"
#include "apxm/Dialect/AIS/IR/AISOps.h"
#include "apxm/Dialect/AIS/Support/AISDebug.h"

#include "mlir/IR/Builders.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/Support/FormatVariadic.h"

namespace mlir::ais {
#define GEN_PASS_DEF_NORMALIZEAGENTGRAPH
#include "apxm/Dialect/AIS/Transforms/Passes.h.inc"

namespace {

APXM_AIS_DEBUG_SETUP(normalize)

// Declarative normalization predicates
static bool needsLowercaseNormalization(StringRef str) {
  return !llvm::all_of(str, [](char c) {
    return !std::isupper(static_cast<unsigned char>(c));
  });
}

static bool needsDeduplication(ValueRange values) {
  llvm::SmallDenseSet<Value, 8> seen;
  return llvm::any_of(values, [&](Value v) { return !seen.insert(v).second; });
}

struct NormalizeAgentGraphPass : impl::NormalizeAgentGraphBase<NormalizeAgentGraphPass> {
  void runOnOperation() override {
    APXM_AIS_DEBUG_HEADER(NormalizeAgentGraph);
    ModuleOp module = getOperation();

    struct Statistics {
      uint64_t contextDedups = 0;
      uint64_t stringNorms = 0;
    } stats;

    // Phase 1: Deduplicate reasoning contexts
    APXM_AIS_DEBUG("Deduplicating reasoning contexts...");
    module.walk([&](RsnOp op) {
      if (needsDeduplication(op.getContext())) {
        deduplicateReasoningContext(op);
        stats.contextDedups++;
        APXM_AIS_DEBUG("  Deduplicated context in: " << op.getTemplateStrAttr());
      }
    });

    // Phase 2: Normalize string attributes
    APXM_AIS_DEBUG("Normalizing string attributes...");
    const std::array kNormalizedAttrs = {"space", "capability"};

    module.walk([&](Operation* op) {
      for (StringRef attrName : kNormalizedAttrs) {
        if (normalizeStringAttribute(op, attrName)) {
          stats.stringNorms++;
        }
      }
    });

    // Module-level metadata
    const uint64_t totalNormalized = stats.contextDedups + stats.stringNorms;
    module->setAttr("ais.graph_normalized",
                    AISGraphNormalizedAttr::get(module.getContext(), totalNormalized));

    APXM_AIS_INFO(llvm::formatv("Normalized {0} attributes (ctx_dedup={1}, str_norm={2})",
                               totalNormalized, stats.contextDedups, stats.stringNorms));
    APXM_AIS_DEBUG_FOOTER(NormalizeAgentGraph);
  }

private:
  static bool normalizeStringAttribute(Operation* op, StringRef attrName) {
    auto attr = op->getAttrOfType<StringAttr>(attrName);
    if (!attr || !needsLowercaseNormalization(attr.getValue()))
      return false;

    SmallString<64> normalized(attr.getValue());
    for (char &ch : normalized)
      ch = llvm::toLower(ch);

    OpBuilder builder(op);
    op->setAttr(attrName, builder.getStringAttr(normalized));
    return true;
  }

  static void deduplicateReasoningContext(RsnOp op) {
    // Preserve operand order while removing duplicates
    llvm::SmallDenseSet<Value, 8> seen;
    SmallVector<Value> uniqueContext;
    uniqueContext.reserve(op.getContext().size());

    llvm::copy_if(op.getContext(), std::back_inserter(uniqueContext),
                  [&](Value v) { return seen.insert(v).second; });

    op->setOperands(uniqueContext);
  }
};

}  // namespace

std::unique_ptr<Pass> createNormalizeAgentGraphPass() {
  return std::make_unique<NormalizeAgentGraphPass>();
}

}  // namespace mlir::ais
