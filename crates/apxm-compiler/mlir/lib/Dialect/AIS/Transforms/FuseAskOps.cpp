/**
 * @file  FuseAskOps.cpp
 * @brief Batches producer-consumer `ais.ask` chains into single operations.
 *
 * The pass looks for patterns:
 *   %a = ais.ask "template A" ...
 *   %b = ais.ask "template B" [%a, ...]
 *
 * AND patterns with string interpolation (merge chains):
 *   %a = ais.ask "template A" ...
 *   %s1 = ais.const_str "Using: "
 *   %m1 = ais.merge %s1, %a
 *   %s2 = ais.const_str ", explain..."
 *   %m2 = ais.merge %m1, %s2
 *   %b = ais.ask "" [%m2]
 *
 * and replaces them with one `ais.ask` whose template is the concatenation
 * of all templates and static strings.  This removes an LLM round-trip
 * (500-2000 ms) and is the highest-ROI optimisation in the AIS pipeline.
 *
 * Fusion is guarded by:
 *   - single use chain from producer to consumer (through merges)
 *   - identical dialect attribute compatibility
 *
 * On success the pass increments `ais.fused_pairs` on the module so that
 * later stages know how much parallelism was removed.
 *
 * Note: Only AskOp is fusible (LOW latency). ThinkOp/ReasonOp are not fused
 * because they have different semantics (extended thinking, structured output).
 */

#include "ais/Dialect/AIS/Transforms/Passes.h"

#include "ais/Dialect/AIS/IR/AISAttributes.h"
#include "ais/Dialect/AIS/IR/AISOps.h"
#include "ais/Dialect/AIS/Support/AISDebug.h"

#include "mlir/IR/Builders.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/StringExtras.h"

namespace mlir::ais {
#define GEN_PASS_DEF_FUSEASKOPS
#include "ais/Dialect/AIS/Transforms/Passes.h.inc"

namespace {

APXM_AIS_DEBUG_SETUP(fusion)

/// Result of tracing through a merge chain to find an AskOp producer
struct MergeChainTrace {
  AskOp producer;                          // The found AskOp producer
  SmallVector<std::string> stringParts;    // Static strings in order (for template building)
  SmallVector<Operation*> intermediateOps; // MergeOps and ConstStrOps to erase
  SmallVector<Value> otherContext;         // Non-producer context values to preserve
};

/// Trace through merge operations to find an AskOp producer.
/// Returns nullopt if the chain is not fusible (multiple uses, unknown ops, etc.)
static std::optional<MergeChainTrace> traceToAskProducer(Value startOperand, AskOp consumer) {
  MergeChainTrace result;
  SmallVector<std::pair<Value, bool>> worklist; // (value, isBeforeProducer)
  llvm::SmallPtrSet<Operation*, 8> visited;

  // Track string parts with their position indicator
  SmallVector<std::pair<std::string, int>> orderedStrings; // (string, position)
  int posCounter = 0;

  worklist.push_back({startOperand, true});

  while (!worklist.empty()) {
    auto [val, beforeProducer] = worklist.pop_back_val();
    Operation* defOp = val.getDefiningOp();

    if (!defOp)
      continue;

    if (visited.contains(defOp))
      continue;
    visited.insert(defOp);

    if (auto askOp = dyn_cast<AskOp>(defOp)) {
      // Found an AskOp - check if it's fusible
      if (!askOp->hasOneUse()) {
        APXM_AIS_DEBUG("  Not fusible: AskOp has multiple uses");
        return std::nullopt;
      }
      if (result.producer) {
        // Already found a producer - can only fuse one
        APXM_AIS_DEBUG("  Not fusible: Multiple AskOp producers in chain");
        return std::nullopt;
      }
      result.producer = askOp;
      // Collect producer's context
      for (Value ctx : askOp.getOperands()) {
        result.otherContext.push_back(ctx);
      }
    } else if (auto mergeOp = dyn_cast<MergeOp>(defOp)) {
      // MergeOp - check single use and recurse into operands
      if (!mergeOp->hasOneUse()) {
        APXM_AIS_DEBUG("  Not fusible: MergeOp has multiple uses");
        return std::nullopt;
      }
      result.intermediateOps.push_back(mergeOp);
      // Process operands in order (left-to-right for string concatenation)
      for (Value operand : mergeOp.getTokens()) {
        worklist.push_back({operand, beforeProducer});
      }
    } else if (auto constStrOp = dyn_cast<ConstStrOp>(defOp)) {
      // Static string - collect for template building
      orderedStrings.push_back({constStrOp.getValue().str(), posCounter++});
      result.intermediateOps.push_back(constStrOp);
    } else {
      // Unknown operation in chain - not fusible
      APXM_AIS_DEBUG("  Not fusible: Unknown op in chain: " << defOp->getName());
      return std::nullopt;
    }
  }

  if (!result.producer) {
    return std::nullopt;
  }

  // Sort strings by position and extract
  llvm::sort(orderedStrings, [](const auto& a, const auto& b) {
    return a.second < b.second;
  });
  for (const auto& [str, _] : orderedStrings) {
    result.stringParts.push_back(str);
  }

  return result;
}

// Declarative fusion condition predicate (direct connection)
static bool isFusibleProducer(AskOp producer, AskOp consumer) {
  return producer && producer->hasOneUse() &&
         producer->getUses().begin()->getOwner() == consumer.getOperation();
}

// Template fusion strategy - combines producer template, interpolation strings, and consumer template
static std::string fuseTemplates(StringRef producerTemplate,
                                  ArrayRef<std::string> interpolationStrings,
                                  StringRef consumerTemplate) {
  SmallString<256> fused;
  fused.append(producerTemplate);
  fused.append("\n---\n");

  // Add interpolation strings (these were the string concat parts)
  for (const auto& str : interpolationStrings) {
    fused.append(str);
  }

  if (!consumerTemplate.empty()) {
    fused.append(consumerTemplate);
  }

  return std::string(fused);
}

// Legacy template fusion (for direct connections without merge chains)
static std::string fuseTemplates(StringRef producerTemplate, StringRef consumerTemplate) {
  return fuseTemplates(producerTemplate, {}, consumerTemplate);
}

struct FuseAskOpsPass : impl::FuseAskOpsBase<FuseAskOpsPass> {
  void runOnOperation() override {
    APXM_AIS_DEBUG_HEADER(FuseAskOps);
    ModuleOp module = getOperation();

    struct Statistics {
      uint64_t scanned = 0;
      uint64_t fusedDirect = 0;      // Direct ask->ask fusion
      uint64_t fusedMergeChain = 0;  // Fusion through merge chains
    } stats;

    SmallVector<Operation*> opsToErase;

    // Single-pass fusion with clear termination conditions
    module.walk([&](AskOp consumer) {
      stats.scanned++;

      // First, try direct fusion (legacy path - ask result used directly by consumer)
      auto directProducer = llvm::find_if(consumer.getOperands(), [&](Value operand) {
        return isFusibleProducer(operand.getDefiningOp<AskOp>(), consumer);
      });

      if (directProducer != consumer.getOperands().end()) {
        // Direct fusion path
        AskOp producer = (*directProducer).getDefiningOp<AskOp>();
        APXM_AIS_DEBUG("  Direct fusion: [" << producer.getTemplateStrAttr() << "] + ["
                                            << consumer.getTemplateStrAttr() << "]");

        SmallVector<Value> fusedContext;
        fusedContext.reserve(producer.getOperands().size() + consumer.getOperands().size() - 1);
        llvm::append_range(fusedContext, producer.getOperands());
        llvm::copy_if(consumer.getOperands(), std::back_inserter(fusedContext),
                      [&](Value ctx) { return ctx != producer.getResult(); });

        OpBuilder builder(consumer);
        auto fusedTemplate = fuseTemplates(producer.getTemplateStrAttr().getValue(),
                                           consumer.getTemplateStrAttr().getValue());

        auto fusedOp = builder.create<AskOp>(
            consumer.getLoc(), consumer.getType(),
            builder.getStringAttr(fusedTemplate), fusedContext);

        // Transfer attributes
        for (NamedAttribute attr : consumer->getAttrs()) {
          if (attr.getName() != "template_str" && attr.getName() != "operandSegmentSizes") {
            fusedOp->setAttr(attr.getName(), attr.getValue());
          }
        }

        fusedOp->setAttr("ais.fused_from",
          AISFusedFromAttr::get(module.getContext(),
            builder.getArrayAttr({
              builder.getStringAttr(llvm::join_items(".", "producer", producer.getTemplateStrAttr())),
              builder.getStringAttr(llvm::join_items(".", "consumer", consumer.getTemplateStrAttr()))
            })));

        consumer.replaceAllUsesWith(fusedOp.getResult());
        opsToErase.push_back(consumer);
        opsToErase.push_back(producer);
        stats.fusedDirect++;
        return WalkResult::advance();
      }

      // Second, try fusion through merge chains (string interpolation patterns)
      for (Value operand : consumer.getOperands()) {
        // Skip if operand is directly an AskOp (would have been caught above)
        if (operand.getDefiningOp<AskOp>())
          continue;

        // Check if operand comes from a merge chain containing an AskOp
        if (auto trace = traceToAskProducer(operand, consumer)) {
          AskOp producer = trace->producer;
          APXM_AIS_DEBUG("  Merge chain fusion: [" << producer.getTemplateStrAttr()
                         << "] + " << trace->stringParts.size() << " strings + ["
                         << consumer.getTemplateStrAttr() << "]");

          // Build fused context: producer's context + consumer's other context
          SmallVector<Value> fusedContext;
          fusedContext.reserve(trace->otherContext.size() + consumer.getOperands().size());
          llvm::append_range(fusedContext, trace->otherContext);
          llvm::copy_if(consumer.getOperands(), std::back_inserter(fusedContext),
                        [&](Value ctx) { return ctx != operand; });

          OpBuilder builder(consumer);
          auto fusedTemplate = fuseTemplates(producer.getTemplateStrAttr().getValue(),
                                             trace->stringParts,
                                             consumer.getTemplateStrAttr().getValue());

          auto fusedOp = builder.create<AskOp>(
              consumer.getLoc(), consumer.getType(),
              builder.getStringAttr(fusedTemplate), fusedContext);

          // Transfer attributes
          for (NamedAttribute attr : consumer->getAttrs()) {
            if (attr.getName() != "template_str" && attr.getName() != "operandSegmentSizes") {
              fusedOp->setAttr(attr.getName(), attr.getValue());
            }
          }

          fusedOp->setAttr("ais.fused_from",
            AISFusedFromAttr::get(module.getContext(),
              builder.getArrayAttr({
                builder.getStringAttr(llvm::join_items(".", "producer", producer.getTemplateStrAttr())),
                builder.getStringAttr("merge_chain"),
                builder.getStringAttr(llvm::join_items(".", "consumer", consumer.getTemplateStrAttr()))
              })));

          consumer.replaceAllUsesWith(fusedOp.getResult());
          opsToErase.push_back(consumer);
          opsToErase.push_back(producer);
          // Also mark intermediate ops for erasure
          for (Operation* op : trace->intermediateOps) {
            opsToErase.push_back(op);
          }
          stats.fusedMergeChain++;
          return WalkResult::advance();
        }
      }

      return WalkResult::advance();
    });

    // Safe bulk erasure (reverse order to handle dependencies)
    for (auto it = opsToErase.rbegin(); it != opsToErase.rend(); ++it) {
      Operation* op = *it;
      if (op->use_empty()) {
        op->erase();
      }
    }

    // Module-level metadata
    uint64_t totalFused = stats.fusedDirect + stats.fusedMergeChain;
    module->setAttr("ais.fused_pairs",
                    AISFusedPairsAttr::get(module.getContext(), totalFused));

    APXM_AIS_INFO("Scanned " << stats.scanned << " ASK ops, fused "
                  << stats.fusedDirect << " direct + "
                  << stats.fusedMergeChain << " merge chains = "
                  << totalFused << " total");
    APXM_AIS_DEBUG_FOOTER(FuseAskOps);
  }
};

}  // namespace

std::unique_ptr<Pass> createFuseAskOpsPass() {
  return std::make_unique<FuseAskOpsPass>();
}

}  // namespace mlir::ais
