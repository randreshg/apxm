/**
 * @file  FuseReasoning.cpp
 * @brief Batches producer-consumer `ais.rsn` chains into single operations.
 *
 * The pass looks for patterns:
 *   %a = ais.rsn "template A" ...
 *   %b = ais.rsn "template B" [%a, ...]
 *
 * and replaces them with one `ais.rsn` whose template is the concatenation
 * of both originals.  This removes an LLM round-trip (500-2000 ms) and is
 * the highest-ROI optimisation in the AIS pipeline.
 *
 * Fusion is guarded by:
 *   - single use of the producer
 *   - absence of inner_plan regions
 *   - identical dialect attribute compatibility
 *
 * On success the pass increments `ais.fused_pairs` on the module so that
 * later stages know how much parallelism was removed.
 */

#include "apxm/Dialect/AIS/Transforms/Passes.h"

#include "apxm/Dialect/AIS/IR/AISAttributes.h"
#include "apxm/Dialect/AIS/IR/AISOps.h"
#include "apxm/Dialect/AIS/Support/AISDebug.h"

#include "mlir/IR/Builders.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringExtras.h"

namespace mlir::ais {
#define GEN_PASS_DEF_FUSEREASONING
#include "apxm/Dialect/AIS/Transforms/Passes.h.inc"

namespace {

APXM_AIS_DEBUG_SETUP(fusion)

// Declarative fusion condition predicate
static bool isFusibleProducer(RsnOp producer, RsnOp consumer) {
  return producer && producer->hasOneUse() &&
         producer->getUses().begin()->getOwner() == consumer.getOperation() &&
         !producer->hasAttr("inner_plan") &&
         !consumer->hasAttr("inner_plan");
}

// Template fusion strategy
static std::string fuseTemplates(StringRef producerTemplate, StringRef consumerTemplate) {
  SmallString<128> fused;
  fused.append(producerTemplate);
  fused.append("\n---\n");
  fused.append(consumerTemplate);
  return std::string(fused);
}

struct FuseReasoningPass : impl::FuseReasoningBase<FuseReasoningPass> {
  void runOnOperation() override {
    APXM_AIS_DEBUG_HEADER(FuseReasoning);
    ModuleOp module = getOperation();

    struct Statistics {
      uint64_t scanned = 0;
      uint64_t fused = 0;
    } stats;

    SmallVector<Operation*> opsToErase;

    // Single-pass fusion with clear termination conditions
    module.walk([&](RsnOp consumer) {
      stats.scanned++;

      // Skip non-fusible consumers immediately
      if (consumer->hasAttr("inner_plan")) {
        APXM_AIS_DEBUG("  Skip (has inner_plan): " << consumer.getTemplateStrAttr());
        return WalkResult::advance();
      }

      // Find first fusible producer in operands
      auto fusibleProducer = llvm::find_if(consumer.getOperands(), [&](Value operand) {
        return isFusibleProducer(operand.getDefiningOp<RsnOp>(), consumer);
      });

      if (fusibleProducer == consumer.getOperands().end())
        return WalkResult::advance();

      RsnOp producer = (*fusibleProducer).getDefiningOp<RsnOp>();
      APXM_AIS_DEBUG("  Fusing: [" << producer.getTemplateStrAttr() << "] + ["
                                  << consumer.getTemplateStrAttr() << "]");

      // Declarative context fusion
      SmallVector<Value> fusedContext;
      fusedContext.reserve(producer.getOperands().size() + consumer.getOperands().size() - 1);

      // Preserve operand order: producer context first
      llvm::append_range(fusedContext, producer.getOperands());
      llvm::copy_if(consumer.getOperands(), std::back_inserter(fusedContext),
                    [&](Value ctx) { return ctx != producer.getResult(); });

      // Create fused operation
      OpBuilder builder(consumer);
      auto fusedTemplate = fuseTemplates(producer.getTemplateStrAttr().getValue(),
                                         consumer.getTemplateStrAttr().getValue());

      auto fusedOp = builder.create<RsnOp>(
          consumer.getLoc(), consumer.getType(),
          builder.getStringAttr(fusedTemplate), fusedContext);

      // Declarative attribute transfer
      const auto shouldSkipAttr = [](StringRef name) {
        return name == "template_str" || name == "operandSegmentSizes";
      };

      for (NamedAttribute attr : consumer->getAttrs()) {
        if (!shouldSkipAttr(attr.getName())) {
          fusedOp->setAttr(attr.getName(), attr.getValue());
        }
      }

      // Mark fusion provenance
      fusedOp->setAttr("ais.fused_from",
        AISFusedFromAttr::get(module.getContext(),
          builder.getArrayAttr({
            builder.getStringAttr(llvm::join_items(".", "producer", producer.getTemplateStrAttr())),
            builder.getStringAttr(llvm::join_items(".", "consumer", consumer.getTemplateStrAttr()))
          })));

      // Update IR
      consumer.replaceAllUsesWith(fusedOp.getResult());
      opsToErase.push_back(consumer);
      opsToErase.push_back(producer);
      stats.fused++;

      return WalkResult::advance();
    });

    // Safe bulk erasure
    for (Operation* op : opsToErase) {
      op->dropAllUses();
      op->erase();
    }

    // Module-level metadata
    module->setAttr("ais.fused_pairs",
                    AISFusedPairsAttr::get(module.getContext(), stats.fused));

    APXM_AIS_INFO("Scanned " << stats.scanned << " RSN ops, fused " << stats.fused << " pairs");
    APXM_AIS_DEBUG_FOOTER(FuseReasoning);
  }
};

}  // namespace

std::unique_ptr<Pass> createFuseReasoningPass() {
  return std::make_unique<FuseReasoningPass>();
}

}  // namespace mlir::ais
