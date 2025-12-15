/**
 * @file  CapabilityScheduling.cpp
 * @brief Assigns execution tiers and cost estimates to every AIS operation.
 *
 * The pass walks the module once, classifies operations heuristically
 * (keyword matching for `inv`, context-size for `rsn`), and attaches
 * three attributes:
 *   - ais.tier           – io / compute / reasoning / memory / general
 *   - ais.intent         – capability / reasoning / goal / general
 *   - ais.estimated_cost – integer cost for the scheduler
 *   - ais.parallel_safe  – present only when the op is speculation-safe
 *
 * All values are derived from the command-line options declared in
 * Passes.td, so the same pipeline can be re-run with different
 * thresholds without recompiling.
 */

#include "apxm/Dialect/AIS/Transforms/Passes.h"

#include "apxm/Dialect/AIS/IR/AISAttributes.h"
#include "apxm/Dialect/AIS/IR/AISOps.h"
#include "apxm/Dialect/AIS/Support/AISDebug.h"

#include "mlir/IR/Builders.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/TypeSwitch.h"

#include <algorithm>
#include <array>
#include <string>

namespace mlir::ais {
#define GEN_PASS_DEF_CAPABILITYSCHEDULING
#include "apxm/Dialect/AIS/Transforms/Passes.h.inc"

namespace {

APXM_AIS_DEBUG_SETUP(scheduling)

// Declarative capability tier classification
struct TierMapping {
  AISTierKind tier;
  std::array<llvm::StringLiteral, 3> keywords;
  size_t keywordCount;
};

constexpr std::array<TierMapping, 4> kTierMappings = {{
    {AISTierKind::io, {"search", "fetch", "http"}, 3},
    {AISTierKind::compute, {"calc", "math", "solve"}, 3},
    {AISTierKind::reasoning, {"reason", "plan", ""}, 2},
    {AISTierKind::memory, {"mem", "store", ""}, 2},
}};

AISTierKind classifyCapabilityTier(StringRef capability) {
  const std::string lowered = capability.lower();
  StringRef loweredRef(lowered);

  auto matchesKeywords = [&](const TierMapping& mapping) {
    return std::any_of(mapping.keywords.begin(),
                       mapping.keywords.begin() + mapping.keywordCount,
                       [&](StringRef keyword) { return loweredRef.contains(keyword); });
  };

  if (const auto match = llvm::find_if(kTierMappings, matchesKeywords);
      match != kTierMappings.end())
    return match->tier;

  return AISTierKind::general;
}

struct CapabilitySchedulingPass : impl::CapabilitySchedulingBase<CapabilitySchedulingPass> {
  using CapabilitySchedulingBase::CapabilitySchedulingBase;

  void runOnOperation() override {
    APXM_AIS_DEBUG_HEADER(CapabilityScheduling);

    ModuleOp module = getOperation();
    struct Statistics {
      unsigned annotated = 0;
      unsigned invocations = 0;
      unsigned reasonings = 0;
      unsigned plans = 0;
    } stats;

    // Declarative operation annotation
    auto annotateOperation = [&](Operation* op) -> bool {
      return TypeSwitch<Operation*, bool>(op)
          .Case<InvOp>([&](auto inv) {
            annotateInvocation(inv);
            stats.invocations++;
            APXM_AIS_DEBUG("  INV: " << inv.getCapabilityAttr() << " -> tier="
                            << stringifyAISTierKind(classifyCapabilityTier(
                                   inv.getCapabilityAttr().getValue())));
            return true;
          })
          .Case<RsnOp>([&](auto rsn) {
            annotateReasoning(rsn);
            stats.reasonings++;
            APXM_AIS_DEBUG("  RSN: ctx_size=" << rsn.getContext().size()
                            << " parallel_safe=" << (rsn.getContext().size() <= parallelThreshold));
            return true;
          })
          .Case<PlanOp>([&](auto plan) {
            annotatePlan(plan);
            stats.plans++;
            return true;
          })
          .Default(false);
    };

    // Single-pass annotation with statistics collection
    module.walk([&](Operation* op) {
      if (annotateOperation(op)) stats.annotated++;
    });

    // Module-level metadata
    module->setAttr("ais.scheduling_annotations",
                    AISAnnotationsAttr::get(module.getContext(), stats.annotated));

    APXM_AIS_INFO("Annotated " << stats.annotated << " ops"
                  << " (inv=" << stats.invocations
                  << ", rsn=" << stats.reasonings
                  << ", plan=" << stats.plans << ")");
    APXM_AIS_DEBUG_FOOTER(CapabilityScheduling);
  }

private:
  void annotateInvocation(InvOp op) const {
    const AISTierKind tier = classifyCapabilityTier(op.getCapabilityAttr().getValue());
    const unsigned cost = baseCost + contextWeight;

    op->setAttr("ais.tier", AISTierAttr::get(op->getContext(), tier));
    op->setAttr("ais.intent", AISIntentAttr::get(op->getContext(), AISIntentKind::capability));
    op->setAttr("ais.estimated_cost",
                AISEstimatedCostAttr::get(op->getContext(), static_cast<int32_t>(cost)));
  }

  void annotateReasoning(RsnOp op) const {
    const size_t contextSize = op.getContext().size();
    const unsigned cost = baseCost + static_cast<unsigned>(contextSize) * contextWeight;
    const bool isParallelSafe = (contextSize <= parallelThreshold);

    op->setAttr("ais.tier", AISTierAttr::get(op->getContext(), AISTierKind::reasoning));
    op->setAttr("ais.intent", AISIntentAttr::get(op->getContext(), AISIntentKind::reasoning));
    op->setAttr("ais.estimated_cost",
                AISEstimatedCostAttr::get(op->getContext(), static_cast<int32_t>(cost)));

    if (isParallelSafe)
      op->setAttr("ais.parallel_safe", AISParallelSafeAttr::get(op->getContext()));
    else
      op->removeAttr("ais.parallel_safe");
  }

  void annotatePlan(PlanOp op) const {
    const unsigned contextCost = std::max(1u, contextWeight / 2)
                               * static_cast<unsigned>(op.getContext().size());
    const unsigned cost = baseCost + contextCost;

    op->setAttr("ais.intent", AISIntentAttr::get(op->getContext(), AISIntentKind::goal));
    op->setAttr("ais.estimated_cost",
                AISEstimatedCostAttr::get(op->getContext(), static_cast<int32_t>(cost)));
  }
};

}  // namespace

std::unique_ptr<Pass> createCapabilitySchedulingPass() {
  return std::make_unique<CapabilitySchedulingPass>();
}

}  // namespace mlir::ais
