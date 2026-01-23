/**
 * @file  BuildPrompt.cpp
 * @brief Generates placeholder templates for LLM operations with empty template_str.
 *
 * This pass addresses the case where LLM operations (ask/think/reason) have
 * empty template strings but non-empty context arrays. For example:
 *
 *   DSL:   ask(user_input)
 *   MLIR:  %r = ais.ask "" [%user_input : !ais.token] : !ais.token
 *
 * Without this pass, the empty template_str causes broken prompts at runtime.
 * This pass transforms it to:
 *
 *   %r = ais.ask "{0}" [%user_input : !ais.token] : !ais.token
 *
 * The "{0}" placeholder is then substituted by the runtime with context[0].
 *
 * This pass works alongside the InstructionConfig system:
 * - BuildPrompt: Ensures template_str is never empty when context exists
 * - InstructionConfig: Maps operation types to system prompts at runtime
 */

#include "ais/Dialect/AIS/Transforms/Passes.h"

#include "ais/Dialect/AIS/IR/AISOps.h"
#include "ais/Dialect/AIS/Support/AISDebug.h"

#include "mlir/IR/Builders.h"
#include "llvm/ADT/TypeSwitch.h"

namespace mlir::ais {
#define GEN_PASS_DEF_BUILDPROMPT
#include "ais/Dialect/AIS/Transforms/Passes.h.inc"

namespace {

APXM_AIS_DEBUG_SETUP(build_prompt)

struct BuildPromptPass : impl::BuildPromptBase<BuildPromptPass> {
  using BuildPromptBase::BuildPromptBase;

  void runOnOperation() override {
    APXM_AIS_DEBUG_HEADER(BuildPrompt);
    ModuleOp module = getOperation();
    unsigned modified = 0;

    module.walk([&](Operation *op) {
      llvm::TypeSwitch<Operation *>(op)
          .Case<AskOp>([&](AskOp askOp) {
            if (processLlmOp(askOp))
              modified++;
          })
          .Case<ThinkOp>([&](ThinkOp thinkOp) {
            if (processLlmOp(thinkOp))
              modified++;
          })
          .Case<ReasonOp>([&](ReasonOp reasonOp) {
            if (processLlmOp(reasonOp))
              modified++;
          });
    });

    if (modified > 0) {
      module->setAttr("ais.prompts_built",
                      IntegerAttr::get(IntegerType::get(module.getContext(), 64),
                                       modified));
      APXM_AIS_INFO("Built prompts for " << modified << " operations");
    } else {
      APXM_AIS_DEBUG("No operations required prompt building");
    }

    APXM_AIS_DEBUG_FOOTER(BuildPrompt);
  }

private:
  /// Process an LLM operation (AskOp, ThinkOp, or ReasonOp).
  /// Returns true if the operation was modified.
  template <typename LlmOpT>
  bool processLlmOp(LlmOpT op) {
    StringRef currentTemplate = op.getTemplateStrAttr().getValue();

    // Only process if template is empty AND context exists
    if (!currentTemplate.empty()) {
      APXM_AIS_DEBUG("  Skipping op with non-empty template: \""
                     << currentTemplate << "\"");
      return false;
    }

    if (op.getContext().empty()) {
      APXM_AIS_DEBUG("  Skipping op with empty context");
      return false;
    }

    if (!generatePlaceholders) {
      APXM_AIS_DEBUG("  Placeholder generation disabled");
      return false;
    }

    // Generate placeholder: "{0}" means "use context[0] as prompt"
    OpBuilder builder(op);
    op.setTemplateStrAttr(builder.getStringAttr("{0}"));

    APXM_AIS_INFO("  Generated {0} placeholder for " << op->getName()
                                                      << " with "
                                                      << op.getContext().size()
                                                      << " context operands");
    return true;
  }
};

} // namespace

std::unique_ptr<Pass> createBuildPromptPass() {
  return std::make_unique<BuildPromptPass>();
}

} // namespace mlir::ais
