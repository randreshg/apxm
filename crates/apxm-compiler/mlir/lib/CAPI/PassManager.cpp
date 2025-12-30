/**
 * @file  PassManager.cpp
 * @brief Imperative pass pipeline builder for the C API.
 *
 * Creates an `mlir::PassManager`, exposes `addPassByName` for textual
 * pipelines, and provides typed helpers (`apxm_pass_manager_add_*`) for
 * hosts that prefer compile-time safety.  The manager is independent of
 * the C++ static registry: passes are looked up in the AIS library only.
 */

#include "ais/CAPI/PassManager.h"
#include "ais/CAPI/Module.h"
#include "ais/Dialect/AIS/Transforms/Passes.h"
#include "mlir/Transforms/Passes.h"
#include "llvm/ADT/SmallPtrSet.h"

extern "C" {

ApxmPassManager* apxm_pass_manager_create(ApxmCompilerContext* ctx) {
  return ctx ? new (std::nothrow) ApxmPassManager(ctx) : nullptr;
}

void apxm_pass_manager_destroy(ApxmPassManager* pm) {
  delete pm;
}

void apxm_pass_manager_clear(ApxmPassManager* pm) {
  if (!pm) return;

  pm->pass_manager = std::make_unique<mlir::PassManager>(pm->context->mlir_context.get());
  pm->registered_passes.clear();
}

bool apxm_pass_manager_run(ApxmPassManager* pm, ApxmModule* module) {
  if (!pm || !module || !module->module) {
    return false;
  }
  return mlir::succeeded(pm->pass_manager->run(*module->module));
}

void apxm_pass_manager_add_inline(ApxmPassManager* pm);

bool apxm_pass_manager_has_pass(ApxmPassManager* pm, const char* pass_name) {
  if (!pm || !pass_name) return false;

  // Simple implementation - in real system would use pass registry
  static const char* known_passes[] = {
    "normalize", "fuse-reasoning", "scheduling",
    "canonicalizer", "cse", "symbol-dce", "inline", "lower-to-async",
    "unconsumed-value-warning"
  };

  for (auto name : known_passes) {
    if (strcmp(name, pass_name) == 0) {
      return true;
    }
  }
  return false;
}

bool apxm_pass_manager_add_pass_by_name(ApxmPassManager* pm, const char* pass_name) {
  if (!pm || !pass_name) return false;

  if (strcmp(pass_name, "normalize") == 0) {
    apxm_pass_manager_add_normalize(pm);
    return true;
  }
  if (strcmp(pass_name, "fuse-reasoning") == 0) {
    apxm_pass_manager_add_fuse_reasoning(pm);
    return true;
  }
  if (strcmp(pass_name, "scheduling") == 0) {
    apxm_pass_manager_add_scheduling(pm);
    return true;
  }
  if (strcmp(pass_name, "canonicalizer") == 0) {
    apxm_pass_manager_add_canonicalizer(pm);
    return true;
  }
  if (strcmp(pass_name, "cse") == 0) {
    apxm_pass_manager_add_cse(pm);
    return true;
  }
  if (strcmp(pass_name, "symbol-dce") == 0) {
    apxm_pass_manager_add_symbol_dce(pm);
    return true;
  }
  if (strcmp(pass_name, "inline") == 0) {
    apxm_pass_manager_add_inline(pm);
    return true;
  }
  if (strcmp(pass_name, "lower-to-async") == 0) {
    apxm_pass_manager_add_lower_to_async(pm);
    return true;
  }
  if (strcmp(pass_name, "unconsumed-value-warning") == 0) {
    apxm_pass_manager_add_unconsumed_value_warning(pm);
    return true;
  }

  return false;
}

// Analysis Passes
void apxm_pass_manager_add_unconsumed_value_warning(ApxmPassManager* pm) {
  if (pm) pm->pass_manager->addPass(mlir::ais::createUnconsumedValueWarningPass());
}

// Transform Passes
void apxm_pass_manager_add_normalize(ApxmPassManager* pm) {
  if (pm) pm->pass_manager->addPass(mlir::ais::createNormalizeAgentGraphPass());
}

void apxm_pass_manager_add_fuse_reasoning(ApxmPassManager* pm) {
  if (pm) pm->pass_manager->addPass(mlir::ais::createFuseReasoningPass());
}

void apxm_pass_manager_add_scheduling(ApxmPassManager* pm) {
  if (pm) pm->pass_manager->addPass(mlir::ais::createCapabilitySchedulingPass());
}

// Optimization Passes
void apxm_pass_manager_add_canonicalizer(ApxmPassManager* pm) {
  if (pm) pm->pass_manager->addPass(mlir::createCanonicalizerPass());
}

void apxm_pass_manager_add_cse(ApxmPassManager* pm) {
  if (pm) pm->pass_manager->addPass(mlir::createCSEPass());
}

void apxm_pass_manager_add_symbol_dce(ApxmPassManager* pm) {
  if (pm) pm->pass_manager->addPass(mlir::createSymbolDCEPass());
}

void apxm_pass_manager_add_inline(ApxmPassManager* pm) {
  if (pm) pm->pass_manager->addPass(mlir::createInlinerPass());
}

// Lowering Passes
void apxm_pass_manager_add_lower_to_async(ApxmPassManager* pm) {
#if defined(APXM_HAS_AIS_TO_ASYNC)
  if (pm) pm->pass_manager->addPass(mlir::ais::createAISToAsyncPass());
#else
  (void)pm;
#endif
}

} // extern "C"
