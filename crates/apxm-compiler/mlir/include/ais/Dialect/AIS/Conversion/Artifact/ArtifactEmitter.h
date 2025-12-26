#ifndef APXM_DIALECT_AIS_CONVERSION_ARTIFACT_ARTIFACTEMITTER_H
#define APXM_DIALECT_AIS_CONVERSION_ARTIFACT_ARTIFACTEMITTER_H

#include "mlir/IR/BuiltinOps.h"
#include <cstdint>
#include <string>
#include <vector>

namespace mlir::ais {

struct ArtifactEmitOptions {
  std::string moduleName;
  bool emitDebugJson = false;
  std::string targetVersion;
};

class ArtifactEmitter {
public:
  explicit ArtifactEmitter(const ArtifactEmitOptions &options = {});

  LogicalResult emitModule(ModuleOp module);

  const std::vector<uint8_t> &getBuffer() const { return buffer; }

private:
  ArtifactEmitOptions options;
  std::vector<uint8_t> buffer;
};

} // namespace mlir::ais

#endif // APXM_DIALECT_AIS_CONVERSION_ARTIFACT_ARTIFACTEMITTER_H
