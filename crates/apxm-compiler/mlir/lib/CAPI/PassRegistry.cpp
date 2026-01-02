/**
 * @file  PassRegistry.cpp
 * @brief Introspection API that lists every pass known to the library.
 *
 * The list is hard-coded to avoid linking in the full MLIR registry;
 * clients can query name, description and category at runtime to build
 * UI menus or validate user-supplied pipelines without touching C++.
 */

#include "ais/CAPI/PassRegistry.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/SmallVector.h"

namespace {

struct PassDescriptor {
  const char* name;
  const char* description;
  ApxmPassCategory category;
};

// Generated from Rust pass definitions - see apxm-ais/src/passes/mod.rs
constexpr PassDescriptor kPassDescriptors[] = {
  #include "ais/CAPI/PassDescriptors.inc"
};

class PassRegistry {
public:
  static PassRegistry& instance() {
    static PassRegistry registry;
    return registry;
  }

  size_t getCount() const {
    return descriptors.size();
  }

  const PassDescriptor* getDescriptor(size_t index) const {
    return index < descriptors.size() ? descriptors[index] : nullptr;
  }

  const PassDescriptor* findDescriptor(llvm::StringRef name) const {
    auto it = nameMap.find(name);
    return it != nameMap.end() ? it->second : nullptr;
  }

private:
  PassRegistry() {
    for (const auto& desc : kPassDescriptors) {
      descriptors.push_back(&desc);
      nameMap[desc.name] = &desc;
    }
  }

  llvm::SmallVector<const PassDescriptor*, 8> descriptors;
  llvm::StringMap<const PassDescriptor*> nameMap;
};

llvm::SmallVector<ApxmPassInfo, 8> infoCache;

const ApxmPassInfo* makePassInfo(const PassDescriptor* descriptor) {
  if (!descriptor) return nullptr;

  ApxmPassInfo info;
  info.name = descriptor->name;
  info.description = descriptor->description;
  info.category = descriptor->category;

  infoCache.push_back(info);
  return &infoCache.back();
}

} // anonymous namespace

extern "C" {

size_t apxm_pass_registry_get_count() {
  return PassRegistry::instance().getCount();
}

const ApxmPassInfo* apxm_pass_registry_get_pass(size_t index) {
  auto* descriptor = PassRegistry::instance().getDescriptor(index);
  return makePassInfo(descriptor);
}

const ApxmPassInfo* apxm_pass_registry_find_pass(const char* name) {
  if (!name) return nullptr;
  auto* descriptor = PassRegistry::instance().findDescriptor(name);
  return makePassInfo(descriptor);
}

} // extern "C"
