#include "apxm/Dialect/AIS/Conversion/Artifact/ArtifactEmitter.h"

#include "apxm/Dialect/AIS/IR/AISOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/AttributeSupport.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Value.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/raw_ostream.h"
#include <cstring>
#include <optional>
#include <string>
#include <utility>
#include <vector>

using namespace mlir;
using namespace mlir::ais;
using llvm::dyn_cast;
using llvm::isa;

namespace {

enum class OperationKind : uint32_t {
  Inv = 0,
  Rsn = 1,
  QMem = 2,
  UMem = 3,
  Plan = 4,
  WaitAll = 5,
  Merge = 6,
  Fence = 7,
  Exc = 8,
  Communicate = 9,
  Reflect = 10,
  Verify = 11,
  Err = 12,
  ReturnOp = 13,
  Jump = 14,
  BranchOnValue = 15,
  LoopStart = 16,
  LoopEnd = 17,
  TryCatch = 18,
  ConstStr = 19,
};

enum class DependencyKind : uint8_t {
  Data = 0,
  Effect = 1,
  Control = 2,
};

struct ArtifactValue;

struct ArtifactValue {
  enum class Kind : uint8_t {
    Null = 0,
    Bool = 1,
    Integer = 2,
    Float = 3,
    String = 4,
    Array = 5,
    Object = 6,
    Token = 7,
  };

  Kind kind = Kind::Null;
  bool boolValue = false;
  int64_t intValue = 0;
  double floatValue = 0.0;
  std::string stringValue;
  std::vector<ArtifactValue> arrayValues;
  std::vector<std::pair<std::string, ArtifactValue>> objectValues;
  uint64_t tokenId = 0;

  static ArtifactValue null() { return ArtifactValue(); }

  static ArtifactValue boolean(bool value) {
    ArtifactValue v;
    v.kind = Kind::Bool;
    v.boolValue = value;
    return v;
  }

  static ArtifactValue integer(int64_t value) {
    ArtifactValue v;
    v.kind = Kind::Integer;
    v.intValue = value;
    return v;
  }

  static ArtifactValue floating(double value) {
    ArtifactValue v;
    v.kind = Kind::Float;
    v.floatValue = value;
    return v;
  }

  static ArtifactValue string(std::string value) {
    ArtifactValue v;
    v.kind = Kind::String;
    v.stringValue = std::move(value);
    return v;
  }

  static ArtifactValue array(std::vector<ArtifactValue> values) {
    ArtifactValue v;
    v.kind = Kind::Array;
    v.arrayValues = std::move(values);
    return v;
  }

  static ArtifactValue object(std::vector<std::pair<std::string, ArtifactValue>> entries) {
    ArtifactValue v;
    v.kind = Kind::Object;
    v.objectValues = std::move(entries);
    return v;
  }

  static ArtifactValue token(uint64_t value) {
    ArtifactValue v;
    v.kind = Kind::Token;
    v.tokenId = value;
    return v;
  }
};

struct ArtifactNodeMetadata {
  uint32_t priority = 0;
  std::optional<uint64_t> estimatedLatency;
};

struct ArtifactNode {
  uint64_t id = 0;
  OperationKind opType = OperationKind::Inv;
  std::vector<std::pair<std::string, ArtifactValue>> attributes;
  std::vector<uint64_t> inputTokens;
  std::vector<uint64_t> outputTokens;
  ArtifactNodeMetadata metadata;
};

struct ArtifactEdge {
  uint64_t from = 0;
  uint64_t to = 0;
  uint64_t token = 0;
  DependencyKind dependency = DependencyKind::Data;
};

struct ArtifactDag {
  std::vector<ArtifactNode> nodes;
  std::vector<ArtifactEdge> edges;
  std::vector<uint64_t> entryNodes;
  std::vector<uint64_t> exitNodes;
  std::string moduleName;
};

struct DagBuildState {
  uint64_t nextTokenId = 1;
  uint64_t nextNodeId = 1;
  llvm::DenseMap<Value, uint64_t> tokenIds;
  llvm::DenseMap<Value, uint64_t> producerNodeIds;
  std::vector<ArtifactEdge> edges;

  uint64_t getTokenId(Value value) {
    auto it = tokenIds.find(value);
    if (it != tokenIds.end())
      return it->second;

    uint64_t id = nextTokenId++;
    tokenIds.insert({value, id});
    return id;
  }

  void addEdge(uint64_t producerNodeId, uint64_t consumerNodeId,
               uint64_t tokenId, DependencyKind dependency = DependencyKind::Data) {
    ArtifactEdge edge;
    edge.from = producerNodeId;
    edge.to = consumerNodeId;
    edge.token = tokenId;
    edge.dependency = dependency;
    edges.push_back(std::move(edge));
  }
};

class ArtifactSerializer {
public:
  void writeU8(uint8_t value) { buffer.push_back(value); }

  void writeU32(uint32_t value) {
    for (int i = 0; i < 4; ++i)
      buffer.push_back(static_cast<uint8_t>((value >> (i * 8)) & 0xFF));
  }

  void writeU64(uint64_t value) {
    for (int i = 0; i < 8; ++i)
      buffer.push_back(static_cast<uint8_t>((value >> (i * 8)) & 0xFF));
  }

  void writeI64(int64_t value) { writeU64(static_cast<uint64_t>(value)); }

  void writeF64(double value) {
    static_assert(sizeof(double) == sizeof(uint64_t), "Unexpected double size");
    uint64_t bits;
    std::memcpy(&bits, &value, sizeof(double));
    writeU64(bits);
  }

  void writeBool(bool value) { writeU8(value ? 1 : 0); }

  void writeString(StringRef value) {
    writeU64(value.size());
    buffer.insert(buffer.end(), value.begin(), value.end());
  }

  void writeValue(const ArtifactValue &value) {
    writeU8(static_cast<uint8_t>(value.kind));
    switch (value.kind) {
    case ArtifactValue::Kind::Null:
      break;
    case ArtifactValue::Kind::Bool:
      writeBool(value.boolValue);
      break;
    case ArtifactValue::Kind::Integer:
      writeI64(value.intValue);
      break;
    case ArtifactValue::Kind::Float:
      writeF64(value.floatValue);
      break;
    case ArtifactValue::Kind::String:
      writeString(value.stringValue);
      break;
    case ArtifactValue::Kind::Array:
      writeU64(value.arrayValues.size());
      for (const auto &element : value.arrayValues)
        writeValue(element);
      break;
    case ArtifactValue::Kind::Object:
      writeU64(value.objectValues.size());
      for (const auto &entry : value.objectValues) {
        writeString(entry.first);
        writeValue(entry.second);
      }
      break;
    case ArtifactValue::Kind::Token:
      writeU64(value.tokenId);
      break;
    }
  }

  void writeNode(const ArtifactNode &node) {
    writeU64(node.id);
    writeU32(static_cast<uint32_t>(node.opType));

    writeU64(node.attributes.size());
    for (const auto &attr : node.attributes) {
      writeString(attr.first);
      writeValue(attr.second);
    }

    writeU64(node.inputTokens.size());
    for (uint64_t token : node.inputTokens)
      writeU64(token);

    writeU64(node.outputTokens.size());
    for (uint64_t token : node.outputTokens)
      writeU64(token);

    writeU32(node.metadata.priority);
    writeBool(node.metadata.estimatedLatency.has_value());
    if (node.metadata.estimatedLatency)
      writeU64(*node.metadata.estimatedLatency);
  }

  void writeDag(const ArtifactDag &dag) {
    writeString(dag.moduleName);

    writeU64(dag.nodes.size());
    for (const auto &node : dag.nodes)
      writeNode(node);

    writeU64(dag.edges.size());
    for (const auto &edge : dag.edges) {
      writeU64(edge.from);
      writeU64(edge.to);
      writeU64(edge.token);
      writeU8(static_cast<uint8_t>(edge.dependency));
    }

    writeU64(dag.entryNodes.size());
    for (uint64_t id : dag.entryNodes)
      writeU64(id);

    writeU64(dag.exitNodes.size());
    for (uint64_t id : dag.exitNodes)
      writeU64(id);
  }

  std::vector<uint8_t> takeBuffer() { return std::move(buffer); }

private:
  std::vector<uint8_t> buffer;
};

std::optional<OperationKind> mapOperation(Operation *op) {
  return TypeSwitch<Operation *, std::optional<OperationKind>>(op)
      .Case<ConstStrOp>([](auto) { return OperationKind::ConstStr; })
      .Case<QMemOp>([](auto) { return OperationKind::QMem; })
      .Case<UMemOp>([](auto) { return OperationKind::UMem; })
      .Case<InvOp>([](auto) { return OperationKind::Inv; })
      .Case<RsnOp>([](auto) { return OperationKind::Rsn; })
      .Case<ReflectOp>([](auto) { return OperationKind::Reflect; })
      .Case<VerifyOp>([](auto) { return OperationKind::Verify; })
      .Case<PlanOp>([](auto) { return OperationKind::Plan; })
      .Case<ExcOp>([](auto) { return OperationKind::Exc; })
      .Case<WaitAllOp>([](auto) { return OperationKind::WaitAll; })
      .Case<MergeOp>([](auto) { return OperationKind::Merge; })
      .Case<FenceOp>([](auto) { return OperationKind::Fence; })
      .Case<CommunicateOp>([](auto) { return OperationKind::Communicate; })
      .Case<ErrOp>([](auto) { return OperationKind::Err; })
      .Case<ReturnOp>([](auto) { return OperationKind::ReturnOp; })
      .Case<func::ReturnOp>([](auto) { return OperationKind::ReturnOp; })
      .Case<JumpOp>([](auto) { return OperationKind::Jump; })
      .Case<BranchOnValueOp>([](auto) { return OperationKind::BranchOnValue; })
      .Case<LoopStartOp>([](auto) { return OperationKind::LoopStart; })
      .Case<LoopEndOp>([](auto) { return OperationKind::LoopEnd; })
      .Case<TryCatchOp>([](auto) { return OperationKind::TryCatch; })
      .Default([](Operation *) { return std::nullopt; });
}

ArtifactValue convertAttribute(Attribute attr) {
  if (!attr)
    return ArtifactValue::null();

  if (auto strAttr = dyn_cast<StringAttr>(attr))
    return ArtifactValue::string(strAttr.getValue().str());

  if (auto intAttr = dyn_cast<IntegerAttr>(attr)) {
    if (intAttr.getType().isInteger(1))
      return ArtifactValue::boolean(!intAttr.getValue().isZero());
    return ArtifactValue::integer(intAttr.getInt());
  }

  if (auto floatAttr = dyn_cast<FloatAttr>(attr))
    return ArtifactValue::floating(floatAttr.getValueAsDouble());

  if (isa<UnitAttr>(attr))
    return ArtifactValue::boolean(true);

  if (auto arrayAttr = dyn_cast<ArrayAttr>(attr)) {
    std::vector<ArtifactValue> values;
    values.reserve(arrayAttr.size());
    for (Attribute element : arrayAttr)
      values.push_back(convertAttribute(element));
    return ArtifactValue::array(std::move(values));
  }

  if (auto dictAttr = dyn_cast<DictionaryAttr>(attr)) {
    std::vector<std::pair<std::string, ArtifactValue>> entries;
    entries.reserve(dictAttr.size());
    for (NamedAttribute named : dictAttr.getValue()) {
      entries.emplace_back(named.getName().str(), convertAttribute(named.getValue()));
    }
    return ArtifactValue::object(std::move(entries));
  }

  std::string fallback;
  llvm::raw_string_ostream os(fallback);
  attr.print(os);
  os.flush();
  return ArtifactValue::string(fallback);
}

void computeEntryAndExit(const ArtifactDag &dag, std::vector<uint64_t> &entry,
                         std::vector<uint64_t> &exit) {
  llvm::DenseMap<uint64_t, uint64_t> incoming;
  llvm::DenseMap<uint64_t, uint64_t> outgoing;

  for (const auto &node : dag.nodes) {
    incoming.try_emplace(node.id, 0);
    outgoing.try_emplace(node.id, 0);
  }

  for (const auto &edge : dag.edges) {
    incoming[edge.to]++;
    outgoing[edge.from]++;
  }

  for (const auto &node : dag.nodes) {
    if (incoming.lookup(node.id) == 0)
      entry.push_back(node.id);
    if (outgoing.lookup(node.id) == 0)
      exit.push_back(node.id);
  }
}

LogicalResult emitNode(Operation *op, DagBuildState &state, ArtifactDag &dag) {
  auto kind = mapOperation(op);
  if (!kind)
    return op->emitError("Unsupported AIS operation for artifact emission"), failure();

  ArtifactNode node;
  node.id = state.nextNodeId++;
  node.opType = *kind;

  for (NamedAttribute named : op->getAttrs()) {
    StringRef key = named.getName().getValue();
    StringRef emitKey = key == "parameters" ? "params" : key;
    node.attributes.emplace_back(emitKey.str(), convertAttribute(named.getValue()));
  }

  if (isa<ais::PlanOp, ais::RsnOp>(op)) {
    node.attributes.emplace_back("inner_plan_supported", ArtifactValue::boolean(true));
  }

  for (Value operand : op->getOperands()) {
    uint64_t tokenId = state.getTokenId(operand);
    node.inputTokens.push_back(tokenId);
    auto producerIt = state.producerNodeIds.find(operand);
    if (producerIt != state.producerNodeIds.end()) {
      state.addEdge(producerIt->second, node.id, tokenId);
    }
  }

  for (Value result : op->getResults()) {
    uint64_t tokenId = state.getTokenId(result);
    node.outputTokens.push_back(tokenId);
    state.producerNodeIds.insert({result, node.id});
  }

  dag.nodes.push_back(std::move(node));
  return success();
}

LogicalResult emitFunction(func::FuncOp func, const ArtifactEmitOptions &options,
                          ArtifactDag &dag) {
  DagBuildState state;

  for (Block &block : func) {
    for (Operation &op : block) {
      if (failed(emitNode(&op, state, dag)))
        return failure();
    }
  }

  dag.edges = std::move(state.edges);

  dag.entryNodes.clear();
  dag.exitNodes.clear();
  computeEntryAndExit(dag, dag.entryNodes, dag.exitNodes);

  if (!options.moduleName.empty())
    dag.moduleName = options.moduleName;
  else
    dag.moduleName = func.getName().str();

  return success();
}

} // namespace

ArtifactEmitter::ArtifactEmitter(const ArtifactEmitOptions &options)
    : options(options) {}

LogicalResult ArtifactEmitter::emitModule(ModuleOp module) {
  buffer.clear();

  auto funcs = module.getOps<func::FuncOp>();
  if (funcs.empty()) {
    module.emitError("Module does not contain any functions to emit");
    return failure();
  }

  ArtifactDag dag;
  if (failed(emitFunction(*funcs.begin(), options, dag)))
    return failure();

  ArtifactSerializer serializer;
  serializer.writeU32(1); // Wire format version
  serializer.writeDag(dag);
  buffer = serializer.takeBuffer();
  return success();
}
