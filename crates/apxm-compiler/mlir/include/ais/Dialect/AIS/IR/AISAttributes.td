/**
 * @file  AISAttributes.td
 * @brief Attribute definitions for the AIS dialect.
 *
 * The file defines two kinds of attributes:
 * 1. Enum attributes that mirror C++ enums (AISTierKind, AISIntentKind).
 *    The helper templates `AIS_I32Enum` and `AIS_I32EnumAttr` remove
 *    repetition and guarantee a uniform assembly format `<value>`.
 * 2. Free-standing attributes that capture optimisation metadata produced
 *    by passes (cost, parallel safety, fusion provenance, counters...).
 *    These attributes are *not* tied to a specific operation so that
 *    multiple passes can incrementally annotate the IR.
 */

#ifndef APXM_AIS_ATTRIBUTES_TD
#define APXM_AIS_ATTRIBUTES_TD

include "mlir/IR/AttrTypeBase.td"
include "mlir/IR/EnumAttr.td"
include "mlir/IR/BuiltinAttributes.td"
include "ais/Dialect/AIS/IR/AISDialect.td"

// Enum helper mirroring the ARTS style.
class AIS_I32Enum<string name, string description, list<I32EnumAttrCase> cases>
    : I32EnumAttr<name, description, cases> {
  let cppNamespace = "::mlir::ais";
  let genSpecializedAttr = 0;
}

class AIS_I32EnumAttr<string mnemonic, AIS_I32Enum enumInfo>
    : EnumAttr<AIS_Dialect, enumInfo, mnemonic> {
  let cppNamespace = "::mlir::ais";
  let assemblyFormat = "`<` $value `>`";
}

// Tier classification for capabilities and reasoning ops.
def AISTierKind : AIS_I32Enum<"AISTierKind",
    "Capability tier classification",
    [
      I32EnumAttrCase<"io", 0, "io">,
      I32EnumAttrCase<"compute", 1, "compute">,
      I32EnumAttrCase<"reasoning", 2, "reasoning">,
      I32EnumAttrCase<"memory", 3, "memory">,
      I32EnumAttrCase<"general", 4, "general">
    ]>;

def AISTierAttr : AIS_I32EnumAttr<"tier", AISTierKind> {
  let cppClassName = "AISTierAttr";
}

// Intent classification used by scheduling heuristics.
def AISIntentKind : AIS_I32Enum<"AISIntentKind",
    "Agent intent classification",
    [
      I32EnumAttrCase<"capability", 0, "capability">,
      I32EnumAttrCase<"reasoning", 1, "reasoning">,
      I32EnumAttrCase<"goal", 2, "goal">,
      I32EnumAttrCase<"general", 3, "general">
    ]>;

def AISIntentAttr : AIS_I32EnumAttr<"intent", AISIntentKind> {
  let cppClassName = "AISIntentAttr";
}

// Relative cost used by the scheduler.
def AISEstimatedCostAttr : AttrDef<AIS_Dialect, "estimated_cost"> {
  let summary = "Estimated relative cost (heuristic)";
  let parameters = (ins "int32_t":$value);
  let mnemonic = "estimated_cost";
  let cppClassName = "AISEstimatedCostAttr";
  let assemblyFormat = "`<` $value `>`";
}

// Marker for parallel-safe reasoning.
def AISParallelSafeAttr : AttrDef<AIS_Dialect, "parallel_safe"> {
  let summary = "Marks reasoning calls safe for parallel execution";
  let mnemonic = "parallel_safe";
  let cppClassName = "AISParallelSafeAttr";
  let assemblyFormat = "";
}

// Provenance for fused reasoning.
def AISFusedFromAttr : AttrDef<AIS_Dialect, "fused_from"> {
  let summary = "Records producer/consumer pair fused into a reasoning op";
  let parameters = (ins "::mlir::ArrayAttr":$sources);
  let mnemonic = "fused_from";
  let cppClassName = "AISFusedFromAttr";
  let assemblyFormat = "`<` $sources `>`";
}

// Module-level counters emitted by passes.
def AISAnnotationsAttr : AttrDef<AIS_Dialect, "scheduling_annotations"> {
  let summary = "Count of ops annotated by scheduling";
  let parameters = (ins "int64_t":$value);
  let mnemonic = "scheduling_annotations";
  let cppClassName = "AISAnnotationsAttr";
  let assemblyFormat = "`<` $value `>`";
}

def AISFusedPairsAttr : AttrDef<AIS_Dialect, "fused_pairs"> {
  let summary = "Number of reasoning pairs fused";
  let parameters = (ins "int64_t":$value);
  let mnemonic = "fused_pairs";
  let cppClassName = "AISFusedPairsAttr";
  let assemblyFormat = "`<` $value `>`";
}

def AISGraphNormalizedAttr : AttrDef<AIS_Dialect, "graph_normalized"> {
  let summary = "Number of normalization edits";
  let parameters = (ins "int64_t":$value);
  let mnemonic = "graph_normalized";
  let cppClassName = "AISGraphNormalizedAttr";
  let assemblyFormat = "`<` $value `>`";
}

#endif // APXM_AIS_ATTRIBUTES_TD
