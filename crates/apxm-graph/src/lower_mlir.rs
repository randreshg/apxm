use crate::{ApxmGraph, GraphError, GraphNode};
use apxm_core::constants::graph::{attrs as graph_attrs, metadata as graph_meta};
use apxm_core::types::AISOperationType;
use apxm_core::types::{Number, Value};
use std::collections::{BTreeSet, HashMap};

#[derive(Clone, Debug)]
enum MlirValueType {
    Token,
    Handle { space: String },
    Goal,
}

#[derive(Clone, Debug)]
struct MlirValueRef {
    ssa: String,
    ty: MlirValueType,
}

struct LoweringState {
    lines: Vec<String>,
    next_temp: u64,
}

impl LoweringState {
    fn new() -> Self {
        Self {
            lines: Vec::new(),
            next_temp: 0,
        }
    }

    fn emit(&mut self, line: impl Into<String>) {
        self.lines.push(line.into());
    }

    fn fresh_value(&mut self, prefix: &str) -> String {
        self.next_temp += 1;
        format!("%{}_{}", prefix, self.next_temp)
    }
}

pub fn lower_to_mlir(graph: &ApxmGraph) -> Result<String, GraphError> {
    graph.validate()?;

    let nodes_by_id = graph
        .nodes
        .iter()
        .map(|node| (node.id, node))
        .collect::<HashMap<_, _>>();
    let order = topo_order(graph, &nodes_by_id)?;

    let mut incoming_by_target: HashMap<u64, Vec<u64>> = HashMap::new();
    let mut outgoing_counts: HashMap<u64, usize> = HashMap::new();
    for edge in &graph.edges {
        incoming_by_target
            .entry(edge.to)
            .or_default()
            .push(edge.from);
        *outgoing_counts.entry(edge.from).or_default() += 1;
    }

    let is_entry = graph
        .metadata
        .get(graph_meta::IS_ENTRY)
        .and_then(Value::as_boolean)
        .unwrap_or(true);

    let mut state = LoweringState::new();
    let mut produced_values: HashMap<u64, MlirValueRef> = HashMap::new();
    let mut arg_values = Vec::with_capacity(graph.parameters.len());
    for (index, _) in graph.parameters.iter().enumerate() {
        arg_values.push(MlirValueRef {
            ssa: format!("%arg{index}"),
            ty: MlirValueType::Token,
        });
    }

    for node_id in &order {
        let node = nodes_by_id.get(node_id).ok_or_else(|| {
            GraphError::Lowering(format!("node id {node_id} missing during lowering"))
        })?;

        let mut inputs = incoming_by_target
            .get(node_id)
            .into_iter()
            .flatten()
            .map(|source_id| {
                produced_values.get(source_id).cloned().ok_or_else(|| {
                    GraphError::Lowering(format!(
                        "node {} references source {} with no produced value",
                        node.id, source_id
                    ))
                })
            })
            .collect::<Result<Vec<_>, _>>()?;

        if inputs.is_empty() && !arg_values.is_empty() {
            inputs.extend(arg_values.clone());
        }

        let node_result = emit_node(&mut state, node, inputs.clone())?;
        if let Some(value) = node_result {
            produced_values.insert(*node_id, value);
            continue;
        }

        if outgoing_counts.get(node_id).copied().unwrap_or(0) > 0 {
            let bridge = emit_bridge_token(&mut state, node.id, &inputs)?;
            produced_values.insert(*node_id, bridge);
        }
    }

    let exit_ids = graph
        .nodes
        .iter()
        .filter(|node| outgoing_counts.get(&node.id).copied().unwrap_or(0) == 0)
        .map(|node| node.id)
        .collect::<Vec<_>>();

    let mut return_candidates = exit_ids
        .iter()
        .filter_map(|id| produced_values.get(id).cloned())
        .collect::<Vec<_>>();

    if return_candidates.is_empty() {
        for node_id in order.iter().rev() {
            if let Some(value) = produced_values.get(node_id) {
                return_candidates.push(value.clone());
                break;
            }
        }
    }

    let return_value = if return_candidates.is_empty() {
        emit_const_token(&mut state, "result")
    } else {
        let token_values = return_candidates
            .into_iter()
            .map(|value| ensure_token(&mut state, value))
            .collect::<Result<Vec<_>, _>>()?;

        if token_values.len() == 1 {
            token_values
                .into_iter()
                .next()
                .ok_or_else(|| GraphError::Lowering("missing return value".to_string()))?
        } else {
            let output = state.fresh_value("ret_merge");
            let operands = token_values
                .iter()
                .map(|value| value.ssa.clone())
                .collect::<Vec<_>>()
                .join(", ");
            let types = vec!["!ais.token"; token_values.len()].join(", ");
            state.emit(format!(
                "    {output} = ais.merge {operands} : {types} -> !ais.token"
            ));
            MlirValueRef {
                ssa: output,
                ty: MlirValueType::Token,
            }
        }
    };

    state.emit(format!("    func.return {} : !ais.token", return_value.ssa));

    let function_name = sanitize_symbol_name(&graph.name);
    let args = graph
        .parameters
        .iter()
        .enumerate()
        .map(|(index, parameter)| {
            format!(
                "%arg{index}: !ais.token {{ais.param_name = {}, ais.param_type = {}}}",
                quote_string(&parameter.name),
                quote_string(&parameter.type_name)
            )
        })
        .collect::<Vec<_>>()
        .join(", ");

    let function_attrs = if is_entry {
        " attributes {ais.entry}"
    } else {
        ""
    };

    let mut mlir = String::new();
    mlir.push_str("module {\n");
    mlir.push_str(&format!(
        "  func.func @{}({}) -> !ais.token{} {{\n",
        function_name, args, function_attrs
    ));
    for line in state.lines {
        mlir.push_str(&line);
        mlir.push('\n');
    }
    mlir.push_str("  }\n");
    mlir.push_str("}\n");

    Ok(mlir)
}

fn topo_order(
    graph: &ApxmGraph,
    nodes_by_id: &HashMap<u64, &GraphNode>,
) -> Result<Vec<u64>, GraphError> {
    let mut in_degree = nodes_by_id
        .keys()
        .map(|id| (*id, 0usize))
        .collect::<HashMap<_, _>>();
    let mut outgoing: HashMap<u64, Vec<u64>> = HashMap::new();

    for edge in &graph.edges {
        outgoing.entry(edge.from).or_default().push(edge.to);
        *in_degree.entry(edge.to).or_insert(0) += 1;
    }

    let mut ready = in_degree
        .iter()
        .filter_map(|(id, degree)| if *degree == 0 { Some(*id) } else { None })
        .collect::<BTreeSet<_>>();
    let mut order = Vec::with_capacity(nodes_by_id.len());

    while let Some(next_id) = ready.pop_first() {
        order.push(next_id);
        if let Some(neighbors) = outgoing.get(&next_id) {
            for neighbor in neighbors {
                if let Some(degree) = in_degree.get_mut(neighbor) {
                    *degree = degree.saturating_sub(1);
                    if *degree == 0 {
                        ready.insert(*neighbor);
                    }
                }
            }
        }
    }

    if order.len() != nodes_by_id.len() {
        return Err(GraphError::Lowering(
            "topological ordering failed (graph may not be acyclic)".to_string(),
        ));
    }

    Ok(order)
}

fn emit_node(
    state: &mut LoweringState,
    node: &GraphNode,
    inputs: Vec<MlirValueRef>,
) -> Result<Option<MlirValueRef>, GraphError> {
    match node.op {
        AISOperationType::ConstStr => {
            let value = get_string_attr(
                &node.attributes,
                &[
                    "value",
                    "text",
                    "const",
                    graph_attrs::TEMPLATE_STR,
                    graph_attrs::PROMPT,
                ],
            )
            .unwrap_or_default();
            let result = format!("%n{}", node.id);
            let attrs = extra_attr_dict(
                &node.attributes,
                &[
                    "value",
                    "text",
                    "const",
                    graph_attrs::TEMPLATE_STR,
                    graph_attrs::PROMPT,
                ],
            );
            state.emit(format!(
                "    {result} = ais.const_str {}{} : !ais.token",
                quote_string(&value),
                attrs
            ));
            Ok(Some(MlirValueRef {
                ssa: result,
                ty: MlirValueType::Token,
            }))
        }
        AISOperationType::Ask => {
            let template = get_non_empty_template(&node.attributes);
            let result = format!("%n{}", node.id);
            let attrs = extra_attr_dict(
                &node.attributes,
                &[graph_attrs::TEMPLATE_STR, graph_attrs::PROMPT],
            );
            let context = format_context_brackets(&inputs);
            state.emit(format!(
                "    {result} = ais.ask {}{}{} : !ais.token",
                quote_string(&template),
                context,
                attrs
            ));
            Ok(Some(MlirValueRef {
                ssa: result,
                ty: MlirValueType::Token,
            }))
        }
        AISOperationType::Think => {
            let template = get_non_empty_template(&node.attributes);
            let result = format!("%n{}", node.id);
            let attrs = extra_attr_dict(
                &node.attributes,
                &[graph_attrs::TEMPLATE_STR, graph_attrs::PROMPT],
            );
            let context = format_context_brackets(&inputs);
            state.emit(format!(
                "    {result} = ais.think {}{}{} : !ais.token",
                quote_string(&template),
                context,
                attrs
            ));
            Ok(Some(MlirValueRef {
                ssa: result,
                ty: MlirValueType::Token,
            }))
        }
        AISOperationType::Reason => {
            let template = get_non_empty_template(&node.attributes);
            let result = format!("%n{}", node.id);
            let attrs = extra_attr_dict(
                &node.attributes,
                &[graph_attrs::TEMPLATE_STR, graph_attrs::PROMPT],
            );
            let context = format_context_brackets(&inputs);
            state.emit(format!(
                "    {result} = ais.reason {}{}{} : !ais.token",
                quote_string(&template),
                context,
                attrs
            ));
            Ok(Some(MlirValueRef {
                ssa: result,
                ty: MlirValueType::Token,
            }))
        }
        AISOperationType::QMem => {
            let query =
                get_string_attr(&node.attributes, &[graph_attrs::QUERY]).unwrap_or_default();
            let sid = get_string_attr(&node.attributes, &["sid", "stage", "scope"])
                .unwrap_or_else(|| "default".to_string());
            let space = normalize_memory_space(
                &get_string_attr(
                    &node.attributes,
                    &[graph_attrs::SPACE, graph_attrs::MEMORY_TIER],
                )
                .unwrap_or_else(|| "stm".to_string()),
            );
            let limit = get_u64_attr(&node.attributes, "limit");
            let result = format!("%n{}", node.id);
            let attrs = extra_attr_dict(
                &node.attributes,
                &[
                    graph_attrs::QUERY,
                    "sid",
                    "stage",
                    "scope",
                    graph_attrs::SPACE,
                    graph_attrs::MEMORY_TIER,
                    "limit",
                ],
            );
            let limit_str = limit
                .map(|value| format!(" limit {value}"))
                .unwrap_or_default();
            let handle_type = format!("!ais.handle<{space}>");

            state.emit(format!(
                "    {result} = ais.qmem {} stage {} in {}{}{} : {}",
                quote_string(&query),
                quote_string(&sid),
                quote_string(&space),
                limit_str,
                attrs,
                handle_type
            ));
            Ok(Some(MlirValueRef {
                ssa: result,
                ty: MlirValueType::Handle { space },
            }))
        }
        AISOperationType::UMem => {
            let space = normalize_memory_space(
                &get_string_attr(
                    &node.attributes,
                    &[graph_attrs::SPACE, graph_attrs::MEMORY_TIER],
                )
                .unwrap_or_else(|| "stm".to_string()),
            );
            let attrs = extra_attr_dict(
                &node.attributes,
                &[graph_attrs::SPACE, graph_attrs::MEMORY_TIER],
            );

            let source = if let Some(input) = inputs.first() {
                ensure_token(state, input.clone())?
            } else {
                emit_const_token(state, "memory")
            };

            state.emit(format!(
                "    ais.umem {} into {}{} : !ais.token",
                source.ssa,
                quote_string(&space),
                attrs
            ));
            Ok(None)
        }
        AISOperationType::Inv => {
            let capability = get_string_attr(&node.attributes, &[graph_attrs::CAPABILITY])
                .unwrap_or_else(|| "unknown_capability".to_string());
            let params_json = get_string_attr(&node.attributes, &[graph_attrs::PARAMS_JSON])
                .unwrap_or_else(|| "{}".to_string());
            let attrs = extra_attr_dict(
                &node.attributes,
                &[graph_attrs::CAPABILITY, graph_attrs::PARAMS_JSON],
            );
            let result = format!("%n{}", node.id);
            state.emit(format!(
                "    {result} = ais.inv {} ({}){} : !ais.token",
                quote_string(&capability),
                quote_string(&params_json),
                attrs
            ));
            Ok(Some(MlirValueRef {
                ssa: result,
                ty: MlirValueType::Token,
            }))
        }
        AISOperationType::BranchOnValue => {
            let mut condition = if let Some(input) = inputs.first() {
                input.clone()
            } else {
                emit_const_token(state, "branch")
            };
            if matches!(condition.ty, MlirValueType::Goal) {
                condition = ensure_token(state, condition)?;
            }

            let true_label = get_string_attr(&node.attributes, &["true_label"])
                .unwrap_or_else(|| "true".to_string());
            let false_label = get_string_attr(&node.attributes, &["false_label"])
                .unwrap_or_else(|| "false".to_string());
            let attrs = extra_attr_dict(&node.attributes, &["true_label", "false_label"]);

            state.emit(format!(
                "    ais.branch_on_value {}, {}, {}{} : {}",
                condition.ssa,
                quote_string(&true_label),
                quote_string(&false_label),
                attrs,
                format_type(&condition.ty)
            ));
            Ok(None)
        }
        AISOperationType::Switch => {
            let discriminant = if let Some(input) = inputs.first() {
                ensure_token(state, input.clone())?
            } else {
                emit_const_token(state, "switch")
            };
            let labels = get_string_array_attr(&node.attributes, "case_labels");
            let case_labels = if labels.is_empty() {
                vec!["default_case".to_string()]
            } else {
                labels
            };
            let attrs = extra_attr_dict(&node.attributes, &["case_labels"]);
            let result = format!("%n{}", node.id);

            state.emit(format!(
                "    {result} = ais.switch {} : !ais.token",
                discriminant.ssa
            ));
            for label in &case_labels {
                state.emit(format!("        case {} {{", quote_string(label)));
                state.emit(format!(
                    "            ais.yield {} : !ais.token",
                    discriminant.ssa
                ));
                state.emit("        }");
            }
            state.emit("        default {");
            state.emit(format!(
                "            ais.yield {} : !ais.token",
                discriminant.ssa
            ));
            if attrs.is_empty() {
                state.emit("        } -> !ais.token");
            } else {
                state.emit(format!("        }} -> !ais.token{}", attrs));
            }

            Ok(Some(MlirValueRef {
                ssa: result,
                ty: MlirValueType::Token,
            }))
        }
        AISOperationType::WaitAll => {
            let tokens = inputs
                .into_iter()
                .map(|value| ensure_token(state, value))
                .collect::<Result<Vec<_>, _>>()?;
            let result = format!("%n{}", node.id);
            let attrs = extra_attr_dict(&node.attributes, &[]);

            if tokens.is_empty() {
                state.emit(format!(
                    "    {result} = ais.wait_all{} -> !ais.token",
                    attrs
                ));
            } else {
                let operands = tokens
                    .iter()
                    .map(|token| token.ssa.clone())
                    .collect::<Vec<_>>()
                    .join(", ");
                let types = vec!["!ais.token"; tokens.len()].join(", ");
                state.emit(format!(
                    "    {result} = ais.wait_all {operands} : {types}{attrs} -> !ais.token"
                ));
            }
            Ok(Some(MlirValueRef {
                ssa: result,
                ty: MlirValueType::Token,
            }))
        }
        AISOperationType::Merge => {
            let tokens = inputs
                .into_iter()
                .map(|value| ensure_token(state, value))
                .collect::<Result<Vec<_>, _>>()?;
            let result = format!("%n{}", node.id);
            let attrs = extra_attr_dict(&node.attributes, &[]);

            if tokens.is_empty() {
                state.emit(format!("    {result} = ais.merge{} -> !ais.token", attrs));
            } else {
                let operands = tokens
                    .iter()
                    .map(|token| token.ssa.clone())
                    .collect::<Vec<_>>()
                    .join(", ");
                let types = vec!["!ais.token"; tokens.len()].join(", ");
                state.emit(format!(
                    "    {result} = ais.merge {operands} : {types}{attrs} -> !ais.token"
                ));
            }
            Ok(Some(MlirValueRef {
                ssa: result,
                ty: MlirValueType::Token,
            }))
        }
        AISOperationType::Fence => {
            let attrs = extra_attr_dict(&node.attributes, &[]);
            state.emit(format!("    ais.fence{attrs}"));
            Ok(None)
        }
        AISOperationType::Plan => {
            let goal = get_string_attr(&node.attributes, &[graph_attrs::GOAL])
                .unwrap_or_else(|| "goal".to_string());
            let attrs = extra_attr_dict(&node.attributes, &[graph_attrs::GOAL]);
            let result = format!("%n{}", node.id);
            let context = format_context_parens(&inputs);

            state.emit(format!(
                "    {result} = ais.plan {}{}{} : !ais.goal<0>",
                quote_string(&goal),
                context,
                attrs
            ));
            Ok(Some(MlirValueRef {
                ssa: result,
                ty: MlirValueType::Goal,
            }))
        }
        AISOperationType::Reflect => {
            let trace_id = get_string_attr(
                &node.attributes,
                &[graph_attrs::TRACE_ID, graph_attrs::TRACE],
            )
            .unwrap_or_else(|| graph_attrs::TRACE.to_string());
            let attrs = extra_attr_dict(
                &node.attributes,
                &[graph_attrs::TRACE_ID, graph_attrs::TRACE],
            );
            let result = format!("%n{}", node.id);
            let context = format_context_parens(&inputs);

            state.emit(format!(
                "    {result} = ais.reflect {}{}{} : !ais.token",
                quote_string(&trace_id),
                context,
                attrs
            ));
            Ok(Some(MlirValueRef {
                ssa: result,
                ty: MlirValueType::Token,
            }))
        }
        AISOperationType::Verify => {
            let template = get_string_attr(
                &node.attributes,
                &[graph_attrs::TEMPLATE_STR, graph_attrs::PROMPT],
            )
            .unwrap_or_else(|| "Verify claim against evidence".to_string());
            let attrs = extra_attr_dict(
                &node.attributes,
                &[graph_attrs::TEMPLATE_STR, graph_attrs::PROMPT],
            );

            let claim = if let Some(input) = inputs.first() {
                input.clone()
            } else {
                emit_const_token(state, "claim")
            };
            let evidence = if let Some(input) = inputs.get(1) {
                input.clone()
            } else {
                claim.clone()
            };

            let result = format!("%n{}", node.id);
            state.emit(format!(
                "    {result} = ais.verify {} : {} vs {} : {} with {}{} : !ais.token",
                claim.ssa,
                format_type(&claim.ty),
                evidence.ssa,
                format_type(&evidence.ty),
                quote_string(&template),
                attrs
            ));
            Ok(Some(MlirValueRef {
                ssa: result,
                ty: MlirValueType::Token,
            }))
        }
        unsupported => Err(GraphError::Lowering(format!(
            "unsupported operation '{}' in MLIR lowering",
            unsupported
        ))),
    }
}

fn emit_bridge_token(
    state: &mut LoweringState,
    node_id: u64,
    inputs: &[MlirValueRef],
) -> Result<MlirValueRef, GraphError> {
    let result = format!("%n{node_id}");
    let tokens = inputs
        .iter()
        .cloned()
        .map(|value| ensure_token(state, value))
        .collect::<Result<Vec<_>, _>>()?;

    if tokens.is_empty() {
        state.emit(format!("    {result} = ais.wait_all -> !ais.token"));
    } else {
        let operands = tokens
            .iter()
            .map(|token| token.ssa.clone())
            .collect::<Vec<_>>()
            .join(", ");
        let types = vec!["!ais.token"; tokens.len()].join(", ");
        state.emit(format!(
            "    {result} = ais.wait_all {operands} : {types} -> !ais.token"
        ));
    }

    Ok(MlirValueRef {
        ssa: result,
        ty: MlirValueType::Token,
    })
}

fn ensure_token(
    state: &mut LoweringState,
    value: MlirValueRef,
) -> Result<MlirValueRef, GraphError> {
    if matches!(value.ty, MlirValueType::Token) {
        return Ok(value);
    }

    let result = state.fresh_value("tok");
    let source_type = format_type(&value.ty);
    state.emit(format!(
        "    {result} = ais.ask \"{{0}}\" [{} : {}] : !ais.token",
        value.ssa, source_type
    ));

    Ok(MlirValueRef {
        ssa: result,
        ty: MlirValueType::Token,
    })
}

fn emit_const_token(state: &mut LoweringState, value: &str) -> MlirValueRef {
    let result = state.fresh_value("const");
    state.emit(format!(
        "    {result} = ais.const_str {} : !ais.token",
        quote_string(value)
    ));
    MlirValueRef {
        ssa: result,
        ty: MlirValueType::Token,
    }
}

fn format_context_brackets(values: &[MlirValueRef]) -> String {
    if values.is_empty() {
        return String::new();
    }
    let operands = values
        .iter()
        .map(|value| value.ssa.clone())
        .collect::<Vec<_>>()
        .join(", ");
    let types = values
        .iter()
        .map(|value| format_type(&value.ty).to_string())
        .collect::<Vec<_>>()
        .join(", ");
    format!(" [{operands} : {types}]")
}

fn format_context_parens(values: &[MlirValueRef]) -> String {
    if values.is_empty() {
        return String::new();
    }
    let operands = values
        .iter()
        .map(|value| value.ssa.clone())
        .collect::<Vec<_>>()
        .join(", ");
    let types = values
        .iter()
        .map(|value| format_type(&value.ty).to_string())
        .collect::<Vec<_>>()
        .join(", ");
    format!(" ({operands} : {types})")
}

fn format_type(value_type: &MlirValueType) -> String {
    match value_type {
        MlirValueType::Token => "!ais.token".to_string(),
        MlirValueType::Handle { space } => format!("!ais.handle<{space}>"),
        MlirValueType::Goal => "!ais.goal<0>".to_string(),
    }
}

fn get_non_empty_template(attributes: &HashMap<String, Value>) -> String {
    get_string_attr(
        attributes,
        &[
            graph_attrs::TEMPLATE_STR,
            graph_attrs::PROMPT,
            graph_attrs::TEMPLATE,
        ],
    )
    .filter(|value| !value.trim().is_empty())
    .unwrap_or_else(|| "{0}".to_string())
}

fn get_string_attr(attributes: &HashMap<String, Value>, keys: &[&str]) -> Option<String> {
    keys.iter()
        .find_map(|key| attributes.get(*key))
        .and_then(value_to_string)
}

fn get_u64_attr(attributes: &HashMap<String, Value>, key: &str) -> Option<u64> {
    attributes.get(key).and_then(Value::as_u64)
}

fn get_string_array_attr(attributes: &HashMap<String, Value>, key: &str) -> Vec<String> {
    attributes
        .get(key)
        .and_then(Value::as_array)
        .map(|values| {
            values
                .iter()
                .filter_map(value_to_string)
                .collect::<Vec<String>>()
        })
        .unwrap_or_default()
}

fn value_to_string(value: &Value) -> Option<String> {
    match value {
        Value::String(text) => Some(text.clone()),
        Value::Bool(flag) => Some(flag.to_string()),
        Value::Number(Number::Integer(value)) => Some(value.to_string()),
        Value::Number(Number::Float(value)) => Some(value.to_string()),
        Value::Array(_) | Value::Object(_) => value.to_json().ok().map(|json| json.to_string()),
        Value::Null => Some(String::new()),
        Value::Token(id) => Some(id.to_string()),
    }
}

fn extra_attr_dict(attributes: &HashMap<String, Value>, consumed: &[&str]) -> String {
    let mut items = attributes
        .iter()
        .filter_map(|(key, value)| {
            if consumed.contains(&key.as_str()) || !is_valid_attr_name(key) {
                return None;
            }
            Some(format!("{key} = {}", value_to_mlir_attr(value)))
        })
        .collect::<Vec<_>>();

    items.sort();
    if items.is_empty() {
        String::new()
    } else {
        format!(" {{{}}}", items.join(", "))
    }
}

fn value_to_mlir_attr(value: &Value) -> String {
    match value {
        Value::Null => quote_string("null"),
        Value::Bool(flag) => {
            if *flag {
                "true".to_string()
            } else {
                "false".to_string()
            }
        }
        Value::Number(Number::Integer(number)) => format!("{number} : i64"),
        Value::Number(Number::Float(number)) => {
            if number.is_finite() {
                if number.fract() == 0.0 {
                    format!("{number:.1} : f64")
                } else {
                    format!("{number} : f64")
                }
            } else {
                quote_string(&number.to_string())
            }
        }
        Value::String(text) => quote_string(text),
        Value::Array(values) => {
            let rendered = values
                .iter()
                .map(value_to_mlir_attr)
                .collect::<Vec<_>>()
                .join(", ");
            format!("[{rendered}]")
        }
        Value::Object(map) => {
            let json = serde_json::to_string(map).unwrap_or_else(|_| "{}".to_string());
            quote_string(&json)
        }
        Value::Token(id) => format!("{id} : i64"),
    }
}

fn is_valid_attr_name(name: &str) -> bool {
    let mut chars = name.chars();
    let Some(first) = chars.next() else {
        return false;
    };
    if !(first.is_ascii_alphabetic() || first == '_' || first == '$') {
        return false;
    }
    chars.all(|ch| ch.is_ascii_alphanumeric() || ch == '_' || ch == '.' || ch == '$')
}

fn normalize_memory_space(value: &str) -> String {
    match value.trim().to_ascii_lowercase().as_str() {
        "stm" => "stm".to_string(),
        "ltm" => "ltm".to_string(),
        "episodic" => "episodic".to_string(),
        _ => "stm".to_string(),
    }
}

fn quote_string(value: &str) -> String {
    let mut escaped = String::with_capacity(value.len() + 2);
    escaped.push('"');
    for ch in value.chars() {
        match ch {
            '\\' => escaped.push_str("\\\\"),
            '"' => escaped.push_str("\\\""),
            '\n' => escaped.push_str("\\n"),
            '\r' => escaped.push_str("\\r"),
            '\t' => escaped.push_str("\\t"),
            _ => escaped.push(ch),
        }
    }
    escaped.push('"');
    escaped
}

fn sanitize_symbol_name(name: &str) -> String {
    let mut symbol = String::with_capacity(name.len().max(8));
    for ch in name.chars() {
        if ch.is_ascii_alphanumeric() || ch == '_' || ch == '.' {
            symbol.push(ch);
        } else {
            symbol.push('_');
        }
    }
    if symbol.is_empty() {
        return "graph_main".to_string();
    }
    let starts_valid = symbol
        .chars()
        .next()
        .map(|ch| ch.is_ascii_alphabetic() || ch == '_')
        .unwrap_or(false);
    if starts_valid {
        symbol
    } else {
        format!("g_{symbol}")
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use apxm_core::types::DependencyType;
    use std::collections::HashMap;

    #[test]
    fn lowers_basic_const_ask_graph() {
        let graph = ApxmGraph {
            name: "research_flow".to_string(),
            nodes: vec![
                GraphNode {
                    id: 1,
                    name: "seed".to_string(),
                    op: AISOperationType::ConstStr,
                    attributes: HashMap::from([(
                        "value".to_string(),
                        Value::String("hello".to_string()),
                    )]),
                },
                GraphNode {
                    id: 2,
                    name: "ask".to_string(),
                    op: AISOperationType::Ask,
                    attributes: HashMap::from([(
                        graph_attrs::TEMPLATE_STR.to_string(),
                        Value::String("Summarize {0}".to_string()),
                    )]),
                },
            ],
            edges: vec![crate::GraphEdge {
                from: 1,
                to: 2,
                dependency: DependencyType::Data,
            }],
            parameters: vec![],
            metadata: HashMap::new(),
        };

        let mlir = lower_to_mlir(&graph).expect("graph lowers to mlir");
        assert!(mlir.contains("func.func @research_flow() -> !ais.token attributes {ais.entry}"));
        assert!(mlir.contains("ais.const_str \"hello\""));
        assert!(mlir.contains("ais.ask \"Summarize {0}\" [%n1 : !ais.token] : !ais.token"));
        assert!(mlir.contains("func.return %n2 : !ais.token"));
    }

    #[test]
    fn lowers_ask_with_tool_attributes() {
        let graph = ApxmGraph {
            name: "tools".to_string(),
            nodes: vec![GraphNode {
                id: 1,
                name: "ask".to_string(),
                op: AISOperationType::Ask,
                attributes: HashMap::from([
                    (
                        graph_attrs::TEMPLATE_STR.to_string(),
                        Value::String("Use tools".to_string()),
                    ),
                    (graph_attrs::TOOLS_ENABLED.to_string(), Value::Bool(true)),
                    (
                        graph_attrs::TOOLS.to_string(),
                        Value::Array(vec![
                            Value::String("bash".to_string()),
                            Value::String("read".to_string()),
                        ]),
                    ),
                ]),
            }],
            edges: vec![],
            parameters: vec![],
            metadata: HashMap::new(),
        };

        let mlir = lower_to_mlir(&graph).expect("graph lowers to mlir");
        assert!(mlir.contains("tools_enabled = true"));
        assert!(mlir.contains("tools = [\"bash\", \"read\"]"));
    }

    #[test]
    fn lowers_switch_with_default_case() {
        let graph = ApxmGraph {
            name: "switch".to_string(),
            nodes: vec![
                GraphNode {
                    id: 1,
                    name: "input".to_string(),
                    op: AISOperationType::ConstStr,
                    attributes: HashMap::from([(
                        "value".to_string(),
                        Value::String("technical".to_string()),
                    )]),
                },
                GraphNode {
                    id: 2,
                    name: "route".to_string(),
                    op: AISOperationType::Switch,
                    attributes: HashMap::new(),
                },
            ],
            edges: vec![crate::GraphEdge {
                from: 1,
                to: 2,
                dependency: DependencyType::Data,
            }],
            parameters: vec![],
            metadata: HashMap::new(),
        };

        let mlir = lower_to_mlir(&graph).expect("graph lowers to mlir");
        assert!(mlir.contains("ais.switch %n1 : !ais.token"));
        assert!(mlir.contains("case \"default_case\""));
        assert!(mlir.contains("default {"));
    }
}
