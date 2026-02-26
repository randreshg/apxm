//! Module API for the Apxm compiler.
//!
//! Modules are the building blocks of Apxm programs.
//! They contain a set of functions, variables, and types that can be used by other modules.
//!
//! Modules can be imported using the `import` keyword, which allows the functions, variables,
//! and types defined in the imported module to be used in the current module.

use crate::api::Context;
use crate::codegen::artifact::parse_wire_dags;
use crate::ffi;
use apxm_artifact::{Artifact, ArtifactMetadata};
use apxm_core::error::builder::ErrorBuilder;
use apxm_core::error::codes::ErrorCode;
use apxm_core::error::compiler::{CompilerError, Result};
use apxm_graph::ApxmGraph;
use std::ffi::{CStr, CString};
use std::fs;
use std::marker::PhantomData;
use std::os::raw::c_char;
use std::path::Path;
use std::ptr;

pub(crate) fn invalid_input_error(message: impl Into<String>) -> CompilerError {
    CompilerError::InvalidInput(Box::new(ErrorBuilder::generic(
        ErrorCode::InternalError,
        message,
    )))
}

pub struct Module {
    raw: *mut ffi::ApxmModule,
    _context: PhantomData<Context>,
}

impl Module {
    /// # Safety
    ///
    /// The caller must ensure that `raw` is a valid pointer to an `ApxmModule`
    /// and that ownership is properly transferred to the returned `Module`.
    pub unsafe fn from_raw(raw: *mut ffi::ApxmModule) -> Self {
        Self {
            raw,
            _context: PhantomData,
        }
    }

    /// Parses the given source string into a module.
    pub fn parse(context: &Context, source: &str) -> Result<Self> {
        let c_source = CString::new(source)
            .map_err(|e| invalid_input_error(format!("Invalid source string: {}", e)))?;

        let raw = ffi::handle_null_result(
            unsafe { ffi::apxm_module_parse(context.as_ptr(), c_source.as_ptr()) },
            "module parsing",
        )?;

        Ok(unsafe { Self::from_raw(raw) })
    }

    /// Parses DSL by canonicalizing to ApxmGraph first, then lowering to MLIR.
    pub fn parse_dsl(context: &Context, source: &str, filename: &str) -> Result<Self> {
        let graph = Self::parse_dsl_graph(context, source, filename)?;
        let mlir_text = graph
            .to_mlir()
            .map_err(|e| invalid_input_error(format!("Graph lowering failed: {e}")))?;
        Self::parse(context, &mlir_text)
    }

    /// Parses DSL source and returns canonical graph payload as ApxmGraph.
    pub fn parse_dsl_graph(context: &Context, source: &str, filename: &str) -> Result<ApxmGraph> {
        let c_source = CString::new(source)
            .map_err(|e| invalid_input_error(format!("Invalid DSL source: {}", e)))?;
        let c_filename = CString::new(filename)
            .map_err(|e| invalid_input_error(format!("Invalid filename: {}", e)))?;

        let raw = ffi::handle_null_result(
            unsafe {
                ffi::apxm_parse_dsl_to_graph_json(
                    context.as_ptr(),
                    c_source.as_ptr(),
                    c_filename.as_ptr(),
                )
            },
            "DSL graph lowering",
        )?;

        let graph_json = unsafe { CStr::from_ptr(raw) }
            .to_string_lossy()
            .into_owned();
        unsafe { ffi::apxm_string_free(raw) };

        ApxmGraph::from_json(&graph_json)
            .map_err(|e| invalid_input_error(format!("Invalid canonical graph from DSL: {e}")))
    }

    /// Parses the given DSL source file into a module.
    pub fn parse_dsl_file(context: &Context, path: &Path) -> Result<Self> {
        let source = fs::read_to_string(path).map_err(CompilerError::Io)?;
        let path_str = path
            .to_str()
            .ok_or_else(|| invalid_input_error("Invalid path encoding".to_string()))?;
        Self::parse_dsl(context, &source, path_str)
    }

    /// Parses the given source file into a module.
    pub fn parse_file(context: &Context, path: &Path) -> Result<Self> {
        let source = fs::read_to_string(path).map_err(CompilerError::Io)?;
        Self::parse(context, &source)
    }

    /// Verifies the module's structure and syntax.
    pub fn verify(&self) -> Result<()> {
        ffi::handle_bool_result(
            unsafe { ffi::apxm_module_verify(self.raw) },
            "module verification",
        )
    }

    pub fn to_string(&self) -> Result<String> {
        let c_str = ffi::handle_null_result(
            unsafe { ffi::apxm_module_to_string(self.raw) },
            "module serialization",
        )?;

        let result = unsafe {
            std::ffi::CStr::from_ptr(c_str)
                .to_string_lossy()
                .into_owned()
        };

        unsafe { ffi::apxm_string_free(c_str as *mut _) };

        Ok(result)
    }

    pub fn as_ptr(&self) -> *mut ffi::ApxmModule {
        self.raw
    }

    pub fn generate_artifact_bytes(&self) -> Result<Vec<u8>> {
        self.generate_artifact_bytes_with_name(None)
    }

    pub fn generate_artifact_bytes_with_name(&self, module_name: Option<&str>) -> Result<Vec<u8>> {
        let payload = self.emit_artifact_payload(module_name)?;
        let dags = parse_wire_dags(&payload)?;

        // Find entry DAG for metadata naming
        let entry_dag = dags.iter().find(|d| d.metadata.is_entry);
        let name = entry_dag
            .and_then(|d| d.metadata.name.clone())
            .or_else(|| dags.first().and_then(|d| d.metadata.name.clone()));

        let metadata = ArtifactMetadata::new(name, crate::VERSION);
        Artifact::new(metadata, dags)
            .to_bytes()
            .map_err(|err| invalid_input_error(err.to_string()))
    }

    pub fn generate_artifact_to_path<P: AsRef<Path>>(&self, path: P) -> Result<()> {
        let bytes = self.generate_artifact_bytes()?;
        std::fs::write(path, &bytes).map_err(CompilerError::Io)
    }

    fn emit_artifact_payload(&self, module_name: Option<&str>) -> Result<Vec<u8>> {
        let module_name_cstr = module_name
            .map(|name| {
                CString::new(name)
                    .map_err(|e| invalid_input_error(format!("Invalid module name: {e}")))
            })
            .transpose()?;

        let c_options = ffi::ApxmArtifactOptions {
            module_name: module_name_cstr
                .as_ref()
                .map(|c| c.as_ptr())
                .unwrap_or(ptr::null()),
            emit_debug_json: false,
            target_version: ptr::null(),
        };

        let raw = ffi::handle_null_result(
            unsafe { ffi::apxm_codegen_emit_artifact(self.raw, &c_options) },
            "artifact generation",
        )?;

        unsafe { copy_artifact_buffer(raw) }
    }
}

unsafe fn copy_artifact_buffer(ptr: *mut c_char) -> Result<Vec<u8>> {
    struct BufferGuard(*mut c_char);

    impl Drop for BufferGuard {
        fn drop(&mut self) {
            if !self.0.is_null() {
                unsafe { ffi::apxm_codegen_free(self.0) };
            }
        }
    }

    let guard = BufferGuard(ptr);
    if guard.0.is_null() {
        return Err(invalid_input_error("Artifact emitter returned null buffer"));
    }

    let mut len_buf = [0u8; 8];
    unsafe {
        len_buf.copy_from_slice(std::slice::from_raw_parts(guard.0 as *const u8, 8));
    }
    let len = u64::from_le_bytes(len_buf) as usize;

    let data = unsafe {
        let data_ptr = guard.0.add(std::mem::size_of::<u64>()) as *const u8;
        std::slice::from_raw_parts(data_ptr, len)
    };
    Ok(data.to_vec())
}

impl Drop for Module {
    fn drop(&mut self) {
        if !self.raw.is_null() {
            unsafe {
                ffi::apxm_module_destroy(self.raw);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::api::Context;
    use apxm_core::types::AISOperationType;

    #[test]
    fn dsl_canonicalizes_to_graph_before_mlir() {
        let context = Context::new().expect("create context");
        let source = r#"
            agent Hello {
                @entry flow main() -> str {
                    ask("hello world") -> result
                    return result
                }
            }
        "#;

        let graph = Module::parse_dsl_graph(&context, source, "hello.ais").expect("graph parse");
        assert_eq!(graph.name, "Hello_main");
        assert_eq!(graph.nodes.len(), 1);
        assert_eq!(graph.nodes[0].op, AISOperationType::Ask);
    }

    #[test]
    fn dsl_module_parse_uses_graph_lowering_path() {
        let context = Context::new().expect("create context");
        let source = r#"
            agent Hello {
                @entry flow main() -> str {
                    ask("hello world") -> result
                    return result
                }
            }
        "#;

        let module = Module::parse_dsl(&context, source, "hello.ais").expect("dsl parse");
        let text = module.to_string().expect("module stringify");
        assert!(text.contains("ais.ask"));
    }

    #[test]
    fn dsl_flow_calls_are_canonicalized_into_graph_nodes() {
        let context = Context::new().expect("create context");
        let source = r#"
            agent Researcher {
                flow research(topic: str) -> str {
                    think("Research this topic thoroughly: " + topic) -> findings
                    return findings
                }
            }

            agent Coordinator {
                @entry flow main() -> str {
                    ask("What topic should we investigate?") -> topic
                    Researcher.research(topic) -> findings
                    ask("Summarize the findings") -> summary
                    return summary
                }
            }
        "#;

        let graph = Module::parse_dsl_graph(&context, source, "multi.ais").expect("graph parse");
        assert_eq!(graph.nodes.len(), 3);
        assert_eq!(graph.nodes[0].op, AISOperationType::Ask);
        assert_eq!(graph.nodes[1].op, AISOperationType::Think);
        assert_eq!(graph.nodes[2].op, AISOperationType::Ask);
    }

    #[test]
    fn dsl_structured_control_if_parallel_switch_lowers_through_graph() {
        let context = Context::new().expect("create context");
        let source = r#"
            agent Router {
                @entry flow main(style: str) -> str {
                    if (style) {
                        ask("formal style selected") -> branch_result
                    } else {
                        ask("casual style selected") -> branch_result
                    }

                    parallel {
                        think("parallel branch A")
                        think("parallel branch B")
                    }

                    switch style {
                        case "formal" => ask("Use formal tone")
                        case "casual" => ask("Use casual tone")
                        default => ask("Use neutral tone")
                    } -> routed

                    ask("Final: " + routed) -> output
                    return output
                }
            }
        "#;

        let graph =
            Module::parse_dsl_graph(&context, source, "structured.ais").expect("graph parse");
        assert!(
            graph
                .nodes
                .iter()
                .any(|node| node.op == AISOperationType::BranchOnValue)
        );
        assert!(
            graph
                .nodes
                .iter()
                .any(|node| node.op == AISOperationType::WaitAll)
        );
        assert!(
            graph
                .nodes
                .iter()
                .any(|node| node.op == AISOperationType::Switch)
        );

        let module = Module::parse_dsl(&context, source, "structured.ais").expect("dsl parse");
        let mlir = module.to_string().expect("module stringify");
        assert!(mlir.contains("ais.branch_on_value"));
        assert!(mlir.contains("ais.wait_all"));
        assert!(mlir.contains("ais.switch"));
    }

    #[test]
    fn dsl_structured_control_loop_try_catch_lowers_through_graph() {
        let context = Context::new().expect("create context");
        let source = r#"
            agent Controller {
                @entry flow main(items: str) -> str {
                    loop(item in items) {
                        ask("Inspect item: " + item)
                    }

                    try {
                        return ask("primary path")
                    } catch {
                        return ask("recovery path")
                    }
                }
            }
        "#;

        let graph = Module::parse_dsl_graph(&context, source, "loop_try.ais").expect("graph parse");
        assert!(
            graph
                .nodes
                .iter()
                .any(|node| node.op == AISOperationType::LoopStart)
        );
        assert!(
            graph
                .nodes
                .iter()
                .any(|node| node.op == AISOperationType::LoopEnd)
        );
        assert!(
            graph
                .nodes
                .iter()
                .any(|node| node.op == AISOperationType::TryCatch)
        );

        let module = Module::parse_dsl(&context, source, "loop_try.ais").expect("dsl parse");
        let mlir = module.to_string().expect("module stringify");
        assert!(mlir.contains("ais.loop_start"));
        assert!(mlir.contains("ais.loop_end"));
        assert!(mlir.contains("ais.try_catch"));
    }
}
