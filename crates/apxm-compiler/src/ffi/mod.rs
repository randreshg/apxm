//! FFI bindings for the Apxm compiler.

pub mod conversions;
pub mod error;
pub mod raw;
pub mod utils;

pub use error::collect_errors;
pub use raw::{
    ApxmArtifactOptions, ApxmCodegenOptions, ApxmCompilerContext, ApxmModule, ApxmPassManager,
    apxm_codegen_emit_artifact, apxm_codegen_emit_rust, apxm_codegen_emit_rust_with_options,
    apxm_codegen_free, apxm_compiler_context_create, apxm_compiler_context_destroy,
    apxm_module_destroy, apxm_module_parse, apxm_module_to_string, apxm_module_verify,
    apxm_parse_dsl, apxm_parse_dsl_file, apxm_pass_manager_add_pass_by_name,
    apxm_pass_manager_clear, apxm_pass_manager_create, apxm_pass_manager_destroy,
    apxm_pass_manager_run, apxm_pass_registry_find_pass, apxm_pass_registry_get_count,
    apxm_pass_registry_get_pass, apxm_string_free,
};
pub use utils::{handle_bool_result, handle_null_result};
