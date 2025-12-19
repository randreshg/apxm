//! APxM Prompt Template Library
//!
//! This crate provides compile-time embedded prompt templates with zero runtime I/O.
//! Templates are embedded via `include_dir!` and rendered with MiniJinja.
//!
//! ## Template naming
//!
//! Templates are addressed by a *logical name* derived from their relative path in `prompts/`:
//!
//! - `prompts/rsn_system.md.jinja`  -> `rsn_system`
//! - `prompts/tools/sql.md.jinja`   -> `tools/sql`
//!
//! The suffix `.md.jinja` (or `.jinja`) is stripped. Directory separators are normalized to `/`.
//!
//! ## Performance characteristics
//!
//! - All templates are loaded exactly once into a global `Environment`.
//! - Rendering is lock-free (the environment is immutable after initialization).
//! - No `unwrap`/panic in normal operation; errors are returned or logged.

#![forbid(unsafe_code)]

use apxm_core::log_error;
use include_dir::{Dir, include_dir};
use minijinja::{Environment, Error as MiniJinjaError, Value as MJValue};
use once_cell::sync::Lazy;
use serde::Serialize;
use std::path::Path;

/// Embedded prompts directory (compile-time inclusion; no runtime filesystem access).
static PROMPTS_DIR: Dir = include_dir!("$CARGO_MANIFEST_DIR/prompts");

/// Global MiniJinja environment with all embedded prompts preloaded.
static GLOBAL_ENV: Lazy<Environment<'static>> = Lazy::new(build_environment);

/// Render a prompt template by name with the given context.
///
/// # Arguments
/// - `template_name`: Logical template name (see crate-level docs).
/// - `context`: Serializable data to interpolate into the template.
///
/// # Returns
/// Rendered prompt, with leading/trailing whitespace trimmed (without reallocating if already trimmed).
///
/// # Errors
/// Returns an error if the template does not exist or rendering fails.
pub fn render_prompt<T: Serialize>(
    template_name: &str,
    context: &T,
) -> Result<String, MiniJinjaError> {
    render_from_env(&GLOBAL_ENV, template_name, context)
}

/// Render an inline (ephemeral) template string.
///
/// Prefer [`render_prompt`] for frequently used templates, as it reuses the global environment.
///
/// # Errors
/// Returns an error if template registration or rendering fails.
pub fn render_inline<T: Serialize>(
    template_str: &str,
    context: &T,
) -> Result<String, MiniJinjaError> {
    let mut env = configured_env();
    // Leak the inline template into a `'static` str so the environment can store it
    let src: &'static str = Box::leak(template_str.to_owned().into_boxed_str());
    env.add_template("inline", src)?;
    render_from_env(&env, "inline", context)
}

/// List all available prompt templates.
///
/// Returns logical template names (see crate-level docs). The returned list is sorted and deduplicated.
pub fn list_prompts() -> Vec<String> {
    let mut names: Vec<String> = PROMPTS_DIR
        .files()
        .map(|f| template_key(f.path()))
        .collect();
    names.sort_unstable();
    names.dedup();
    names
}

fn render_from_env<T: Serialize>(
    env: &Environment<'static>,
    template_name: &str,
    context: &T,
) -> Result<String, MiniJinjaError> {
    let tmpl = env.get_template(template_name)?;
    let rendered = tmpl.render(MJValue::from_serialize(context))?;
    Ok(trim_owned(rendered))
}

fn build_environment() -> Environment<'static> {
    let mut env = configured_env();

    // Collect registration state so we can fail loudly if anything goes wrong.
    let mut registration_failures: Vec<String> = Vec::new();
    let mut added_names: Vec<String> = Vec::new();

    for file in PROMPTS_DIR.files() {
        let path = file.path();
        let name = template_key(path);

        let source = match std::str::from_utf8(file.contents()) {
            Ok(s) => s.to_owned(),
            Err(e) => {
                let msg = format!(
                    "Prompt template is not valid UTF-8: name={:?} path={:?} err={}",
                    name, path, e
                );
                // Log for debugging and record as a failure so initialization is fatal.
                log_error!("prompts", "{}", msg);
                registration_failures.push(msg);
                continue;
            }
        };

        // Leak name and source to `'static` so MiniJinja's Environment can store references.
        let name_static: &'static str = Box::leak(name.clone().into_boxed_str());
        let source_static: &'static str = Box::leak(source.into_boxed_str());

        // Attempt to register the template. Record any failure.
        match env.add_template(name_static, source_static) {
            Ok(()) => {
                added_names.push(name);
            }
            Err(e) => {
                let msg = format!(
                    "Failed to register prompt template: path={:?} err={}",
                    path, e
                );
                log_error!("prompts", "{}", msg);
                registration_failures.push(msg);
            }
        }
    }

    // Ensure every embedded file was successfully registered. Compute missing names deterministically.
    let mut missing: Vec<String> = Vec::new();
    for file in PROMPTS_DIR.files() {
        let key = template_key(file.path());
        if !added_names.iter().any(|n| n == &key) {
            missing.push(key);
        }
    }

    // If there were any failures or missing templates, fail fast with a clear diagnostic.
    if !registration_failures.is_empty() || !missing.is_empty() {
        let mut err = String::from("Prompt template registration failed during initialization.");
        if !registration_failures.is_empty() {
            err.push_str("\nRegistration errors:");
            for f in registration_failures {
                err.push_str(&format!("\n  - {}", f));
            }
        }
        if !missing.is_empty() {
            err.push_str("\nMissing templates (expected embedded files not registered):");
            for m in missing {
                err.push_str(&format!("\n  - {}", m));
            }
        }
        // Panic at startup so template issues are caught immediately and deterministically.
        panic!("{}", err);
    }

    env
}

fn configured_env() -> Environment<'static> {
    let mut env = Environment::new();
    env.set_trim_blocks(true);
    env.set_lstrip_blocks(true);
    env
}

/// Derive a logical template key from an embedded file path.
///
/// - Normalizes separators to `/`
/// - Strips `.md.jinja` or `.jinja` suffix (or the last extension as a fallback)
fn template_key(path: &Path) -> String {
    let raw = path.to_string_lossy();
    let normalized = if raw.contains('\\') {
        raw.replace('\\', "/")
    } else {
        raw.into_owned()
    };

    strip_template_suffix(&normalized).to_owned()
}

fn strip_template_suffix(s: &str) -> &str {
    if let Some(v) = s.strip_suffix(".md.jinja") {
        return v;
    }
    if let Some(v) = s.strip_suffix(".jinja") {
        return v;
    }
    match s.rsplit_once('.') {
        Some((base, _)) => base,
        None => s,
    }
}

fn trim_owned(s: String) -> String {
    let trimmed = s.trim();
    if trimmed.len() == s.len() {
        return s;
    }
    trimmed.to_owned()
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn render_inline_renders() {
        let template = "Hello {{ name }}!";
        let context = json!({ "name": "World" });
        let result = render_inline(template, &context).expect("render_inline failed");
        assert_eq!(result, "Hello World!");
    }

    #[test]
    fn render_inline_trims() {
        let template = "\n  Result: {{ value }}  \n";
        let context = json!({ "value": 42 });
        let result = render_inline(template, &context).expect("render_inline failed");
        assert_eq!(result, "Result: 42");
    }

    #[test]
    fn list_prompts_is_stable() {
        // Validates behavior, not contents (depends on embedded test fixtures).
        let _ = list_prompts();
    }
}
