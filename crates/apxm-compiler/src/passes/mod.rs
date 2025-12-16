//! APXM Compiler Pass System
//!
//! This module contains the pass system that transforms and optimizes
//! intermediate representations of AI operations.
//!
//! # Components
//!
//! - [`PassManager`]: Runs optimization passes
//! - [`build_pipeline`]: Creates compilation pipelines
//! - Registry functions: Pass management

mod manager;
mod pipeline;
mod registry;

pub use manager::PassManager;
pub use pipeline::build_pipeline;
pub use registry::{find_pass, get_pass_count, get_pass_info, list_passes};
