//! Logging utilities for APXM compiler and runtime.
//!
//! This module provides a centralized logging interface that can be:
//! - Disabled at compile time for production builds
//! - Configured to use different backends (env_logger, tracing, etc.)
//! - Used consistently across all crates

use std::fmt::Display;

/// Logging level for compiler diagnostics
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum Level {
    /// Critical errors that prevent compilation
    Error,
    /// Warnings about potential issues
    Warn,
    /// Informational messages
    Info,
    /// Debugging information
    Debug,
    /// Detailed tracing information
    Trace,
}

/// Log a message at the specified level.
#[inline]
pub fn log<M: Display>(level: Level, module: &str, message: M) {
    // Convert APXM logging levels into tracing levels.
    match level {
        Level::Error => tracing::event!(
            tracing::Level::ERROR,
            module = module,
            message = format!("{message}")
        ),
        Level::Warn => tracing::event!(
            tracing::Level::WARN,
            module = module,
            message = format!("{message}")
        ),
        Level::Info => tracing::event!(
            tracing::Level::INFO,
            module = module,
            message = format!("{message}")
        ),
        Level::Debug => tracing::event!(
            tracing::Level::DEBUG,
            module = module,
            message = format!("{message}")
        ),
        Level::Trace => tracing::event!(
            tracing::Level::TRACE,
            module = module,
            message = format!("{message}")
        ),
    }
}

/// Macros for logging at specific levels.
#[macro_export]
macro_rules! log_error {
    ($module:expr, $($arg:tt)*) => {
        tracing::event!(tracing::Level::ERROR, module = $module, $($arg)*)
    }
}

#[macro_export]
macro_rules! log_warn {
    ($module:expr, $($arg:tt)*) => {
        tracing::event!(tracing::Level::WARN, module = $module, $($arg)*)
    }
}

#[macro_export]
macro_rules! log_info {
    ($module:expr, $($arg:tt)*) => {
        tracing::event!(tracing::Level::INFO, module = $module, $($arg)*)
    }
}

#[macro_export]
macro_rules! log_debug {
    ($module:expr, $($arg:tt)*) => {
        tracing::event!(tracing::Level::DEBUG, module = $module, $($arg)*)
    }
}

#[macro_export]
macro_rules! log_trace {
    ($module:expr, $($arg:tt)*) => {
        tracing::event!(tracing::Level::TRACE, module = $module, $($arg)*)
    }
}

// =============================================================================
// APXM Runtime Tracing Macros
// =============================================================================
// These macros provide structured tracing for runtime execution with worker
// context, operation tracking, and performance monitoring.
//
// When the `no-trace` feature is enabled, all macros compile to nothing
// for zero overhead in production/benchmark builds.

// ---- With tracing enabled (default) ----

/// Trace scheduler-level events (worker lifecycle, ready queue, etc.)
#[cfg(not(feature = "no-trace"))]
#[macro_export]
macro_rules! apxm_sched {
    ($level:ident, $($arg:tt)*) => {
        tracing::$level!(target: "apxm::scheduler", $($arg)*)
    }
}

/// Trace operation dispatch/completion with worker context
#[cfg(not(feature = "no-trace"))]
#[macro_export]
macro_rules! apxm_op {
    ($level:ident, worker = $worker:expr, $($arg:tt)*) => {
        tracing::$level!(target: "apxm::ops", worker = $worker, $($arg)*)
    };
    ($level:ident, $($arg:tt)*) => {
        tracing::$level!(target: "apxm::ops", $($arg)*)
    }
}

/// Trace LLM requests and responses
#[cfg(not(feature = "no-trace"))]
#[macro_export]
macro_rules! apxm_llm {
    ($level:ident, worker = $worker:expr, $($arg:tt)*) => {
        tracing::$level!(target: "apxm::llm", worker = $worker, $($arg)*)
    };
    ($level:ident, $($arg:tt)*) => {
        tracing::$level!(target: "apxm::llm", $($arg)*)
    }
}

/// Trace token production and consumption
#[cfg(not(feature = "no-trace"))]
#[macro_export]
macro_rules! apxm_token {
    ($level:ident, worker = $worker:expr, $($arg:tt)*) => {
        tracing::$level!(target: "apxm::tokens", worker = $worker, $($arg)*)
    };
    ($level:ident, $($arg:tt)*) => {
        tracing::$level!(target: "apxm::tokens", $($arg)*)
    }
}

/// Trace artifact loading and DAG operations
#[cfg(not(feature = "no-trace"))]
#[macro_export]
macro_rules! apxm_dag {
    ($level:ident, $($arg:tt)*) => {
        tracing::$level!(target: "apxm::dag", $($arg)*)
    }
}

// ---- With tracing disabled (no-trace feature) ----

/// Trace scheduler-level events - compiles to nothing when no-trace is enabled
#[cfg(feature = "no-trace")]
#[macro_export]
macro_rules! apxm_sched {
    ($level:ident, $($arg:tt)*) => {};
}

/// Trace operation dispatch/completion - compiles to nothing when no-trace is enabled
#[cfg(feature = "no-trace")]
#[macro_export]
macro_rules! apxm_op {
    ($level:ident, worker = $worker:expr, $($arg:tt)*) => {};
    ($level:ident, $($arg:tt)*) => {};
}

/// Trace LLM requests and responses - compiles to nothing when no-trace is enabled
#[cfg(feature = "no-trace")]
#[macro_export]
macro_rules! apxm_llm {
    ($level:ident, worker = $worker:expr, $($arg:tt)*) => {};
    ($level:ident, $($arg:tt)*) => {};
}

/// Trace token production and consumption - compiles to nothing when no-trace is enabled
#[cfg(feature = "no-trace")]
#[macro_export]
macro_rules! apxm_token {
    ($level:ident, worker = $worker:expr, $($arg:tt)*) => {};
    ($level:ident, $($arg:tt)*) => {};
}

/// Trace artifact loading and DAG operations - compiles to nothing when no-trace is enabled
#[cfg(feature = "no-trace")]
#[macro_export]
macro_rules! apxm_dag {
    ($level:ident, $($arg:tt)*) => {};
}

