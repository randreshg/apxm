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
        tracing::event!(tracing::Level::ERROR, module = $module, $($arg)*);
    }
}

#[macro_export]
macro_rules! log_warn {
    ($module:expr, $($arg:tt)*) => {
        tracing::event!(tracing::Level::WARN, module = $module, $($arg)*);
    }
}

#[macro_export]
macro_rules! log_info {
    ($module:expr, $($arg:tt)*) => {
        tracing::event!(tracing::Level::INFO, module = $module, $($arg)*);
    }
}

#[macro_export]
macro_rules! log_debug {
    ($module:expr, $($arg:tt)*) => {
        tracing::event!(tracing::Level::DEBUG, module = $module, $($arg)*);
    }
}

#[macro_export]
macro_rules! log_trace {
    ($module:expr, $($arg:tt)*) => {
        tracing::event!(tracing::Level::TRACE, module = $module, $($arg)*);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_log_error() {
        log_error!("test", "This is an error message");
    }

    #[test]
    fn test_log_warn() {
        log_warn!("test", "This is a warning message");
    }

    #[test]
    fn test_log_info() {
        log_info!("test", "This is an info message");
    }

    #[test]
    fn test_log_debug() {
        log_debug!("test", "This is a debug message");
    }

    #[test]
    fn test_log_trace() {
        log_trace!("test", "This is a trace message");
    }

    #[test]
    fn test_logging_levels() {
        assert!(Level::Error < Level::Warn);
        assert!(Level::Warn < Level::Info);
        assert!(Level::Info < Level::Debug);
        assert!(Level::Debug < Level::Trace);
    }
}
