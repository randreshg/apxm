//! Operation executor trait.
//!
//! Executors run AIS operations given their metadata and input values.

use thiserror::Error;

use crate::types::{AISOperation, Value};

/// Errors produced by executors.
#[derive(Debug, Error, Clone, PartialEq)]
pub enum ExecutionError {
    #[error("invalid input: {0}")]
    InvalidInput(String),
    #[error("execution failed: {0}")]
    Failed(String),
}

/// Contract for all operation executors.
pub trait OperationExecutor: Send + Sync {
    /// Executes an operation with the provided inputs and returns the resulting value.
    fn execute(&self, op: &AISOperation, inputs: Vec<Value>) -> Result<Value, ExecutionError>;
}

#[cfg(test)]
mod tests {
    use super::*;

    struct AddExecutor;

    impl OperationExecutor for AddExecutor {
        fn execute(&self, op: &AISOperation, inputs: Vec<Value>) -> Result<Value, ExecutionError> {
            if op.op_type != crate::types::AISOperationType::Inv {
                return Err(ExecutionError::InvalidInput("wrong op".into()));
            }
            if inputs.len() != 2 {
                return Err(ExecutionError::InvalidInput("need two inputs".into()));
            }
            let a = inputs[0]
                .as_number()
                .and_then(|n| Some(n.as_f64()))
                .ok_or_else(|| ExecutionError::InvalidInput("invalid first".into()))?;
            let b = inputs[1]
                .as_number()
                .and_then(|n| Some(n.as_f64()))
                .ok_or_else(|| ExecutionError::InvalidInput("invalid second".into()))?;
            Ok(Value::from(a + b))
        }
    }

    #[test]
    fn test_execute_success() {
        let executor = AddExecutor;
        let op = AISOperation::new(1, crate::types::AISOperationType::Inv);
        let res = executor
            .execute(&op, vec![Value::from(1.0f64), Value::from(2.0f64)])
            .expect("add");
        assert!(matches!(res, Value::Number(_)));
    }

    #[test]
    fn test_execute_failure() {
        let executor = AddExecutor;
        let op = AISOperation::new(1, crate::types::AISOperationType::Plan);
        let err = executor.execute(&op, vec![]).expect_err("should fail");
        assert!(matches!(err, ExecutionError::InvalidInput(_)));
    }

    fn assert_send_sync<T: Send + Sync>() {}

    #[test]
    fn trait_is_send_sync() {
        assert_send_sync::<Box<dyn OperationExecutor>>();
    }
}
