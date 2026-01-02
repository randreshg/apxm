//! Effect metadata for AIS operations interacting with the AAM.

use apxm_core::types::operations::AISOperationType;
use std::collections::HashSet;

/// Logical component of the Agent Abstract Machine/memory touched by an operation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum AamComponent {
    Beliefs,
    Goals,
    Capabilities,
    ShortTermMemory,
    LongTermMemory,
    Episodic,
}

/// Read/write set for an operation.
#[derive(Debug, Clone, Default)]
pub struct OperationEffects {
    pub reads: HashSet<AamComponent>,
    pub writes: HashSet<AamComponent>,
    pub has_side_effects: bool,
}

impl OperationEffects {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn read(mut self, component: AamComponent) -> Self {
        self.reads.insert(component);
        self
    }

    pub fn write(mut self, component: AamComponent) -> Self {
        self.writes.insert(component);
        self.has_side_effects = true;
        self
    }

    pub fn can_reorder_with(&self, other: &OperationEffects) -> bool {
        self.reads.is_disjoint(&other.writes)
            && self.writes.is_disjoint(&other.reads)
            && self.writes.is_disjoint(&other.writes)
    }
}

/// Convenience helper mirroring the legacy runtime mapping.
pub fn operation_effects(op: &AISOperationType) -> OperationEffects {
    use AamComponent::*;

    match op {
        AISOperationType::QMem => OperationEffects::new().read(Beliefs).read(ShortTermMemory),
        AISOperationType::UMem => OperationEffects::new()
            .write(Beliefs)
            .write(ShortTermMemory),
        // Ask: reads beliefs only (simple Q&A)
        AISOperationType::Ask => OperationEffects::new().read(Beliefs),
        // Think: reads beliefs only (extended thinking, no side effects)
        AISOperationType::Think => OperationEffects::new().read(Beliefs),
        // Reason: reads AND writes beliefs + goals (structured reasoning)
        AISOperationType::Reason => OperationEffects::new()
            .read(Beliefs)
            .write(Beliefs)
            .write(Goals),
        AISOperationType::Plan => OperationEffects::new().read(Goals).write(Goals),
        AISOperationType::Reflect => OperationEffects::new().read(Episodic),
        AISOperationType::Verify => OperationEffects::new().read(Beliefs),
        AISOperationType::Inv => OperationEffects::new().read(Capabilities),
        AISOperationType::Fence => OperationEffects::new()
            .write(Beliefs)
            .write(ShortTermMemory)
            .write(LongTermMemory)
            .write(Goals),
        _ => OperationEffects::new(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reorder_conflicts() {
        // Reason writes to Beliefs, QMem reads Beliefs -> cannot reorder
        let reason = operation_effects(&AISOperationType::Reason);
        let qmem = operation_effects(&AISOperationType::QMem);
        assert!(!reason.can_reorder_with(&qmem));
    }
}
