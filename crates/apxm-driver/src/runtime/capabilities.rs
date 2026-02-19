//! Capability registry configuration for the runtime.

use crate::{config::ApXmConfig, error::DriverError};
use apxm_runtime::CapabilitySystem;

pub fn configure_capability_registry(
    capability_system: &CapabilitySystem,
    config: &ApXmConfig,
) -> Result<(), DriverError> {
    let tools_config = config.tools_config();
    apxm_tools::register_standard_tools(capability_system, &tools_config)
        .map_err(DriverError::Runtime)
}
