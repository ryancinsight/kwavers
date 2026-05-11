//! Registry for connected transducer devices.

use super::hardware::TransducerHardware;
use super::types::DeviceId;
use crate::core::error::{KwaversError, KwaversResult};
use std::collections::HashMap;
use std::sync::{Arc, Mutex};

type DeviceHandle = Arc<Mutex<Box<dyn TransducerHardware>>>; // dyn: runtime hardware driver boundary.

/// Device manager for discovering and managing multiple transducers.
pub struct DeviceManager {
    /// Connected devices.
    devices: HashMap<DeviceId, DeviceHandle>,
    /// Device discovery enabled.
    discovery_enabled: bool,
}

impl std::fmt::Debug for DeviceManager {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("DeviceManager")
            .field("device_count", &self.devices.len())
            .field("discovery_enabled", &self.discovery_enabled)
            .finish()
    }
}

impl DeviceManager {
    /// Create new device manager.
    /// # Errors
    /// - Returns [`Err`] if an internal constraint is violated.
    ///
    #[must_use]
    pub fn new() -> Self {
        Self {
            devices: HashMap::new(),
            discovery_enabled: false,
        }
    }

    /// Register a device.
    /// # Errors
    /// - Returns [`KwaversError::InvalidInput`] if the precondition for invalid or out-of-range input parameters is violated.
    ///
    pub fn register_device(
        &mut self,
        device_id: DeviceId,
        device: Box<dyn TransducerHardware>, // dyn: runtime hardware driver boundary.
    ) -> KwaversResult<()> {
        if self.devices.contains_key(&device_id) {
            return Err(KwaversError::InvalidInput(format!(
                "Device {device_id} already registered"
            )));
        }

        self.devices.insert(device_id, Arc::new(Mutex::new(device)));
        Ok(())
    }

    /// Get device by ID.
    #[must_use]
    pub fn get_device(&self, device_id: &str) -> Option<DeviceHandle> {
        self.devices.get(device_id).cloned()
    }

    /// List all device IDs.
    /// # Errors
    /// - Returns [`Err`] if an internal constraint is violated.
    ///
    #[must_use]
    pub fn list_devices(&self) -> Vec<DeviceId> {
        self.devices.keys().cloned().collect()
    }

    /// Get number of connected devices.
    /// # Errors
    /// - Returns [`Err`] if an internal constraint is violated.
    ///
    #[must_use]
    pub fn device_count(&self) -> usize {
        self.devices.len()
    }

    /// Enable automatic device discovery.
    /// # Errors
    /// - Returns [`Err`] if an internal constraint is violated.
    ///
    pub fn enable_discovery(&mut self) -> KwaversResult<()> {
        self.discovery_enabled = true;
        Ok(())
    }

    /// Disable automatic device discovery.
    /// # Errors
    /// - Returns [`Err`] if an internal constraint is violated.
    ///
    pub fn disable_discovery(&mut self) {
        self.discovery_enabled = false;
    }

    /// Remove device from registry.
    /// # Errors
    /// - Returns [`KwaversError::InvalidInput`] if the precondition for invalid or out-of-range input parameters is violated.
    ///
    pub fn unregister_device(&mut self, device_id: &str) -> KwaversResult<()> {
        if self.devices.remove(device_id).is_some() {
            Ok(())
        } else {
            Err(KwaversError::InvalidInput(format!(
                "Device {device_id} not found"
            )))
        }
    }
}

impl Default for DeviceManager {
    fn default() -> Self {
        Self::new()
    }
}
