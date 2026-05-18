//! Hardware communication trait for ultrasound transducers.

use super::types::{
    DeviceTelemetry, DeviceTransducerSpecification, HardwareCommand, HardwareResponse,
    TransducerState,
};
use crate::core::error::KwaversResult;

/// Transducer hardware interface.
///
/// Implementations hide protocol-specific details while preserving one
/// application-layer API for physical devices and simulators.
pub trait TransducerHardware: Send + Sync {
    /// Get transducer specification.
    /// # Errors
    /// - Returns [`Err`] if an internal constraint is violated.
    ///
    fn specification(&self) -> &DeviceTransducerSpecification;

    /// Get current transducer state.
    /// # Errors
    /// - Returns [`Err`] if an internal constraint is violated.
    ///
    fn state(&self) -> TransducerState;

    /// Send hardware command.
    /// # Errors
    /// - Returns [`Err`] if an internal constraint is violated.
    ///
    fn send_command(&mut self, command: HardwareCommand) -> KwaversResult<HardwareResponse>;

    /// Check if device is connected.
    /// # Errors
    /// - Returns [`Err`] if an internal constraint is violated.
    ///
    fn is_connected(&self) -> bool;

    /// Perform device calibration.
    /// # Errors
    /// - Returns [`Err`] if an internal constraint is violated.
    ///
    fn calibrate(&mut self) -> KwaversResult<()>;

    /// Get real-time telemetry.
    /// # Errors
    /// - Returns [`Err`] if an internal constraint is violated.
    ///
    fn get_telemetry(&self) -> KwaversResult<DeviceTelemetry>;

    /// Close device connection.
    /// # Errors
    /// - Returns [`Err`] if an internal constraint is violated.
    ///
    fn disconnect(&mut self) -> KwaversResult<()>;

    /// Get last error message.
    fn last_error(&self) -> Option<String>;
}
