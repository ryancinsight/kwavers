//! Hardware communication trait for ultrasound transducers.

use super::types::{
    DeviceTelemetry, HardwareCommand, HardwareResponse, TransducerSpecification, TransducerState,
};
use crate::core::error::KwaversResult;

/// Transducer hardware interface.
///
/// Implementations hide protocol-specific details while preserving one
/// application-layer API for physical devices and simulators.
pub trait TransducerHardware: Send + Sync {
    /// Get transducer specification.
    fn specification(&self) -> &TransducerSpecification;

    /// Get current transducer state.
    fn state(&self) -> TransducerState;

    /// Send hardware command.
    fn send_command(&mut self, command: HardwareCommand) -> KwaversResult<HardwareResponse>;

    /// Check if device is connected.
    fn is_connected(&self) -> bool;

    /// Perform device calibration.
    fn calibrate(&mut self) -> KwaversResult<()>;

    /// Get real-time telemetry.
    fn get_telemetry(&self) -> KwaversResult<DeviceTelemetry>;

    /// Close device connection.
    fn disconnect(&mut self) -> KwaversResult<()>;

    /// Get last error message.
    fn last_error(&self) -> Option<String>;
}
