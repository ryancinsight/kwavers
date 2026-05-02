//! Hardware transducer interface and device abstraction.
//!
//! This module provides a unified hardware abstraction layer for ultrasound
//! transducers and medical ultrasound devices. Device types and commands live
//! in `types`, the dynamic hardware boundary lives in `hardware`, test/sim
//! behavior lives in `mock`, and device registry state lives in `manager`.

mod hardware;
mod manager;
mod mock;
mod types;

pub use hardware::TransducerHardware;
pub use manager::DeviceManager;
pub use mock::MockTransducer;
pub use types::{
    CommunicationProtocol, DeviceId, DeviceStatus, DeviceTelemetry, HardwareCommand,
    HardwareResponse, TransducerSpecification, TransducerState,
};

#[cfg(test)]
mod tests;
