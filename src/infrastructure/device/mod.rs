//! Hardware Device Abstraction Layer
//!
//! This module provides a unified interface for controlling ultrasound transducers
//! and other medical ultrasound hardware, abstracting away protocol-specific details.
//!
//! ## Architecture
//!
//! The device layer sits between the application layer and the physical hardware:
//!
//! ```text
//! Clinical Application (therapy planning, real-time reconstruction)
//!             ↓
//! Device Manager (discovery, lifecycle management)
//!             ↓
//! Transducer Hardware Interface (USB, Ethernet, mock)
//!             ↓
//! Physical Transducers / Simulation
//! ```
//!
//! ## Key Responsibilities
//!
//! - **Hardware Abstraction**: Provide uniform API for different hardware platforms
//! - **Device Discovery**: Automatic detection and enumeration of connected devices
//! - **Lifecycle Management**: Safe connection, configuration, and disconnection
//! - **Command Protocol**: High-level commands translated to hardware-specific messages
//! - **Safety Interlocks**: Hardware-enforced safety limits and monitoring
//! - **Telemetry**: Real-time monitoring of device status and performance
//!
//! ## Usage Example
//!
//! ```rust,no_run
//! use kwavers::infrastructure::device::{DeviceManager, MockTransducer};
//!
//! // Create device manager
//! let mut manager = DeviceManager::new();
//!
//! // Register a mock transducer for testing
//! let transducer = Box::new(MockTransducer::new(
//!     "HIFU-1.5".to_string(),
//!     "Manufacturer".to_string(),
//! ));
//! manager.register_device("device_1".to_string(), transducer)?;
//!
//! // Interact with device
//! if let Some(device) = manager.get_device("device_1") {
//!     let mut dev = device.lock().unwrap();
//!     dev.send_command(HardwareCommand::SetPower(50.0))?;
//! }
//! # Ok::<(), kwavers::core::error::KwaversError>(())
//! ```

pub mod transducer_interface;

pub use transducer_interface::{
    CommunicationProtocol, DeviceId, DeviceManager, DeviceStatus, DeviceTelemetry, HardwareCommand,
    HardwareResponse, MockTransducer, TransducerHardware, TransducerSpecification, TransducerState,
};
