//! Hardware Transducer Interface and Device Abstraction
//!
//! This module provides a unified hardware abstraction layer for ultrasound transducers
//! and medical ultrasound devices, enabling integration with various hardware platforms
//! (clinical systems, research prototypes, simulations).
//!
//! ## Hardware Abstraction Architecture
//!
//! ```
//! Application Layer
//!     ↓
//! Device Manager (enumerates, configures, manages lifetime)
//!     ↓
//! Hardware Interface (communication protocol abstraction)
//!     ↓
//! Device Drivers (USB, Ethernet, PCI-e, mock for simulation)
//!     ↓
//! Physical Hardware / Simulator
//! ```
//!
//! ## Transducer Capabilities
//!
//! - **Frequency Control**: 0.5-15 MHz
//! - **Power Adjustment**: 0-100W acoustic output
//! - **Focal Configuration**: Manual/automatic focus control
//! - **Element Excitation**: Individual element control for phased arrays
//! - **Safety Interlocks**: Hardware-enforced limits
//!
//! ## Communication Protocols
//!
//! - **USB 2.0/3.0**: Research and portable systems
//! - **Ethernet (10/100/1000)**: Clinical and multi-site systems
//! - **PCI-e**: High-speed GPU acceleration
//! - **Mock**: Simulation and testing
//!
//! ## References
//! - IEEE 1451: Smart Sensor Interface Standard
//! - IEC 60601-2-37: Therapeutic ultrasound equipment
//! - DICOM SR3 (Supplemental Report 3): Ultrasound acquisition context

use crate::core::error::{KwaversError, KwaversResult};
use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

/// Unique device identifier
pub type DeviceId = String;

/// Device communication protocol
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CommunicationProtocol {
    /// USB 2.0/3.0 connection
    USB,
    /// Ethernet (10/100/1000 Mbps)
    Ethernet,
    /// PCI-e high-speed interface
    PCIe,
    /// Simulation/mock device
    Mock,
}

impl std::fmt::Display for CommunicationProtocol {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::USB => write!(f, "USB"),
            Self::Ethernet => write!(f, "Ethernet"),
            Self::PCIe => write!(f, "PCI-e"),
            Self::Mock => write!(f, "Mock"),
        }
    }
}

/// Transducer physical characteristics
#[derive(Debug, Clone)]
pub struct TransducerSpecification {
    /// Transducer model name
    pub model: String,
    /// Manufacturer identifier
    pub manufacturer: String,
    /// Serial number for traceability
    pub serial_number: String,
    /// Frequency range (Hz)
    pub frequency_range: (f64, f64),
    /// Maximum acoustic power (W)
    pub max_power: f64,
    /// Number of elements (1 for single-element, >1 for arrays)
    pub num_elements: usize,
    /// Focal length (mm, None for unfocused)
    pub focal_length_mm: Option<f64>,
    /// Element diameter (mm)
    pub element_diameter_mm: f64,
    /// Calibration date
    pub calibration_date: String,
    /// Calibration valid until
    pub calibration_expiry: String,
}

/// Transducer operational state
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TransducerState {
    /// Transducer disconnected or offline
    Offline,
    /// Connected, idle, ready for operation
    Idle,
    /// Currently transmitting ultrasound
    Transmitting,
    /// Receiving echoes or backscatter
    Receiving,
    /// Error state - requires attention
    Error,
    /// Calibration in progress
    Calibrating,
}

impl std::fmt::Display for TransducerState {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Offline => write!(f, "Offline"),
            Self::Idle => write!(f, "Idle"),
            Self::Transmitting => write!(f, "Transmitting"),
            Self::Receiving => write!(f, "Receiving"),
            Self::Error => write!(f, "Error"),
            Self::Calibrating => write!(f, "Calibrating"),
        }
    }
}

/// Hardware control commands
#[derive(Debug, Clone)]
pub enum HardwareCommand {
    /// Set output power (0-100%)
    SetPower(f64),
    /// Set operating frequency (Hz)
    SetFrequency(f64),
    /// Enable/disable transmission
    SetTransmissionEnabled(bool),
    /// Set element excitation pattern (phased arrays)
    SetElementExcitation(Vec<f64>),
    /// Perform automatic calibration
    Calibrate,
    /// Reset device to default state
    Reset,
    /// Query device status
    GetStatus,
    /// Set safety parameter
    SetSafetyLimit(String, f64),
}

/// Hardware response from device
#[derive(Debug, Clone)]
pub enum HardwareResponse {
    /// Acknowledgment of command
    Acknowledged,
    /// Status information
    Status(DeviceStatus),
    /// Telemetry data from device
    Telemetry(DeviceTelemetry),
    /// Device error
    Error(String),
}

/// Real-time device status
#[derive(Debug, Clone)]
pub struct DeviceStatus {
    /// Transducer state
    pub state: TransducerState,
    /// Current operating frequency (Hz)
    pub current_frequency: f64,
    /// Current output power (0-100%)
    pub current_power_percent: f64,
    /// Temperature at transducer (°C)
    pub temperature_c: f64,
    /// Time since last calibration (seconds)
    pub time_since_calibration_s: f64,
    /// Error flags (bitfield)
    pub error_flags: u32,
    /// Uptime since last reset (seconds)
    pub uptime_s: f64,
}

/// Real-time telemetry from device
#[derive(Debug, Clone)]
pub struct DeviceTelemetry {
    /// Timestamp of measurement
    pub timestamp: Instant,
    /// Acoustic power measured by on-device sensor (W)
    pub measured_power_w: f64,
    /// Transducer temperature (°C)
    pub temperature_c: f64,
    /// Acoustic impedance (Rayl)
    pub acoustic_impedance: f64,
    /// Back-reflection coefficient (0-1)
    pub reflection_coefficient: f64,
    /// Current draw from power supply (A)
    pub current_draw_a: f64,
}

/// Transducer hardware interface trait
///
/// This trait defines the interface for communicating with physical ultrasound transducers.
/// Implementations handle protocol-specific details (USB, Ethernet, etc.) while maintaining
/// a uniform API for the application layer.
pub trait TransducerHardware: Send + Sync {
    /// Get transducer specification
    fn specification(&self) -> &TransducerSpecification;

    /// Get current transducer state
    fn state(&self) -> TransducerState;

    /// Send hardware command
    fn send_command(&mut self, command: HardwareCommand) -> KwaversResult<HardwareResponse>;

    /// Check if device is connected
    fn is_connected(&self) -> bool;

    /// Perform device calibration
    fn calibrate(&mut self) -> KwaversResult<()>;

    /// Get real-time telemetry
    fn get_telemetry(&self) -> KwaversResult<DeviceTelemetry>;

    /// Close device connection
    fn disconnect(&mut self) -> KwaversResult<()>;

    /// Get last error message
    fn last_error(&self) -> Option<String>;
}

/// Mock transducer for simulation and testing
#[derive(Debug)]
pub struct MockTransducer {
    /// Device specification
    spec: TransducerSpecification,
    /// Current state
    state: TransducerState,
    /// Current frequency
    current_frequency: f64,
    /// Current power
    current_power_percent: f64,
    /// Last error
    last_error: Option<String>,
    /// Creation time
    created_at: Instant,
}

impl MockTransducer {
    /// Create new mock transducer
    pub fn new(model: String, manufacturer: String) -> Self {
        Self {
            spec: TransducerSpecification {
                model,
                manufacturer,
                serial_number: "MOCK-001".to_string(),
                frequency_range: (0.5e6, 15.0e6),
                max_power: 100.0,
                num_elements: 64,
                focal_length_mm: Some(50.0),
                element_diameter_mm: 0.5,
                calibration_date: "2026-01-30".to_string(),
                calibration_expiry: "2027-01-30".to_string(),
            },
            state: TransducerState::Idle,
            current_frequency: 1.5e6,
            current_power_percent: 0.0,
            last_error: None,
            created_at: Instant::now(),
        }
    }
}

impl TransducerHardware for MockTransducer {
    fn specification(&self) -> &TransducerSpecification {
        &self.spec
    }

    fn state(&self) -> TransducerState {
        self.state
    }

    fn send_command(&mut self, command: HardwareCommand) -> KwaversResult<HardwareResponse> {
        match command {
            HardwareCommand::SetPower(power) => {
                if power < 0.0 || power > 100.0 {
                    self.last_error = Some("Power must be 0-100%".to_string());
                    return Err(KwaversError::InvalidInput(
                        "Invalid power range".to_string(),
                    ));
                }
                self.current_power_percent = power;
                if power > 0.0 && self.state == TransducerState::Idle {
                    self.state = TransducerState::Transmitting;
                }
                Ok(HardwareResponse::Acknowledged)
            }
            HardwareCommand::SetFrequency(freq) => {
                if freq < self.spec.frequency_range.0 || freq > self.spec.frequency_range.1 {
                    self.last_error = Some(format!(
                        "Frequency out of range [{:.1} MHz, {:.1} MHz]",
                        self.spec.frequency_range.0 / 1e6,
                        self.spec.frequency_range.1 / 1e6
                    ));
                    return Err(KwaversError::InvalidInput("Invalid frequency".to_string()));
                }
                self.current_frequency = freq;
                Ok(HardwareResponse::Acknowledged)
            }
            HardwareCommand::SetTransmissionEnabled(enabled) => {
                self.state = if enabled {
                    TransducerState::Transmitting
                } else {
                    TransducerState::Idle
                };
                Ok(HardwareResponse::Acknowledged)
            }
            HardwareCommand::Calibrate => {
                self.state = TransducerState::Calibrating;
                std::thread::sleep(Duration::from_millis(100)); // Simulate calibration time
                self.state = TransducerState::Idle;
                Ok(HardwareResponse::Acknowledged)
            }
            HardwareCommand::GetStatus => Ok(HardwareResponse::Status(DeviceStatus {
                state: self.state,
                current_frequency: self.current_frequency,
                current_power_percent: self.current_power_percent,
                temperature_c: 25.0,
                time_since_calibration_s: self.created_at.elapsed().as_secs_f64(),
                error_flags: 0,
                uptime_s: self.created_at.elapsed().as_secs_f64(),
            })),
            HardwareCommand::Reset => {
                self.current_power_percent = 0.0;
                self.state = TransducerState::Idle;
                Ok(HardwareResponse::Acknowledged)
            }
            _ => Err(KwaversError::InvalidInput(
                "Unsupported command for mock device".to_string(),
            )),
        }
    }

    fn is_connected(&self) -> bool {
        true
    }

    fn calibrate(&mut self) -> KwaversResult<()> {
        self.state = TransducerState::Calibrating;
        std::thread::sleep(Duration::from_millis(100));
        self.state = TransducerState::Idle;
        Ok(())
    }

    fn get_telemetry(&self) -> KwaversResult<DeviceTelemetry> {
        Ok(DeviceTelemetry {
            timestamp: Instant::now(),
            measured_power_w: self.current_power_percent * self.spec.max_power / 100.0,
            temperature_c: 25.0 + (self.current_power_percent / 100.0) * 10.0,
            acoustic_impedance: 1.5e6,
            reflection_coefficient: 0.05,
            current_draw_a: 1.0 + (self.current_power_percent / 100.0) * 4.0,
        })
    }

    fn disconnect(&mut self) -> KwaversResult<()> {
        self.state = TransducerState::Offline;
        Ok(())
    }

    fn last_error(&self) -> Option<String> {
        self.last_error.clone()
    }
}

/// Device manager for discovering and managing multiple transducers
pub struct DeviceManager {
    /// Connected devices
    devices: HashMap<DeviceId, Arc<Mutex<Box<dyn TransducerHardware>>>>,
    /// Device discovery enabled
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
    /// Create new device manager
    pub fn new() -> Self {
        Self {
            devices: HashMap::new(),
            discovery_enabled: false,
        }
    }

    /// Register a device
    pub fn register_device(
        &mut self,
        device_id: DeviceId,
        device: Box<dyn TransducerHardware>,
    ) -> KwaversResult<()> {
        if self.devices.contains_key(&device_id) {
            return Err(KwaversError::InvalidInput(format!(
                "Device {} already registered",
                device_id
            )));
        }

        self.devices.insert(device_id, Arc::new(Mutex::new(device)));
        Ok(())
    }

    /// Get device by ID
    pub fn get_device(&self, device_id: &str) -> Option<Arc<Mutex<Box<dyn TransducerHardware>>>> {
        self.devices.get(device_id).cloned()
    }

    /// List all device IDs
    pub fn list_devices(&self) -> Vec<DeviceId> {
        self.devices.keys().cloned().collect()
    }

    /// Get number of connected devices
    pub fn device_count(&self) -> usize {
        self.devices.len()
    }

    /// Enable automatic device discovery
    pub fn enable_discovery(&mut self) -> KwaversResult<()> {
        self.discovery_enabled = true;
        // Would perform actual device discovery here in real implementation
        Ok(())
    }

    /// Disable automatic device discovery
    pub fn disable_discovery(&mut self) {
        self.discovery_enabled = false;
    }

    /// Remove device from registry
    pub fn unregister_device(&mut self, device_id: &str) -> KwaversResult<()> {
        if self.devices.remove(device_id).is_some() {
            Ok(())
        } else {
            Err(KwaversError::InvalidInput(format!(
                "Device {} not found",
                device_id
            )))
        }
    }
}

impl Default for DeviceManager {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_communication_protocol_display() {
        assert_eq!(CommunicationProtocol::USB.to_string(), "USB");
        assert_eq!(CommunicationProtocol::Ethernet.to_string(), "Ethernet");
        assert_eq!(CommunicationProtocol::Mock.to_string(), "Mock");
    }

    #[test]
    fn test_transducer_state_display() {
        assert_eq!(TransducerState::Idle.to_string(), "Idle");
        assert_eq!(TransducerState::Transmitting.to_string(), "Transmitting");
    }

    #[test]
    fn test_mock_transducer_creation() {
        let transducer = MockTransducer::new("TEST-1.5".to_string(), "TestCorp".to_string());
        assert_eq!(transducer.state(), TransducerState::Idle);
        assert!(transducer.is_connected());
    }

    #[test]
    fn test_mock_transducer_set_power() {
        let mut transducer = MockTransducer::new("TEST-1.5".to_string(), "TestCorp".to_string());
        let response = transducer.send_command(HardwareCommand::SetPower(50.0));
        assert!(response.is_ok());
        assert!(transducer.current_power_percent - 50.0 < 0.01);
    }

    #[test]
    fn test_mock_transducer_invalid_power() {
        let mut transducer = MockTransducer::new("TEST-1.5".to_string(), "TestCorp".to_string());
        let response = transducer.send_command(HardwareCommand::SetPower(150.0));
        assert!(response.is_err());
        assert!(transducer.last_error().is_some());
    }

    #[test]
    fn test_mock_transducer_set_frequency() {
        let mut transducer = MockTransducer::new("TEST-1.5".to_string(), "TestCorp".to_string());
        let response = transducer.send_command(HardwareCommand::SetFrequency(5.0e6));
        assert!(response.is_ok());
        assert!((transducer.current_frequency - 5.0e6).abs() < 1.0);
    }

    #[test]
    fn test_device_manager_registration() {
        let mut manager = DeviceManager::new();
        let transducer = Box::new(MockTransducer::new(
            "TEST-1.5".to_string(),
            "TestCorp".to_string(),
        ));
        let result = manager.register_device("device_1".to_string(), transducer);
        assert!(result.is_ok());
        assert_eq!(manager.device_count(), 1);
    }

    #[test]
    fn test_device_manager_duplicate_registration() {
        let mut manager = DeviceManager::new();
        let transducer1 = Box::new(MockTransducer::new(
            "TEST-1.5".to_string(),
            "TestCorp".to_string(),
        ));
        let transducer2 = Box::new(MockTransducer::new(
            "TEST-2.0".to_string(),
            "TestCorp".to_string(),
        ));

        let _ = manager.register_device("device_1".to_string(), transducer1);
        let result = manager.register_device("device_1".to_string(), transducer2);
        assert!(result.is_err());
    }

    #[test]
    fn test_device_manager_list_devices() {
        let mut manager = DeviceManager::new();
        for i in 0..3 {
            let transducer = Box::new(MockTransducer::new(
                format!("TEST-{}", i),
                "TestCorp".to_string(),
            ));
            let _ = manager.register_device(format!("device_{}", i), transducer);
        }

        let devices = manager.list_devices();
        assert_eq!(devices.len(), 3);
    }

    #[test]
    fn test_mock_transducer_calibration() {
        let mut transducer = MockTransducer::new("TEST-1.5".to_string(), "TestCorp".to_string());
        let result = transducer.calibrate();
        assert!(result.is_ok());
        assert_eq!(transducer.state(), TransducerState::Idle);
    }

    #[test]
    fn test_mock_transducer_telemetry() {
        let mut transducer = MockTransducer::new("TEST-1.5".to_string(), "TestCorp".to_string());
        let _ = transducer.send_command(HardwareCommand::SetPower(50.0));
        let telemetry = transducer.get_telemetry();
        assert!(telemetry.is_ok());

        let telem = telemetry.unwrap();
        assert!(telem.measured_power_w > 0.0);
        assert!(telem.temperature_c > 20.0);
    }

    #[test]
    fn test_device_specification_creation() {
        let spec = TransducerSpecification {
            model: "TEST-1.5".to_string(),
            manufacturer: "TestCorp".to_string(),
            serial_number: "SN-001".to_string(),
            frequency_range: (0.5e6, 10.0e6),
            max_power: 50.0,
            num_elements: 128,
            focal_length_mm: Some(60.0),
            element_diameter_mm: 0.3,
            calibration_date: "2026-01-30".to_string(),
            calibration_expiry: "2027-01-30".to_string(),
        };

        assert_eq!(spec.model, "TEST-1.5");
        assert_eq!(spec.num_elements, 128);
        assert!(spec.focal_length_mm.is_some());
    }
}
