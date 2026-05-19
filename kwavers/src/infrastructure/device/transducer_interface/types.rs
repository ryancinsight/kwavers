use std::time::Instant;

/// Unique device identifier.
pub type DeviceId = String;

/// Device communication protocol.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CommunicationProtocol {
    /// USB 2.0/3.0 connection.
    USB,
    /// Ethernet (10/100/1000 Mbps).
    Ethernet,
    /// PCI-e high-speed interface.
    PCIe,
    /// Simulation/mock device.
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

/// Transducer physical characteristics.
#[derive(Debug, Clone)]
pub struct DeviceTransducerSpecification {
    /// Transducer model name.
    pub model: String,
    /// Manufacturer identifier.
    pub manufacturer: String,
    /// Serial number for traceability.
    pub serial_number: String,
    /// Frequency range (Hz).
    pub frequency_range: (f64, f64),
    /// Maximum acoustic power (W).
    pub max_power: f64,
    /// Number of elements (1 for single-element, >1 for arrays).
    pub num_elements: usize,
    /// Focal length (mm), `None` for unfocused.
    pub focal_length_mm: Option<f64>,
    /// Element diameter (mm).
    pub element_diameter_mm: f64,
    /// Calibration date.
    pub calibration_date: String,
    /// Calibration valid until.
    pub calibration_expiry: String,
}

/// Transducer operational state.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TransducerState {
    /// Transducer disconnected or offline.
    Offline,
    /// Connected, idle, ready for operation.
    Idle,
    /// Currently transmitting ultrasound.
    Transmitting,
    /// Receiving echoes or backscatter.
    Receiving,
    /// Error state — requires attention.
    Error,
    /// Calibration in progress.
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

/// Hardware control commands.
#[derive(Debug, Clone)]
pub enum HardwareCommand {
    /// Set output power (0–100%).
    SetPower(f64),
    /// Set operating frequency (Hz).
    SetFrequency(f64),
    /// Enable/disable transmission.
    SetTransmissionEnabled(bool),
    /// Set element excitation pattern (phased arrays).
    SetElementExcitation(Vec<f64>),
    /// Perform automatic calibration.
    Calibrate,
    /// Reset device to default state.
    Reset,
    /// Query device status.
    GetStatus,
    /// Set safety parameter.
    SetSafetyLimit(String, f64),
}

/// Hardware response from device.
#[derive(Debug, Clone)]
pub enum HardwareResponse {
    /// Acknowledgment of command.
    Acknowledged,
    /// Status information.
    Status(TransducerDeviceStatus),
    /// Telemetry data from device.
    Telemetry(DeviceTelemetry),
    /// Device error.
    Error(String),
}

/// Real-time device status.
#[derive(Debug, Clone)]
pub struct TransducerDeviceStatus {
    /// Transducer state.
    pub state: TransducerState,
    /// Current operating frequency (Hz).
    pub current_frequency: f64,
    /// Current output power (0–100%).
    pub current_power_percent: f64,
    /// Temperature at transducer (°C).
    pub temperature_c: f64,
    /// Time since last calibration (seconds).
    pub time_since_calibration_s: f64,
    /// Error flags (bitfield).
    pub error_flags: u32,
    /// Uptime since last reset (seconds).
    pub uptime_s: f64,
}

/// Real-time telemetry from device.
#[derive(Debug, Clone)]
pub struct DeviceTelemetry {
    /// Timestamp of measurement.
    pub timestamp: Instant,
    /// Acoustic power measured by on-device sensor (W).
    pub measured_power_w: f64,
    /// Transducer temperature (°C).
    pub temperature_c: f64,
    /// Acoustic impedance (Rayl).
    pub acoustic_impedance: f64,
    /// Back-reflection coefficient (0–1).
    pub reflection_coefficient: f64,
    /// Current draw from power supply (A).
    pub current_draw_a: f64,
}
