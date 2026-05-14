//! In-memory transducer implementation for simulation and tests.

use super::hardware::TransducerHardware;
use super::types::{
    DeviceStatus, DeviceTelemetry, HardwareCommand, HardwareResponse, TransducerSpecification,
    TransducerState,
};
use crate::core::error::{KwaversError, KwaversResult};
use std::time::Instant;

/// Mock transducer for simulation and testing.
#[derive(Debug)]
pub struct MockTransducer {
    /// Device specification.
    pub(super) spec: TransducerSpecification,
    /// Current state.
    pub(super) state: TransducerState,
    /// Current frequency.
    pub(super) current_frequency: f64,
    /// Current power.
    pub(super) current_power_percent: f64,
    /// Last error.
    pub(super) last_error: Option<String>,
    /// Creation time.
    pub(super) created_at: Instant,
}

impl MockTransducer {
    /// Create new mock transducer.
    #[must_use]
    pub fn new(model: String, manufacturer: String) -> Self {
        Self {
            spec: TransducerSpecification {
                model,
                manufacturer,
                serial_number: "MOCK-001".to_owned(),
                frequency_range: (0.5e6, 15.0e6),
                max_power: 100.0,
                num_elements: 64,
                focal_length_mm: Some(50.0),
                element_diameter_mm: 0.5,
                calibration_date: "2026-01-30".to_owned(),
                calibration_expiry: "2027-01-30".to_owned(),
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
                if !(0.0..=100.0).contains(&power) {
                    self.last_error = Some("Power must be 0-100%".to_owned());
                    return Err(KwaversError::InvalidInput("Invalid power range".to_owned()));
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
                    return Err(KwaversError::InvalidInput("Invalid frequency".to_owned()));
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
                self.calibrate()?;
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
                "Unsupported command for mock device".to_owned(),
            )),
        }
    }

    fn is_connected(&self) -> bool {
        true
    }

    fn calibrate(&mut self) -> KwaversResult<()> {
        self.state = TransducerState::Calibrating;
        self.state = TransducerState::Idle;
        Ok(())
    }

    fn get_telemetry(&self) -> KwaversResult<DeviceTelemetry> {
        Ok(DeviceTelemetry {
            timestamp: Instant::now(),
            measured_power_w: self.current_power_percent * self.spec.max_power / 100.0,
            temperature_c: (self.current_power_percent / 100.0).mul_add(10.0, 25.0),
            acoustic_impedance: 1.5e6,
            reflection_coefficient: 0.05,
            current_draw_a: (self.current_power_percent / 100.0).mul_add(4.0, 1.0),
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
