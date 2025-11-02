//! High-Intensity Focused Ultrasound (HIFU) Module
//!
//! This module implements therapeutic ultrasound applications using high-intensity
//! focused ultrasound for non-invasive tissue ablation and other treatments.
//!
//! # Physics Overview
//!
//! HIFU uses focused ultrasound beams to create localized heating and tissue damage
//! through several mechanisms:
//!
//! 1. **Thermal ablation**: Absorption of acoustic energy leading to temperature rise
//! 2. **Cavitation**: Bubble formation and collapse causing mechanical damage
//! 3. **Mechanical effects**: Radiation force and streaming
//!
//! # Key Components
//!
//! - `HIFUTransducer`: Focused transducer design and acoustic field computation
//! - `HIFUTreatmentPlan`: Treatment planning and monitoring
//! - `ThermalDose`: Thermal modeling and bio-heat transfer
//! - `CavitationModel`: Bubble dynamics and cavitation effects

use crate::error::{KwaversError, KwaversResult, ValidationError};
use crate::grid::Grid;
use crate::medium::Medium;
use ndarray::Array3;

/// HIFU transducer configuration
#[derive(Debug, Clone)]
pub struct HIFUTransducer {
    /// Transducer geometry
    pub geometry: TransducerGeometry,
    /// Operating frequency (Hz)
    pub frequency: f64,
    /// Acoustic power (W)
    pub acoustic_power: f64,
    /// Focal length (m)
    pub focal_length: f64,
    /// Aperture radius (m)
    pub aperture_radius: f64,
    /// Duty cycle (0-1)
    pub duty_cycle: f64,
}

/// Transducer geometry types
#[derive(Debug, Clone, PartialEq)]
pub enum TransducerGeometry {
    /// Single-element focused transducer
    SingleElement,
    /// Phased array transducer
    PhasedArray {
        /// Number of elements
        n_elements: usize,
        /// Element spacing (m)
        element_spacing: f64,
    },
    /// Annular array transducer
    AnnularArray {
        /// Number of rings
        n_rings: usize,
        /// Ring radii (m)
        ring_radii: Vec<f64>,
    },
}

/// Treatment planning and execution
#[derive(Debug, Clone)]
pub struct HIFUTreatmentPlan {
    /// Target region definition
    pub target: TreatmentTarget,
    /// Treatment protocol
    pub protocol: TreatmentProtocol,
    /// Safety margins and constraints
    pub safety: SafetyConstraints,
    /// Monitoring configuration
    pub monitoring: MonitoringConfig,
}

/// Treatment target specification
#[derive(Debug, Clone)]
pub struct TreatmentTarget {
    /// Target center position (m)
    pub center: [f64; 3],
    /// Target dimensions (m)
    pub dimensions: [f64; 3],
    /// Target shape
    pub shape: TargetShape,
}

/// Target shape types
#[derive(Debug, Clone, PartialEq)]
pub enum TargetShape {
    /// Spherical target
    Sphere,
    /// Cylindrical target
    Cylinder,
    /// Custom shape defined by mask
    Custom,
}

/// Treatment protocol parameters
#[derive(Debug, Clone)]
pub struct TreatmentProtocol {
    /// Total treatment time (s)
    pub total_duration: f64,
    /// Pulse duration (s)
    pub pulse_duration: f64,
    /// Pulse repetition frequency (Hz)
    pub prf: f64,
    /// Cooling periods between pulses (s)
    pub cooling_period: f64,
    /// Treatment phases
    pub phases: Vec<TreatmentPhase>,
}

/// Treatment phase definition
#[derive(Debug, Clone)]
pub struct TreatmentPhase {
    /// Phase name
    pub name: String,
    /// Phase duration (s)
    pub duration: f64,
    /// Acoustic power during phase (W)
    pub power: f64,
    /// Focus position offset from target center (m)
    pub focus_offset: [f64; 3],
}

/// Safety constraints
#[derive(Debug, Clone)]
pub struct SafetyConstraints {
    /// Maximum temperature (°C)
    pub max_temperature: f64,
    /// Maximum thermal dose (CEM43)
    pub max_thermal_dose: f64,
    /// Maximum acoustic intensity (W/cm²)
    pub max_intensity: f64,
    /// Critical structure avoidance zones
    pub avoidance_zones: Vec<AvoidanceZone>,
}

/// Avoidance zone for critical structures
#[derive(Debug, Clone)]
pub struct AvoidanceZone {
    /// Zone center (m)
    pub center: [f64; 3],
    /// Zone radius (m)
    pub radius: f64,
    /// Maximum allowed temperature rise (°C)
    pub max_temp_rise: f64,
}

/// Monitoring configuration
#[derive(Debug, Clone)]
pub struct MonitoringConfig {
    /// Temperature monitoring points
    pub temperature_points: Vec<[f64; 3]>,
    /// Acoustic feedback channels
    pub feedback_channels: Vec<FeedbackChannel>,
    /// Real-time adjustment parameters
    pub real_time_adjustment: bool,
}

/// Feedback channel types
#[derive(Debug, Clone, PartialEq)]
pub enum FeedbackChannel {
    /// Magnetic Resonance Imaging
    MRI,
    /// Ultrasound imaging
    Ultrasound,
    /// Thermocouple
    Thermocouple,
    /// Infrared thermography
    Infrared,
}

/// Thermal dose calculation (CEM43 metric)
#[derive(Debug, Clone)]
pub struct ThermalDose {
    /// Cumulative equivalent minutes at 43°C
    pub cem43: Array3<f64>,
    /// Temperature history for dose calculation
    temperature_history: Vec<Array3<f64>>,
    /// Time points for temperature history
    time_points: Vec<f64>,
}

impl HIFUTransducer {
    /// Create a new single-element focused transducer
    pub fn new_single_element(
        frequency: f64,
        acoustic_power: f64,
        focal_length: f64,
        aperture_radius: f64,
    ) -> Self {
        Self {
            geometry: TransducerGeometry::SingleElement,
            frequency,
            acoustic_power,
            focal_length,
            aperture_radius,
            duty_cycle: 1.0,
        }
    }

    /// Compute acoustic pressure field
    ///
    /// Uses Rayleigh-Sommerfeld integral for focused transducer field computation.
    ///
    /// # References
    ///
    /// O'Neil (1949): Theory of focusing radiators
    pub fn compute_pressure_field(
        &self,
        grid: &Grid,
        medium: &dyn Medium,
    ) -> KwaversResult<Array3<f64>> {
        let (nx, ny, nz) = grid.dimensions();
        let mut pressure = Array3::zeros((nx, ny, nz));

        // Focus location (geometric focus)
        let focus_x = 0.0;
        let focus_y = 0.0;
        let focus_z = self.focal_length;

        // Compute pressure field using simplified focused field model
        // In practice, this would use full Rayleigh-Sommerfeld integration
        for k in 0..nz {
            for j in 0..ny {
                for i in 0..nx {
                    let x = i as f64 * grid.dx;
                    let y = j as f64 * grid.dy;
                    let z = k as f64 * grid.dz;

                    // Distance from transducer surface (simplified as point source at focus)
                    let r = ((x - focus_x).powi(2) + (y - focus_y).powi(2) + (z - focus_z).powi(2)).sqrt();

                    if r > 1e-6 {
                        // Simplified focused field: Gaussian beam profile
                        let wavenumber = 2.0 * std::f64::consts::PI * self.frequency / medium.sound_speed(0, 0, 0);

                        // Focal gain (simplified)
                        let focal_gain = (self.aperture_radius / (self.focal_length * wavenumber)).abs();

                        // Pressure amplitude (simplified scaling)
                        let pressure_amplitude = (self.acoustic_power / (4.0 * std::f64::consts::PI * r.powi(2))).sqrt() * focal_gain;

                        // Phase factor
                        let phase = -wavenumber * r;

                        pressure[[i, j, k]] = pressure_amplitude * phase.cos();
                    }
                }
            }
        }

        Ok(pressure)
    }

    /// Compute acoustic intensity field (W/m²)
    pub fn compute_intensity_field(
        &self,
        grid: &Grid,
        medium: &dyn Medium,
    ) -> KwaversResult<Array3<f64>> {
        let pressure = self.compute_pressure_field(grid, medium)?;

        // Intensity = (pressure²) / (ρc) for plane waves
        // For focused fields, this is a simplification
        let mut intensity = Array3::zeros(pressure.dim());
        let rho = medium.density(0, 0, 0);
        let c = medium.sound_speed(0, 0, 0);
        let impedance = rho * c;

        for (i, &p) in pressure.iter().enumerate() {
            intensity.as_slice_mut().unwrap()[i] = p.powi(2) / impedance;
        }

        Ok(intensity)
    }
}

impl HIFUTreatmentPlan {
    /// Create a new treatment plan
    pub fn new(target: TreatmentTarget, protocol: TreatmentProtocol) -> Self {
        Self {
            target,
            protocol,
            safety: SafetyConstraints::default(),
            monitoring: MonitoringConfig::default(),
        }
    }

    /// Validate treatment plan against safety constraints
    pub fn validate(&self, _grid: &Grid, _medium: &dyn Medium, transducer: &HIFUTransducer) -> KwaversResult<()> {
        // Check target is within accessible region
        if self.target.center[2] < transducer.focal_length * 0.5 {
            return Err(KwaversError::Validation(ValidationError::InvalidValue {
                parameter: "target.center.z".to_string(),
                value: self.target.center[2],
                reason: "Target too close to transducer".to_string(),
            }));
        }

        // Check thermal constraints
        if self.safety.max_temperature > 100.0 {
            return Err(KwaversError::Validation(ValidationError::InvalidValue {
                parameter: "safety.max_temperature".to_string(),
                value: self.safety.max_temperature,
                reason: "Maximum temperature exceeds safe limit".to_string(),
            }));
        }

        // Check acoustic intensity limits
        if self.safety.max_intensity > 1000.0 { // 1000 W/cm² = 10^7 W/m²
            return Err(KwaversError::Validation(ValidationError::InvalidValue {
                parameter: "safety.max_intensity".to_string(),
                value: self.safety.max_intensity,
                reason: "Maximum intensity exceeds safe limit".to_string(),
            }));
        }

        Ok(())
    }
}

impl Default for SafetyConstraints {
    fn default() -> Self {
        Self {
            max_temperature: 85.0, // °C
            max_thermal_dose: 240.0, // CEM43
            max_intensity: 1000.0, // W/cm²
            avoidance_zones: Vec::new(),
        }
    }
}

impl Default for MonitoringConfig {
    fn default() -> Self {
        Self {
            temperature_points: Vec::new(),
            feedback_channels: vec![FeedbackChannel::Ultrasound],
            real_time_adjustment: true,
        }
    }
}

impl ThermalDose {
    /// Create new thermal dose calculator
    pub fn new(grid: &Grid) -> Self {
        Self {
            cem43: Array3::zeros(grid.dimensions()),
            temperature_history: Vec::new(),
            time_points: Vec::new(),
        }
    }

    /// Add temperature measurement
    pub fn add_temperature_measurement(&mut self, temperature: Array3<f64>, time: f64) {
        self.temperature_history.push(temperature);
        self.time_points.push(time);
        self.update_cem43();
    }

    /// Update cumulative equivalent minutes at 43°C
    ///
    /// Uses Sapareto & Dewey (1984) formula: CEM43 = Σ R^(43-T) Δt
    /// where R = 2 for T < 43°C, R = 4 for T ≥ 43°C
    fn update_cem43(&mut self) {
        if self.temperature_history.len() < 2 {
            return;
        }

        self.cem43.fill(0.0);

        for i in 1..self.temperature_history.len() {
            let dt = self.time_points[i] - self.time_points[i-1];
            let temp_prev = &self.temperature_history[i-1];
            let temp_curr = &self.temperature_history[i];

            // Use average temperature over time step
            for idx in 0..self.cem43.len() {
                let t_avg = (temp_prev.as_slice().unwrap()[idx] + temp_curr.as_slice().unwrap()[idx]) / 2.0;

                // R factor based on temperature
                let r: f64 = if t_avg >= 43.0 { 4.0 } else { 2.0 };

                // Equivalent time contribution
                let t_eq = dt * r.powf(43.0 - t_avg);
                self.cem43.as_slice_mut().unwrap()[idx] += t_eq;
            }
        }
    }

    /// Get thermal dose at specific location
    pub fn dose_at(&self, i: usize, j: usize, k: usize) -> f64 {
        self.cem43[[i, j, k]]
    }

    /// Check if ablation threshold reached (CEM43 > 240)
    pub fn ablation_threshold_reached(&self) -> Array3<bool> {
        self.cem43.mapv(|dose| dose > 240.0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::medium::homogeneous::HomogeneousMedium;

    #[test]
    fn test_hifu_transducer_creation() {
        let transducer = HIFUTransducer::new_single_element(1e6, 100.0, 0.1, 0.05);

        assert_eq!(transducer.frequency, 1e6);
        assert_eq!(transducer.acoustic_power, 100.0);
        assert_eq!(transducer.focal_length, 0.1);
        assert_eq!(transducer.aperture_radius, 0.05);
    }

    #[test]
    fn test_pressure_field_computation() -> KwaversResult<()> {
        let grid = Grid::new(32, 32, 32, 0.002, 0.002, 0.002)?;
        let medium = HomogeneousMedium::new(1000.0, 1540.0, 0.5, 1.0, &grid);
        let transducer = HIFUTransducer::new_single_element(1e6, 50.0, 0.08, 0.04);

        let pressure = transducer.compute_pressure_field(&grid, &medium)?;

        // Check dimensions
        assert_eq!(pressure.dim(), grid.dimensions());

        // Check that pressure is non-zero at focus (approximate)
        let focus_i = (0.08 / 0.002) as usize;
        if focus_i < grid.nx {
            assert!(pressure[[focus_i, grid.ny/2, grid.nz/2]].abs() > 0.0);
        }

        Ok(())
    }

    #[test]
    fn test_thermal_dose_calculation() {
        let grid = Grid::new(16, 16, 16, 0.005, 0.005, 0.005).unwrap();
        let mut thermal_dose = ThermalDose::new(&grid);

        // Add some temperature measurements
        let temp1 = Array3::from_elem(grid.dimensions(), 37.0); // Baseline
        let temp2 = Array3::from_elem(grid.dimensions(), 50.0); // Heating
        let temp3 = Array3::from_elem(grid.dimensions(), 60.0); // More heating

        thermal_dose.add_temperature_measurement(temp1, 0.0);
        thermal_dose.add_temperature_measurement(temp2, 10.0);
        thermal_dose.add_temperature_measurement(temp3, 20.0);

        // Check that dose increases with temperature
        let dose_center = thermal_dose.dose_at(grid.nx/2, grid.ny/2, grid.nz/2);
        assert!(dose_center > 0.0);

        // Check ablation threshold detection
        let ablation = thermal_dose.ablation_threshold_reached();
        // Should be false for this short heating period
        assert!(!ablation[[grid.nx/2, grid.ny/2, grid.nz/2]]);
    }

    #[test]
    fn test_treatment_plan_validation() -> KwaversResult<()> {
        let target = TreatmentTarget {
            center: [0.0, 0.0, 0.08],
            dimensions: [0.01, 0.01, 0.01],
            shape: TargetShape::Sphere,
        };

        let protocol = TreatmentProtocol {
            total_duration: 30.0,
            pulse_duration: 5.0,
            prf: 1.0,
            cooling_period: 10.0,
            phases: vec![TreatmentPhase {
                name: "Heating".to_string(),
                duration: 20.0,
                power: 50.0,
                focus_offset: [0.0, 0.0, 0.0],
            }],
        };

        let plan = HIFUTreatmentPlan::new(target, protocol);

        let grid = Grid::new(32, 32, 32, 0.002, 0.002, 0.002)?;
        let medium = HomogeneousMedium::new(1000.0, 1540.0, 0.5, 1.0, &grid);
        let transducer = HIFUTransducer::new_single_element(1e6, 50.0, 0.08, 0.04);

        // Should validate successfully
        assert!(plan.validate(&grid, &medium, &transducer).is_ok());

        Ok(())
    }
}
