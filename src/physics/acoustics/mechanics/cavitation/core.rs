//! Core cavitation mechanics functionality
//!
//! This module provides fundamental cavitation detection and modeling
//! based on acoustic pressure thresholds and bubble dynamics.

use crate::domain::core::error::KwaversResult;
use ndarray::{Array3, Zip};

/// Core cavitation detection and modeling
pub trait CavitationCore: Send + Sync {
    /// Detect cavitation based on pressure threshold
    fn detect_cavitation(&self, pressure: f64, threshold: f64) -> bool;

    /// Calculate cavitation index
    fn cavitation_index(&self, pressure: f64, vapor_pressure: f64, ambient_pressure: f64) -> f64;

    /// Update cavitation state
    fn update(&mut self, pressure_field: &Array3<f64>, dt: f64) -> KwaversResult<()>;
}

/// Cavitation threshold models
#[derive(Debug, Clone, Copy)]
pub enum ThresholdModel {
    /// Blake threshold (static pressure)
    Blake,
    /// Neppiras threshold (acoustic pressure)
    Neppiras,
    /// Apfel-Holland mechanical index
    MechanicalIndex,
    /// Flynn threshold (violent collapse)
    Flynn,
}

/// Cavitation state at a point
#[derive(Debug, Clone, Copy)]
pub struct CavitationState {
    /// Whether cavitation is occurring
    pub is_cavitating: bool,
    /// Cavitation intensity (0-1)
    pub intensity: f64,
    /// Time since cavitation onset \[s\]
    pub duration: f64,
    /// Peak negative pressure reached \[Pa\]
    pub peak_negative_pressure: f64,
    /// Mechanical index value
    pub mechanical_index: f64,
}

impl Default for CavitationState {
    fn default() -> Self {
        Self {
            is_cavitating: false,
            intensity: 0.0,
            duration: 0.0,
            peak_negative_pressure: 0.0,
            mechanical_index: 0.0,
        }
    }
}

/// Calculate Blake threshold pressure
/// Based on Blake (1949): "The onset of cavitation in liquids"
#[must_use]
pub fn blake_threshold(
    surface_tension: f64,  // [N/m]
    initial_radius: f64,   // [m]
    ambient_pressure: f64, // [Pa]
    vapor_pressure: f64,   // [Pa]
) -> f64 {
    // P_Blake = P_0 + P_v - 2σ/R_0
    ambient_pressure + vapor_pressure - 2.0 * surface_tension / initial_radius
}

/// Calculate Neppiras threshold
/// Based on Neppiras (1980): "Acoustic cavitation"
#[must_use]
pub fn neppiras_threshold(
    ambient_pressure: f64, // [Pa]
    vapor_pressure: f64,   // [Pa]
    surface_tension: f64,  // [N/m]
    nucleus_radius: f64,   // [m]
) -> f64 {
    // Threshold for transient cavitation
    let hydrostatic = ambient_pressure - vapor_pressure;
    let surface = 2.0 * surface_tension / nucleus_radius;

    0.5 * (hydrostatic + surface)
}

/// Calculate Flynn threshold for violent cavitation
/// Based on Flynn (1964): "Physics of Acoustic Cavitation in Liquids"
#[must_use]
pub fn flynn_threshold(
    ambient_pressure: f64, // [Pa]
    vapor_pressure: f64,   // [Pa]
    surface_tension: f64,  // [N/m]
    nucleus_radius: f64,   // [m]
) -> f64 {
    // Flynn criterion for transient cavitation
    // P_Flynn = 0.83 * (P_0 + 2σ/R_n)
    0.83 * (ambient_pressure + 2.0 * surface_tension / nucleus_radius) - vapor_pressure
}

/// Calculate mechanical index (MI)
/// MI = `P_neg` / `sqrt(f_c)` where `P_neg` in `MPa` and `f_c` in `MHz`
#[must_use]
pub fn mechanical_index(peak_negative_pressure: f64, center_frequency: f64) -> f64 {
    let p_mpa = peak_negative_pressure.abs() / 1e6; // Convert Pa to MPa
    let f_mhz = center_frequency / 1e6; // Convert Hz to MHz

    p_mpa / f_mhz.sqrt()
}

/// Flynn's criterion for violent collapse
/// Based on Flynn (1964): "Physics of acoustic cavitation in liquids"
#[must_use]
pub fn flynn_criterion(
    max_radius: f64,     // [m]
    initial_radius: f64, // [m]
) -> bool {
    // Violent collapse when R_max/R_0 > 2
    max_radius / initial_radius > 2.0
}

/// Cavitation dose accumulation
#[derive(Debug, Clone)]
pub struct CavitationDose {
    /// Accumulated dose value
    pub total_dose: f64,
    /// Time history of cavitation events
    pub time_history: Vec<f64>,
    /// Intensity history
    pub intensity_history: Vec<f64>,
}

impl CavitationDose {
    #[must_use]
    pub fn new() -> Self {
        Self {
            total_dose: 0.0,
            time_history: Vec::new(),
            intensity_history: Vec::new(),
        }
    }

    /// Update dose with new cavitation event
    pub fn update(&mut self, intensity: f64, dt: f64, time: f64) {
        self.total_dose += intensity * dt;
        self.time_history.push(time);
        self.intensity_history.push(intensity);
    }

    /// Calculate time-weighted average intensity
    #[must_use]
    pub fn average_intensity(&self) -> f64 {
        if self.intensity_history.is_empty() {
            0.0
        } else {
            self.intensity_history.iter().sum::<f64>() / self.intensity_history.len() as f64
        }
    }
}

/// Rectified diffusion model for bubble growth
/// Based on Eller & Flynn (1965): "Rectified diffusion during nonlinear pulsations"
#[must_use]
pub fn rectified_diffusion_rate(
    radius: f64,            // [m]
    ambient_pressure: f64,  // [Pa]
    acoustic_pressure: f64, // [Pa]
    frequency: f64,         // [Hz]
    diffusivity: f64,       // [m²/s]
    concentration: f64,     // [mol/m³]
) -> f64 {
    // Rectified diffusion growth rate using Eller-Flynn model
    // This is a standard approximation widely used in bubble dynamics
    // Full model would include shell elasticity for contrast agents
    //
    // References:
    // - Eller & Flynn (1965): "Rectified diffusion during nonlinear pulsations"
    // - Church (1988): "A theoretical study of cavitation generated by acoustic beam"
    let pressure_ratio = acoustic_pressure / ambient_pressure;
    let peclet = radius * radius * frequency / diffusivity;

    // Growth rate ∝ R * D * C * (P_ac/P_0) * √Pe
    4.0 * std::f64::consts::PI
        * radius
        * diffusivity
        * concentration
        * pressure_ratio
        * peclet.sqrt()
}

impl Default for CavitationDose {
    fn default() -> Self {
        Self::new()
    }
}

/// Main cavitation model implementation
#[derive(Debug, Clone)]
pub struct CavitationModel {
    /// Threshold model to use
    pub threshold_model: ThresholdModel,
    /// Surface tension [N/m]
    pub surface_tension: f64,
    /// Initial bubble radius \[m\]
    pub initial_radius: f64,
    /// Ambient pressure \[Pa\]
    pub ambient_pressure: f64,
    /// Vapor pressure \[Pa\]
    pub vapor_pressure: f64,
    /// Current cavitation states
    pub states: Array3<CavitationState>,
    /// Cavitation dose accumulator
    pub dose: CavitationDose,
}

impl CavitationModel {
    /// Create new cavitation model
    #[must_use]
    pub fn new(grid_shape: (usize, usize, usize)) -> Self {
        Self {
            threshold_model: ThresholdModel::MechanicalIndex,
            surface_tension: 0.0728,    // Water at 20°C
            initial_radius: 1e-6,       // 1 micron
            ambient_pressure: 101325.0, // 1 atm
            vapor_pressure: 2339.0,     // Water at 20°C
            states: Array3::default(grid_shape),
            dose: CavitationDose::new(),
        }
    }

    /// Update cavitation state based on pressure field
    pub fn update(&mut self, pressure_field: &Array3<f64>, frequency: f64, dt: f64, time: f64) {
        let threshold = match self.threshold_model {
            ThresholdModel::Blake => blake_threshold(
                self.surface_tension,
                self.initial_radius,
                self.ambient_pressure,
                self.vapor_pressure,
            ),
            ThresholdModel::Neppiras => neppiras_threshold(
                self.ambient_pressure,
                self.vapor_pressure,
                self.surface_tension,
                self.initial_radius,
            ),
            _ => 0.5 * self.ambient_pressure, // Default threshold
        };

        for ((i, j, k), p) in pressure_field.indexed_iter() {
            let state = &mut self.states[[i, j, k]];

            // Check for cavitation
            let was_cavitating = state.is_cavitating;
            state.is_cavitating = *p < -threshold;

            // Update state
            if state.is_cavitating {
                state.duration += dt;
                state.peak_negative_pressure = state.peak_negative_pressure.min(*p);
                state.mechanical_index = mechanical_index(*p, frequency);
                state.intensity =
                    (state.peak_negative_pressure.abs() / self.ambient_pressure).min(1.0);

                // Update dose
                self.dose.update(state.intensity, dt, time);
            } else if was_cavitating {
                // Just stopped cavitating
                state.duration = 0.0;
                state.intensity = 0.0;
            }
        }
    }
}

impl CavitationCore for CavitationModel {
    fn detect_cavitation(&self, pressure: f64, threshold: f64) -> bool {
        pressure < -threshold
    }

    fn cavitation_index(&self, pressure: f64, vapor_pressure: f64, ambient_pressure: f64) -> f64 {
        (ambient_pressure + pressure - vapor_pressure) / (ambient_pressure - vapor_pressure)
    }

    fn update(&mut self, pressure_field: &Array3<f64>, dt: f64) -> KwaversResult<()> {
        // Update cavitation states based on pressure field
        Zip::from(&mut self.states)
            .and(pressure_field)
            .for_each(|state, &pressure| {
                let threshold = match self.threshold_model {
                    ThresholdModel::Blake => blake_threshold(
                        self.surface_tension,
                        self.initial_radius,
                        self.ambient_pressure,
                        self.vapor_pressure,
                    ),
                    ThresholdModel::Neppiras => neppiras_threshold(
                        self.ambient_pressure,
                        self.vapor_pressure,
                        self.surface_tension,
                        self.initial_radius,
                    ),
                    ThresholdModel::Flynn => flynn_threshold(
                        self.ambient_pressure,
                        self.vapor_pressure,
                        self.surface_tension,
                        self.initial_radius,
                    ),
                    ThresholdModel::MechanicalIndex => {
                        // MI-based threshold uses pressure directly
                        // Threshold is when MI > 0.7 (FDA guideline)
                        // For 1 MHz, this is approximately -0.7 MPa
                        -0.7e6 // -0.7 MPa
                    }
                };

                if pressure < threshold {
                    // Cavitation occurring
                    if !state.is_cavitating {
                        state.is_cavitating = true;
                        state.duration = 0.0;
                    }
                    state.duration += dt;
                    state.intensity = ((threshold - pressure) / threshold).min(1.0);
                    state.peak_negative_pressure = state.peak_negative_pressure.min(pressure);
                } else {
                    // No cavitation
                    state.is_cavitating = false;
                    state.intensity = 0.0;
                }
            });

        // Update dose with average intensity
        let total_intensity: f64 = self
            .states
            .iter()
            .filter(|s| s.is_cavitating)
            .map(|s| s.intensity)
            .sum();
        let avg_intensity = total_intensity / self.states.len() as f64;
        let time = dt; // This should be accumulated time, but we don't have it in this trait
        self.dose.update(avg_intensity, dt, time);

        Ok(())
    }
}
