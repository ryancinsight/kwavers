//! Core cavitation mechanics functionality
//!
//! This module provides fundamental cavitation detection and modeling
//! based on acoustic pressure thresholds and bubble dynamics.

use crate::error::KwaversResult;
use ndarray::{Array1, Array3};

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
    /// Time since cavitation onset [s]
    pub duration: f64,
    /// Peak negative pressure reached [Pa]
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

/// Calculate mechanical index (MI)
/// MI = P_neg / sqrt(f_c) where P_neg in MPa and f_c in MHz
pub fn mechanical_index(peak_negative_pressure: f64, center_frequency: f64) -> f64 {
    let p_mpa = peak_negative_pressure.abs() / 1e6; // Convert Pa to MPa
    let f_mhz = center_frequency / 1e6; // Convert Hz to MHz

    p_mpa / f_mhz.sqrt()
}

/// Flynn's criterion for violent collapse
/// Based on Flynn (1964): "Physics of acoustic cavitation in liquids"
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
pub fn rectified_diffusion_rate(
    radius: f64,            // [m]
    ambient_pressure: f64,  // [Pa]
    acoustic_pressure: f64, // [Pa]
    frequency: f64,         // [Hz]
    diffusivity: f64,       // [m²/s]
    concentration: f64,     // [mol/m³]
) -> f64 {
    // Simplified rectified diffusion growth rate
    let pressure_ratio = acoustic_pressure / ambient_pressure;
    let peclet = radius * radius * frequency / diffusivity;

    // Growth rate proportional to pressure amplitude and Peclet number
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
