//! k-Wave Configuration Types
//!
//! This module contains all configuration structures and enums for k-Wave compatibility,
//! following the Single Responsibility Principle (GRASP).

use ndarray::Array3;

use super::sensors::SensorConfig;

/// k-Wave simulation configuration matching MATLAB interface
#[derive(Debug, Clone)]
pub struct KWaveConfig {
    /// Number of time steps
    pub nt: usize,
    /// Time step size \[s\]
    pub dt: f64,
    /// Enable power law absorption
    pub absorption_mode: AbsorptionMode,
    /// Enable nonlinear acoustics
    pub nonlinearity: bool,
    /// PML size in grid points
    pub pml_size: usize,
    /// PML absorption coefficient
    pub pml_alpha: f64,
    /// Data recording options
    pub sensor_mask: Option<Array3<bool>>,
    /// Perfectly matched layer inside domain
    pub pml_inside: bool,
    /// Smooth source terms
    pub smooth_sources: bool,
    pub sensor_config: SensorConfig,
}

/// Absorption models supported by k-Wave (enhanced for exact parity)
#[derive(Debug, Clone)]
pub enum AbsorptionMode {
    /// No absorption
    Lossless,
    /// Stokes absorption (frequency squared)
    Stokes,
    /// Power law absorption: α = α₀ω^y
    PowerLaw { alpha_coeff: f64, alpha_power: f64 },
    /// Multi-relaxation absorption model for complex media
    /// References: Szabo, T. L. (1995). "Time domain wave equations for lossy media"
    MultiRelaxation {
        tau: Vec<f64>,     // Relaxation times [s]
        weights: Vec<f64>, // Relaxation weights [dimensionless]
    },
    /// Causal absorption with configurable relaxation times
    /// References: Chen, W. & Holm, S. (2003). "Modified Szabo's wave equation models"
    Causal {
        relaxation_times: Vec<f64>, // Multiple relaxation times [s]
        alpha_0: f64,               // Low-frequency absorption [Np/m]
    },
}

impl Default for KWaveConfig {
    fn default() -> Self {
        Self {
            nt: 1000,
            dt: 1e-7, // 100 ns default time step
            absorption_mode: AbsorptionMode::Lossless,
            nonlinearity: false,
            pml_size: 20,
            pml_alpha: 2.0,
            sensor_mask: None,
            pml_inside: true,
            smooth_sources: true,
            sensor_config: SensorConfig::default(),
        }
    }
}
