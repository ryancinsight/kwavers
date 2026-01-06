//! Spectral Solver Configuration
//!
//! This module contains all configuration structures and enums for the generalized
//! spectral solver.

use ndarray::Array3;
use serde::{Deserialize, Serialize};

use crate::boundary::{CPMLConfig, PMLConfig};
use crate::recorder::SensorConfig;
use crate::solver::spectral_correction::SpectralCorrectionConfig;

/// Compatibility modes for the spectral solver
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum CompatibilityMode {
    /// Optimal mode using latest literature approaches (default)
    Optimal,
    /// k-Wave parity mode (MATLAB compatibility)
    KWave,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type")]
pub enum BoundaryConfig {
    /// Perfectly Matched Layer
    PML(PMLConfig),
    /// Convolutional Perfectly Matched Layer
    CPML(CPMLConfig),
    /// No absorbing boundary
    None,
}

impl Default for BoundaryConfig {
    fn default() -> Self {
        Self::PML(PMLConfig::default())
    }
}

/// Configuration for the generalized spectral solver
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SpectralConfig {
    /// Number of time steps
    pub nt: usize,
    /// Time step size [s]
    pub dt: f64,
    /// Compatibility mode for solver logic
    pub compatibility_mode: CompatibilityMode,
    /// Spectral (k-space) correction configuration
    pub spectral_correction: SpectralCorrectionConfig,
    /// Absorption mode
    pub absorption_mode: AbsorptionMode,
    /// Enable nonlinear acoustics
    pub nonlinearity: bool,
    /// Boundary condition configuration
    pub boundary: BoundaryConfig,
    /// Data recording options
    pub sensor_mask: Option<Array3<bool>>,
    /// Perfectly matched layer inside domain
    pub pml_inside: bool,
    /// Smooth source terms
    pub smooth_sources: bool,
    /// Anti-aliasing filter configuration
    pub anti_aliasing: AntiAliasingConfig,
    pub sensor_config: SensorConfig,
}

impl Default for SpectralConfig {
    fn default() -> Self {
        Self {
            nt: 1000,
            dt: 1e-7, // 100 ns default time step
            compatibility_mode: CompatibilityMode::Optimal,
            spectral_correction: SpectralCorrectionConfig::default(),
            absorption_mode: AbsorptionMode::Lossless,
            nonlinearity: false,
            boundary: BoundaryConfig::PML(PMLConfig {
                thickness: 20,
                ..PMLConfig::default()
            }),
            sensor_mask: None,
            pml_inside: true,
            smooth_sources: true,
            anti_aliasing: AntiAliasingConfig::default(),
            sensor_config: SensorConfig::default(),
        }
    }
}

/// Anti-aliasing filter configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AntiAliasingConfig {
    /// Enable anti-aliasing filter
    pub enabled: bool,
    /// Cutoff frequency as fraction of Nyquist (0.0 to 1.0)
    pub cutoff: f64,
    /// Filter order/steepness
    pub order: u32,
}

impl Default for AntiAliasingConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            cutoff: 0.95,
            order: 8,
        }
    }
}

/// Absorption models supported by the spectral solver
#[derive(Debug, Clone, Serialize, Deserialize)]
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
