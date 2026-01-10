//! Absorption models for acoustic simulations
//!
//! Numerical models for simulating absorption and dispersion in various media.

use serde::{Deserialize, Serialize};

/// Absorption models supported by solvers
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Default)]
pub enum AbsorptionMode {
    /// No absorption
    #[default]
    Lossless,
    /// Stokes absorption (frequency squared)
    Stokes,
    /// Power law absorption: α = α₀ω^y
    PowerLaw {
        /// Absorption coefficient [Np/m/MHz^y]
        alpha_coeff: f64,
        /// Power law exponent
        alpha_power: f64,
    },
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
