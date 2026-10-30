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
    /// Power law absorption: α(f) = α₀ · (f / 1 MHz)^y
    PowerLaw {
        /// Absorption coefficient at 1 MHz [dB/(MHz^y·cm)], when the caller is
        /// requesting one explicitly.
        ///
        /// `Some` is a uniform request and applies at every voxel. `None` hands
        /// ownership to the medium, which the solver then reads per voxel —
        /// the only route by which a heterogeneous medium's spatial variation
        /// survives. The two cases are distinguishable precisely because they
        /// are not both an `f64`: a sentinel zero could not tell "unset" from
        /// "deliberately lossless", and conflating them made a caller's
        /// coefficient silently unreachable (ADR 120).
        ///
        /// The spectral solvers convert this raw k-Wave coefficient to the
        /// SI coefficient `Np/((rad/s)^y·m)` at the solver boundary.
        alpha_coeff: Option<f64>,
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

// Power-law unit conversions (forward, at-frequency, and inverse forms) are
// owned by `kwavers_medium::absorption::power_law` beside `PowerLawAbsorption`
// (KW-POWER-LAW-INVERSE-FORMS); this module keeps only the `AbsorptionMode`
// model enum. Import converters from
// `kwavers_medium::absorption::power_law`.
