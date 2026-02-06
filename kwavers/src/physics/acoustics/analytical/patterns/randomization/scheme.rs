//! Randomization schemes for phase modulation

/// Randomization schemes
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum RandomizationScheme {
    /// Time-varying phase randomization
    Temporal {
        /// Switching period (seconds)
        period: f64,
    },

    /// Spatially-varying phase randomization
    Spatial {
        /// Correlation length (meters)
        correlation: f64,
    },

    /// Combined spatial and temporal
    SpatioTemporal {
        /// Temporal switching period
        period: f64,
        /// Spatial correlation length
        correlation: f64,
    },

    /// Frequency-dependent randomization
    Frequency {
        /// Bandwidth for randomization (Hz)
        bandwidth: f64,
    },

    /// Amplitude-weighted randomization
    Amplitude {
        /// Weighting factor
        weight: f64,
    },
}

impl Default for RandomizationScheme {
    fn default() -> Self {
        Self::Temporal { period: 1e-3 }
    }
}
