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

#[cfg(test)]
mod tests {
    use super::*;

    /// Default scheme is Temporal with period 1 ms.
    #[test]
    fn default_is_temporal_1ms() {
        match RandomizationScheme::default() {
            RandomizationScheme::Temporal { period } => {
                assert!((period - 1e-3).abs() < 1e-15,
                    "default period must be 1e-3, got {period}");
            }
            other => panic!("expected Temporal, got {other:?}"),
        }
    }

    /// Spatial variant stores correlation length.
    #[test]
    fn spatial_stores_correlation() {
        let s = RandomizationScheme::Spatial { correlation: 2.5e-3 };
        match s {
            RandomizationScheme::Spatial { correlation } => {
                assert!((correlation - 2.5e-3).abs() < 1e-18);
            }
            _ => panic!("expected Spatial"),
        }
    }

    /// SpatioTemporal stores both fields.
    #[test]
    fn spatiotemporal_stores_both_fields() {
        let st = RandomizationScheme::SpatioTemporal { period: 1e-4, correlation: 1e-3 };
        match st {
            RandomizationScheme::SpatioTemporal { period, correlation } => {
                assert!((period - 1e-4).abs() < 1e-18);
                assert!((correlation - 1e-3).abs() < 1e-18);
            }
            _ => panic!("expected SpatioTemporal"),
        }
    }

    /// Clone produces an equal copy.
    #[test]
    fn clone_produces_equal_copy() {
        let original = RandomizationScheme::Frequency { bandwidth: 5e4 };
        let cloned = original;
        assert_eq!(original, cloned);
    }
}
