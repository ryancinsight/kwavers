//! Reproducible sensitivity-screening configuration.

use core::num::NonZeroU32;
use tyche_core::Seed;

/// Configuration for deterministic correlation screening.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SensitivityConfig {
    /// Number of Latin-hypercube samples.
    pub sample_count: NonZeroU32,
    /// Stable Tyche study seed.
    pub seed: Seed,
}

impl Default for SensitivityConfig {
    fn default() -> Self {
        Self {
            sample_count: NonZeroU32::new(1000)
                .expect("invariant: the default sample count is non-zero"),
            seed: Seed::new(0),
        }
    }
}
