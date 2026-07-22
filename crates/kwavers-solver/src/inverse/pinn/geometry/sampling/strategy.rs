//! Interior experimental-design selection.

use serde::{Deserialize, Serialize};

/// Reproducible experimental design for PINN interior collocation points.
#[non_exhaustive]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum CollocationSamplingStrategy {
    /// Independent domain-separated Tyche counter points.
    Uniform,
    /// Allocation-free random-access Latin hypercube.
    LatinHypercube,
    /// Digitally shifted low-discrepancy Sobol sequence.
    Sobol,
}
