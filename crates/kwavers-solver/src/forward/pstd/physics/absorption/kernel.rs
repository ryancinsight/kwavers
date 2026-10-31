use super::strata::ExponentStrata;
use leto::Array3;

/// Precomputed spectral absorption arrays for power-law fractional Laplacian.
///
/// Allocated only when `AbsorptionMode::PowerLaw` is active; `None` for lossless mode.
/// This avoids 4 × N³ × 8-byte allocations per lossless simulation.
pub(crate) struct AbsorptionKernel {
    /// Absorption coefficient field τ = −2 α₀ c₀^(y−1) [Treeby & Cox 2010 Eq. 19]
    pub tau: Array3<f64>,
    /// Dispersion coefficient field η = 2 α₀ c₀^y tan(πy/2) [Eq. 20]; zero
    /// when the power law omits dispersion.
    pub eta: Array3<f64>,
    /// Per-cell α_SI coefficient [Np/((rad/s)^y·m)].
    ///
    /// Converts to physical absorption at center angular frequency ω_c via
    /// `α(ω_c) [Np/m] = alpha_si · ω_c^y` with `y` from [`Self::spectrum`].
    /// Used by `PSTDSolver::compute_acoustic_heat_source()` for thermal coupling.
    pub alpha_si: Array3<f64>,
    /// Exponent `y` that `tau`, `eta` and `alpha_si` share, with the spectral
    /// operators built for it.
    pub spectrum: ExponentSpectrum,
}

/// Power-law exponent `y` of `α(ω) = α_SI ω^y` and the fractional-Laplacian
/// symbols `|k|^(y−2)`, `|k|^(y−1)` built for it, stored on the r2c half
/// spectrum (`z ∈ [0, nz/2]`) with the DC bin set to zero.
pub(crate) enum ExponentSpectrum {
    /// One exponent for every voxel and one symbol pair (k-Wave's model).
    Uniform {
        /// The exponent.
        exponent: f64,
        /// `|k|^(y−2)` [Treeby & Cox 2010 Eq. 10].
        nabla1: Array3<f64>,
        /// `|k|^(y−1)` [Treeby & Cox 2010 Eq. 10].
        nabla2: Array3<f64>,
    },
    /// A per-voxel exponent, blended between exponent strata.
    Stratified(ExponentStrata),
}

impl ExponentSpectrum {
    /// Exponent of voxel `[i, j, k]`.
    pub(crate) fn exponent_at(&self, index: [usize; 3]) -> f64 {
        match self {
            Self::Uniform { exponent, .. } => *exponent,
            Self::Stratified(strata) => strata.exponent_at(index),
        }
    }
}
