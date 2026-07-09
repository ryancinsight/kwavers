use super::strata::ExponentStrata;
use leto::Array3;

/// Precomputed spectral absorption arrays for power-law fractional Laplacian.
///
/// Allocated only when `AbsorptionMode::PowerLaw` is active; `None` for lossless mode.
/// This avoids 4 × N³ × 8-byte allocations per lossless simulation.
pub(crate) struct AbsorptionKernel {
    /// Absorption coefficient field τ = −2 α₀ c₀^(y−1) [Treeby & Cox 2010 Eq. 19]
    pub tau: Array3<f64>,
    /// Dispersion coefficient field η = 2 α₀ c₀^y tan(πy/2) [Eq. 20]
    pub eta: Array3<f64>,
    /// Spectral nabla1 operator |k|^(y−2) in FFT wavenumber order [Eq. 10]
    pub nabla1: Array3<f64>,
    /// Spectral nabla2 operator |k|^(y−1) in FFT wavenumber order [Eq. 10]
    pub nabla2: Array3<f64>,
    /// Per-cell α_SI coefficient [Np/((rad/s)^y·m)].
    ///
    /// Converts to physical absorption at center angular frequency ω_c via
    /// `α(ω_c) [Np/m] = alpha_si · ω_c^y`.
    /// Used by `PSTDSolver::compute_acoustic_heat_source()` for thermal coupling.
    pub alpha_si: Array3<f64>,
    /// Stratified spectral operators for a **spatially-varying** exponent y(x).
    ///
    /// `None` when y is uniform (the `nabla1`/`nabla2` single-symbol path is used,
    /// matching k-Wave); `Some` when the medium carries per-voxel exponents (e.g.
    /// a CT-derived tissue medium), in which case the apply path blends per-voxel
    /// across exponent strata — beyond k-Wave's single global exponent.
    pub strata: Option<ExponentStrata>,
}
