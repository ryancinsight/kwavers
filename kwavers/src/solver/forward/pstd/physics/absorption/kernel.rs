use ndarray::Array3;

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
}
