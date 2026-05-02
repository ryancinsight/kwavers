//! `SpectralOperator` trait definition.

use crate::core::error::KwaversResult;
use ndarray::{Array1, Array3, ArrayView3};

/// Trait for spectral operators
///
/// Spectral operators perform operations in Fourier (k-space) domain,
/// enabling spectral accuracy for smooth functions.
///
/// # Thread Safety
///
/// All implementations must be `Send + Sync` to enable parallel computation.
pub trait SpectralOperator: Send + Sync {
    /// Apply operator in k-space
    ///
    /// Computes the spectral derivative:
    /// 1. Forward FFT: u(x) → û(k)
    /// 2. Multiply by ik: ∂û/∂x = ik_x û(k)
    /// 3. Inverse FFT: ∂u/∂x = F⁻¹{ik_x û(k)}
    fn apply_kspace(&self, field: ArrayView3<f64>) -> KwaversResult<Array3<f64>>;

    /// Get wavenumber grids — returns (k_x, k_y, k_z)
    fn wavenumber_grid(&self) -> (Array1<f64>, Array1<f64>, Array1<f64>);

    /// Get the Nyquist wavenumber: k_max = π/Δx
    fn nyquist_wavenumber(&self) -> (f64, f64, f64);

    /// Apply anti-aliasing filter — removes components above 2/3 Nyquist
    fn apply_antialias_filter(&self, field: ArrayView3<f64>) -> KwaversResult<Array3<f64>>;
}
