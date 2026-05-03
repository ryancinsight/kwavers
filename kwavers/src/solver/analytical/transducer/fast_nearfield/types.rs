//! FNM configuration and supporting types.

use crate::math::fft::Complex64;
use ndarray::Array2;

/// Configuration for Fast Nearfield Method
#[derive(Debug, Clone)]
pub struct FNMConfig {
    /// Grid spacing in x-direction (m)
    pub dx: f64,
    /// Grid spacing in y-direction (m)
    pub dy: f64,
    /// Number of angular spectrum points (Nx, Ny)
    pub angular_spectrum_size: (usize, usize),
    /// Maximum k-space extent (as fraction of Nyquist)
    pub k_max_factor: f64,
    /// Use separable approximation for faster computation
    pub separable_approximation: bool,
}

impl Default for FNMConfig {
    fn default() -> Self {
        Self {
            dx: 0.1e-3, // 0.1 mm
            dy: 0.1e-3, // 0.1 mm
            angular_spectrum_size: (512, 512),
            k_max_factor: 2.0,
            separable_approximation: false,
        }
    }
}

/// Angular spectrum factors for a given z-plane
#[derive(Debug, Clone)]
pub struct AngularSpectrumFactors {
    /// Z distance (m)
    pub z: f64,
    /// Angular spectrum of Green's function (complex)
    pub green_spectrum: Array2<Complex64>,
    /// kx coordinates
    pub kx: Vec<f64>,
    /// ky coordinates
    pub ky: Vec<f64>,
}
