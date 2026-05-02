//! Pseudospectral derivative operator using FFT.
//!
//! # Wavenumber Convention
//!
//! For an N-point grid, wavenumbers are arranged as:
//! ```text
//! k = [0, 1, 2, ..., N/2-1, -N/2, -N/2+1, ..., -1] * (2π / L)
//! ```
//! where L = N * Δx is the domain length.
//!
//! # Reference
//!
//! - Liu, Q. H. (1997). Microwave Opt. Technol. Lett., 15(3), 158-165.

use super::trait_def::SpectralOperator;
use crate::core::error::{KwaversResult, NumericalError};
use crate::math::fft::{Complex64, Shape1D, FFT_CACHE_1D};
use ndarray::{Array1, Array3, ArrayView3, Axis};
use std::f64::consts::PI;
use std::sync::Arc;

/// Pseudospectral derivative operator using FFT
///
/// Computes spatial derivatives using the Fourier differentiation theorem,
/// providing spectral accuracy (exponential convergence) for smooth functions.
#[derive(Debug)]
pub struct PseudospectralDerivative {
    /// Wavenumber grid in X direction (rad/m)
    kx: Array1<f64>,
    /// Wavenumber grid in Y direction (rad/m)
    ky: Array1<f64>,
    /// Wavenumber grid in Z direction (rad/m)
    kz: Array1<f64>,
    /// Grid spacing in X (m)
    dx: f64,
    /// Grid spacing in Y (m)
    dy: f64,
    /// Grid spacing in Z (m)
    dz: f64,
}

impl PseudospectralDerivative {
    /// Create a new pseudospectral derivative operator
    ///
    /// # Arguments
    ///
    /// * `nx/ny/nz` - Number of grid points per direction
    /// * `dx/dy/dz` - Grid spacings (meters)
    ///
    /// # Errors
    ///
    /// Returns error if any grid spacing is non-positive.
    pub fn new(nx: usize, ny: usize, nz: usize, dx: f64, dy: f64, dz: f64) -> KwaversResult<Self> {
        if dx <= 0.0 || dy <= 0.0 || dz <= 0.0 {
            return Err(NumericalError::InvalidGridSpacing { dx, dy, dz }.into());
        }

        Ok(Self {
            kx: Self::wavenumber_vector(nx, dx),
            ky: Self::wavenumber_vector(ny, dy),
            kz: Self::wavenumber_vector(nz, dz),
            dx,
            dy,
            dz,
        })
    }

    /// Generate wavenumber vector for FFT.
    ///
    /// k[i] = 2π·i / (N·d)        for i = 0..N/2
    /// k[i] = 2π·(i−N) / (N·d)   for i = N/2..N
    pub(super) fn wavenumber_vector(n: usize, d: f64) -> Array1<f64> {
        let mut k = Array1::zeros(n);
        let dk = 2.0 * PI / ((n as f64) * d);

        for i in 0..n / 2 {
            k[i] = (i as f64) * dk;
        }
        for i in n / 2..n {
            k[i] = ((i as i64) - (n as i64)) as f64 * dk;
        }

        k
    }

    /// Compute spectral derivative in X direction: ∂u/∂x = F⁻¹{ik_x F{u}}
    ///
    /// # Spectral Accuracy
    ///
    /// For smooth periodic functions, achieves O(exp(-cN)) convergence.
    /// Validated: ∂(sin(kx))/∂x = k·cos(kx) with L∞ error < 1e-12.
    pub fn derivative_x(&self, field: ArrayView3<f64>) -> KwaversResult<Array3<f64>> {
        let (nx, ny, nz) = field.dim();

        if nx != self.kx.len() {
            return Err(NumericalError::InvalidGridSpacing {
                dx: self.dx,
                dy: self.dy,
                dz: self.dz,
            }
            .into());
        }

        let mut derivative = Array3::zeros((nx, ny, nz));
        let fft = FFT_CACHE_1D.get_or_create(Shape1D { n: nx });
        let ifft = Arc::clone(&fft);

        for j in 0..ny {
            for k in 0..nz {
                let mut buffer = Array1::<Complex64>::from_iter(
                    field
                        .index_axis(Axis(1), j)
                        .index_axis(Axis(1), k)
                        .iter()
                        .map(|&x| Complex64::new(x, 0.0)),
                );

                fft.forward_complex_inplace(&mut buffer);

                for (idx, kx_val) in self.kx.iter().enumerate() {
                    buffer[idx] *= Complex64::new(0.0, *kx_val);
                }

                // apollo-fft applies 1/N normalisation; no extra scale needed.
                ifft.inverse_complex_inplace(&mut buffer);

                for (idx, val) in buffer.iter().enumerate() {
                    derivative[[idx, j, k]] = val.re;
                }
            }
        }

        Ok(derivative)
    }

    /// Compute spectral derivative in Y direction: ∂u/∂y = F⁻¹{ik_y F{u}}
    pub fn derivative_y(&self, field: ArrayView3<f64>) -> KwaversResult<Array3<f64>> {
        let (nx, ny, nz) = field.dim();

        if ny != self.ky.len() {
            return Err(NumericalError::InvalidGridSpacing {
                dx: self.dx,
                dy: self.dy,
                dz: self.dz,
            }
            .into());
        }

        let mut derivative = Array3::zeros((nx, ny, nz));
        let fft = FFT_CACHE_1D.get_or_create(Shape1D { n: ny });
        let ifft = Arc::clone(&fft);

        for i in 0..nx {
            for k in 0..nz {
                let mut buffer = Array1::<Complex64>::from_iter(
                    field
                        .index_axis(Axis(0), i)
                        .index_axis(Axis(1), k)
                        .iter()
                        .map(|&x| Complex64::new(x, 0.0)),
                );

                fft.forward_complex_inplace(&mut buffer);

                for (idx, ky_val) in self.ky.iter().enumerate() {
                    buffer[idx] *= Complex64::new(0.0, *ky_val);
                }

                // apollo-fft applies 1/N normalisation; no extra scale needed.
                ifft.inverse_complex_inplace(&mut buffer);

                for (idx, val) in buffer.iter().enumerate() {
                    derivative[[i, idx, k]] = val.re;
                }
            }
        }

        Ok(derivative)
    }

    /// Compute spectral derivative in Z direction: ∂u/∂z = F⁻¹{ik_z F{u}}
    pub fn derivative_z(&self, field: ArrayView3<f64>) -> KwaversResult<Array3<f64>> {
        let (nx, ny, nz) = field.dim();

        if nz != self.kz.len() {
            return Err(NumericalError::InvalidGridSpacing {
                dx: self.dx,
                dy: self.dy,
                dz: self.dz,
            }
            .into());
        }

        let mut derivative = Array3::zeros((nx, ny, nz));
        let fft = FFT_CACHE_1D.get_or_create(Shape1D { n: nz });
        let ifft = Arc::clone(&fft);

        for i in 0..nx {
            for j in 0..ny {
                let mut buffer = Array1::<Complex64>::from_iter(
                    field
                        .index_axis(Axis(0), i)
                        .index_axis(Axis(0), j)
                        .iter()
                        .map(|&x| Complex64::new(x, 0.0)),
                );

                fft.forward_complex_inplace(&mut buffer);

                for (idx, kz_val) in self.kz.iter().enumerate() {
                    buffer[idx] *= Complex64::new(0.0, *kz_val);
                }

                // apollo-fft applies 1/N normalisation; no extra scale needed.
                ifft.inverse_complex_inplace(&mut buffer);

                for (idx, val) in buffer.iter().enumerate() {
                    derivative[[i, j, idx]] = val.re;
                }
            }
        }

        Ok(derivative)
    }
}

impl SpectralOperator for PseudospectralDerivative {
    fn apply_kspace(&self, field: ArrayView3<f64>) -> KwaversResult<Array3<f64>> {
        self.derivative_x(field)
    }

    fn wavenumber_grid(&self) -> (Array1<f64>, Array1<f64>, Array1<f64>) {
        (self.kx.clone(), self.ky.clone(), self.kz.clone())
    }

    fn nyquist_wavenumber(&self) -> (f64, f64, f64) {
        (PI / self.dx, PI / self.dy, PI / self.dz)
    }

    fn apply_antialias_filter(&self, _field: ArrayView3<f64>) -> KwaversResult<Array3<f64>> {
        Err(NumericalError::NotImplemented {
            feature: "Anti-aliasing filter (requires FFT integration)".to_string(),
        }
        .into())
    }
}
