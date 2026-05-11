//! Spectral Derivative Operators for Pseudospectral Methods
//!
//! Implements high-order accurate spatial derivative operators using spectral
//! methods (Fourier-based). Achieves spectral accuracy (exponential convergence)
//! for smooth fields.
//!
//! ## Mathematical Foundation
//!
//! ```text
//! ∂u/∂x = F⁻¹[i·kₓ·F[u]]
//! ```
//!
//! ## References
//!
//! - Boyd, J. P. (2001). Chebyshev and Fourier Spectral Methods
//! - Trefethen, L. N. (2000). Spectral Methods in MATLAB
//! - Canuto et al. (2006). Spectral Methods: Fundamentals in Single Domains

use crate::core::error::{KwaversError, KwaversResult};
use crate::math::fft::{Complex64, Fft1d, Shape1D, FFT_CACHE_1D};
use ndarray::parallel::prelude::*;
use ndarray::{Array1, Array3, ArrayView3, Axis};
use std::sync::Arc;

/// Spectral derivative operator for 3D fields.
///
/// FFT plans and i*k*dealiasing multipliers are pre-computed at construction,
/// eliminating per-call plan creation overhead.
pub struct SpectralDerivativeOperator {
    pub(super) nx: usize,
    pub(super) ny: usize,
    pub(super) nz: usize,

    fft_x: Arc<Fft1d>,
    ifft_x: Arc<Fft1d>,
    fft_y: Arc<Fft1d>,
    ifft_y: Arc<Fft1d>,
    fft_z: Arc<Fft1d>,
    ifft_z: Arc<Fft1d>,

    ikd_x: Vec<Complex64>,
    ikd_y: Vec<Complex64>,
    ikd_z: Vec<Complex64>,
}

impl std::fmt::Debug for SpectralDerivativeOperator {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("SpectralDerivativeOperator")
            .field("nx", &self.nx)
            .field("ny", &self.ny)
            .field("nz", &self.nz)
            .finish()
    }
}

impl Clone for SpectralDerivativeOperator {
    fn clone(&self) -> Self {
        Self {
            nx: self.nx,
            ny: self.ny,
            nz: self.nz,
            fft_x: Arc::clone(&self.fft_x),
            ifft_x: Arc::clone(&self.ifft_x),
            fft_y: Arc::clone(&self.fft_y),
            ifft_y: Arc::clone(&self.ifft_y),
            fft_z: Arc::clone(&self.fft_z),
            ifft_z: Arc::clone(&self.ifft_z),
            ikd_x: self.ikd_x.clone(),
            ikd_y: self.ikd_y.clone(),
            ikd_z: self.ikd_z.clone(),
        }
    }
}

impl SpectralDerivativeOperator {
    /// Create new spectral derivative operator.
    ///
    /// # Panics
    ///
    /// If any grid dimension is 0.
    pub fn new(nx: usize, ny: usize, nz: usize, dx: f64, dy: f64, dz: f64) -> Self {
        assert!(
            nx > 0 && ny > 0 && nz > 0,
            "Grid dimensions must be positive"
        );

        let kx = Self::compute_wavenumbers(nx, dx);
        let ky = Self::compute_wavenumbers(ny, dy);
        let kz = Self::compute_wavenumbers(nz, dz);

        let dealiasing_filter_x = Self::compute_dealiasing_filter(nx, dx);
        let dealiasing_filter_y = Self::compute_dealiasing_filter(ny, dy);
        let dealiasing_filter_z = Self::compute_dealiasing_filter(nz, dz);

        let ikd_x: Vec<Complex64> = (0..nx)
            .map(|i| Complex64::new(0.0, kx[i] * dealiasing_filter_x[i]))
            .collect();
        let ikd_y: Vec<Complex64> = (0..ny)
            .map(|i| Complex64::new(0.0, ky[i] * dealiasing_filter_y[i]))
            .collect();
        let ikd_z: Vec<Complex64> = (0..nz)
            .map(|i| Complex64::new(0.0, kz[i] * dealiasing_filter_z[i]))
            .collect();

        let fft_x = FFT_CACHE_1D.get_or_create(Shape1D { n: nx });
        let ifft_x = Arc::clone(&fft_x);
        let fft_y = FFT_CACHE_1D.get_or_create(Shape1D { n: ny });
        let ifft_y = Arc::clone(&fft_y);
        let fft_z = FFT_CACHE_1D.get_or_create(Shape1D { n: nz });
        let ifft_z = Arc::clone(&fft_z);

        Self {
            nx,
            ny,
            nz,
            fft_x,
            ifft_x,
            fft_y,
            ifft_y,
            fft_z,
            ifft_z,
            ikd_x,
            ikd_y,
            ikd_z,
        }
    }

    /// Compute wavenumber array: k[n] = 2π·n/(N·Δx) for n < N/2, 2π·(n-N)/(N·Δx) otherwise.
    fn compute_wavenumbers(n: usize, dx: f64) -> Array1<f64> {
        let mut k = Array1::zeros(n);
        let norm = 2.0 * std::f64::consts::PI / (n as f64 * dx);
        for i in 0..n / 2 {
            k[i] = i as f64 * norm;
        }
        for i in n / 2..n {
            k[i] = (i as f64 - n as f64) * norm;
        }
        k
    }

    /// Compute 2/3-rule dealiasing filter (sets |k| > 2π/(3Δx) to zero).
    /// # Errors
    /// - Returns [`Err`] if an internal constraint is violated.
    ///
    fn compute_dealiasing_filter(n: usize, dx: f64) -> Array1<f64> {
        let mut filter = Array1::ones(n);
        let cutoff = 2.0 * std::f64::consts::PI / (3.0 * dx);
        let k = Self::compute_wavenumbers(n, dx);
        for i in 0..n {
            if k[i].abs() > cutoff {
                filter[i] = 0.0;
            }
        }
        filter
    }

    /// Compute x-derivative via spectral method (FFT along x pencils).
    /// # Errors
    /// - Propagates any [`KwaversError`] returned by called functions.
    ///
    pub fn derivative_x(&self, field: &ArrayView3<f64>) -> KwaversResult<Array3<f64>> {
        self.validate_field(field)?;
        let mut derivative = Array3::zeros([self.nx, self.ny, self.nz]);
        self.derivative_along_x_impl(field, &mut derivative)?;
        self.validate_output(&derivative)?;
        Ok(derivative)
    }

    /// Compute y-derivative via spectral method (FFT along y pencils).
    /// # Errors
    /// - Propagates any [`KwaversError`] returned by called functions.
    ///
    pub fn derivative_y(&self, field: &ArrayView3<f64>) -> KwaversResult<Array3<f64>> {
        self.validate_field(field)?;
        let mut derivative = Array3::zeros([self.nx, self.ny, self.nz]);
        self.derivative_along_y_impl(field, &mut derivative)?;
        self.validate_output(&derivative)?;
        Ok(derivative)
    }

    /// Compute z-derivative via spectral method (FFT along z pencils).
    /// # Errors
    /// - Propagates any [`KwaversError`] returned by called functions.
    ///
    pub fn derivative_z(&self, field: &ArrayView3<f64>) -> KwaversResult<Array3<f64>> {
        self.validate_field(field)?;
        let mut derivative = Array3::zeros([self.nx, self.ny, self.nz]);
        self.derivative_along_z_impl(field, &mut derivative)?;
        self.validate_output(&derivative)?;
        Ok(derivative)
    }

    #[inline]
    fn validate_field(&self, field: &ArrayView3<f64>) -> KwaversResult<()> {
        if field.shape() != [self.nx, self.ny, self.nz] {
            return Err(KwaversError::InvalidInput(format!(
                "Field shape {:?} mismatch grid {:?}",
                field.shape(),
                &[self.nx, self.ny, self.nz]
            )));
        }
        if !field.iter().all(|&x| x.is_finite()) {
            return Err(KwaversError::InvalidInput(
                "Input field contains NaN or Inf values".into(),
            ));
        }
        Ok(())
    }

    #[inline]
    fn validate_output(&self, derivative: &Array3<f64>) -> KwaversResult<()> {
        if !derivative.iter().all(|&x| x.is_finite()) {
            return Err(KwaversError::InvalidInput(
                "Output field contains NaN or Inf values (numerical instability)".into(),
            ));
        }
        Ok(())
    }

    /// Parallelises over j (Axis 1): each rayon thread processes all nz pencils for one j.
    /// # Errors
    /// - Returns [`Err`] if an internal constraint is violated.
    ///
    fn derivative_along_x_impl(
        &self,
        field: &ArrayView3<f64>,
        derivative: &mut Array3<f64>,
    ) -> KwaversResult<()> {
        let nx = self.nx;
        let nz = self.nz;
        let fft = &*self.fft_x;
        let ikd = &self.ikd_x;

        derivative
            .axis_iter_mut(Axis(1))
            .into_par_iter()
            .enumerate()
            .for_each(|(j, mut slice)| {
                let mut line = Array1::<Complex64>::from_elem(nx, Complex64::default());
                for l in 0..nz {
                    for i in 0..nx {
                        line[i] = Complex64::new(field[[i, j, l]], 0.0);
                    }
                    fft.forward_complex_inplace(&mut line);
                    for (i, &ikd_val) in ikd.iter().enumerate() {
                        line[i] *= ikd_val;
                    }
                    fft.inverse_complex_inplace(&mut line);
                    for i in 0..nx {
                        slice[[i, l]] = line[i].re;
                    }
                }
            });

        Ok(())
    }

    /// Parallelises over i (Axis 0): each rayon thread processes all nz pencils for one i.
    /// # Errors
    /// - Returns [`Err`] if an internal constraint is violated.
    ///
    fn derivative_along_y_impl(
        &self,
        field: &ArrayView3<f64>,
        derivative: &mut Array3<f64>,
    ) -> KwaversResult<()> {
        let ny = self.ny;
        let nz = self.nz;
        let fft = &*self.fft_y;
        let ikd = &self.ikd_y;

        derivative
            .axis_iter_mut(Axis(0))
            .into_par_iter()
            .enumerate()
            .for_each(|(i, mut slice)| {
                let mut line = Array1::<Complex64>::from_elem(ny, Complex64::default());
                for l in 0..nz {
                    for j in 0..ny {
                        line[j] = Complex64::new(field[[i, j, l]], 0.0);
                    }
                    fft.forward_complex_inplace(&mut line);
                    for (j, &ikd_val) in ikd.iter().enumerate() {
                        line[j] *= ikd_val;
                    }
                    fft.inverse_complex_inplace(&mut line);
                    for j in 0..ny {
                        slice[[j, l]] = line[j].re;
                    }
                }
            });

        Ok(())
    }

    /// Parallelises over i (Axis 0): each rayon thread processes all ny pencils for one i.
    /// # Errors
    /// - Returns [`Err`] if an internal constraint is violated.
    ///
    fn derivative_along_z_impl(
        &self,
        field: &ArrayView3<f64>,
        derivative: &mut Array3<f64>,
    ) -> KwaversResult<()> {
        let ny = self.ny;
        let nz = self.nz;
        let fft = &*self.fft_z;
        let ikd = &self.ikd_z;

        derivative
            .axis_iter_mut(Axis(0))
            .into_par_iter()
            .enumerate()
            .for_each(|(i, mut slice)| {
                let mut line = Array1::<Complex64>::from_elem(nz, Complex64::default());
                for j in 0..ny {
                    for l in 0..nz {
                        line[l] = Complex64::new(field[[i, j, l]], 0.0);
                    }
                    fft.forward_complex_inplace(&mut line);
                    for (l, &ikd_val) in ikd.iter().enumerate() {
                        line[l] *= ikd_val;
                    }
                    fft.inverse_complex_inplace(&mut line);
                    for l in 0..nz {
                        slice[[j, l]] = line[l].re;
                    }
                }
            });

        Ok(())
    }
}
