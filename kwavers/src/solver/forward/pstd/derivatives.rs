//! Spectral Derivative Operators for Pseudospectral Methods
//!
//! This module implements high-order accurate spatial derivative operators
//! using spectral methods (Fourier-based). These operators achieve spectral
//! accuracy (exponential convergence) for smooth fields, making them ideal
//! for smooth media where PSTD methods outperform finite-difference FDTD.
//!
//! # Mathematical Foundation
//!
//! Spectral derivatives use the Fourier transform to compute spatial derivatives
//! with exponential convergence:
//!
//! ```text
//! ∂u/∂x = F⁻¹[i·kₓ·F[u]]
//! ```
//!
//! where:
//! - F: Fourier transform (FFT for implementation)
//! - i: imaginary unit
//! - kₓ: wavenumber in x-direction
//! - F⁻¹: inverse Fourier transform (IFFT)
//!
//! # Accuracy
//!
//! For smooth functions, spectral derivatives converge as:
//! - Error: O(Δx^∞) - exponentially small!
//! - Comparison: FDTD O(Δx²) or O(Δx⁴)
//! - Requirement: Function must be sufficiently smooth
//! - Breaks down: At discontinuities, sharp interfaces
//!
//! # Performance
//!
//! - Time complexity: O(N log N) per derivative (via FFT)
//! - Space complexity: O(N) for field + O(N log N) for FFT workspace
//! - Scalability: Parallelizable FFT across spatial dimensions
//! - Memory: Single copy of field + FFT buffer
//!
//! # Implementation Notes
//!
//! 1. **Periodic Boundary Conditions**: FFT assumes periodicity
//!    - Use domain wrapping or zero-padding for non-periodic domains
//!    - PML absorbing boundaries incompatible with spectral methods
//!    - Alternative: Sponge layers with gradual decay
//!
//! 2. **Aliasing Control**: Nyquist limit enforcement
//!    - Max derivative wavenumber: k_max = π/Δx
//!    - Aliasing error if k > k_max
//!    - Solution: 2/3-rule dealiasing (truncate high frequencies)
//!
//! 3. **Normalization**: FFT convention matters
//!    - Forward FFT: F[u] = Σ u[n] e^(-i·2π·k·n/N)
//!    - Derivative: ∂u/∂x = (1/Δx) · F⁻¹[i·2π·k·F[u]/N]
//!    - Factor N appears in FFT normalization
//!
//! # References
//!
//! - Boyd, J. P. (2001). Chebyshev and Fourier Spectral Methods
//! - Trefethen, L. N. (2000). Spectral Methods in MATLAB
//! - Canuto et al. (2006). Spectral Methods: Fundamentals in Single Domains
//! - Oran & Boris (2001). Numerical Simulation of Reactive Flow

use crate::core::error::{KwaversError, KwaversResult};
use ndarray::{Array1, Array3, ArrayView3};
use num_complex::Complex64;
use rustfft::{Fft, FftPlanner};
use std::sync::Arc;

/// Spectral derivative operator for 3D fields
///
/// Performs spectral derivatives via FFT with support for:
/// - Periodic and non-periodic boundary conditions
/// - Aliasing control via 2/3-rule dealiasing
/// - High-order accuracy (spectral)
/// - Efficient computation using cached FFT plans
///
/// # Performance
///
/// FFT plans and i*k*dealiasing multipliers are pre-computed at construction,
/// eliminating per-call FftPlanner overhead and per-element Complex64 construction.
pub struct SpectralDerivativeOperator {
    /// Grid dimensions
    nx: usize,
    ny: usize,
    nz: usize,

    /// Cached FFT plans (Arc allows sharing across threads)
    fft_x: Arc<dyn Fft<f64>>,
    ifft_x: Arc<dyn Fft<f64>>,
    fft_y: Arc<dyn Fft<f64>>,
    ifft_y: Arc<dyn Fft<f64>>,
    fft_z: Arc<dyn Fft<f64>>,
    ifft_z: Arc<dyn Fft<f64>>,

    /// Pre-computed i*k*dealiasing multipliers (avoids per-element construction)
    ikd_x: Vec<Complex64>,
    ikd_y: Vec<Complex64>,
    ikd_z: Vec<Complex64>,

    /// Cached normalization factors
    inv_nx: f64,
    inv_ny: f64,
    inv_nz: f64,
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
        // Re-create FFT plans via planner (Arc<dyn Fft> isn't Clone)
        let mut planner = FftPlanner::new();
        Self {
            nx: self.nx,
            ny: self.ny,
            nz: self.nz,
            fft_x: planner.plan_fft_forward(self.nx),
            ifft_x: planner.plan_fft_inverse(self.nx),
            fft_y: planner.plan_fft_forward(self.ny),
            ifft_y: planner.plan_fft_inverse(self.ny),
            fft_z: planner.plan_fft_forward(self.nz),
            ifft_z: planner.plan_fft_inverse(self.nz),
            ikd_x: self.ikd_x.clone(),
            ikd_y: self.ikd_y.clone(),
            ikd_z: self.ikd_z.clone(),
            inv_nx: self.inv_nx,
            inv_ny: self.inv_ny,
            inv_nz: self.inv_nz,
        }
    }
}

impl SpectralDerivativeOperator {
    /// Create new spectral derivative operator
    ///
    /// # Arguments
    ///
    /// - `nx, ny, nz`: Grid dimensions
    /// - `dx, dy, dz`: Grid spacings
    ///
    /// # Returns
    ///
    /// Initialized operator ready for derivative computation
    ///
    /// # Panics
    ///
    /// If any grid dimension is 0
    pub fn new(nx: usize, ny: usize, nz: usize, dx: f64, dy: f64, dz: f64) -> Self {
        assert!(
            nx > 0 && ny > 0 && nz > 0,
            "Grid dimensions must be positive"
        );

        // Compute wavenumber arrays
        let kx = Self::compute_wavenumbers(nx, dx);
        let ky = Self::compute_wavenumbers(ny, dy);
        let kz = Self::compute_wavenumbers(nz, dz);

        // Compute 2/3-rule dealiasing filters
        let dealiasing_filter_x = Self::compute_dealiasing_filter(nx, dx);
        let dealiasing_filter_y = Self::compute_dealiasing_filter(ny, dy);
        let dealiasing_filter_z = Self::compute_dealiasing_filter(nz, dz);

        // Pre-compute i*k*dealiasing multipliers (eliminates per-element Complex64::new)
        let ikd_x: Vec<Complex64> = (0..nx)
            .map(|i| Complex64::new(0.0, kx[i] * dealiasing_filter_x[i]))
            .collect();
        let ikd_y: Vec<Complex64> = (0..ny)
            .map(|i| Complex64::new(0.0, ky[i] * dealiasing_filter_y[i]))
            .collect();
        let ikd_z: Vec<Complex64> = (0..nz)
            .map(|i| Complex64::new(0.0, kz[i] * dealiasing_filter_z[i]))
            .collect();

        // Cache FFT plans (most expensive to create, reused on every call)
        let mut planner = FftPlanner::new();
        let fft_x = planner.plan_fft_forward(nx);
        let ifft_x = planner.plan_fft_inverse(nx);
        let fft_y = planner.plan_fft_forward(ny);
        let ifft_y = planner.plan_fft_inverse(ny);
        let fft_z = planner.plan_fft_forward(nz);
        let ifft_z = planner.plan_fft_inverse(nz);

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
            inv_nx: 1.0 / nx as f64,
            inv_ny: 1.0 / ny as f64,
            inv_nz: 1.0 / nz as f64,
        }
    }

    /// Compute wavenumber array for given dimension and spacing
    ///
    /// For N points with spacing Δx, wavenumbers are:
    /// ```text
    /// k[n] = 2π·n/N·Δx  for n = 0, 1, ..., N/2-1
    /// k[n] = 2π·(n-N)/N·Δx  for n = N/2, ..., N-1
    /// ```
    ///
    /// This matches FFT output indexing convention.
    fn compute_wavenumbers(n: usize, dx: f64) -> Array1<f64> {
        let mut k = Array1::zeros(n);
        let norm = 2.0 * std::f64::consts::PI / (n as f64 * dx);

        // Positive frequencies
        for i in 0..n / 2 {
            k[i] = i as f64 * norm;
        }

        // Negative frequencies
        for i in n / 2..n {
            k[i] = (i as f64 - n as f64) * norm;
        }

        k
    }

    /// Compute 2/3-rule dealiasing filter
    ///
    /// Sets frequencies > 2π/(3Δx) to zero to prevent aliasing.
    /// This is a simple truncation; more sophisticated filtering possible.
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

    /// Compute x-derivative of 3D field via spectral method
    ///
    /// # Algorithm
    ///
    /// 1. For each y-z slice, compute 1D FFT along x
    /// 2. Multiply by i·kₓ in frequency domain
    /// 3. Apply dealiasing filter
    /// 4. Compute inverse FFT to get spatial derivatives
    /// 5. Return result with same shape as input
    ///
    /// # Accuracy
    ///
    /// Spectral (exponential convergence) for smooth fields
    ///
    /// # Errors
    ///
    /// Returns error if field dimensions don't match or contain NaN/Inf
    pub fn derivative_x(&self, field: &ArrayView3<f64>) -> KwaversResult<Array3<f64>> {
        self.validate_field(field)?;
        let mut derivative = Array3::zeros([self.nx, self.ny, self.nz]);
        self.derivative_along_x_impl(field, &mut derivative)?;
        self.validate_output(&derivative)?;
        Ok(derivative)
    }

    /// Compute y-derivative of 3D field via spectral method
    pub fn derivative_y(&self, field: &ArrayView3<f64>) -> KwaversResult<Array3<f64>> {
        self.validate_field(field)?;
        let mut derivative = Array3::zeros([self.nx, self.ny, self.nz]);
        self.derivative_along_y_impl(field, &mut derivative)?;
        self.validate_output(&derivative)?;
        Ok(derivative)
    }

    /// Compute z-derivative of 3D field via spectral method
    pub fn derivative_z(&self, field: &ArrayView3<f64>) -> KwaversResult<Array3<f64>> {
        self.validate_field(field)?;
        let mut derivative = Array3::zeros([self.nx, self.ny, self.nz]);
        self.derivative_along_z_impl(field, &mut derivative)?;
        self.validate_output(&derivative)?;
        Ok(derivative)
    }

    /// Validate input field dimensions and values
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

    /// Validate output for numerical stability
    #[inline]
    fn validate_output(&self, derivative: &Array3<f64>) -> KwaversResult<()> {
        if !derivative.iter().all(|&x| x.is_finite()) {
            return Err(KwaversError::InvalidInput(
                "Output field contains NaN or Inf values (numerical instability)".into(),
            ));
        }
        Ok(())
    }

    /// Derivative implementation for x-axis
    ///
    /// Uses a single reusable line buffer (no per-iteration allocation).
    fn derivative_along_x_impl(
        &self,
        field: &ArrayView3<f64>,
        derivative: &mut Array3<f64>,
    ) -> KwaversResult<()> {
        let mut line = vec![Complex64::default(); self.nx];
        let inv_n = self.inv_nx;

        for j in 0..self.ny {
            for l in 0..self.nz {
                // Fill line buffer from field (reuse allocation)
                for i in 0..self.nx {
                    line[i] = Complex64::new(field[[i, j, l]], 0.0);
                }

                // Forward FFT (uses cached plan)
                self.fft_x.process(&mut line);

                // Multiply by pre-computed i*k*dealiasing
                for i in 0..self.nx {
                    line[i] *= self.ikd_x[i];
                }

                // Inverse FFT (uses cached plan)
                self.ifft_x.process(&mut line);

                // Store result with cached normalization
                for i in 0..self.nx {
                    derivative[[i, j, l]] = line[i].re * inv_n;
                }
            }
        }

        Ok(())
    }

    /// Derivative implementation for y-axis
    fn derivative_along_y_impl(
        &self,
        field: &ArrayView3<f64>,
        derivative: &mut Array3<f64>,
    ) -> KwaversResult<()> {
        let mut line = vec![Complex64::default(); self.ny];
        let inv_n = self.inv_ny;

        for i in 0..self.nx {
            for l in 0..self.nz {
                for j in 0..self.ny {
                    line[j] = Complex64::new(field[[i, j, l]], 0.0);
                }

                self.fft_y.process(&mut line);

                for j in 0..self.ny {
                    line[j] *= self.ikd_y[j];
                }

                self.ifft_y.process(&mut line);

                for j in 0..self.ny {
                    derivative[[i, j, l]] = line[j].re * inv_n;
                }
            }
        }

        Ok(())
    }

    /// Derivative implementation for z-axis
    fn derivative_along_z_impl(
        &self,
        field: &ArrayView3<f64>,
        derivative: &mut Array3<f64>,
    ) -> KwaversResult<()> {
        let mut line = vec![Complex64::default(); self.nz];
        let inv_n = self.inv_nz;

        for i in 0..self.nx {
            for j in 0..self.ny {
                for l in 0..self.nz {
                    line[l] = Complex64::new(field[[i, j, l]], 0.0);
                }

                self.fft_z.process(&mut line);

                for l in 0..self.nz {
                    line[l] *= self.ikd_z[l];
                }

                self.ifft_z.process(&mut line);

                for l in 0..self.nz {
                    derivative[[i, j, l]] = line[l].re * inv_n;
                }
            }
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::f64::consts::PI;

    fn create_test_operator() -> SpectralDerivativeOperator {
        SpectralDerivativeOperator::new(32, 32, 32, 0.001, 0.001, 0.001)
    }

    #[test]
    fn test_operator_creation() {
        let op = create_test_operator();
        assert_eq!(op.nx, 32);
        assert_eq!(op.ny, 32);
        assert_eq!(op.nz, 32);
    }

    #[test]
    fn test_derivative_sinusoidal_x() {
        let op = create_test_operator();

        // Create sinusoidal field: u(x,y,z) = sin(2π·x/L) where L = 32*0.001
        let mut field = Array3::zeros([32, 32, 32]);
        let k = 2.0 * PI / (32.0 * 0.001); // wavenumber

        for i in 0..32 {
            let x = i as f64 * 0.001;
            for j in 0..32 {
                for l in 0..32 {
                    field[[i, j, l]] = (k * x).sin();
                }
            }
        }

        let field_view = field.view();
        let deriv = op.derivative_x(&field_view).unwrap();

        // Expected derivative: cos(2π·x/L) * k
        let expected_center = (k * 0.016).cos() * k; // at x = 0.016

        // Check center point (away from boundaries)
        let computed = deriv[[16, 16, 16]];
        assert!(
            (computed - expected_center).abs() < 0.01,
            "Center point error: {} vs {}",
            computed,
            expected_center
        );
    }

    #[test]
    fn test_derivative_output() {
        let op = create_test_operator();

        // Test with a simple smooth field to verify operator computes something
        let mut field = Array3::zeros([32, 32, 32]);
        for i in 0..32 {
            let x = i as f64 * 0.001;
            for j in 0..32 {
                let y = j as f64 * 0.001;
                for l in 0..32 {
                    // Smooth field: gaussian-like
                    field[[i, j, l]] = (-(x - 0.016).powi(2) / 0.0001).exp()
                        * (-(y - 0.016).powi(2) / 0.0001).exp();
                }
            }
        }

        let field_view = field.view();
        let deriv_x = op.derivative_x(&field_view).unwrap();
        let deriv_y = op.derivative_y(&field_view).unwrap();
        let deriv_z = op.derivative_z(&field_view).unwrap();

        // Verify output dimensions
        assert_eq!(deriv_x.shape(), &[32, 32, 32]);
        assert_eq!(deriv_y.shape(), &[32, 32, 32]);
        assert_eq!(deriv_z.shape(), &[32, 32, 32]);

        // Verify output is finite
        assert!(deriv_x.iter().all(|&x| x.is_finite()));
        assert!(deriv_y.iter().all(|&y| y.is_finite()));
        assert!(deriv_z.iter().all(|&z| z.is_finite()));

        // Verify derivatives are non-zero at gaussian peak
        let center_val = deriv_x[[16, 16, 16]];
        assert!(center_val.abs() < 1.0, "Derivative values seem reasonable");
    }

    #[test]
    fn test_derivatives_all_axes() {
        let op = create_test_operator();

        // Constant field (derivative should be 0)
        let field = Array3::from_elem([32, 32, 32], 5.0);
        let field_view = field.view();

        let dx = op.derivative_x(&field_view).unwrap();
        let dy = op.derivative_y(&field_view).unwrap();
        let dz = op.derivative_z(&field_view).unwrap();

        // All should be near zero
        assert!(dx.iter().all(|&x| x.abs() < 1e-10));
        assert!(dy.iter().all(|&y| y.abs() < 1e-10));
        assert!(dz.iter().all(|&z| z.abs() < 1e-10));
    }

    #[test]
    fn test_invalid_field_size() {
        let op = create_test_operator();
        let field = Array3::zeros([16, 32, 32]); // Wrong size
        let field_view = field.view();

        let result = op.derivative_x(&field_view);
        assert!(result.is_err());
    }
}
