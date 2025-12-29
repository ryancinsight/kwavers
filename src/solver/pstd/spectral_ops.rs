//! Spectral operations for PSTD solver

use crate::error::KwaversResult;
use crate::grid::Grid;
use crate::utils::{fft_3d_array, ifft_3d_array};
use ndarray::{Array3, Zip};
use num_complex::Complex;
use std::f64::consts::PI;

/// Spectral operations handler
#[derive(Debug)]
pub struct SpectralOperations {
    pub kx: Array3<f64>,
    pub ky: Array3<f64>,
    pub kz: Array3<f64>,
    pub k_squared: Array3<f64>,
    pub kappa: Option<Array3<f64>>,
}

impl SpectralOperations {
    /// Create new spectral operations
    pub fn new(grid: &Grid) -> Self {
        let (kx, ky, kz) = Self::compute_wavenumbers(grid);
        let k_squared = Self::compute_k_squared(&kx, &ky, &kz);

        Self {
            kx,
            ky,
            kz,
            k_squared,
            kappa: None,
        }
    }

    /// Set k-space correction factors for heterogeneous media
    pub fn set_kspace_correction(&mut self, kappa: Array3<f64>) {
        self.kappa = Some(kappa);
    }

    /// Compute wavenumber arrays
    fn compute_wavenumbers(grid: &Grid) -> (Array3<f64>, Array3<f64>, Array3<f64>) {
        let mut kx = Array3::zeros((grid.nx, grid.ny, grid.nz));
        let mut ky = Array3::zeros((grid.nx, grid.ny, grid.nz));
        let mut kz = Array3::zeros((grid.nx, grid.ny, grid.nz));

        let kx_1d = grid.compute_kx();
        let ky_1d = grid.compute_ky();
        let kz_1d = grid.compute_kz();

        for i in 0..grid.nx {
            for j in 0..grid.ny {
                for k in 0..grid.nz {
                    kx[[i, j, k]] = kx_1d[i];
                    ky[[i, j, k]] = ky_1d[j];
                    kz[[i, j, k]] = kz_1d[k];
                }
            }
        }

        (kx, ky, kz)
    }

    /// Compute k² for Laplacian
    fn compute_k_squared(kx: &Array3<f64>, ky: &Array3<f64>, kz: &Array3<f64>) -> Array3<f64> {
        let mut k_squared = Array3::zeros(kx.dim());
        Zip::from(&mut k_squared)
            .and(kx)
            .and(ky)
            .and(kz)
            .for_each(|k2, &kx, &ky, &kz| {
                *k2 = kx * kx + ky * ky + kz * kz;
            });
        k_squared
    }

    /// Compute gradient in spectral space
    pub fn compute_gradient(
        &self,
        field: &Array3<f64>,
        _grid: &Grid,
    ) -> KwaversResult<(Array3<f64>, Array3<f64>, Array3<f64>)> {
        let field_hat = fft_3d_array(field);

        let grad_x_hat = &field_hat * &self.kx.mapv(|k| Complex::new(0.0, k));
        let grad_y_hat = &field_hat * &self.ky.mapv(|k| Complex::new(0.0, k));
        let grad_z_hat = &field_hat * &self.kz.mapv(|k| Complex::new(0.0, k));

        // Apply k-space correction if available
        let (grad_x_hat, grad_y_hat, grad_z_hat) = if let Some(ref kappa) = self.kappa {
            let kappa_complex = kappa.mapv(|k| Complex::new(k, 0.0));
            (
                grad_x_hat * &kappa_complex,
                grad_y_hat * &kappa_complex,
                grad_z_hat * &kappa_complex,
            )
        } else {
            (grad_x_hat, grad_y_hat, grad_z_hat)
        };

        Ok((
            ifft_3d_array(&grad_x_hat),
            ifft_3d_array(&grad_y_hat),
            ifft_3d_array(&grad_z_hat),
        ))
    }

    /// Compute Laplacian in spectral space
    pub fn compute_laplacian(
        &self,
        field: &Array3<f64>,
        _grid: &Grid,
    ) -> KwaversResult<Array3<f64>> {
        let field_hat = fft_3d_array(field);

        let mut laplacian_hat = field_hat.clone();
        Zip::from(&mut laplacian_hat)
            .and(&self.k_squared)
            .for_each(|l, &k2| *l *= -k2);

        if let Some(ref kappa) = self.kappa {
            Zip::from(&mut laplacian_hat)
                .and(kappa)
                .for_each(|l, &k| *l *= k);
        }

        Ok(ifft_3d_array(&laplacian_hat))
    }

    /// Apply anti-aliasing (2/3 rule)
    /// Compute divergence of a vector field
    #[must_use]
    pub fn compute_divergence(
        &self,
        vx: &Array3<f64>,
        vy: &Array3<f64>,
        vz: &Array3<f64>,
    ) -> Array3<f64> {
        use crate::utils::{fft_3d_array, ifft_3d_array};
        use num_complex::Complex;

        // Transform to k-space
        let vx_hat = fft_3d_array(vx);
        let vy_hat = fft_3d_array(vy);
        let vz_hat = fft_3d_array(vz);

        // Compute divergence in k-space: div(v) = ikx*vx + iky*vy + ikz*vz
        let mut div_hat = Array3::zeros(vx_hat.raw_dim());
        let i = Complex::new(0.0, 1.0);
        for ((idx, d), &vx) in div_hat.indexed_iter_mut().zip(vx_hat.iter()) {
            let kx = self.kx[idx];
            let ky = self.ky[idx];
            let kz = self.kz[idx];
            let vy = vy_hat[idx];
            let vz = vz_hat[idx];
            *d = i * (kx * vx + ky * vy + kz * vz);
        }

        // Transform back to real space
        ifft_3d_array(&div_hat)
    }

    pub fn apply_antialiasing(&self, field_hat: &mut Array3<Complex<f64>>, grid: &Grid) {
        let kx_max = 2.0 * std::f64::consts::PI / grid.dx / 3.0;
        let ky_max = 2.0 * std::f64::consts::PI / grid.dy / 3.0;
        let kz_max = 2.0 * std::f64::consts::PI / grid.dz / 3.0;

        Zip::from(field_hat)
            .and(&self.kx)
            .and(&self.ky)
            .and(&self.kz)
            .for_each(|f, &kx, &ky, &kz| {
                if kx.abs() > kx_max || ky.abs() > ky_max || kz.abs() > kz_max {
                    *f = Complex::new(0.0, 0.0);
                }
            });
    }
}

/// k-Space pseudospectral operator for exact k-Wave compatibility
///
/// Implements power-law absorption with dispersion correction:
/// α(ω) = α₀ · |ω|^y where y ∈ [0, 3]
///
/// # Mathematical Foundation
///
/// The k-space operator applies absorption and dispersion in frequency domain:
/// ```text
/// ∂p/∂t = -c₀·∇·u - α(ω)*p + S
/// ∂u/∂t = -∇p/ρ₀
/// ```
///
/// With k-space correction for arbitrary absorption laws.
#[derive(Debug, Clone)]
pub struct KSpaceOperator {
    /// Wavenumber arrays for each spatial dimension
    kx: Array3<f64>,
    ky: Array3<f64>,
    kz: Array3<f64>,

    /// Power-law absorption operator: exp(-α(ω)·Δt)
    absorption_operator: Array3<Complex<f64>>,

    /// Dispersion correction operator for causal absorption
    dispersion_correction: Array3<Complex<f64>>,

    /// Grid spacing for finite difference corrections
    dx: f64,
    dy: f64,
    dz: f64,

    /// Sound speed (assumed homogeneous for k-space method)
    c0: f64,
}

impl KSpaceOperator {
    /// Create new k-space operator with power-law absorption
    ///
    /// # Arguments
    /// * `grid_size` - (nx, ny, nz) grid dimensions
    /// * `grid_spacing` - (dx, dy, dz) spatial steps
    /// * `c0` - Reference sound speed
    /// * `alpha_coeff` - Absorption coefficient α₀
    /// * `alpha_power` - Power law exponent y ∈ [0, 3]
    /// * `dt` - Time step for operator precomputation
    pub fn new(
        grid_size: (usize, usize, usize),
        grid_spacing: (f64, f64, f64),
        c0: f64,
        alpha_coeff: f64,
        alpha_power: f64,
        dt: f64,
    ) -> Self {
        let (nx, ny, nz) = grid_size;
        let (dx, dy, dz) = grid_spacing;

        // Initialize k-space wavenumber grids
        let mut kx = Array3::zeros((nx, ny, nz));
        let mut ky = Array3::zeros((nx, ny, nz));
        let mut kz = Array3::zeros((nx, ny, nz));

        // Compute wavenumbers with proper FFT ordering
        Self::compute_wavenumbers(&mut kx, &mut ky, &mut kz, grid_size, grid_spacing);

        // Precompute absorption operator
        let absorption_operator =
            Self::compute_absorption_operator(&kx, &ky, &kz, c0, alpha_coeff, alpha_power, dt);

        // Precompute dispersion correction
        let dispersion_correction =
            Self::compute_dispersion_correction(&kx, &ky, &kz, c0, alpha_coeff, alpha_power, dt);

        Self {
            kx,
            ky,
            kz,
            absorption_operator,
            dispersion_correction,
            dx,
            dy,
            dz,
            c0,
        }
    }

    /// Apply k-space absorption to pressure field (in-place)
    ///
    /// Multiplies by exp(-α(ω)·Δt) in frequency domain
    pub fn apply_absorption(&self, pressure_fft: &mut Array3<Complex<f64>>) {
        *pressure_fft *= &self.absorption_operator;
    }

    /// Apply dispersion correction for causal absorption
    ///
    /// Corrects phase velocity to maintain causality with power-law absorption
    pub fn apply_dispersion(&self, pressure_fft: &mut Array3<Complex<f64>>) {
        *pressure_fft *= &self.dispersion_correction;
    }

    /// Get grid spacing for finite difference corrections
    pub fn grid_spacing(&self) -> (f64, f64, f64) {
        (self.dx, self.dy, self.dz)
    }

    /// Get reference sound speed
    pub fn sound_speed(&self) -> f64 {
        self.c0
    }

    /// Get k-space operators (read-only access)
    pub fn k_operators(&self) -> (&Array3<f64>, &Array3<f64>, &Array3<f64>) {
        (&self.kx, &self.ky, &self.kz)
    }

    /// Compute k-space gradient (returns ∇p in frequency domain)
    ///
    /// Uses exact spectral derivatives: ∇_FFT = i·k·FFT(field)
    pub fn k_space_gradient(
        &self,
        pressure_fft: &Array3<Complex<f64>>,
    ) -> (
        Array3<Complex<f64>>,
        Array3<Complex<f64>>,
        Array3<Complex<f64>>,
    ) {
        let i = Complex::new(0.0, 1.0);

        let grad_x = pressure_fft * &self.kx.mapv(|k| i * k);
        let grad_y = pressure_fft * &self.ky.mapv(|k| i * k);
        let grad_z = pressure_fft * &self.kz.mapv(|k| i * k);

        (grad_x, grad_y, grad_z)
    }

    /// Compute k-space Laplacian (returns ∇²p in frequency domain)
    pub fn k_space_laplacian(&self, pressure_fft: &Array3<Complex<f64>>) -> Array3<Complex<f64>> {
        let k_squared =
            &self.kx.mapv(|k| k * k) + &self.ky.mapv(|k| k * k) + &self.kz.mapv(|k| k * k);
        pressure_fft * &k_squared.mapv(|k2| Complex::new(-k2, 0.0))
    }

    // Private implementation methods

    fn compute_wavenumbers(
        kx: &mut Array3<f64>,
        ky: &mut Array3<f64>,
        kz: &mut Array3<f64>,
        grid_size: (usize, usize, usize),
        grid_spacing: (f64, f64, f64),
    ) {
        let (nx, ny, nz) = grid_size;
        let (dx, dy, dz) = grid_spacing;

        // k-space sampling following FFT conventions
        for i in 0..nx {
            for j in 0..ny {
                for k in 0..nz {
                    // FFT frequency indexing with proper Nyquist handling
                    let kx_val = if i <= nx / 2 {
                        2.0 * PI * i as f64 / (nx as f64 * dx)
                    } else {
                        2.0 * PI * (i as f64 - nx as f64) / (nx as f64 * dx)
                    };

                    let ky_val = if j <= ny / 2 {
                        2.0 * PI * j as f64 / (ny as f64 * dy)
                    } else {
                        2.0 * PI * (j as f64 - ny as f64) / (ny as f64 * dy)
                    };

                    let kz_val = if k <= nz / 2 {
                        2.0 * PI * k as f64 / (nz as f64 * dz)
                    } else {
                        2.0 * PI * (k as f64 - nz as f64) / (nz as f64 * dz)
                    };

                    kx[[i, j, k]] = kx_val;
                    ky[[i, j, k]] = ky_val;
                    kz[[i, j, k]] = kz_val;
                }
            }
        }
    }

    fn compute_absorption_operator(
        kx: &Array3<f64>,
        ky: &Array3<f64>,
        kz: &Array3<f64>,
        c0: f64,
        alpha_coeff: f64,
        alpha_power: f64,
        dt: f64,
    ) -> Array3<Complex<f64>> {
        let mut absorption = Array3::zeros(kx.raw_dim());

        for ((i, j, k), abs_val) in absorption.indexed_iter_mut() {
            let k_mag =
                (kx[[i, j, k]].powi(2) + ky[[i, j, k]].powi(2) + kz[[i, j, k]].powi(2)).sqrt();
            let omega = c0 * k_mag;

            if omega > 1e-16 {
                // Power-law absorption: α(ω) = α₀ · |ω|^y
                // Convert from dB/(MHz^y cm) to Nepers/m for proper units
                let freq_hz = omega / (2.0 * PI);
                let freq_mhz = freq_hz / 1e6;
                let alpha_db_per_cm = alpha_coeff * freq_mhz.powf(alpha_power);

                // Convert dB/cm to Nepers/m
                let alpha_np_per_m = alpha_db_per_cm * (10.0_f64.ln() / 20.0) * 100.0;

                *abs_val = Complex::new((-alpha_np_per_m * dt).exp(), 0.0);
            } else {
                // DC component (ω=0) has zero absorption
                *abs_val = Complex::new(1.0, 0.0);
            }
        }

        absorption
    }

    fn compute_dispersion_correction(
        kx: &Array3<f64>,
        ky: &Array3<f64>,
        kz: &Array3<f64>,
        c0: f64,
        alpha_coeff: f64,
        alpha_power: f64,
        dt: f64,
    ) -> Array3<Complex<f64>> {
        let mut dispersion = Array3::zeros(kx.raw_dim());

        // Dispersion correction for causal absorption (Treeby & Cox 2010)
        for ((i, j, k), disp_val) in dispersion.indexed_iter_mut() {
            let k_mag =
                (kx[[i, j, k]].powi(2) + ky[[i, j, k]].powi(2) + kz[[i, j, k]].powi(2)).sqrt();

            if k_mag > 1e-16 {
                let omega = c0 * k_mag;

                // Phase correction for power-law dispersion using Kramers-Kronig relation
                // For power-law absorption α(ω) ∝ |ω|^y, the dispersion correction is:
                // Δφ(ω) = tan(πy/2) * α(ω) / (2*c₀) for y ≠ 1
                // For y = 1: Δφ(ω) = (α₀/(2*c₀)) * sgn(ω) * log|ω|
                // Note: α(ω) includes the frequency dependence |ω|^y

                // Convert absorption coefficient with frequency dependence
                let freq_hz = omega / (2.0 * PI);
                let freq_mhz = freq_hz / 1e6;
                let alpha_db_per_cm = alpha_coeff * freq_mhz.powf(alpha_power);
                let alpha_np_per_m = alpha_db_per_cm * (10.0_f64.ln() / 20.0) * 100.0;

                let phase_correction = if (alpha_power - 1.0).abs() < f64::EPSILON {
                    // y = 1 case: Hilbert transform gives logarithmic dispersion
                    // Δφ(ω) = (α₀/(2*c₀)) * sgn(ω) * log|ω|
                    // Use the same coefficient scaling as absorption
                    let sign_omega = if omega > 0.0 { 1.0 } else { -1.0 };
                    -dt * alpha_np_per_m * sign_omega * omega.abs().ln() / (2.0 * c0)
                } else {
                    // General case y ≠ 1: Δφ(ω) = tan(πy/2) * α(ω) / (2*c₀)
                    -dt * (PI * alpha_power / 2.0).tan() * alpha_np_per_m / (2.0 * c0)
                };

                *disp_val = Complex::new(0.0, phase_correction).exp();
            } else {
                // No dispersion at DC
                *disp_val = Complex::new(1.0, 0.0);
            }
        }

        dispersion
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_k_space_operator_creation() {
        let operator = KSpaceOperator::new(
            (64, 64, 64),
            (1e-4, 1e-4, 1e-4),
            1500.0, // c0
            0.75,   // alpha_coeff
            1.5,    // alpha_power
            1e-8,   // dt
        );

        // Verify wavenumber symmetry properties
        let (nx, _ny, _nz) = (64, 64, 64);

        // DC component should be zero
        assert_eq!(operator.kx[[0, 0, 0]], 0.0);
        assert_eq!(operator.ky[[0, 0, 0]], 0.0);
        assert_eq!(operator.kz[[0, 0, 0]], 0.0);

        // Nyquist frequency handling
        if nx % 2 == 0 {
            let nyq_x = operator.kx[[nx / 2, 0, 0]];
            assert!(nyq_x > 0.0, "Nyquist frequency should be positive");
        }
    }

    #[test]
    fn test_absorption_operator_properties() {
        let operator =
            KSpaceOperator::new((32, 32, 32), (1e-4, 1e-4, 1e-4), 1500.0, 0.75, 1.5, 1e-8);

        // All absorption values should have magnitude ≤ 1 (physical requirement)
        for abs_val in operator.absorption_operator.iter() {
            assert!(
                abs_val.norm() <= 1.0,
                "Absorption must be ≤ 1 for stability"
            );
        }

        // Check that we have some reasonable absorption values
        let mut has_non_zero = false;
        for abs_val in operator.absorption_operator.iter() {
            if abs_val.norm() > 1e-10 {
                has_non_zero = true;
                break;
            }
        }
        assert!(
            has_non_zero,
            "Should have significant non-zero absorption values"
        );
    }

    #[test]
    fn test_k_space_gradient_accuracy() {
        let operator = KSpaceOperator::new(
            (8, 8, 8), // Smaller size for simpler test
            (1e-4, 1e-4, 1e-4),
            1500.0,
            0.75,
            1.5,
            1e-8,
        );

        // Test gradient properties on a simple k-space field
        let mut test_field_fft = Array3::zeros((8, 8, 8));

        // Set a single mode at (1,0,0) - should give a pure x-derivative
        test_field_fft[[1, 0, 0]] = Complex::new(1.0, 0.0);

        let (grad_x, grad_y, grad_z) = operator.k_space_gradient(&test_field_fft);

        // For k-mode at (1,0,0), gradient should be i*kx at that mode
        let k_operators = operator.k_operators();
        let expected_kx = k_operators.0[[1, 0, 0]];
        let expected_grad = Complex::new(0.0, expected_kx); // i*kx

        assert_relative_eq!(grad_x[[1, 0, 0]].re, expected_grad.re, epsilon = 1e-10);
        assert_relative_eq!(grad_x[[1, 0, 0]].im, expected_grad.im, epsilon = 1e-10);

        // y and z gradients should be zero for this k-mode
        assert_relative_eq!(grad_y[[1, 0, 0]].norm(), 0.0, epsilon = 1e-10);
        assert_relative_eq!(grad_z[[1, 0, 0]].norm(), 0.0, epsilon = 1e-10);
    }

    #[test]
    fn test_dispersion_correction_y_equals_1() {
        let operator =
            KSpaceOperator::new((32, 32, 32), (1e-4, 1e-4, 1e-4), 1500.0, 0.75, 1.0, 1e-8);

        // Check a non-DC component
        let dispersion_val = operator.dispersion_correction[[1, 1, 1]];

        // For small phase corrections, exp(iθ) ≈ 1 + iθ
        // The test expects the correction to be detectable
        // Since θ is small, check that it's not exactly identity
        assert!(
            dispersion_val != Complex::new(1.0, 0.0),
            "Dispersion correction should not be exactly identity"
        );

        // Check that imaginary part is non-zero (phase correction applied)
        assert!(
            dispersion_val.im.abs() > 1e-12,
            "Dispersion correction should have non-zero imaginary part, got {}",
            dispersion_val.im.abs()
        );

        // DC component should have no dispersion
        let dc_disp = operator.dispersion_correction[[0, 0, 0]];
        assert_relative_eq!(dc_disp.re, 1.0, epsilon = 1e-10);
        assert_relative_eq!(dc_disp.im, 0.0, epsilon = 1e-10);
    }

    #[test]
    fn test_dispersion_correction_general_case() {
        let operator =
            KSpaceOperator::new((32, 32, 32), (1e-4, 1e-4, 1e-4), 1500.0, 0.75, 1.5, 1e-8);

        // Check a non-DC component
        let dispersion_val = operator.dispersion_correction[[1, 1, 1]];

        // For small phase corrections, exp(iθ) ≈ 1 + iθ
        // The test expects the correction to be detectable
        // Since θ is small, check that it's not exactly identity
        assert!(
            dispersion_val != Complex::new(1.0, 0.0),
            "Dispersion correction should not be exactly identity"
        );

        // Check that imaginary part is non-zero (phase correction applied)
        assert!(
            dispersion_val.im.abs() > 1e-12,
            "Dispersion correction should have non-zero imaginary part, got {}",
            dispersion_val.im.abs()
        );

        // DC component should have no dispersion
        let dc_disp = operator.dispersion_correction[[0, 0, 0]];
        assert_relative_eq!(dc_disp.re, 1.0, epsilon = 1e-10);
        assert_relative_eq!(dc_disp.im, 0.0, epsilon = 1e-10);
    }

    #[test]
    fn test_absorption_dc_component() {
        let operator =
            KSpaceOperator::new((32, 32, 32), (1e-4, 1e-4, 1e-4), 1500.0, 0.75, 1.5, 1e-8);

        // DC component should have no absorption
        let dc_abs = operator.absorption_operator[[0, 0, 0]];
        assert_relative_eq!(dc_abs.re, 1.0, epsilon = 1e-10);
        assert_relative_eq!(dc_abs.im, 0.0, epsilon = 1e-10);
    }
}
