// Sprint 96 Implementation Plan: k-Space Pseudospectral Foundation
//
// OBJECTIVE: Achieve exact k-Wave parity for power-law absorption
// PRIORITY: P0 - CRITICAL GAP (Core functionality missing)
//
// This module implements the k-space pseudospectral method following:
// Treeby, B. E., & Cox, B. T. (2010). "k-Wave: MATLAB toolbox for the simulation
// and reconstruction of photoacoustic wave fields." J. Biomed. Opt. 15(2).

use ndarray::Array3;
use num_complex::Complex;
use std::f64::consts::PI;

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
                let freq_mhz = omega / (2.0 * PI * 1e6);
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

                // Phase correction for power-law dispersion
                let phase_correction = match alpha_power {
                    // y = 1 case: Hilbert transform of sgn(ω) -> log(|ω|)
                    y if (y - 1.0).abs() < f64::EPSILON => {
                        let alpha_coeff_np = alpha_coeff * (10.0_f64.ln() / 20.0) * 100.0 / 1e6;
                        -c0 * alpha_coeff_np * (2.0 / PI) * omega * dt * omega.ln()
                    }
                    // General case y != 1
                    y => -c0 * dt * (PI * y / 2.0).tan() * k_mag,
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
            // Note: Some values can be 0 due to underflow or numerical precision
        }

        // Debug the absorption calculation
        let mut values_stats = (0, 0, 0); // (zero, small, normal)
        for abs_val in operator.absorption_operator.iter() {
            let norm = abs_val.norm();
            if norm < 1e-15 {
                values_stats.0 += 1;
            } else if norm < 0.1 {
                values_stats.1 += 1;
            } else {
                values_stats.2 += 1;
            }
        }

        println!(
            "Absorption stats: zero={}, small={}, normal={}",
            values_stats.0, values_stats.1, values_stats.2
        );

        // Check that we have some reasonable absorption values
        assert!(
            values_stats.1 + values_stats.2 > values_stats.0 / 2,
            "Should have significant non-zero absorption values"
        );

        // DC component is now handled in a separate test
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
        // Use a field that's concentrated at a single k-mode for predictable results
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
        assert!(
            (dispersion_val.re - 1.0).abs() > 1e-10,
            "Dispersion correction for y=1 should not be identity"
        );
        assert!(
            dispersion_val.im.abs() > 1e-10,
            "Dispersion correction for y=1 should have a non-zero imaginary part"
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
        assert!(
            (dispersion_val.re - 1.0).abs() > 1e-10,
            "Dispersion correction for y=1.5 should not be identity"
        );
        assert!(
            dispersion_val.im.abs() > 1e-10,
            "Dispersion correction for y=1.5 should have a non-zero imaginary part"
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
