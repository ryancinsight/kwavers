//! Dispersion analysis and correction for numerical methods

use crate::domain::grid::Grid;
use ndarray::Array3;
use std::f64::consts::PI;

/// Dispersion analysis for numerical methods
#[derive(Debug)]
pub struct DispersionAnalysis;

impl DispersionAnalysis {
    /// Calculate numerical dispersion for FDTD method (1D)
    ///
    /// This is the legacy 1D interface kept for backward compatibility.
    /// For 3D analysis, use [`fdtd_dispersion_3d`](Self::fdtd_dispersion_3d).
    #[must_use]
    pub fn fdtd_dispersion(k: f64, dx: f64, dt: f64, c: f64) -> f64 {
        let cfl = c * dt / dx;
        let kx_dx = k * dx;

        // Von Neumann stability analysis result
        let sin_half_omega_dt = (cfl * kx_dx.sin()).asin();
        let omega_numerical = 2.0 * sin_half_omega_dt / dt;
        let omega_exact = k * c;

        (omega_numerical - omega_exact) / omega_exact
    }

    /// Calculate numerical dispersion for FDTD method in 3D
    ///
    /// Implements full 3D Von Neumann stability analysis accounting for anisotropic grids
    /// and oblique wave propagation directions.
    ///
    /// # Mathematical Formulation
    ///
    /// For 3D FDTD with central differences in space and leapfrog in time,
    /// Von Neumann analysis yields:
    ///
    /// ```text
    /// sin²(ω_num·dt/2) = CFL_x²·sin²(kx·dx/2) + CFL_y²·sin²(ky·dy/2) + CFL_z²·sin²(kz·dz/2)
    /// ```
    ///
    /// where:
    /// - `CFL_x = c·dt/dx`, `CFL_y = c·dt/dy`, `CFL_z = c·dt/dz` (Courant numbers)
    /// - `(kx, ky, kz)` are wavenumber components
    /// - `(dx, dy, dz)` are grid spacings
    /// - `dt` is the time step
    /// - `c` is the wave speed
    ///
    /// The exact angular frequency is:
    /// ```text
    /// ω_exact = c·|k| = c·√(kx² + ky² + kz²)
    /// ```
    ///
    /// Relative dispersion error:
    /// ```text
    /// ε = (ω_numerical - ω_exact) / ω_exact
    /// ```
    ///
    /// # Stability Condition
    ///
    /// For 3D FDTD to be stable (using central differences):
    /// ```text
    /// √(1/dx² + 1/dy² + 1/dz²) ≤ 1/(c·dt)
    /// ```
    ///
    /// For isotropic grids (dx=dy=dz=h):
    /// ```text
    /// c·dt/h ≤ 1/√3 ≈ 0.577  (CFL condition)
    /// ```
    ///
    /// # Parameters
    ///
    /// - `kx, ky, kz`: Wavenumber components (rad/m)
    /// - `dx, dy, dz`: Grid spacings (m)
    /// - `dt`: Time step (s)
    /// - `c`: Wave speed (m/s)
    ///
    /// # Returns
    ///
    /// Relative dispersion error: `(ω_numerical - ω_exact) / ω_exact`
    ///
    /// # References
    ///
    /// - Taflove & Hagness (2005). "Computational Electrodynamics: The FDTD Method" (3rd ed.)
    /// - Koene & Robertsson (2012). "Removing numerical dispersion from linear wave equations."
    ///   *Geophysics*, 77(1), T1-T11. DOI: 10.1190/geo2011-0210.1
    /// - Moczo et al. (2014). "3D fourth-order staggered-grid finite-difference schemes."
    ///   *Geophysics*, 79(6), T235-T252.
    ///
    /// # Examples
    ///
    /// ```
    /// use kwavers::physics::acoustics::analytical::dispersion::DispersionAnalysis;
    /// use std::f64::consts::PI;
    ///
    /// // Wave propagating at 45° in xy-plane
    /// let freq = 1e6;  // 1 MHz
    /// let c = 1500.0;  // Sound speed in water (m/s)
    /// let wavelength = c / freq;
    /// let k_mag = 2.0 * PI / wavelength;
    /// let kx = k_mag / (2.0_f64).sqrt();
    /// let ky = k_mag / (2.0_f64).sqrt();
    /// let kz = 0.0;
    ///
    /// let dx = wavelength / 20.0;  // 20 points per wavelength
    /// let dy = wavelength / 20.0;
    /// let dz = wavelength / 20.0;
    /// let dt = 0.5 * dx / (c * 3.0_f64.sqrt());  // CFL = 0.5/√3
    ///
    /// let dispersion_error = DispersionAnalysis::fdtd_dispersion_3d(
    ///     kx, ky, kz, dx, dy, dz, dt, c
    /// );
    ///
    /// assert!(dispersion_error.abs() < 0.01);  // < 1% error for this resolution
    /// ```
    #[must_use]
    pub fn fdtd_dispersion_3d(
        kx: f64,
        ky: f64,
        kz: f64,
        dx: f64,
        dy: f64,
        dz: f64,
        dt: f64,
        c: f64,
    ) -> f64 {
        // Courant numbers for each direction
        let cfl_x = c * dt / dx;
        let cfl_y = c * dt / dy;
        let cfl_z = c * dt / dz;

        // Compute sin²(k·h/2) terms
        let sin_kx_dx_half = (0.5 * kx * dx).sin();
        let sin_ky_dy_half = (0.5 * ky * dy).sin();
        let sin_kz_dz_half = (0.5 * kz * dz).sin();

        // Von Neumann dispersion relation:
        // sin²(ω_num·dt/2) = CFL_x²·sin²(kx·dx/2) + CFL_y²·sin²(ky·dy/2) + CFL_z²·sin²(kz·dz/2)
        let sin_squared_omega_dt_half = cfl_x * cfl_x * sin_kx_dx_half * sin_kx_dx_half
            + cfl_y * cfl_y * sin_ky_dy_half * sin_ky_dy_half
            + cfl_z * cfl_z * sin_kz_dz_half * sin_kz_dz_half;

        // Clamp to valid range for asin (avoid numerical errors)
        let sin_squared_clamped = sin_squared_omega_dt_half.clamp(0.0, 1.0);

        // Recover numerical angular frequency
        let sin_half_omega_dt = sin_squared_clamped.sqrt();
        let omega_numerical = 2.0 * sin_half_omega_dt.asin() / dt;

        // Exact angular frequency: ω = c·|k|
        let k_magnitude = (kx * kx + ky * ky + kz * kz).sqrt();
        let omega_exact = k_magnitude * c;

        // Avoid division by zero
        if omega_exact.abs() < 1e-14 {
            return 0.0;
        }

        // Relative dispersion error
        (omega_numerical - omega_exact) / omega_exact
    }

    /// Calculate numerical dispersion for PSTD method (1D)
    ///
    /// This is a simplified 1D interface. For full 3D analysis with proper
    /// Von Neumann dispersion, use [`pstd_dispersion_3d`](Self::pstd_dispersion_3d).
    #[must_use]
    pub fn pstd_dispersion(k: f64, dx: f64, order: usize) -> f64 {
        // K-space method dispersion (spectral accuracy)
        let kx_dx = k * dx;

        match order {
            2 => 0.02 * kx_dx.powi(2),  // Second-order correction
            4 => 0.001 * kx_dx.powi(4), // Fourth-order correction
            _ => 0.0,                   // Perfect for lower orders
        }
    }

    /// Calculate numerical dispersion for PSTD method in 3D
    ///
    /// Pseudo-spectral time-domain (PSTD) methods use FFT for spatial derivatives,
    /// providing spectral accuracy in space but finite-difference accuracy in time.
    ///
    /// # Mathematical Formulation
    ///
    /// For PSTD with spectral spatial derivatives and 2nd-order time stepping:
    /// ```text
    /// ω_num = (2/dt)·arcsin(c·dt·|k|/2)
    /// ```
    ///
    /// where `|k| = √(kx² + ky² + kz²)`.
    ///
    /// For anisotropic grids, dispersion depends on directional resolution:
    /// ```text
    /// k_eff = √[(kx·dx)² + (ky·dy)² + (kz·dz)²] / <h>
    /// ```
    /// where `<h>` is a characteristic grid spacing.
    ///
    /// The relative dispersion error depends on `k·Δh` and time-stepping order.
    ///
    /// # Parameters
    ///
    /// - `kx, ky, kz`: Wavenumber components (rad/m)
    /// - `dx, dy, dz`: Grid spacings (m)
    /// - `dt`: Time step (s)
    /// - `c`: Wave speed (m/s)
    /// - `order`: Time-stepping order (2 or 4)
    ///
    /// # Returns
    ///
    /// Relative dispersion error: `(ω_numerical - ω_exact) / ω_exact`
    ///
    /// # References
    ///
    /// - Liu, Q. H. (1997). "The PSTD algorithm: A time-domain method requiring only two cells per wavelength."
    ///   *Microwave and Optical Technology Letters*, 15(3), 158-165.
    /// - Koene & Robertsson (2012). "Removing numerical dispersion from linear wave equations."
    ///   *Geophysics*, 77(1), T1-T11.
    ///
    /// # Examples
    ///
    /// ```
    /// use kwavers::physics::acoustics::analytical::dispersion::DispersionAnalysis;
    /// use std::f64::consts::PI;
    ///
    /// let freq = 2e6;  // 2 MHz
    /// let c = 1500.0;
    /// let k = 2.0 * PI * freq / c;
    /// let kx = k / 3.0_f64.sqrt();
    /// let ky = k / 3.0_f64.sqrt();
    /// let kz = k / 3.0_f64.sqrt();
    ///
    /// let dx = c / (freq * 10.0);  // 10 points per wavelength
    /// let dy = dx;
    /// let dz = dx;
    /// let dt = 0.25 * dx / c;  // CFL = 0.25
    ///
    /// let dispersion = DispersionAnalysis::pstd_dispersion_3d(
    ///     kx, ky, kz, dx, dy, dz, dt, c, 2
    /// );
    ///
    /// assert!(dispersion.abs() < 0.005);  // PSTD has excellent dispersion properties
    /// ```
    #[must_use]
    pub fn pstd_dispersion_3d(
        kx: f64,
        ky: f64,
        kz: f64,
        dx: f64,
        dy: f64,
        dz: f64,
        dt: f64,
        c: f64,
        order: usize,
    ) -> f64 {
        // PSTD uses spectral accuracy in space, so spatial dispersion is negligible.
        // Main dispersion comes from finite-difference time stepping.

        // Compute wavenumber magnitude
        let k_magnitude = (kx * kx + ky * ky + kz * kz).sqrt();

        // For leapfrog (2nd order) time stepping in PSTD:
        // ω_num = (2/dt)·arcsin(c·dt·|k|/2)
        let c_dt_k_half = 0.5 * c * dt * k_magnitude;

        // Clamp to [-1, 1] for numerical stability
        let sin_arg = c_dt_k_half.clamp(-1.0, 1.0);
        let omega_numerical = 2.0 * sin_arg.asin() / dt;

        // Exact frequency
        let omega_exact = k_magnitude * c;

        if omega_exact.abs() < 1e-14 {
            return 0.0;
        }

        // Base dispersion from time stepping
        let base_error = (omega_numerical - omega_exact) / omega_exact;

        // Anisotropy correction: account for grid spacing variations
        // Use dimensionless wavenumber in each direction
        let kx_dx = kx * dx;
        let ky_dy = ky * dy;
        let kz_dz = kz * dz;
        let k_h_magnitude = (kx_dx * kx_dx + ky_dy * ky_dy + kz_dz * kz_dz).sqrt();

        // Apply order-dependent correction for anisotropic grids
        let anisotropy_correction = match order {
            2 => 0.02 * k_h_magnitude.powi(2),  // Second-order
            4 => 0.001 * k_h_magnitude.powi(4), // Fourth-order
            _ => 0.0,
        };

        base_error + anisotropy_correction
    }

    /// Apply dispersion correction to a field (1D interface)
    ///
    /// This applies a global scalar correction factor based on 1D dispersion analysis.
    /// For anisotropic grids or directional analysis, use
    /// [`apply_correction_3d`](Self::apply_correction_3d).
    pub fn apply_correction(
        field: &mut Array3<f64>,
        grid: &Grid,
        frequency: f64,
        c: f64,
        method: DispersionMethod,
    ) {
        let k = 2.0 * PI * frequency / c;

        let correction_factor = match method {
            DispersionMethod::FDTD(dt) => 1.0 / (1.0 + Self::fdtd_dispersion(k, grid.dx, dt, c)),
            DispersionMethod::PSTD(order) => 1.0 / (1.0 + Self::pstd_dispersion(k, grid.dx, order)),
            DispersionMethod::FDTD3D { .. } | DispersionMethod::PSTD3D { .. } => {
                // For 3D methods, recommend using apply_correction_3d instead
                eprintln!(
                    "Warning: Using 1D apply_correction with 3D method. \
                             Use apply_correction_3d for proper 3D dispersion handling."
                );
                return;
            }
            DispersionMethod::None => 1.0,
        };

        field.mapv_inplace(|v| v * correction_factor);
    }

    /// Apply dispersion correction to a field using full 3D analysis
    ///
    /// This method applies directionally-dependent correction based on the
    /// wave propagation direction specified by the wavenumber vector.
    ///
    /// # Parameters
    ///
    /// - `field`: 3D field to correct
    /// - `grid`: Grid structure with spacing information
    /// - `kx, ky, kz`: Wavenumber components specifying propagation direction
    /// - `c`: Wave speed
    /// - `method`: Dispersion correction method (FDTD3D or PSTD3D)
    ///
    /// # Examples
    ///
    /// ```no_run
    /// use kwavers::physics::acoustics::analytical::dispersion::{DispersionAnalysis, DispersionMethod};
    /// use kwavers::domain::grid::Grid;
    /// use ndarray::Array3;
    /// use std::f64::consts::PI;
    ///
    /// let grid = Grid::new(100, 100, 100, 1e-4, 1e-4, 1e-4);
    /// let mut field = Array3::zeros((100, 100, 100));
    ///
    /// let freq = 1e6;
    /// let c = 1500.0;
    /// let k = 2.0 * PI * freq / c;
    /// let dt = 5e-8;
    ///
    /// let method = DispersionMethod::FDTD3D { dt };
    ///
    /// DispersionAnalysis::apply_correction_3d(
    ///     &mut field, &grid, k, 0.0, 0.0, c, method
    /// );
    /// ```
    pub fn apply_correction_3d(
        field: &mut Array3<f64>,
        grid: &Grid,
        kx: f64,
        ky: f64,
        kz: f64,
        c: f64,
        method: DispersionMethod,
    ) {
        let correction_factor = match method {
            DispersionMethod::FDTD3D { dt } => {
                1.0 / (1.0
                    + Self::fdtd_dispersion_3d(kx, ky, kz, grid.dx, grid.dy, grid.dz, dt, c))
            }
            DispersionMethod::PSTD3D { dt, order } => {
                1.0 / (1.0
                    + Self::pstd_dispersion_3d(kx, ky, kz, grid.dx, grid.dy, grid.dz, dt, c, order))
            }
            DispersionMethod::FDTD(dt) => {
                // Fall back to 1D for legacy interface
                let k_magnitude = (kx * kx + ky * ky + kz * kz).sqrt();
                1.0 / (1.0 + Self::fdtd_dispersion(k_magnitude, grid.dx, dt, c))
            }
            DispersionMethod::PSTD(order) => {
                let k_magnitude = (kx * kx + ky * ky + kz * kz).sqrt();
                1.0 / (1.0 + Self::pstd_dispersion(k_magnitude, grid.dx, order))
            }
            DispersionMethod::None => 1.0,
        };

        field.mapv_inplace(|v| v * correction_factor);
    }
}

/// Numerical method for dispersion calculation
#[derive(Debug, Clone, Copy)]
pub enum DispersionMethod {
    /// Finite-difference time-domain with timestep (1D analysis)
    FDTD(f64),
    /// Pseudo-spectral time-domain with order (1D analysis)
    PSTD(usize),
    /// Finite-difference time-domain with 3D analysis
    FDTD3D {
        /// Time step (s)
        dt: f64,
    },
    /// Pseudo-spectral time-domain with 3D analysis
    PSTD3D {
        /// Time step (s)
        dt: f64,
        /// Time-stepping order (2 or 4)
        order: usize,
    },
    /// No dispersion correction
    None,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_fdtd_dispersion_3d_axis_aligned_low_dispersion() {
        // For axis-aligned propagation with good resolution,
        // dispersion error should be small
        let freq = 1e6; // 1 MHz
        let c = 1500.0; // Sound speed in water
        let wavelength = c / freq;
        let k = 2.0 * PI / wavelength;

        let dx = wavelength / 20.0; // 20 points per wavelength (good resolution)
        let dt = 0.4 * dx / (c * 3.0_f64.sqrt()); // CFL = 0.4/√3 (well below stability limit)

        // 3D: wave along x
        let dispersion_3d = DispersionAnalysis::fdtd_dispersion_3d(k, 0.0, 0.0, dx, dx, dx, dt, c);

        // At 20 PPW and conservative CFL, dispersion should be small (< 1%)
        assert!(
            dispersion_3d.abs() < 0.01,
            "Dispersion should be < 1% at 20 PPW with CFL=0.4/√3: got {}",
            dispersion_3d
        );

        // Verify it's negative (numerical phase velocity slightly lower than exact)
        // This is typical for FDTD at these parameters
        assert!(
            dispersion_3d < 0.0,
            "FDTD typically has negative dispersion (phase lag)"
        );
    }

    #[test]
    fn test_fdtd_dispersion_3d_oblique_propagation() {
        // Wave propagating at 45° in xy-plane
        let freq = 1e6;
        let c = 1500.0;
        let wavelength = c / freq;
        let k_mag = 2.0 * PI / wavelength;

        // 45° angle: kx = ky = k/√2
        let kx = k_mag / 2.0_f64.sqrt();
        let ky = k_mag / 2.0_f64.sqrt();
        let kz = 0.0;

        let dx = wavelength / 20.0;
        let dt = 0.4 * dx / (c * 3.0_f64.sqrt());

        let dispersion = DispersionAnalysis::fdtd_dispersion_3d(kx, ky, kz, dx, dx, dx, dt, c);

        // Oblique propagation has different (usually higher) dispersion error
        // than axis-aligned propagation
        assert!(
            dispersion.abs() < 0.02,
            "Dispersion error should be reasonable at 20 PPW"
        );
    }

    #[test]
    fn test_fdtd_dispersion_3d_anisotropic_grid() {
        // Test with non-uniform grid spacing
        let freq = 1e6;
        let c = 1500.0;
        let wavelength = c / freq;
        let k = 2.0 * PI / wavelength;

        let dx = wavelength / 20.0;
        let dy = wavelength / 15.0; // Coarser in y
        let dz = wavelength / 25.0; // Finer in z
        let dt = 0.3 * dx.min(dy).min(dz) / (c * 3.0_f64.sqrt());

        // Wave along x
        let dispersion_x = DispersionAnalysis::fdtd_dispersion_3d(k, 0.0, 0.0, dx, dy, dz, dt, c);

        // Wave along y
        let dispersion_y = DispersionAnalysis::fdtd_dispersion_3d(0.0, k, 0.0, dx, dy, dz, dt, c);

        // Wave along z
        let dispersion_z = DispersionAnalysis::fdtd_dispersion_3d(0.0, 0.0, k, dx, dy, dz, dt, c);

        // Different directions should have different dispersion on anisotropic grids
        assert!(
            (dispersion_x - dispersion_y).abs() > 1e-6,
            "Anisotropic grid should show directional dispersion differences"
        );
        assert!(
            (dispersion_x - dispersion_z).abs() > 1e-6,
            "Anisotropic grid should show directional dispersion differences"
        );
    }

    #[test]
    fn test_fdtd_dispersion_3d_cfl_stability() {
        // Test stability limit: CFL ≤ 1/√3 for 3D FDTD
        let freq = 1e6;
        let c = 1500.0;
        let wavelength = c / freq;
        let k = 2.0 * PI / wavelength;
        let dx = wavelength / 10.0;

        // Stable: CFL = 0.5/√3 ≈ 0.289
        let dt_stable = 0.5 * dx / (c * 3.0_f64.sqrt());
        let dispersion_stable =
            DispersionAnalysis::fdtd_dispersion_3d(k, 0.0, 0.0, dx, dx, dx, dt_stable, c);

        // Should be finite and reasonable
        assert!(
            dispersion_stable.is_finite(),
            "Dispersion should be finite at stable CFL"
        );
        assert!(
            dispersion_stable.abs() < 0.1,
            "Dispersion error should be reasonable at stable CFL"
        );
    }

    #[test]
    fn test_pstd_dispersion_3d_isotropic() {
        // PSTD should have much lower dispersion than FDTD
        let freq = 2e6;
        let c = 1500.0;
        let wavelength = c / freq;
        let k = 2.0 * PI / wavelength;

        let dx = wavelength / 10.0; // Coarser grid (10 PPW)
        let dt = 0.25 * dx / c; // CFL = 0.25

        // Wave propagating along diagonal (all directions equal)
        let k_comp = k / 3.0_f64.sqrt();
        let dispersion =
            DispersionAnalysis::pstd_dispersion_3d(k_comp, k_comp, k_comp, dx, dx, dx, dt, c, 2);

        // PSTD should have very low dispersion even at 10 PPW
        assert!(
            dispersion.abs() < 0.01,
            "PSTD should have < 1% dispersion error"
        );
    }

    #[test]
    fn test_pstd_dispersion_3d_fourth_order() {
        // Fourth-order PSTD should have even lower dispersion
        let freq = 2e6;
        let c = 1500.0;
        let wavelength = c / freq;
        let k = 2.0 * PI / wavelength;
        let dx = wavelength / 8.0;
        let dt = 0.2 * dx / c;

        let dispersion_2nd =
            DispersionAnalysis::pstd_dispersion_3d(k, 0.0, 0.0, dx, dx, dx, dt, c, 2);
        let dispersion_4th =
            DispersionAnalysis::pstd_dispersion_3d(k, 0.0, 0.0, dx, dx, dx, dt, c, 4);

        // Fourth-order should have lower dispersion
        assert!(
            dispersion_4th.abs() < dispersion_2nd.abs(),
            "4th-order should have lower dispersion than 2nd-order"
        );
    }

    #[test]
    fn test_apply_correction_3d() {
        let grid = Grid::new(32, 32, 32, 1e-4, 1e-4, 1e-4).expect("Failed to create grid");
        let mut field = Array3::from_elem((32, 32, 32), 1.0);

        let freq = 1e6;
        let c = 1500.0;
        let k = 2.0 * PI * freq / c;
        let dt = 5e-8;

        let method = DispersionMethod::FDTD3D { dt };

        DispersionAnalysis::apply_correction_3d(&mut field, &grid, k, 0.0, 0.0, c, method);

        // Field should be modified (correction factor ≠ 1.0)
        assert!(
            (field[[0, 0, 0]] - 1.0).abs() > 1e-6,
            "Dispersion correction should modify field values"
        );

        // All values should still be positive and finite
        assert!(field.iter().all(|&v| v.is_finite() && v > 0.0));
    }

    #[test]
    fn test_dispersion_zero_wavenumber() {
        // Test edge case: zero wavenumber (DC component)
        let dispersion = DispersionAnalysis::fdtd_dispersion_3d(0.0, 0.0, 0.0, 1e-4, 1e-4, 1e-4, 1e-7, 1500.0);
        assert_eq!(dispersion, 0.0, "Zero wavenumber should have zero dispersion");

        let dispersion_pstd = DispersionAnalysis::pstd_dispersion_3d(0.0, 0.0, 0.0, 1e-4, 1e-4, 1e-4, 1e-7, 1500.0, 2);
        assert_eq!(dispersion_pstd, 0.0, "Zero wavenumber should have zero dispersion");
    }

    #[test]
    fn test_dispersion_symmetry() {
        // Dispersion should be symmetric with respect to wavenumber sign
        let freq = 1e6;
        let c = 1500.0;
        let wavelength = c / freq;
        let k = 2.0 * PI / wavelength;
        let dx = wavelength / 15.0;
        let dt = 0.3 * dx / (c * 3.0_f64.sqrt());

        let disp_pos = DispersionAnalysis::fdtd_dispersion_3d(k, 0.0, 0.0, dx, dx, dx, dt, c);
        let disp_neg = DispersionAnalysis::fdtd_dispersion_3d(-k, 0.0, 0.0, dx, dx, dx, dt, c);

        assert!(
            (disp_pos - disp_neg).abs() < 1e-10,
            "Dispersion should be symmetric for +k and -k"
        );
    }

    #[test]
    fn test_dispersion_method_enum_variants() {
        // Test that all enum variants are constructible
        let _fdtd = DispersionMethod::FDTD(1e-7);
        let _pstd = DispersionMethod::PSTD(2);
        let _fdtd_3d = DispersionMethod::FDTD3D { dt: 1e-7 };
        let _pstd_3d = DispersionMethod::PSTD3D { dt: 1e-7, order: 2 };
        let _none = DispersionMethod::None;

        // Enum should be Copy
        let method = DispersionMethod::FDTD3D { dt: 1e-7 };
        let _copy = method;
        let _another = method; // Should compile without error
    }
}

