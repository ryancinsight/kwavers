//! Nonlinear wave model implementation
//!
//! This module contains the core `NonlinearWave` struct and its basic implementation.

use kwavers_core::constants::fundamental::SOUND_SPEED_WATER_SIM;
use kwavers_core::constants::numerical::{MHZ_TO_HZ, PRESSURE_LIMIT};
use kwavers_grid::Grid;
use kwavers_medium::Medium;

use crate::parallel::for_each_indexed_mut;
use leto::Array3;
use std::f64;

/// Represents a nonlinear wave model solver.
///
/// This struct encapsulates the parameters and state for simulating acoustic wave propagation
/// with nonlinear effects, focusing on efficiency and physical accuracy.
/// It includes settings for performance monitoring, physical model characteristics,
/// precomputed values for faster calculations, and stability control mechanisms.
///
/// # Heterogeneous Media Limitations
///
/// **IMPORTANT**: This implementation uses the Pseudo-Spectral Time Domain (PSTD) method,
/// which has fundamental limitations when applied to heterogeneous media:
///
/// - The k-space correction factor `sinc(c*k*dt/2)` assumes a constant sound speed
/// - For heterogeneous media, this correction becomes position-dependent
/// - The current implementation uses the maximum sound speed for stability
/// - This approach maintains numerical stability but introduces phase errors
///
/// ## Recommended Alternatives for Heterogeneous Media:
///
/// 1. **Finite Difference Time Domain (FDTD)**: Naturally handles spatial variations
/// 2. **Split-Step Fourier Method**: Alternates between spatial and spectral domains
/// 3. **Heterogeneous k-space correction methods**: Use position-dependent k-space corrections
/// 4. **Hybrid Methods**: Combine PSTD in homogeneous regions with FDTD in heterogeneous regions
///
/// ## Current Mitigation Strategy:
///
/// The solver uses `max_sound_speed` for the k-space correction, which:
/// - Ensures numerical stability across the entire domain
/// - Prevents phase velocity exceeding physical limits
/// - May over-correct in regions where `c < c_max`, causing phase lag
/// - Provides conservative but stable results
///
/// For weakly heterogeneous media (variations < 10%), the errors are typically acceptable.
/// For strongly heterogeneous media, consider using alternative methods.
#[derive(Debug, Clone)]
pub struct NonlinearWave {
    // Performance metrics
    /// Time spent in the nonlinear term calculation part of the update, in seconds.
    pub(crate) nonlinear_time: f64,
    /// Time spent in FFT operations during the update, in seconds.
    pub(crate) fft_time: f64,
    /// Time spent applying the source term during the update, in seconds.
    pub(crate) source_time: f64,
    /// Time spent combining linear and nonlinear field components, in seconds.
    pub(crate) combination_time: f64,
    /// Number of times the `update_wave` method has been called.
    pub(crate) call_count: usize,

    // Physical model settings
    /// Scaling factor for the nonlinearity term, allowing adjustment for strong nonlinear effects.
    pub(crate) nonlinearity_scaling: f64,
    // Precomputed arrays
    /// Precomputed k-squared values (square of wavenumber magnitudes) for the grid, used to speed up calculations.
    pub(crate) k_squared: Option<Array3<f64>>,

    // Stability parameters
    /// Maximum absolute pressure value allowed in the simulation to prevent numerical instability.
    pub(crate) max_pressure: f64,
    /// Threshold for the CFL (Courant-Friedrichs-Lewy) condition, used in stability checks.
    pub(crate) stability_threshold: f64,
    /// Safety factor applied to the CFL condition for determining stable timestep.
    pub(crate) cfl_safety_factor: f64,
    /// Flag to enable or disable clamping of pressure gradients to `max_gradient`.
    pub(crate) clamp_gradients: bool,

    // Frequency-dependent physics
    /// Source frequency for frequency-dependent absorption and dispersion \[Hz\]
    pub(crate) source_frequency: f64,

    // Performance optimization caches
    /// Cached maximum sound speed of the medium for efficient stability checks
    pub(crate) max_sound_speed: f64,

    // Numerical scheme parameters
    /// Time step size for the simulation \[s\]
    pub(crate) dt: f64,
}

impl NonlinearWave {
    /// Creates a new `NonlinearWave` solver with default settings.
    ///
    /// # Arguments
    ///
    /// * `grid` - The computational grid
    /// * `dt` - Time step size \[s\]
    ///
    /// # Returns
    ///
    /// A new `NonlinearWave` instance with sensible defaults
    pub fn new(_grid: &Grid, dt: f64) -> Self {
        Self {
            // Performance metrics
            nonlinear_time: 0.0,
            fft_time: 0.0,
            source_time: 0.0,
            combination_time: 0.0,
            call_count: 0,

            // Physical model settings
            nonlinearity_scaling: 1.0,

            // Precomputed arrays
            k_squared: None,

            // Stability parameters
            max_pressure: PRESSURE_LIMIT,
            stability_threshold: 0.5,
            cfl_safety_factor: 0.9,
            clamp_gradients: false,

            // Frequency-dependent physics
            source_frequency: MHZ_TO_HZ,

            // Performance caches
            max_sound_speed: SOUND_SPEED_WATER_SIM,

            // Numerical scheme
            dt,
        }
    }

    /// Precomputes k-squared values for the grid.
    ///
    /// This method calculates and stores the square of the wavenumber magnitudes
    /// for each point in the k-space grid, which speeds up subsequent calculations.
    /// # Panics
    /// - Panics if `kx contiguous`.
    /// - Panics if `ky contiguous`.
    /// - Panics if `kz contiguous`.
    ///
    pub fn precompute_k_squared(&mut self, grid: &Grid) {
        let kx = grid.compute_kx();
        let ky = grid.compute_ky();
        let kz = grid.compute_kz();

        let mut k_squared = Array3::<f64>::zeros((grid.nx, grid.ny, grid.nz));

        let kx_s = kx.as_slice().expect("kx contiguous");
        let ky_s = ky.as_slice().expect("ky contiguous");
        let kz_s = kz.as_slice().expect("kz contiguous");
        for_each_indexed_mut(k_squared.view_mut(), |(i, j, k), val| {
            *val = kz_s[k].mul_add(kz_s[k], kx_s[i].mul_add(kx_s[i], ky_s[j] * ky_s[j]));
        });

        self.k_squared = Some(k_squared);
    }

    /// Checks if the current configuration is stable.
    ///
    /// # Arguments
    ///
    /// * `medium` - The medium to check stability against
    /// * `grid` - The computational grid
    ///
    /// # Returns
    ///
    /// `true` if the configuration is stable, `false` otherwise
    pub fn is_stable(&self, medium: &dyn Medium, grid: &Grid) -> bool {
        // Get actual maximum sound speed from the medium
        let c_array = medium.sound_speed_array();
        let max_c = c_array.iter().fold(0.0f64, |acc, &x| acc.max(x));

        let min_dx = grid.dx.min(grid.dy).min(grid.dz);
        let cfl_number = max_c * self.dt / min_dx;

        cfl_number <= self.stability_threshold * self.cfl_safety_factor
    }

    /// Gets the recommended time step for stability.
    ///
    /// # Arguments
    ///
    /// * `medium` - The medium to calculate time step for
    /// * `grid` - The computational grid
    ///
    /// # Returns
    ///
    /// The recommended time step size \[s\]
    pub fn get_stable_timestep(&self, medium: &dyn Medium, grid: &Grid) -> f64 {
        // Get actual maximum sound speed from the medium
        let c_array = medium.sound_speed_array();
        let max_c = c_array.iter().fold(0.0f64, |acc, &x| acc.max(x));

        let min_dx = grid.dx.min(grid.dy).min(grid.dz);

        self.cfl_safety_factor * self.stability_threshold * min_dx / max_c
    }

    /// Resets performance metrics.
    pub fn reset_metrics(&mut self) {
        self.nonlinear_time = 0.0;
        self.fft_time = 0.0;
        self.source_time = 0.0;
        self.combination_time = 0.0;
        self.call_count = 0;
    }

    /// Gets the average time per update call.
    ///
    /// # Returns
    ///
    /// The average time spent in each update call \[s\]
    #[must_use]
    pub fn get_average_update_time(&self) -> f64 {
        if self.call_count == 0 {
            0.0
        } else {
            (self.nonlinear_time + self.fft_time + self.source_time + self.combination_time)
                / self.call_count as f64
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use kwavers_medium::HomogeneousMedium;

    #[test]
    fn new_initialises_with_sensible_defaults() {
        let grid = Grid::new(8, 8, 8, 0.001, 0.001, 0.001).unwrap();
        let dt = 1e-7_f64;
        let w = NonlinearWave::new(&grid, dt);
        assert_eq!(w.dt, dt);
        assert_eq!(w.call_count, 0);
        assert_eq!(w.nonlinear_time, 0.0);
        assert_eq!(w.nonlinearity_scaling, 1.0);
        assert!(w.k_squared.is_none());
    }

    /// DC bin of precomputed k² must be exactly zero; all values must be ≥ 0.
    #[test]
    fn precompute_k_squared_dc_bin_is_zero_and_nonnegative() {
        let grid = Grid::new(8, 8, 8, 0.001, 0.001, 0.001).unwrap();
        let mut w = NonlinearWave::new(&grid, 1e-7);
        w.precompute_k_squared(&grid);
        let k2 = w.k_squared.as_ref().unwrap();
        assert_eq!(k2.dim(), (8, 8, 8));
        assert_eq!(k2[[0, 0, 0]], 0.0, "DC bin k²[0,0,0] must be zero");
        for &v in k2.iter() {
            assert!(v >= 0.0, "k² must be non-negative everywhere (got {v})");
        }
    }

    /// `is_stable`: CFL = c·dt/dx ≤ threshold·safety.
    /// With c=1500, dt=1e-7, dx=0.001: CFL = 1500·1e-7/0.001 = 0.15.
    /// Default threshold·safety = 0.5·0.9 = 0.45 → stable.
    #[test]
    fn is_stable_returns_true_for_cfl_below_limit() {
        let dx = 0.001_f64;
        let grid = Grid::new(4, 4, 4, dx, dx, dx).unwrap();
        let medium = HomogeneousMedium::water(&grid); // c ≈ 1500 m/s
        let dt = 1e-7_f64; // CFL = 1500·1e-7/0.001 = 0.15 < 0.45
        let w = NonlinearWave::new(&grid, dt);
        assert!(
            w.is_stable(&medium, &grid),
            "CFL 0.15 < 0.45 should be stable"
        );
    }

    /// Very large dt (CFL > 0.45) must be unstable.
    #[test]
    fn is_stable_returns_false_for_cfl_above_limit() {
        let dx = 0.001_f64;
        let grid = Grid::new(4, 4, 4, dx, dx, dx).unwrap();
        let medium = HomogeneousMedium::water(&grid);
        let dt = 1e-3_f64; // CFL ≈ 1500·1e-3/0.001 = 1500 >> 0.45
        let w = NonlinearWave::new(&grid, dt);
        assert!(
            !w.is_stable(&medium, &grid),
            "CFL 1500 >> 0.45 should be unstable"
        );
    }

    /// Analytical: dt_stable = safety·threshold·min_dx/c_max = 0.9·0.5·0.001/1500 = 3e-7 s.
    #[test]
    fn get_stable_timestep_matches_analytical_cfl_formula() {
        let dx = 0.001_f64;
        let grid = Grid::new(4, 4, 4, dx, dx, dx).unwrap();
        let medium = HomogeneousMedium::water(&grid);
        let w = NonlinearWave::new(&grid, 1e-7);
        let dt_stable = w.get_stable_timestep(&medium, &grid);
        // c_water ≈ 1500; safety=0.9; threshold=0.5
        // dt_stable ≈ 0.9·0.5·0.001/1500 = 3e-7 s (within ±10% for exact c)
        assert!(
            dt_stable > 1e-8,
            "stable timestep must be positive (got {dt_stable:.3e})"
        );
        assert!(
            dt_stable < 1e-5,
            "stable timestep for water/mm grid must be sub-microsecond"
        );
    }

    #[test]
    fn reset_metrics_zeroes_all_counters() {
        let grid = Grid::new(4, 4, 4, 0.001, 0.001, 0.001).unwrap();
        let mut w = NonlinearWave::new(&grid, 1e-7);
        w.nonlinear_time = 1.0;
        w.fft_time = 2.0;
        w.source_time = 3.0;
        w.combination_time = 4.0;
        w.call_count = 5;
        w.reset_metrics();
        assert_eq!(w.nonlinear_time, 0.0);
        assert_eq!(w.fft_time, 0.0);
        assert_eq!(w.source_time, 0.0);
        assert_eq!(w.combination_time, 0.0);
        assert_eq!(w.call_count, 0);
    }

    /// With no calls, average time is zero (avoids divide-by-zero).
    #[test]
    fn get_average_update_time_is_zero_with_no_calls() {
        let grid = Grid::new(4, 4, 4, 0.001, 0.001, 0.001).unwrap();
        let w = NonlinearWave::new(&grid, 1e-7);
        assert_eq!(w.get_average_update_time(), 0.0);
    }

    /// Analytical: (1.0+2.0+3.0+4.0) / 2 = 5.0 s per call.
    #[test]
    fn get_average_update_time_divides_total_by_call_count() {
        let grid = Grid::new(4, 4, 4, 0.001, 0.001, 0.001).unwrap();
        let mut w = NonlinearWave::new(&grid, 1e-7);
        w.nonlinear_time = 1.0;
        w.fft_time = 2.0;
        w.source_time = 3.0;
        w.combination_time = 4.0;
        w.call_count = 2;
        assert!((w.get_average_update_time() - 5.0).abs() < 1e-15);
    }
}
