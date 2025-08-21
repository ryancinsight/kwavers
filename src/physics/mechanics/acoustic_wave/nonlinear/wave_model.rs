//! Nonlinear wave model implementation
//! 
//! This module contains the core NonlinearWave struct and its basic implementation.

use crate::constants::{cfl, performance, stability};
use crate::grid::Grid;
use crate::medium::Medium;
use crate::physics::field_mapping::UnifiedFieldType;
use crate::utils::{fft_3d, ifft_3d};
use crate::KwaversResult;

use log::{debug, info, warn};
use ndarray::{Array3, Array4, ArrayView3, Axis, Zip};
use rustfft::num_complex::Complex;
use std::f64;

use super::multi_frequency::MultiFrequencyConfig;

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
/// 3. **k-Wave Style Methods**: Use heterogeneous k-space corrections
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
    /// Flag to enable or disable adaptive time-stepping for potentially more stable simulations.
    pub(crate) use_adaptive_timestep: bool,

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

    // Iterator optimization settings
    /// Chunk size for cache-friendly processing
    pub(crate) chunk_size: usize,
    /// Whether to use chunked processing for large grids
    pub(crate) use_chunked_processing: bool,

    // Multi-frequency simulation support
    /// Configuration for multi-frequency analysis
    pub(crate) multi_freq_config: Option<MultiFrequencyConfig>,

    // Frequency-dependent physics
    /// Source frequency for frequency-dependent absorption and dispersion [Hz]
    pub(crate) source_frequency: f64,

    // Performance optimization caches
    /// Cached maximum sound speed of the medium for efficient stability checks
    pub(crate) max_sound_speed: f64,

    // Numerical scheme parameters
    /// Time step size for the simulation [s]
    pub(crate) dt: f64,
    /// Spatial step sizes [m]
    pub(crate) dx: f64,
    pub(crate) dy: f64,
    pub(crate) dz: f64,
}

impl NonlinearWave {
    /// Creates a new NonlinearWave solver with default settings.
    ///
    /// # Arguments
    ///
    /// * `grid` - The computational grid
    /// * `dt` - Time step size [s]
    ///
    /// # Returns
    ///
    /// A new NonlinearWave instance with sensible defaults
    pub fn new(grid: &Grid, dt: f64) -> Self {
        let chunk_size = if grid.nx * grid.ny * grid.nz > performance::LARGE_GRID_THRESHOLD {
            performance::CHUNK_SIZE_LARGE
        } else if grid.nx * grid.ny * grid.nz > performance::MEDIUM_GRID_THRESHOLD {
            performance::CHUNK_SIZE_MEDIUM
        } else {
            performance::CHUNK_SIZE_SMALL
        };

        let use_chunked_processing = 
            grid.nx * grid.ny * grid.nz > performance::CHUNKED_PROCESSING_THRESHOLD;

        Self {
            // Performance metrics
            nonlinear_time: 0.0,
            fft_time: 0.0,
            source_time: 0.0,
            combination_time: 0.0,
            call_count: 0,

            // Physical model settings
            nonlinearity_scaling: 1.0,
            use_adaptive_timestep: false,

            // Precomputed arrays
            k_squared: None,

            // Stability parameters
            max_pressure: stability::PRESSURE_LIMIT,
            stability_threshold: cfl::DEFAULT_STABILITY_THRESHOLD,
            cfl_safety_factor: cfl::DEFAULT_SAFETY_FACTOR,
            clamp_gradients: false,

            // Iterator optimization
            chunk_size,
            use_chunked_processing,

            // Multi-frequency support
            multi_freq_config: None,

            // Frequency-dependent physics
            source_frequency: 1e6, // Default 1 MHz

            // Performance caches
            max_sound_speed: 1500.0, // Default water sound speed

            // Numerical scheme
            dt,
            dx: grid.dx,
            dy: grid.dy,
            dz: grid.dz,
        }
    }

    /// Precomputes k-squared values for the grid.
    ///
    /// This method calculates and stores the square of the wavenumber magnitudes
    /// for each point in the k-space grid, which speeds up subsequent calculations.
    pub fn precompute_k_squared(&mut self, grid: &Grid) {
        let kx = grid.get_kx();
        let ky = grid.get_ky();
        let kz = grid.get_kz();

        let mut k_squared = Array3::<f64>::zeros((grid.nx, grid.ny, grid.nz));

        // Use iterators for better performance
        k_squared
            .indexed_iter_mut()
            .for_each(|((i, j, k), val)| {
                *val = kx[i].powi(2) + ky[j].powi(2) + kz[k].powi(2);
            });

        self.k_squared = Some(k_squared);
    }

    /// Updates the maximum sound speed cache.
    ///
    /// This method should be called whenever the medium properties change.
    pub fn update_max_sound_speed(&mut self, medium: &dyn Medium) {
        self.max_sound_speed = medium.get_max_sound_speed();
    }

    /// Checks if the current configuration is stable.
    ///
    /// # Arguments
    ///
    /// * `medium` - The medium to check stability against
    ///
    /// # Returns
    ///
    /// `true` if the configuration is stable, `false` otherwise
    pub fn is_stable(&self, medium: &dyn Medium) -> bool {
        let max_c = medium.get_max_sound_speed();
        let min_dx = self.dx.min(self.dy).min(self.dz);
        let cfl_number = max_c * self.dt / min_dx;
        
        cfl_number <= self.stability_threshold * self.cfl_safety_factor
    }

    /// Gets the recommended time step for stability.
    ///
    /// # Arguments
    ///
    /// * `medium` - The medium to calculate time step for
    ///
    /// # Returns
    ///
    /// The recommended time step size [s]
    pub fn get_stable_timestep(&self, medium: &dyn Medium) -> f64 {
        let max_c = medium.get_max_sound_speed();
        let min_dx = self.dx.min(self.dy).min(self.dz);
        
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
    /// The average time spent in each update call [s]
    pub fn get_average_update_time(&self) -> f64 {
        if self.call_count == 0 {
            0.0
        } else {
            (self.nonlinear_time + self.fft_time + self.source_time + self.combination_time) 
                / self.call_count as f64
        }
    }
}