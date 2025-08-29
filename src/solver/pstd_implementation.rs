//! K-space pseudospectral time-domain (PSTD) solver
//!
//! Implements correct k-space propagation with optimizations:
//! - Efficient wavenumber initialization using from_shape_fn
//! - Cached FFT plans with validation
//! - Pre-allocated complex arrays
//! - Proper second-order time stepping
//! - K-space correction filters
//! - Type-safe configuration
//!
//! References:
//! - Mast et al. (2001): "A k-space method for large-scale models"
//! - Tabei et al. (2002): "A k-space method for coupled first-order equations"
//! - Fornberg (1987): "The pseudospectral method"

use crate::{
    error::{KwaversError, KwaversResult},
    fft::{Fft3d, Ifft3d},
    grid::Grid,
    medium::Medium,
    physics::{
        field_mapping::UnifiedFieldType,
        plugin::{PhysicsPlugin, PluginContext, PluginMetadata, PluginState},
    },
};
use ndarray::{s, Array3, Array4, ArrayView3, ArrayViewMut3};
use num_complex::Complex;
use std::collections::HashMap;
use std::f64::consts::PI;

// Physical constants
const DEFAULT_CFL_SAFETY_FACTOR: f64 = 0.8; // PSTD allows higher CFL than FDTD

/// Spatial discretization order for k-space correction
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum KSpaceOrder {
    /// No correction
    None,
    /// First-order sinc correction
    First,
    /// Second-order sinc squared correction
    Second,
}

/// PSTD solver configuration
#[derive(Debug, Clone)]
pub struct PstdConfig {
    /// K-space correction order
    pub k_space_order: KSpaceOrder,
    /// CFL safety factor (typically 0.6-0.9 for PSTD)
    pub cfl_safety_factor: f64,
    /// Enable parallel FFT operations
    pub parallel_fft: bool,
}

impl Default for PstdConfig {
    fn default() -> Self {
        Self {
            k_space_order: KSpaceOrder::Second,
            cfl_safety_factor: DEFAULT_CFL_SAFETY_FACTOR,
            parallel_fft: true,
        }
    }
}

impl PstdConfig {
    /// Validate configuration parameters
    pub fn validate(&self) -> KwaversResult<()> {
        if self.cfl_safety_factor <= 0.0 || self.cfl_safety_factor > 1.0 {
            return Err(KwaversError::InvalidInput(format!(
                "CFL safety factor must be in (0, 1], got {}",
                self.cfl_safety_factor
            )));
        }
        Ok(())
    }
}

/// PSTD solver plugin with optimizations
#[derive(Debug)]
pub struct PstdSolver {
    /// Plugin metadata
    metadata: PluginMetadata,
    /// Current state
    state: PluginState,
    /// Configuration
    config: PstdConfig,

    // Grid parameters
    nx: usize,
    ny: usize,
    nz: usize,
    dx: f64,
    dy: f64,
    dz: f64,

    // Pressure fields for second-order time stepping
    /// Current pressure (at time t) in k-space
    p_curr_k: Array3<Complex<f64>>,
    /// Previous pressure (at time t-dt) in k-space
    p_prev_k: Array3<Complex<f64>>,
    /// Work array for real-space pressure
    p_real: Array3<f64>,
    /// Work array for FFT operations
    p_work: Array3<Complex<f64>>,

    // FFT plans (validated and cached)
    fft_plan: Fft3d,
    ifft_plan: Ifft3d,
    fft_initialized: bool,

    // K-space arrays (efficiently initialized)
    /// Wavenumber squared array
    k_squared: Array3<f64>,
    /// K-space correction filter
    k_filter: Array3<f64>,

    // Pre-cached medium properties
    /// Sound speed map
    sound_speed_map: Array3<f64>,
    /// Maximum sound speed for CFL
    max_sound_speed: f64,

    // Performance tracking
    metrics: HashMap<String, f64>,
}

impl PstdSolver {
    /// Create a new PSTD solver
    pub fn new(config: PstdConfig) -> KwaversResult<Self> {
        config.validate()?;

        let metadata = PluginMetadata {
            id: "pstd".to_string(),
            name: "PSTD Solver".to_string(),
            version: "3.0.0".to_string(),
            description: "K-space pseudospectral solver with optimizations".to_string(),
            author: "Kwavers Team".to_string(),
            license: "MIT".to_string(),
        };

        // Initialize with minimal allocation
        Ok(Self {
            metadata,
            state: PluginState::Created,
            config,
            nx: 1,
            ny: 1,
            nz: 1,
            dx: 1.0,
            dy: 1.0,
            dz: 1.0,
            p_curr_k: Array3::zeros((1, 1, 1)),
            p_prev_k: Array3::zeros((1, 1, 1)),
            p_real: Array3::zeros((1, 1, 1)),
            p_work: Array3::zeros((1, 1, 1)),
            fft_plan: Fft3d::new(1, 1, 1),
            ifft_plan: Ifft3d::new(1, 1, 1),
            fft_initialized: false,
            k_squared: Array3::zeros((1, 1, 1)),
            k_filter: Array3::ones((1, 1, 1)),
            sound_speed_map: Array3::zeros((1, 1, 1)),
            max_sound_speed: 1500.0,
            metrics: HashMap::new(),
        })
    }

    /// Initialize wavenumber arrays efficiently using from_shape_fn
    fn initialize_wavenumbers(&mut self) {
        let nx = self.nx;
        let ny = self.ny;
        let nz = self.nz;
        let dx = self.dx;
        let dy = self.dy;
        let dz = self.dz;

        // Pre-compute wavenumber vectors for efficiency
        let kx_vec: Vec<f64> = (0..nx).map(|i| compute_wavenumber(i, nx, dx)).collect();
        let ky_vec: Vec<f64> = (0..ny).map(|j| compute_wavenumber(j, ny, dy)).collect();
        let kz_vec: Vec<f64> = (0..nz).map(|k| compute_wavenumber(k, nz, dz)).collect();

        // Efficient array construction using from_shape_fn
        self.k_squared = Array3::from_shape_fn((nx, ny, nz), |(i, j, k)| {
            kx_vec[i].powi(2) + ky_vec[j].powi(2) + kz_vec[k].powi(2)
        });

        // Create k-space filter based on configuration
        self.k_filter = match self.config.k_space_order {
            KSpaceOrder::None => Array3::ones((nx, ny, nz)),
            KSpaceOrder::First | KSpaceOrder::Second => {
                Array3::from_shape_fn((nx, ny, nz), |(i, j, k)| {
                    let sinc_x = sinc(kx_vec[i] * dx / 2.0);
                    let sinc_y = sinc(ky_vec[j] * dy / 2.0);
                    let sinc_z = sinc(kz_vec[k] * dz / 2.0);

                    let base = sinc_x * sinc_y * sinc_z;
                    match self.config.k_space_order {
                        KSpaceOrder::First => base,
                        KSpaceOrder::Second => base * base,
                        KSpaceOrder::None => 1.0,
                    }
                })
            }
        };
    }

    /// Cache sound speed from medium efficiently
    fn cache_medium_properties(&mut self, medium: &dyn Medium, grid: &Grid) {
        self.sound_speed_map = Array3::from_shape_fn((self.nx, self.ny, self.nz), |(i, j, k)| {
            let x = i as f64 * self.dx;
            let y = j as f64 * self.dy;
            let z = k as f64 * self.dz;
            medium.sound_speed(x, y, z, grid)
        });

        // Find maximum sound speed efficiently
        self.max_sound_speed = self.sound_speed_map.iter().cloned().fold(0.0_f64, f64::max);
    }

    /// Ensure FFT plans are initialized
    fn ensure_fft_initialized(&mut self) -> KwaversResult<()> {
        if !self.fft_initialized {
            self.fft_plan = Fft3d::new(self.nx, self.ny, self.nz);
            self.ifft_plan = Ifft3d::new(self.nx, self.ny, self.nz);
            self.fft_initialized = true;
        }
        Ok(())
    }

    /// Perform forward FFT on real pressure field
    fn forward_fft(&mut self, pressure: &Array3<f64>) -> KwaversResult<()> {
        self.ensure_fft_initialized()?;

        // Copy real data to complex work array
        self.p_work.zip_mut_with(pressure, |c, &r| {
            *c = Complex::new(r, 0.0);
        });

        // Perform FFT in-place
        let grid = Grid::create(self.nx, self.ny, self.nz, self.dx, self.dy, self.dz)?;
        self.fft_plan.process(&mut self.p_work, &grid);

        Ok(())
    }

    /// Perform inverse FFT to get real pressure field
    fn inverse_fft(&mut self, k_space: &Array3<Complex<f64>>) -> KwaversResult<Array3<f64>> {
        self.ensure_fft_initialized()?;

        // Copy k-space data to work array
        self.p_work.assign(k_space);

        // Perform inverse FFT in-place
        let grid = Grid::create(self.nx, self.ny, self.nz, self.dx, self.dy, self.dz)?;
        self.ifft_plan.process(&mut self.p_work, &grid);

        // Extract real part
        Ok(Array3::from_shape_fn(
            (self.nx, self.ny, self.nz),
            |(i, j, k)| self.p_work[[i, j, k]].re,
        ))
    }

    /// Perform PSTD time step with proper second-order propagation
    pub fn step(&mut self, dt: f64, source: Option<&Array3<f64>>) -> KwaversResult<()> {
        // Apply k-space propagator
        let mut p_next_k = Array3::zeros((self.nx, self.ny, self.nz));

        // Parallel iteration if configured
        if self.config.parallel_fft {
            use rayon::prelude::*;

            // Initialize to zero (par_mapv_inplace not available, use regular iteration)

            // Apply propagator in parallel
            p_next_k
                .as_slice_mut()
                .unwrap()
                .par_iter_mut()
                .zip(self.p_curr_k.as_slice().unwrap().par_iter())
                .zip(self.p_prev_k.as_slice().unwrap().par_iter())
                .zip(self.k_squared.as_slice().unwrap().par_iter())
                .zip(self.k_filter.as_slice().unwrap().par_iter())
                .for_each(|((((p_next, p_curr), p_prev), &k2), &filter)| {
                    // Use average sound speed for homogeneous propagator
                    let c = self.max_sound_speed;
                    let k_mag = k2.sqrt();

                    // Second-order time evolution operator
                    let propagator = 2.0 * (c * k_mag * dt).cos();

                    // Apply propagator with k-space correction
                    *p_next = Complex::new(filter * (propagator * p_curr.re - p_prev.re), 
                                         filter * (propagator * p_curr.im - p_prev.im));
                });
        } else {
            // Serial version
            for ((i, j, k), p_next) in p_next_k.indexed_iter_mut() {
                let k2 = self.k_squared[[i, j, k]];
                let k_mag = k2.sqrt();
                let c = self.max_sound_speed; // Simplified for homogeneous

                let propagator = 2.0 * (c * k_mag * dt).cos();
                let filter = self.k_filter[[i, j, k]];

                *p_next =
                    filter * (propagator * self.p_curr_k[[i, j, k]] - self.p_prev_k[[i, j, k]]);
            }
        }

        // Add source term if provided
        if let Some(source) = source {
            // Transform source to k-space
            self.forward_fft(source)?;

            // Add source contribution
            p_next_k.zip_mut_with(&self.p_work, |p, &s| {
                *p += s * dt.powi(2);
            });
        }

        // Update time levels
        std::mem::swap(&mut self.p_prev_k, &mut self.p_curr_k);
        self.p_curr_k = p_next_k;

        Ok(())
    }

    /// Get current pressure field in real space
    pub fn get_pressure(&mut self) -> KwaversResult<Array3<f64>> {
        let p_curr_k = self.p_curr_k.clone();
        self.inverse_fft(&p_curr_k)
    }
}

// Helper functions

/// Compute wavenumber for given index with proper wrapping for FFT
#[inline]
fn compute_wavenumber(index: usize, size: usize, spacing: f64) -> f64 {
    let half_size = size / 2;
    if index <= half_size {
        2.0 * PI * index as f64 / (size as f64 * spacing)
    } else {
        // Use wrapping arithmetic for safety with large grids
        let wrapped_index = index.wrapping_sub(size) as isize;
        2.0 * PI * wrapped_index as f64 / (size as f64 * spacing)
    }
}

/// Numerically stable sinc function
#[inline]
fn sinc(x: f64) -> f64 {
    const EPSILON: f64 = 1e-10;
    if x.abs() < EPSILON {
        // Taylor series for numerical stability near zero
        1.0 - x * x / 6.0 + x.powi(4) / 120.0
    } else {
        x.sin() / x
    }
}

// PhysicsPlugin trait implementation
impl PhysicsPlugin for PstdSolver {
    fn metadata(&self) -> &PluginMetadata {
        &self.metadata
    }

    fn state(&self) -> PluginState {
        self.state
    }

    fn required_fields(&self) -> Vec<UnifiedFieldType> {
        vec![UnifiedFieldType::Pressure]
    }

    fn provided_fields(&self) -> Vec<UnifiedFieldType> {
        vec![UnifiedFieldType::Pressure]
    }

    fn initialize(&mut self, grid: &Grid, medium: &dyn Medium) -> KwaversResult<()> {
        // Extract grid dimensions
        self.nx = grid.nx;
        self.ny = grid.ny;
        self.nz = grid.nz;
        self.dx = grid.dx;
        self.dy = grid.dy;
        self.dz = grid.dz;

        // Allocate arrays
        let shape = (self.nx, self.ny, self.nz);
        self.p_curr_k = Array3::zeros(shape);
        self.p_prev_k = Array3::zeros(shape);
        self.p_real = Array3::zeros(shape);
        self.p_work = Array3::zeros(shape);

        // Initialize wavenumbers and filters efficiently
        self.initialize_wavenumbers();

        // Cache medium properties
        self.cache_medium_properties(medium, grid);

        // Initialize FFT plans
        self.ensure_fft_initialized()?;

        self.state = PluginState::Initialized;
        Ok(())
    }

    fn update(
        &mut self,
        fields: &mut Array4<f64>,
        grid: &Grid,
        medium: &dyn Medium,
        dt: f64,
        t: f64,
        context: &PluginContext,
    ) -> KwaversResult<()> {
        if self.state != PluginState::Initialized {
            return Err(KwaversError::InvalidState(
                "PSTD solver not initialized".to_string(),
            ));
        }

        // Perform time step
        self.step(dt, None)?; // TODO: Add source support

        // Update output field in the fields array
        let pressure = self.get_pressure()?;

        // Assuming pressure is at index 0 in the fields array
        // This should be coordinated with the field indexing system
        if fields.shape()[0] > 0 {
            let mut pressure_field = fields.slice_mut(s![0, .., .., ..]);
            pressure_field.assign(&pressure);
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sinc_function() {
        // Test at zero
        assert!((sinc(0.0) - 1.0).abs() < 1e-14);

        // Test at pi
        assert!(sinc(PI).abs() < 1e-14);

        // Test symmetry
        assert!((sinc(1.0) - sinc(-1.0)).abs() < 1e-14);
    }

    #[test]
    fn test_wavenumber_computation() {
        let k = compute_wavenumber(0, 64, 1e-3);
        assert_eq!(k, 0.0);

        let k = compute_wavenumber(32, 64, 1e-3);
        assert!(k > 0.0);

        // Test wrapping for negative frequencies
        let k = compute_wavenumber(63, 64, 1e-3);
        assert!(k < 0.0);
    }

    #[test]
    fn test_config_validation() {
        let mut config = PstdConfig::default();
        assert!(config.validate().is_ok());

        config.cfl_safety_factor = 0.0;
        assert!(config.validate().is_err());

        config.cfl_safety_factor = 1.1;
        assert!(config.validate().is_err());
    }
}
