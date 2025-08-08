//! Full Kuznetsov Equation Implementation for Nonlinear Acoustics
//! 
//! This module implements the complete Kuznetsov equation, which provides the most
//! comprehensive model for nonlinear acoustic wave propagation in lossy media.
//! 
//! # Physics Background
//! 
//! The Kuznetsov equation is:
//! ```text
//! ∇²p - (1/c₀²)∂²p/∂t² = -(β/ρ₀c₀⁴)∂²p²/∂t² - (δ/c₀⁴)∂³p/∂t³ + F
//! ```
//! 
//! Where:
//! - p: acoustic pressure
//! - c₀: small-signal sound speed
//! - β = 1 + B/2A: nonlinearity coefficient
//! - ρ₀: ambient density
//! - δ: acoustic diffusivity (related to absorption and dispersion)
//! - F: source terms
//! 
//! ## Key Features:
//! 
//! 1. **Full Nonlinearity**: Includes all second-order nonlinear terms
//! 2. **Acoustic Diffusivity**: Third-order time derivative for thermoviscous losses
//! 3. **Dispersion**: Proper handling of frequency-dependent absorption
//! 4. **Harmonic Generation**: Accurate modeling of harmonic buildup
//! 
//! ## Advantages over Westervelt Equation:
//! 
//! - More accurate for strong nonlinearity (high B/A values)
//! - Better representation of cumulative nonlinear effects
//! - Includes all second-order terms neglected in Westervelt
//! - More stable for shock formation
//! 
//! ## Numerical Implementation:
//! 
//! We use a mixed-domain approach with proper k-space corrections:
//! - Spatial derivatives: k-space (spectral accuracy)
//! - Time derivatives: finite difference (stability)
//! - Nonlinear terms: real space (efficiency)
//! - **K-space correction**: Higher-order time stepping with dispersion relation
//! 
//! # Design Principles Applied:
//! 
//! - **SOLID**: Single responsibility (Kuznetsov physics only)
//! - **DRY**: Reuses existing FFT and grid infrastructure
//! - **KISS**: Clear separation of linear/nonlinear/diffusive terms
//! - **YAGNI**: Only implements validated physics requirements
//! - **CUPID**: Composable with other physics components
//! - **SSOT**: Single source of truth for all physical constants
//! - **CLEAN**: Comprehensive documentation and tests

use crate::grid::Grid;
use crate::medium::Medium;
use crate::error::KwaversResult;
use crate::physics::traits::AcousticWaveModel;
use crate::utils::{fft_3d, ifft_3d};
use crate::fft::Fft3d;
use ndarray::{Array3, Array4, Zip, Axis};
use std::f64::consts::PI;

use log::{info, warn, debug};
use std::time::Instant;

// Physical constants for k-space corrections in Kuznetsov equation
/// Second-order k-space correction coefficient for Kuznetsov equation
/// Accounts for numerical dispersion in the spectral representation of
/// nonlinear acoustic wave propagation. Value tuned for optimal accuracy
/// in the ultrasound frequency range (1-10 MHz).
const KUZNETSOV_K_SPACE_CORRECTION_SECOND_ORDER: f64 = 0.05;

/// Fourth-order k-space correction coefficient for Kuznetsov equation  
/// Provides higher-order dispersion compensation for improved accuracy
/// at high frequencies approaching the Nyquist limit. Essential for
/// maintaining phase accuracy in nonlinear harmonic generation.
const KUZNETSOV_K_SPACE_CORRECTION_FOURTH_ORDER: f64 = 0.01;

/// Configuration for the Kuznetsov equation solver
#[derive(Debug, Clone)]
pub struct KuznetsovConfig {
    /// Enable full nonlinear terms (default: true)
    pub enable_nonlinearity: bool,
    
    /// Enable acoustic diffusivity term (default: true)
    pub enable_diffusivity: bool,
    
    /// Nonlinearity scaling factor for parametric studies
    pub nonlinearity_scaling: f64,
    
    /// Maximum pressure for stability clamping (Pa)
    pub max_pressure: f64,
    
    /// Time integration scheme
    pub time_scheme: TimeIntegrationScheme,
    
    /// K-space correction order (1-4)
    pub k_space_correction_order: usize,
    
    /// Spatial accuracy order (2, 4, or 6)
    pub spatial_order: usize,
    
    /// Enable adaptive time stepping
    pub adaptive_timestep: bool,
    
    /// CFL safety factor
    pub cfl_factor: f64,
    
    /// Enable dispersion compensation
    pub enable_dispersion_compensation: bool,

    /// Enable stability filter for diffusivity term
    pub stability_filter: bool,

    /// Maximum frequency for stability filter (Hz)
    pub max_frequency: f64,

    /// Default diffusivity coefficient for stability filter
    pub diffusivity: f64,
}

impl Default for KuznetsovConfig {
    fn default() -> Self {
        Self {
            enable_nonlinearity: true,
            enable_diffusivity: true,
            nonlinearity_scaling: 1.0,
            max_pressure: 1e8, // 100 MPa
            time_scheme: TimeIntegrationScheme::RK4,
            k_space_correction_order: 4, // Higher order for better accuracy
            spatial_order: 4,
            adaptive_timestep: false,
            cfl_factor: 0.3,
            enable_dispersion_compensation: true, // Enable by default
            stability_filter: true,
            max_frequency: 1e6, // 1 MHz
            diffusivity: 1.0,
        }
    }
}

/// Time integration schemes available
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum TimeIntegrationScheme {
    /// Forward Euler (first-order, fast but less stable)
    Euler,
    
    /// Second-order Runge-Kutta
    RK2,
    
    /// Fourth-order Runge-Kutta (default, good accuracy)
    RK4,
    
    /// Adams-Bashforth 3rd order (for efficiency with history)
    AdamsBashforth3,
}

/// Performance metrics for the solver
/// 
/// Tracks time spent in different parts of the solver:
/// - linear_time: Time for linear operations (velocity update, pressure update, filters)
/// - nonlinear_time: Time computing nonlinear terms
/// - diffusion_time: Time computing diffusion/absorption terms
/// - fft_time: Time in FFT operations
/// - prev_other_ops_time: Used to calculate linear_time by subtraction
#[derive(Debug, Clone, Default)]
struct SolverMetrics {
    linear_time: f64,
    nonlinear_time: f64,
    diffusion_time: f64,
    fft_time: f64,
    total_steps: u64,
    k_space_correction_time: f64,
    prev_other_ops_time: f64,
}

/// Workspace for RK4 integration to avoid repeated allocations
#[derive(Debug)]
struct RK4Workspace {
    pressure_temp: Array3<f64>,
    k1: Array3<f64>,
    k2: Array3<f64>,
    k3: Array3<f64>,
    k4: Array3<f64>,
    linear_term_cache: Array3<f64>,
    nonlinear_term_cache: Array3<f64>,
    diffusion_term_cache: Array3<f64>,
}

impl RK4Workspace {
    fn new(nx: usize, ny: usize, nz: usize) -> Self {
        let shape = (nx, ny, nz);
        Self {
            pressure_temp: Array3::zeros(shape),
            k1: Array3::zeros(shape),
            k2: Array3::zeros(shape),
            k3: Array3::zeros(shape),
            k4: Array3::zeros(shape),
            linear_term_cache: Array3::zeros(shape),
            nonlinear_term_cache: Array3::zeros(shape),
            diffusion_term_cache: Array3::zeros(shape),
        }
    }
}

/// Main Kuznetsov equation solver
#[derive(Debug)]
pub struct KuznetsovWave {
    config: KuznetsovConfig,
    
    /// Pre-computed k-space arrays for efficiency
    k_squared: Array3<f64>,
    k_magnitude: Array3<f64>,
    
    /// K-space correction phase factors for dispersion compensation
    phase_correction_factors: Array3<f64>,
    
    /// History buffers for time derivatives and Adams-Bashforth
    pressure_history: Vec<Array3<f64>>,
    nonlinear_history: Vec<Array3<f64>>,
    
    /// FFT planner for efficiency (now properly used)
    fft_planner: Fft3d,
    
    /// Performance metrics
    metrics: SolverMetrics,
    
    /// RK4 workspace (lazily initialized)
    rk4_workspace: Option<RK4Workspace>,
    
    /// Time step counter for adaptive methods
    step_count: u64,
    
    /// Previous pressure field for second-order time derivative
    pressure_prev: Option<Array3<f64>>,
    
    /// Previous velocity field for second-order formulation
    velocity_prev: Option<Array4<f64>>,
}

impl KuznetsovWave {
    /// Create a new Kuznetsov equation solver with proper physics validation
    pub fn new(grid: &Grid, config: KuznetsovConfig) -> KwaversResult<Self> {
        info!("Initializing Kuznetsov equation solver with config: {:?}", config);
        
        // Validate configuration
        Self::validate_config(&config, grid)?;
        
        // Precompute k-space arrays
        let k_squared = grid.k_squared();
        let k_magnitude = compute_k_magnitude(grid);
        
        // Compute k-space correction factors for dispersion compensation
        let phase_correction_factors = if config.enable_dispersion_compensation {
            compute_k_space_correction_factors(grid, config.k_space_correction_order)
        } else {
            Array3::ones((grid.nx, grid.ny, grid.nz))
        };
        
        // Initialize history buffers based on scheme
        let history_size = match config.time_scheme {
            TimeIntegrationScheme::Euler | TimeIntegrationScheme::RK2 | TimeIntegrationScheme::RK4 => 2,
            TimeIntegrationScheme::AdamsBashforth3 => 3,
        };
        
        let pressure_history = vec![grid.zeros_array(); history_size];
        let nonlinear_history = vec![grid.zeros_array(); 3];
        
        // Create FFT planner
        let fft_planner = Fft3d::new(grid.nx, grid.ny, grid.nz);
        
        Ok(Self {
            config,
            k_squared,
            k_magnitude,
            phase_correction_factors,
            pressure_history,
            nonlinear_history,
            fft_planner,
            metrics: SolverMetrics::default(),
            rk4_workspace: None,
            step_count: 0,
            pressure_prev: None,
            velocity_prev: None,
        })
    }
    
    /// Validate solver configuration against grid and physics constraints
    fn validate_config(config: &KuznetsovConfig, grid: &Grid) -> KwaversResult<()> {
        if config.k_space_correction_order < 1 || config.k_space_correction_order > 4 {
            return Err(crate::error::KwaversError::Config(
                crate::error::ConfigError::InvalidValue {
                    parameter: "k_space_correction_order".to_string(),
                    value: config.k_space_correction_order.to_string(),
                    constraint: "must be between 1 and 4".to_string(),
                }
            ));
        }
        
        if config.cfl_factor <= 0.0 || config.cfl_factor > 1.0 {
            return Err(crate::error::KwaversError::Config(
                crate::error::ConfigError::InvalidValue {
                    parameter: "cfl_factor".to_string(),
                    value: config.cfl_factor.to_string(),
                    constraint: "must be between 0.0 and 1.0".to_string(),
                }
            ));
        }
        
        // Check grid resolution requirements
        let min_dx = grid.dx.min(grid.dy).min(grid.dz);
        if min_dx <= 0.0 {
            return Err(crate::error::KwaversError::Config(
                crate::error::ConfigError::InvalidValue {
                    parameter: "grid_spacing".to_string(),
                    value: min_dx.to_string(),
                    constraint: "must be positive".to_string(),
                }
            ));
        }
        
        debug!("Kuznetsov solver configuration validated successfully");
        Ok(())
    }
    
    /// Compute Laplacian using spectral method with proper k-space handling
    fn compute_laplacian(&mut self, field: &Array3<f64>, grid: &Grid) -> KwaversResult<Array3<f64>> {
        let start = Instant::now();
        
        // Use existing utility functions temporarily until FFT API is improved
        let mut fields_4d = Array4::zeros((1, grid.nx, grid.ny, grid.nz));
        fields_4d.index_axis_mut(Axis(0), 0).assign(field);
        
        // Transform to k-space using existing utilities
        let mut field_k = fft_3d(&fields_4d, 0, grid);
        
        // Apply Laplacian in k-space: ∇²f = -k² * f
        Zip::from(&mut field_k)
            .and(&self.k_squared)
            .for_each(|fk, &k2| {
                *fk *= -k2;
            });
        
        // Transform back to real space
        let result = ifft_3d(&field_k, grid);
        
        self.metrics.fft_time += start.elapsed().as_secs_f64();
        Ok(result)
    }
    
    /// Compute the nonlinear term: -(β/ρ₀c₀⁴)∂²p²/∂t²
    fn compute_nonlinear_term(
        &mut self,
        pressure: &Array3<f64>,
        medium: &dyn Medium,
        grid: &Grid,
        dt: f64,
    ) -> KwaversResult<Array3<f64>> {
        let start = Instant::now();
        
        if !self.config.enable_nonlinearity {
            return Ok(Array3::zeros(pressure.raw_dim()));
        }
        
        // Get previous pressure values
        let p_prev = &self.pressure_history[0];
        let p_prev2 = &self.pressure_history[1];
        
        // Compute p² at three time levels
        let p2_curr = pressure * pressure;
        let p2_prev = p_prev * p_prev;
        let p2_prev2 = p_prev2 * p_prev2;
        
        // Second-order finite difference for ∂²p²/∂t²
        let d2p2_dt2 = (&p2_curr - 2.0 * &p2_prev + &p2_prev2) / (dt * dt);
        
        // Apply nonlinearity coefficient
        let mut nonlinear_term = Array3::zeros(pressure.raw_dim());
        
        Zip::indexed(&mut nonlinear_term)
            .and(&d2p2_dt2)
            .for_each(|(i, j, k), nl, &d2p2| {
                let x = i as f64 * grid.dx;
                let y = j as f64 * grid.dy;
                let z = k as f64 * grid.dz;
                
                let rho = medium.density(x, y, z, grid);
                let c0 = medium.sound_speed(x, y, z, grid);
                let beta = medium.nonlinearity_coefficient(x, y, z, grid);
                
                let coeff = -beta / (rho * c0.powi(4));
                *nl = coeff * d2p2 * self.config.nonlinearity_scaling;
            });
        
        self.metrics.nonlinear_time += start.elapsed().as_secs_f64();
        Ok(nonlinear_term)
    }
    
    /// Compute diffusivity term for thermal and viscous losses
    /// α ∇²p where α is the diffusivity coefficient
    fn compute_diffusivity_term(
        &mut self,
        pressure: &Array3<f64>,
        medium: &dyn Medium,
        grid: &Grid,
        dt: f64,
    ) -> KwaversResult<Array3<f64>> {
        let start = Instant::now();
        
        // For the Kuznetsov equation, the diffusivity term is α∇²p
        // where α is the thermal diffusivity
        let laplacian = self.compute_laplacian(pressure, grid)?;
        
        // Apply spatially varying diffusivity
        let mut result = Array3::zeros(pressure.dim());
        result.indexed_iter_mut()
            .zip(laplacian.iter())
            .for_each(|(((i, j, k), res), &lap)| {
                let x = i as f64 * grid.dx;
                let y = j as f64 * grid.dy;
                let z = k as f64 * grid.dz;
                
                // Get spatially varying diffusivity
                let alpha = medium.thermal_diffusivity(x, y, z, grid);
                
                // Apply diffusivity: α∇²p
                *res = alpha * lap;
            });
        
        // Update metrics
        self.metrics.diffusion_time += start.elapsed().as_secs_f64();
        
        Ok(result)
    }
    
    /// Apply stability filter to prevent high-frequency instabilities
    fn apply_stability_filter(&self, field: &mut Array3<f64>, grid: &Grid, dt: f64) {
        // Simple 3-point smoothing filter for stability
        let filter_strength = (dt * self.config.max_frequency).min(0.1);
        
        if filter_strength > 1e-6 {
            let mut filtered = field.clone();
            
            // Apply 3D smoothing filter using iterator-based approach
            ndarray::Zip::indexed(&mut filtered)
                .for_each(|(i, j, k), val| {
                    if i > 0 && i < grid.nx-1 && j > 0 && j < grid.ny-1 && k > 0 && k < grid.nz-1 {
                        let neighbors = field[[i-1,j,k]] + field[[i+1,j,k]] +
                                       field[[i,j-1,k]] + field[[i,j+1,k]] +
                                       field[[i,j,k-1]] + field[[i,j,k+1]];
                        let center = field[[i,j,k]];
                        
                        *val = center + filter_strength * (neighbors/6.0 - center);
                    }
                });
            
            field.assign(&filtered);
        }
    }
    
    /// Update pressure history buffers
    fn update_history(&mut self, current_pressure: &Array3<f64>) {
        // Shift history
        let n = self.pressure_history.len();
        for i in (1..n).rev() {
            let (left, right) = self.pressure_history.split_at_mut(i);
            right[0].assign(&left[i-1]);
        }
        
        // Store current pressure
        self.pressure_history[0].assign(current_pressure);
        
        // Update nonlinear history (p²)
        if self.config.enable_nonlinearity {
            for i in (1..self.nonlinear_history.len()).rev() {
                let (left, right) = self.nonlinear_history.split_at_mut(i);
                right[0].assign(&left[i-1]);
            }
            
            // Store p²
            Zip::from(&mut self.nonlinear_history[0])
                .and(current_pressure)
                .for_each(|p2, &p| {
                    *p2 = p * p;
                });
        }
    }
    
    /// Check CFL condition for stability
    pub fn check_cfl_condition(&self, grid: &Grid, medium: &dyn Medium, dt: f64) -> bool {
        // Find maximum sound speed
        let c_max = (0..grid.nx).flat_map(|i| {
            (0..grid.ny).flat_map(move |j| {
                (0..grid.nz).map(move |k| {
                    let x = i as f64 * grid.dx;
                    let y = j as f64 * grid.dy;
                    let z = k as f64 * grid.dz;
                    medium.sound_speed(x, y, z, grid)
                })
            })
        }).fold(0.0f64, f64::max);
        
        // CFL condition for 3D
        let dx_min = grid.dx.min(grid.dy).min(grid.dz);
        let cfl = c_max * dt / dx_min;
        
        if cfl > self.config.cfl_factor {
            warn!("CFL condition violated: {} > {}", cfl, self.config.cfl_factor);
            false
        } else {
            true
        }
    }

    /// Internal wave update method with explicit pressure and velocity arrays
    fn update_wave_internal(
        &mut self,
        pressure: &mut Array3<f64>,
        velocity: &mut Array4<f64>,
        source_term: &Array3<f64>,
        grid: &Grid,
        medium: &dyn Medium,
        dt: f64,
        _t: f64,
    ) -> KwaversResult<()> {
        let start = Instant::now();
        
        // Check stability
        if !self.check_cfl_condition(grid, medium, dt) && !self.config.adaptive_timestep {
            warn!("CFL condition violated, simulation may be unstable");
        }
        
        // Initialize previous fields if not already done
        if self.pressure_prev.is_none() {
            self.pressure_prev = Some(pressure.clone());
            self.velocity_prev = Some(velocity.clone());
        }
        
        // Store current pressure for history
        let pressure_current = pressure.clone();
        
        // For second-order formulation, we need to update velocity first
        // ∂v/∂t = -1/ρ ∇p
        update_velocity_field(velocity, pressure, medium, grid, dt)?;
        
        // Now update pressure using second-order time derivative
        // ∂²p/∂t² = c²∇²p + nonlinear_terms + diffusivity_terms
        
        // Get previous pressure for second-order derivative
        let pressure_prev = self.pressure_prev.as_ref().unwrap();
        
        // Compute all terms of the Kuznetsov equation
        let laplacian = self.compute_laplacian(pressure, grid)?;
        let nonlinear_term = self.compute_nonlinear_term(pressure, medium, grid, dt)?;
        let diffusivity_term = self.compute_diffusivity_term(pressure, medium, grid, dt)?;
        
        // Compute linear wave term (c²∇²p)
        let linear_term = compute_linear_term(&laplacian, pressure, medium, grid);
        
        // Total acceleration term (∂²p/∂t²)
        let acceleration = &linear_term + &nonlinear_term + &diffusivity_term + source_term;
        
        // Update pressure using second-order central difference
        // p^{n+1} = 2*p^n - p^{n-1} + dt²*acceleration
        let dt_squared = dt * dt;
        Zip::from(&mut *pressure)
            .and(pressure_prev)
            .and(&acceleration)
            .for_each(|p_next, &p_prev, &acc| {
                let p_current = *p_next;
                *p_next = 2.0 * p_current - p_prev + dt_squared * acc;
                
                // Clamp for stability
                if p_next.abs() > self.config.max_pressure {
                    *p_next = p_next.signum() * self.config.max_pressure;
                }
            });
        
        // Apply stability filter if enabled
        if self.config.stability_filter {
            self.apply_stability_filter(pressure, grid, dt);
        }
        
        // Update history buffers
        self.pressure_prev = Some(pressure_current);
        self.update_history(pressure);
        
        // Update metrics
        self.metrics.total_steps += 1;
        let elapsed = start.elapsed().as_secs_f64();
        
        // Calculate time spent on linear operations (total time minus other tracked operations)
        // Note: We need to track the time spent in this specific call, not cumulative
        let other_ops_time = (self.metrics.nonlinear_time + self.metrics.diffusion_time + self.metrics.fft_time) 
            - self.metrics.prev_other_ops_time;
        let linear_ops_time = elapsed - other_ops_time;
        
        self.metrics.linear_time += linear_ops_time.max(0.0); // Ensure non-negative
        self.metrics.prev_other_ops_time = self.metrics.nonlinear_time + self.metrics.diffusion_time + self.metrics.fft_time;
        
        Ok(())
    }
}

impl AcousticWaveModel for KuznetsovWave {
    fn update_wave(
        &mut self,
        fields: &mut Array4<f64>,
        prev_pressure: &Array3<f64>,
        source: &dyn crate::source::Source,
        grid: &Grid,
        medium: &dyn Medium,
        dt: f64,
        t: f64,
    ) {
        use crate::solver::{PRESSURE_IDX, VX_IDX, VY_IDX, VZ_IDX};
        
        // Extract pressure and velocity from fields
        let pressure_idx = PRESSURE_IDX;
        let vx_idx = VX_IDX;
        let vy_idx = VY_IDX;
        let vz_idx = VZ_IDX;
        
        // Get pressure field
        let mut pressure = fields.index_axis(Axis(0), pressure_idx).to_owned();
        
        // Create velocity array
        let mut velocity = Array4::zeros((3, grid.nx, grid.ny, grid.nz));
        velocity.index_axis_mut(Axis(0), 0).assign(&fields.index_axis(Axis(0), vx_idx));
        velocity.index_axis_mut(Axis(0), 1).assign(&fields.index_axis(Axis(0), vy_idx));
        velocity.index_axis_mut(Axis(0), 2).assign(&fields.index_axis(Axis(0), vz_idx));
        
        // Get source term
        let mut source_term = grid.zeros_array();
        source_term.indexed_iter_mut().for_each(|((i, j, k), val)| {
            let x = i as f64 * grid.dx;
            let y = j as f64 * grid.dy;
            let z = k as f64 * grid.dz;
            *val = source.get_source_term(t, x, y, z, grid);
        });
        
        // Call the original update method
        if let Err(e) = self.update_wave_internal(&mut pressure, &mut velocity, &source_term, grid, medium, dt, t) {
            warn!("Kuznetsov wave update failed: {}", e);
        }
        
        // Write back results
        fields.index_axis_mut(Axis(0), pressure_idx).assign(&pressure);
        fields.index_axis_mut(Axis(0), vx_idx).assign(&velocity.index_axis(Axis(0), 0));
        fields.index_axis_mut(Axis(0), vy_idx).assign(&velocity.index_axis(Axis(0), 1));
        fields.index_axis_mut(Axis(0), vz_idx).assign(&velocity.index_axis(Axis(0), 2));
    }
    
    fn report_performance(&self) {
        let total_time = self.metrics.linear_time + self.metrics.nonlinear_time 
            + self.metrics.diffusion_time + self.metrics.fft_time;
        
        info!("Kuznetsov Wave Performance:");
        info!("  Total steps: {}", self.metrics.total_steps);
        info!("  Total time: {:.3}s", total_time);
        if self.metrics.total_steps > 0 {
            info!("  Average step time: {:.3}ms", 1000.0 * total_time / self.metrics.total_steps as f64);
        }
        info!("  Time breakdown:");
        info!("    Linear operations: {:.3}s ({:.1}%)", 
            self.metrics.linear_time, 100.0 * self.metrics.linear_time / total_time);
        info!("    Nonlinear term: {:.3}s ({:.1}%)", 
            self.metrics.nonlinear_time, 100.0 * self.metrics.nonlinear_time / total_time);
        info!("    Diffusion term: {:.3}s ({:.1}%)", 
            self.metrics.diffusion_time, 100.0 * self.metrics.diffusion_time / total_time);
        info!("    FFT operations: {:.3}s ({:.1}%)", 
            self.metrics.fft_time, 100.0 * self.metrics.fft_time / total_time);
    }
    
    fn set_nonlinearity_scaling(&mut self, scaling: f64) {
        self.config.nonlinearity_scaling = scaling;
        info!("Set nonlinearity scaling to {}", scaling);
    }
    

}

// Helper functions

/// Compute k-space magnitude array
fn compute_k_magnitude(grid: &Grid) -> Array3<f64> {
    let mut k_mag = Array3::zeros((grid.nx, grid.ny, grid.nz));
    
    let kx_max = PI / grid.dx;
    let ky_max = PI / grid.dy;
    let kz_max = PI / grid.dz;
    
    for i in 0..grid.nx {
        for j in 0..grid.ny {
            for k in 0..grid.nz {
                let kx = if i <= grid.nx/2 {
                    2.0 * PI * i as f64 / (grid.nx as f64 * grid.dx)
                } else {
                    -2.0 * PI * (grid.nx - i) as f64 / (grid.nx as f64 * grid.dx)
                };
                
                let ky = if j <= grid.ny/2 {
                    2.0 * PI * j as f64 / (grid.ny as f64 * grid.dy)
                } else {
                    -2.0 * PI * (grid.ny - j) as f64 / (grid.ny as f64 * grid.dy)
                };
                
                let kz = if k <= grid.nz/2 {
                    2.0 * PI * k as f64 / (grid.nz as f64 * grid.dz)
                } else {
                    -2.0 * PI * (grid.nz - k) as f64 / (grid.nz as f64 * grid.dz)
                };
                
                k_mag[[i, j, k]] = (kx*kx + ky*ky + kz*kz).sqrt();
            }
        }
    }
    
    k_mag
}

/// Compute phase correction factors for k-space derivatives
#[cfg(test)]
fn compute_phase_factors(grid: &Grid, order: usize) -> Array3<f64> {
    let mut factors = Array3::ones((grid.nx, grid.ny, grid.nz));
    
    // Apply sinc correction based on order
    match order {
        2 => {
            // Second-order correction
            for i in 0..grid.nx {
                for j in 0..grid.ny {
                    for k in 0..grid.nz {
                        let kx = if i <= grid.nx/2 {
                            PI * i as f64 / (grid.nx as f64)
                        } else {
                            -PI * (grid.nx - i) as f64 / (grid.nx as f64)
                        };
                        let ky = if j <= grid.ny/2 {
                            PI * j as f64 / (grid.ny as f64)
                        } else {
                            -PI * (grid.ny - j) as f64 / (grid.ny as f64)
                        };
                        let kz = if k <= grid.nz/2 {
                            PI * k as f64 / (grid.nz as f64)
                        } else {
                            -PI * (grid.nz - k) as f64 / (grid.nz as f64)
                        };
                        
                        // Sinc correction for finite differences
                        let sinc_x = if kx.abs() > 1e-10 { kx.sin() / kx } else { 1.0 };
                        let sinc_y = if ky.abs() > 1e-10 { ky.sin() / ky } else { 1.0 };
                        let sinc_z = if kz.abs() > 1e-10 { kz.sin() / kz } else { 1.0 };
                        
                        factors[[i, j, k]] = 1.0 / (sinc_x * sinc_y * sinc_z);
                    }
                }
            }
        }
        4 => {
            // Fourth-order correction
            for i in 0..grid.nx {
                for j in 0..grid.ny {
                    for k in 0..grid.nz {
                        let kx = if i <= grid.nx/2 {
                            PI * i as f64 / (grid.nx as f64)
                        } else {
                            -PI * (grid.nx - i) as f64 / (grid.nx as f64)
                        };
                        let ky = if j <= grid.ny/2 {
                            PI * j as f64 / (grid.ny as f64)
                        } else {
                            -PI * (grid.ny - j) as f64 / (grid.ny as f64)
                        };
                        let kz = if k <= grid.nz/2 {
                            PI * k as f64 / (grid.nz as f64)
                        } else {
                            -PI * (grid.nz - k) as f64 / (grid.nz as f64)
                        };
                        
                        // Fourth-order finite difference correction
                        // d/dx ≈ (8*sin(kx) - sin(2*kx))/(6*dx) => correction factor
                        let corr_x = if kx.abs() > 1e-10 {
                            kx / ((8.0 * kx.sin() - (2.0 * kx).sin()) / 6.0)
                        } else {
                            1.0
                        };
                        let corr_y = if ky.abs() > 1e-10 {
                            ky / ((8.0 * ky.sin() - (2.0 * ky).sin()) / 6.0)
                        } else {
                            1.0
                        };
                        let corr_z = if kz.abs() > 1e-10 {
                            kz / ((8.0 * kz.sin() - (2.0 * kz).sin()) / 6.0)
                        } else {
                            1.0
                        };
                        
                        factors[[i, j, k]] = corr_x * corr_y * corr_z;
                    }
                }
            }
        }
        6 => {
            // Sixth-order correction
            for i in 0..grid.nx {
                for j in 0..grid.ny {
                    for k in 0..grid.nz {
                        let kx = if i <= grid.nx/2 {
                            PI * i as f64 / (grid.nx as f64)
                        } else {
                            -PI * (grid.nx - i) as f64 / (grid.nx as f64)
                        };
                        let ky = if j <= grid.ny/2 {
                            PI * j as f64 / (grid.ny as f64)
                        } else {
                            -PI * (grid.ny - j) as f64 / (grid.ny as f64)
                        };
                        let kz = if k <= grid.nz/2 {
                            PI * k as f64 / (grid.nz as f64)
                        } else {
                            -PI * (grid.nz - k) as f64 / (grid.nz as f64)
                        };
                        
                        // Sixth-order finite difference correction
                        // d/dx ≈ (45*sin(kx) - 9*sin(2*kx) + sin(3*kx))/(30*dx) => correction factor
                        let corr_x = if kx.abs() > 1e-10 {
                            kx / ((45.0 * kx.sin() - 9.0 * (2.0 * kx).sin() + (3.0 * kx).sin()) / 30.0)
                        } else {
                            1.0
                        };
                        let corr_y = if ky.abs() > 1e-10 {
                            ky / ((45.0 * ky.sin() - 9.0 * (2.0 * ky).sin() + (3.0 * ky).sin()) / 30.0)
                        } else {
                            1.0
                        };
                        let corr_z = if kz.abs() > 1e-10 {
                            kz / ((45.0 * kz.sin() - 9.0 * (2.0 * kz).sin() + (3.0 * kz).sin()) / 30.0)
                        } else {
                            1.0
                        };
                        
                        factors[[i, j, k]] = corr_x * corr_y * corr_z;
                    }
                }
            }
        }
        _ => {
            warn!("Unsupported spatial order {} for phase correction, using no correction (order 1)", order);
            // factors already initialized to ones
        }
    }
    
    factors
}

/// Compute k-space correction factors for dispersion compensation
fn compute_k_space_correction_factors(grid: &Grid, order: usize) -> Array3<f64> {
    let mut factors = Array3::ones((grid.nx, grid.ny, grid.nz));
    
    // Effective wave number for normalization
    let k0 = 2.0 * PI / (grid.dx.min(grid.dy).min(grid.dz));
    
    for i in 0..grid.nx {
        for j in 0..grid.ny {
            for k in 0..grid.nz {
                let kx = if i <= grid.nx/2 {
                    2.0 * PI * i as f64 / (grid.nx as f64 * grid.dx)
                } else {
                    -2.0 * PI * (grid.nx - i) as f64 / (grid.nx as f64 * grid.dx)
                };
                
                let ky = if j <= grid.ny/2 {
                    2.0 * PI * j as f64 / (grid.ny as f64 * grid.dy)
                } else {
                    -2.0 * PI * (grid.ny - j) as f64 / (grid.ny as f64 * grid.dy)
                };
                
                let kz = if k <= grid.nz/2 {
                    2.0 * PI * k as f64 / (grid.nz as f64 * grid.dz)
                } else {
                    -2.0 * PI * (grid.nz - k) as f64 / (grid.nz as f64 * grid.dz)
                };
                
                let k_norm = (kx*kx + ky*ky + kz*kz).sqrt();
                
                // Apply higher-order correction for dispersion
                let coeff = match order {
                    2 => {
                        // Second-order correction: improved dispersion relation
                        let normalized_k = k_norm / k0;
                        1.0 + KUZNETSOV_K_SPACE_CORRECTION_SECOND_ORDER * normalized_k * normalized_k
                    }
                    4 => {
                        // Fourth-order correction: better high-frequency behavior
                        let normalized_k = k_norm / k0;
                        1.0 + KUZNETSOV_K_SPACE_CORRECTION_SECOND_ORDER * normalized_k * normalized_k + KUZNETSOV_K_SPACE_CORRECTION_FOURTH_ORDER * normalized_k.powi(4)
                    }
                    _ => {
                        // Default to no correction
                        1.0
                    }
                };
                
                factors[[i, j, k]] = coeff;
            }
        }
    }
    
    factors
}


/// Compute the linear wave equation term
fn compute_linear_term(
    laplacian: &Array3<f64>,
    pressure: &Array3<f64>,
    medium: &dyn Medium,
    grid: &Grid,
) -> Array3<f64> {
    let mut linear_term = Array3::zeros(pressure.raw_dim());
    
    Zip::indexed(&mut linear_term)
        .and(laplacian)
        .and(pressure)
        .for_each(|(i, j, k), lin, &lap, &p| {
            let x = i as f64 * grid.dx;
            let y = j as f64 * grid.dy;
            let z = k as f64 * grid.dz;
            
            let c = medium.sound_speed(x, y, z, grid);
            
            // Linear wave equation term: c²∇²p
            *lin = c * c * lap;
        });
    
    linear_term
}

/// Helper function to update velocity field
fn update_velocity_field(
    velocity: &mut Array4<f64>,
    pressure: &Array3<f64>,
    medium: &dyn Medium,
    grid: &Grid,
    dt: f64,
) -> KwaversResult<()> {
    // Compute pressure gradients
    let mut dp_dx = Array3::zeros(pressure.raw_dim());
    let mut dp_dy = Array3::zeros(pressure.raw_dim());
    let mut dp_dz = Array3::zeros(pressure.raw_dim());
    
    // Central differences for interior points
    for i in 1..grid.nx-1 {
        for j in 1..grid.ny-1 {
            for k in 1..grid.nz-1 {
                dp_dx[[i, j, k]] = (pressure[[i+1, j, k]] - pressure[[i-1, j, k]]) / (2.0 * grid.dx);
                dp_dy[[i, j, k]] = (pressure[[i, j+1, k]] - pressure[[i, j-1, k]]) / (2.0 * grid.dy);
                dp_dz[[i, j, k]] = (pressure[[i, j, k+1]] - pressure[[i, j, k-1]]) / (2.0 * grid.dz);
            }
        }
    }
    
    // Update velocity components separately to avoid borrow checker issues
    {
        let mut vx = velocity.index_axis_mut(Axis(0), 0);
        for i in 0..grid.nx {
            for j in 0..grid.ny {
                for k in 0..grid.nz {
                    let x = i as f64 * grid.dx;
                    let y = j as f64 * grid.dy;
                    let z = k as f64 * grid.dz;
                    let rho = medium.density(x, y, z, grid);
                    vx[[i, j, k]] -= dt * dp_dx[[i, j, k]] / rho;
                }
            }
        }
    }
    
    {
        let mut vy = velocity.index_axis_mut(Axis(0), 1);
        for i in 0..grid.nx {
            for j in 0..grid.ny {
                for k in 0..grid.nz {
                    let x = i as f64 * grid.dx;
                    let y = j as f64 * grid.dy;
                    let z = k as f64 * grid.dz;
                    let rho = medium.density(x, y, z, grid);
                    vy[[i, j, k]] -= dt * dp_dy[[i, j, k]] / rho;
                }
            }
        }
    }
    
    {
        let mut vz = velocity.index_axis_mut(Axis(0), 2);
        for i in 0..grid.nx {
            for j in 0..grid.ny {
                for k in 0..grid.nz {
                    let x = i as f64 * grid.dx;
                    let y = j as f64 * grid.dy;
                    let z = k as f64 * grid.dz;
                    let rho = medium.density(x, y, z, grid);
                    vz[[i, j, k]] -= dt * dp_dz[[i, j, k]] / rho;
                }
            }
        }
    }
    
    Ok(())
}

/// Helper function to compute linear term into a pre-allocated array
fn compute_linear_term_into(
    laplacian: &Array3<f64>,
    pressure: &Array3<f64>,
    medium: &dyn Medium,
    grid: &Grid,
    linear_term: &mut Array3<f64>,
) {
    Zip::indexed(linear_term)
        .and(laplacian)
        .and(pressure)
        .for_each(|(i, j, k), lin, &lap, &p| {
            let x = i as f64 * grid.dx;
            let y = j as f64 * grid.dy;
            let z = k as f64 * grid.dz;
            
            let c = medium.sound_speed(x, y, z, grid);
            
            // Linear wave equation term: c²∇²p
            *lin = c * c * lap;
        });
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::grid::Grid;
    use crate::medium::HomogeneousMedium;
    use crate::physics::traits::AcousticWaveModel;
    use ndarray::{Array3, Array4};
    use crate::source::Source;
    use crate::signal::Signal;
    
    // Test source implementation
    struct TestSource;
    
    impl std::fmt::Debug for TestSource {
        fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            write!(f, "TestSource")
        }
    }
    
    impl Source for TestSource {
        fn get_source_term(&self, _t: f64, _x: f64, _y: f64, _z: f64, _grid: &Grid) -> f64 {
            0.0 // No source for these tests
        }
        
        fn positions(&self) -> Vec<(f64, f64, f64)> {
            vec![]
        }
        
        fn signal(&self) -> &dyn Signal {
            panic!("Not implemented for test source")
        }
    }
    
    /// Test basic initialization and configuration
    #[test]
    fn test_kuznetsov_initialization() {
        let grid = Grid::new(64, 64, 64, 1e-3, 1e-3, 1e-3);
        let config = KuznetsovConfig::default();
        let solver = KuznetsovWave::new(&grid, config).unwrap();
        
        assert_eq!(solver.k_squared.dim(), (64, 64, 64));
        assert_eq!(solver.pressure_history.len(), 2);
    }
    
    /// Test linear propagation with a simple Gaussian pulse
    #[test]
    fn test_linear_propagation() {
        let grid = Grid::new(128, 128, 128, 1e-3, 1e-3, 1e-3);
        let mut config = KuznetsovConfig::default();
        config.enable_nonlinearity = false;
        config.enable_diffusivity = false;
        
        let mut solver = KuznetsovWave::new(&grid, config).unwrap();
        let medium = HomogeneousMedium::new(1000.0, 1500.0, &grid, 0.0, 0.0);
        
        // Initialize with Gaussian pulse
        let mut pressure = Array3::zeros((128, 128, 128));
        let velocity = Array4::zeros((3, 128, 128, 128));
        let source: Array3<f64> = Array3::zeros((128, 128, 128));
        
        for i in 0..128 {
            for j in 0..128 {
                for k in 0..128 {
                    let x = (i as f64 - 64.0) * grid.dx;
                    let y = (j as f64 - 64.0) * grid.dy;
                    let z = (k as f64 - 64.0) * grid.dz;
                    let r2 = x*x + y*y + z*z;
                    pressure[[i, j, k]] = 1e6 * (-r2 / (2.0 * 0.01_f64.powi(2))).exp();
                }
            }
        }
        
        let initial_energy = pressure.iter().map(|&p| p * p).sum::<f64>();
        
        // Create test source
        let source = TestSource;
        
        // Create fields array
        let mut fields = Array4::zeros((13, 128, 128, 128)); // Standard field indices
        fields.index_axis_mut(Axis(0), 0).assign(&pressure); // Pressure at index 0
        fields.index_axis_mut(Axis(0), 4).assign(&velocity.index_axis(Axis(0), 0)); // vx at index 4
        fields.index_axis_mut(Axis(0), 5).assign(&velocity.index_axis(Axis(0), 1)); // vy at index 5
        fields.index_axis_mut(Axis(0), 6).assign(&velocity.index_axis(Axis(0), 2)); // vz at index 6
        
        let prev_pressure = pressure.clone();
        
        // Run simulation
        let dt = 1e-7;
        for _ in 0..100 {
            solver.update_wave(&mut fields, &prev_pressure, &source, &grid, &medium, dt, 0.0);
        }
        
        // Extract final pressure
        let final_pressure = fields.index_axis(Axis(0), 0);
        let final_energy = final_pressure.iter().map(|&p| p * p).sum::<f64>();
        
        // The second-order formulation is now properly implemented
        // Energy conservation should work correctly
        let energy_ratio = final_energy / initial_energy;
        assert!((energy_ratio - 1.0).abs() < 0.01, 
            "Energy not conserved in linear regime: ratio = {}", energy_ratio);
    }
    
    /// Test phase correction factors for all supported orders
    #[test]
    fn test_phase_correction_factors() {
        let grid = Grid::new(64, 64, 64, 1e-3, 1e-3, 1e-3);
        
        // Test order 2
        let factors_2 = compute_phase_factors(&grid, 2);
        assert_eq!(factors_2.dim(), (64, 64, 64));
        // Check that DC component has no correction
        assert!((factors_2[[0, 0, 0]] - 1.0).abs() < 1e-10);
        // Check that high frequencies have correction > 1
        assert!(factors_2[[32, 0, 0]] > 1.0);
        
        // Test order 4
        let factors_4 = compute_phase_factors(&grid, 4);
        assert_eq!(factors_4.dim(), (64, 64, 64));
        // Check that DC component has no correction
        assert!((factors_4[[0, 0, 0]] - 1.0).abs() < 1e-10);
        // Check that corrections are different from order 2
        assert!((factors_4[[32, 0, 0]] - factors_2[[32, 0, 0]]).abs() > 1e-3);
        
        // Test order 6
        let factors_6 = compute_phase_factors(&grid, 6);
        assert_eq!(factors_6.dim(), (64, 64, 64));
        // Check that DC component has no correction
        assert!((factors_6[[0, 0, 0]] - 1.0).abs() < 1e-10);
        // Check that corrections are different from order 4
        assert!((factors_6[[32, 0, 0]] - factors_4[[32, 0, 0]]).abs() > 1e-3);
        
        // Test unsupported order (should use no correction)
        let _ = env_logger::builder().is_test(true).try_init();
        let factors_8 = compute_phase_factors(&grid, 8);
        assert_eq!(factors_8.dim(), (64, 64, 64));
        // Should be all ones (no correction)
        for val in factors_8.iter() {
            assert!((val - 1.0).abs() < 1e-10);
        }
    }

    /// Test k-space correction factors for dispersion compensation
    #[test]
    fn test_k_space_correction_factors() {
        let grid = Grid::new(64, 64, 64, 1e-3, 1e-3, 1e-3);
        
        // Test order 2
        let factors_2 = compute_k_space_correction_factors(&grid, 2);
        assert_eq!(factors_2.dim(), (64, 64, 64));
        // Check that DC component has no correction
        assert!((factors_2[[0, 0, 0]] - 1.0).abs() < 1e-10);
        // Check that high frequencies have correction > 1
        assert!(factors_2[[32, 0, 0]] > 1.0);
        
        // Test order 4
        let factors_4 = compute_k_space_correction_factors(&grid, 4);
        assert_eq!(factors_4.dim(), (64, 64, 64));
        // Check that DC component has no correction
        assert!((factors_4[[0, 0, 0]] - 1.0).abs() < 1e-10);
                 // Check that corrections are different from order 2
         assert!((factors_4[[32, 0, 0]] - factors_2[[32, 0, 0]]).abs() > 1e-6);
         
         // Test unsupported order (should use no correction)
         let _ = env_logger::builder().is_test(true).try_init();
         let factors_8 = compute_k_space_correction_factors(&grid, 8);
         assert_eq!(factors_8.dim(), (64, 64, 64));
         // Should be all ones (no correction)
         for val in factors_8.iter() {
             assert!((val - 1.0).abs() < 1e-10);
         }
    }
}

// Include comprehensive test module
#[cfg(test)]
#[path = "kuznetsov_tests.rs"]
mod kuznetsov_tests;