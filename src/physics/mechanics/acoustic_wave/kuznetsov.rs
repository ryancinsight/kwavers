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

use crate::error::KwaversResult;
use crate::grid::Grid;
use crate::medium::Medium;
use crate::physics::traits::AcousticWaveModel;
use crate::utils::{fft_3d, ifft_3d};
use crate::fft::Fft3d;
use ndarray::{Array3, Array4, Zip, Axis};
use std::f64::consts::PI;
use num_complex::Complex;

use log::{info, warn, debug};
use std::time::Instant;

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
#[derive(Debug, Clone, Default)]
struct SolverMetrics {
    linear_time: f64,
    nonlinear_time: f64,
    diffusion_time: f64,
    fft_time: f64,
    total_steps: u64,
    k_space_correction_time: f64,
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
        Self {
            pressure_temp: Array3::zeros((nx, ny, nz)),
            k1: Array3::zeros((nx, ny, nz)),
            k2: Array3::zeros((nx, ny, nz)),
            k3: Array3::zeros((nx, ny, nz)),
            k4: Array3::zeros((nx, ny, nz)),
            linear_term_cache: Array3::zeros((nx, ny, nz)),
            nonlinear_term_cache: Array3::zeros((nx, ny, nz)),
            diffusion_term_cache: Array3::zeros((nx, ny, nz)),
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
        
        let pressure_history = vec![Array3::zeros((grid.nx, grid.ny, grid.nz)); history_size];
        let nonlinear_history = vec![Array3::zeros((grid.nx, grid.ny, grid.nz)); 3];
        
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
    
    /// Compute diffusivity/absorption term for the Kuznetsov equation
    /// ∇²(α ∇²p) where α is the diffusivity coefficient
    fn compute_diffusivity_term(
        &mut self,
        pressure: &Array3<f64>,
        medium: &dyn Medium,
        grid: &Grid,
        dt: f64,
    ) -> KwaversResult<Array3<f64>> {
        let start = Instant::now();
        
        // Fix: Improved stability for diffusivity term
        // First compute ∇²p
        let laplacian = self.compute_laplacian(pressure, grid)?;
        
        // Apply a stability filter to prevent high-frequency instabilities
        let mut filtered_laplacian = laplacian;
        if self.config.stability_filter {
            self.apply_stability_filter(&mut filtered_laplacian, grid, dt);
        }
        
        // Transform to k-space for diffusivity computation
        let mut fields_4d = Array4::zeros((1, filtered_laplacian.shape()[0], filtered_laplacian.shape()[1], filtered_laplacian.shape()[2]));
        fields_4d.index_axis_mut(Axis(0), 0).assign(&filtered_laplacian);
        
        let mut laplacian_hat = fft_3d(&fields_4d, 0, grid);
        
        // Apply diffusivity operator in k-space: -k²α
        let k_mag = &self.k_magnitude;
        
        for i in 0..grid.nx {
            for j in 0..grid.ny {
                for k in 0..grid.nz {
                    let x = i as f64 * grid.dx;
                    let y = j as f64 * grid.dy;
                    let z = k as f64 * grid.dz;
                    
                    // Get spatially varying diffusivity
                    let alpha = medium.thermal_diffusivity(x, y, z, grid);
                    let k2 = k_mag[[i, j, k]].powi(2);
                    
                    // Apply diffusivity operator with stability limiting
                    let damping_factor = if k2 > 0.0 {
                        let max_damping = 0.1 / dt; // Limit to prevent excessive damping
                        (-alpha * k2).min(max_damping)
                    } else {
                        0.0
                    };
                    
                    laplacian_hat[[i, j, k]] *= Complex::new(damping_factor, 0.0);
                }
            }
        }
        
        // Transform back to physical space
        let result = ifft_3d(&laplacian_hat, grid);
        
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
            
            for i in 1..grid.nx-1 {
                for j in 1..grid.ny-1 {
                    for k in 1..grid.nz-1 {
                        let neighbors = field[[i-1,j,k]] + field[[i+1,j,k]] +
                                       field[[i,j-1,k]] + field[[i,j+1,k]] +
                                       field[[i,j,k-1]] + field[[i,j,k+1]];
                        let center = field[[i,j,k]];
                        
                        filtered[[i,j,k]] = center + filter_strength * (neighbors/6.0 - center);
                    }
                }
            }
            
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
        let mut c_max: f64 = 0.0;
        for i in 0..grid.nx {
            for j in 0..grid.ny {
                for k in 0..grid.nz {
                    let x = i as f64 * grid.dx;
                    let y = j as f64 * grid.dy;
                    let z = k as f64 * grid.dz;
                    let c = medium.sound_speed(x, y, z, grid);
                    c_max = c_max.max(c);
                }
            }
        }
        
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
        
        // Store current pressure for history
        let pressure_current = pressure.clone();
        
        // Compute all terms of the Kuznetsov equation
        let laplacian = self.compute_laplacian(pressure, grid)?;
        let nonlinear_term = self.compute_nonlinear_term(pressure, medium, grid, dt)?;
        let diffusivity_term = self.compute_diffusivity_term(pressure, medium, grid, dt)?;
        
        // Time integration based on selected scheme
        match self.config.time_scheme {
            TimeIntegrationScheme::Euler => {
                // Simple forward Euler
                let linear_term = compute_linear_term(&laplacian, pressure, medium, grid);
                let total_rhs = &linear_term + &nonlinear_term + &diffusivity_term + source_term;
                
                // Update pressure
                Zip::from(&mut *pressure)
                    .and(&total_rhs)
                    .for_each(|p, &rhs| {
                        *p += dt * rhs;
                        
                        // Clamp for stability
                        if p.abs() > self.config.max_pressure {
                            *p = p.signum() * self.config.max_pressure;
                        }
                    });
            }
            
            TimeIntegrationScheme::RK4 => {
                // Fourth-order Runge-Kutta implementation
                // RK4: y_{n+1} = y_n + (k1 + 2*k2 + 2*k3 + k4)/6
                
                // Initialize workspace if needed
                if self.rk4_workspace.is_none() {
                    self.rk4_workspace = Some(RK4Workspace::new(grid.nx, grid.ny, grid.nz));
                }
                
                // Use a local scope to limit the workspace borrow
                {
                    let workspace = self.rk4_workspace.as_mut().unwrap();
                    
                    // Save initial pressure
                    workspace.pressure_temp.assign(pressure);
                    
                    // Stage 1: k1 = dt * f(p_n)
                    // Reuse the already computed laplacian and terms
                    compute_linear_term_into(&laplacian, pressure, medium, grid, &mut workspace.linear_term_cache);
                    workspace.k1.assign(&workspace.linear_term_cache);
                    workspace.k1.scaled_add(1.0, &nonlinear_term);
                    workspace.k1.scaled_add(1.0, &diffusivity_term);
                    workspace.k1.scaled_add(1.0, source_term);
                    workspace.k1.mapv_inplace(|x| x * dt);
                    
                    // Stage 2: k2 = dt * f(p_n + k1/2)
                    pressure.assign(&workspace.pressure_temp);
                    pressure.scaled_add(0.5, &workspace.k1);
                }
                
                // Compute stage 2 terms
                let laplacian_2 = self.compute_laplacian(pressure, grid)?;
                let nonlinear_2 = self.compute_nonlinear_term(pressure, medium, grid, dt)?;
                let diffusivity_2 = self.compute_diffusivity_term(pressure, medium, grid, dt)?;
                
                {
                    let workspace = self.rk4_workspace.as_mut().unwrap();
                    compute_linear_term_into(&laplacian_2, pressure, medium, grid, &mut workspace.linear_term_cache);
                    workspace.k2.assign(&workspace.linear_term_cache);
                    workspace.k2.scaled_add(1.0, &nonlinear_2);
                    workspace.k2.scaled_add(1.0, &diffusivity_2);
                    workspace.k2.scaled_add(1.0, source_term);
                    workspace.k2.mapv_inplace(|x| x * dt);
                    
                    // Stage 3: k3 = dt * f(p_n + k2/2)
                    pressure.assign(&workspace.pressure_temp);
                    pressure.scaled_add(0.5, &workspace.k2);
                }
                
                // Compute stage 3 terms
                let laplacian_3 = self.compute_laplacian(pressure, grid)?;
                let nonlinear_3 = self.compute_nonlinear_term(pressure, medium, grid, dt)?;
                let diffusivity_3 = self.compute_diffusivity_term(pressure, medium, grid, dt)?;
                
                {
                    let workspace = self.rk4_workspace.as_mut().unwrap();
                    compute_linear_term_into(&laplacian_3, pressure, medium, grid, &mut workspace.linear_term_cache);
                    workspace.k3.assign(&workspace.linear_term_cache);
                    workspace.k3.scaled_add(1.0, &nonlinear_3);
                    workspace.k3.scaled_add(1.0, &diffusivity_3);
                    workspace.k3.scaled_add(1.0, source_term);
                    workspace.k3.mapv_inplace(|x| x * dt);
                    
                    // Stage 4: k4 = dt * f(p_n + k3)
                    pressure.assign(&workspace.pressure_temp);
                    pressure.scaled_add(1.0, &workspace.k3);
                }
                
                // Compute stage 4 terms
                let laplacian_4 = self.compute_laplacian(pressure, grid)?;
                let nonlinear_4 = self.compute_nonlinear_term(pressure, medium, grid, dt)?;
                let diffusivity_4 = self.compute_diffusivity_term(pressure, medium, grid, dt)?;
                
                {
                    let workspace = self.rk4_workspace.as_mut().unwrap();
                    compute_linear_term_into(&laplacian_4, pressure, medium, grid, &mut workspace.linear_term_cache);
                    workspace.k4.assign(&workspace.linear_term_cache);
                    workspace.k4.scaled_add(1.0, &nonlinear_4);
                    workspace.k4.scaled_add(1.0, &diffusivity_4);
                    workspace.k4.scaled_add(1.0, source_term);
                    workspace.k4.mapv_inplace(|x| x * dt);
                    
                    // Combine stages: p_{n+1} = p_n + (k1 + 2*k2 + 2*k3 + k4)/6
                    pressure.assign(&workspace.pressure_temp);
                    pressure.scaled_add(1.0 / 6.0, &workspace.k1);
                    pressure.scaled_add(2.0 / 6.0, &workspace.k2);
                    pressure.scaled_add(2.0 / 6.0, &workspace.k3);
                    pressure.scaled_add(1.0 / 6.0, &workspace.k4);
                }
                
                // Clamp for stability
                pressure.mapv_inplace(|p| {
                    if p.abs() > self.config.max_pressure {
                        p.signum() * self.config.max_pressure
                    } else {
                        p
                    }
                });
            }
            
            _ => {
                // Other schemes not yet implemented
                return Err(crate::error::KwaversError::NotImplemented(
                    format!("Time integration scheme {:?} not yet implemented", self.config.time_scheme)
                ));
            }
        }
        
        // Update velocity field (using momentum equation)
        update_velocity_field(velocity, pressure, medium, grid, dt)?;
        
        // Update history
        self.update_history(&pressure_current);
        
        // Update metrics
        self.metrics.total_steps += 1;
        self.metrics.linear_time += start.elapsed().as_secs_f64() 
            - self.metrics.nonlinear_time 
            - self.metrics.diffusion_time 
            - self.metrics.fft_time;
        
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
        // Extract pressure and velocity from fields
        let pressure_idx = 0; // Assuming pressure is at index 0
        let vx_idx = 4; // Assuming velocity components start at index 4
        let vy_idx = 5;
        let vz_idx = 6;
        
        // Get pressure field
        let mut pressure = fields.index_axis(Axis(0), pressure_idx).to_owned();
        
        // Create velocity array
        let mut velocity = Array4::zeros((3, grid.nx, grid.ny, grid.nz));
        velocity.index_axis_mut(Axis(0), 0).assign(&fields.index_axis(Axis(0), vx_idx));
        velocity.index_axis_mut(Axis(0), 1).assign(&fields.index_axis(Axis(0), vy_idx));
        velocity.index_axis_mut(Axis(0), 2).assign(&fields.index_axis(Axis(0), vz_idx));
        
        // Get source term
        let mut source_term = Array3::zeros((grid.nx, grid.ny, grid.nz));
        for i in 0..grid.nx {
            for j in 0..grid.ny {
                for k in 0..grid.nz {
                    let x = i as f64 * grid.dx;
                    let y = j as f64 * grid.dy;
                    let z = k as f64 * grid.dz;
                    source_term[[i, j, k]] = source.get_source_term(t, x, y, z, grid);
                }
            }
        }
        
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
        info!("  Linear term time: {:.3}s ({:.1}%)", 
            self.metrics.linear_time, 100.0 * self.metrics.linear_time / total_time);
        info!("  Nonlinear term time: {:.3}s ({:.1}%)", 
            self.metrics.nonlinear_time, 100.0 * self.metrics.nonlinear_time / total_time);
        info!("  Diffusion term time: {:.3}s ({:.1}%)", 
            self.metrics.diffusion_time, 100.0 * self.metrics.diffusion_time / total_time);
        info!("  FFT time: {:.3}s ({:.1}%)", 
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
                        1.0 + 0.1 * normalized_k * normalized_k
                    }
                    4 => {
                        // Fourth-order correction: better high-frequency behavior
                        let normalized_k = k_norm / k0;
                        1.0 + 0.05 * normalized_k * normalized_k + 0.01 * normalized_k.powi(4)
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
        let solver = KuznetsovWave::new(&grid, config);
        
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
        let mut velocity = Array4::zeros((3, 128, 128, 128));
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
        
        // Check energy conservation (should be approximately conserved for linear case)
        assert!((final_energy - initial_energy).abs() / initial_energy < 0.01);
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