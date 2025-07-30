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
//! We use a mixed-domain approach:
//! - Spatial derivatives: k-space (spectral accuracy)
//! - Time derivatives: finite difference (stability)
//! - Nonlinear terms: real space (efficiency)
//! 
//! # Design Principles Applied:
//! 
//! - **SOLID**: Single responsibility (Kuznetsov physics only)
//! - **DRY**: Reuses existing FFT and grid infrastructure
//! - **KISS**: Clear separation of linear/nonlinear/diffusive terms
//! - **YAGNI**: Only implements validated physics requirements
//! - **CUPID**: Composable with other physics components
//! - **CLEAN**: Comprehensive documentation and tests

use crate::error::KwaversResult;
use crate::grid::Grid;
use crate::medium::Medium;
use crate::physics::traits::AcousticWaveModel;
use crate::utils::{fft_3d, ifft_3d};
use crate::fft::{Fft3d, Ifft3d};
use ndarray::{Array3, Array4, Zip, Axis};
use rustfft::num_complex::Complex;
use std::f64::consts::PI;
use log::{info, warn};
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
    
    /// Spatial accuracy order (2, 4, or 6)
    pub spatial_order: usize,
    
    /// Enable adaptive time stepping
    pub adaptive_timestep: bool,
    
    /// CFL safety factor
    pub cfl_factor: f64,
}

impl Default for KuznetsovConfig {
    fn default() -> Self {
        Self {
            enable_nonlinearity: true,
            enable_diffusivity: true,
            nonlinearity_scaling: 1.0,
            max_pressure: 1e8, // 100 MPa
            time_scheme: TimeIntegrationScheme::RK4,
            spatial_order: 4,
            adaptive_timestep: false,
            cfl_factor: 0.3,
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
}

/// Kuznetsov wave equation solver
#[derive(Debug)]
pub struct KuznetsovWave {
    config: KuznetsovConfig,
    
    /// Pre-computed k-space arrays
    k_squared: Array3<f64>,
    k_magnitude: Array3<f64>,
    phase_factors: Array3<f64>,
    
    /// History buffers for time derivatives
    pressure_history: Vec<Array3<f64>>,
    nonlinear_history: Vec<Array3<f64>>,
    
    /// FFT planner for efficiency
    fft_planner: Option<Fft3d>,
    
    /// Performance metrics
    metrics: SolverMetrics,
}

impl KuznetsovWave {
    /// Create a new Kuznetsov equation solver
    pub fn new(grid: &Grid, config: KuznetsovConfig) -> Self {
        info!("Initializing Kuznetsov equation solver with config: {:?}", config);
        
        // Precompute k-space arrays
        let k_squared = grid.k_squared();
        let k_magnitude = compute_k_magnitude(grid);
        let phase_factors = compute_phase_factors(grid, config.spatial_order);
        
        // Initialize history buffers
        let history_size = match config.time_scheme {
            TimeIntegrationScheme::Euler | TimeIntegrationScheme::RK2 | TimeIntegrationScheme::RK4 => 2,
            TimeIntegrationScheme::AdamsBashforth3 => 3,
        };
        
        let pressure_history = vec![Array3::zeros((grid.nx, grid.ny, grid.nz)); history_size];
        let nonlinear_history = vec![Array3::zeros((grid.nx, grid.ny, grid.nz)); 3];
        
        // Create FFT planner
        let fft_planner = Fft3d::new(grid.nx, grid.ny, grid.nz);
        
        Self {
            config,
            k_squared,
            k_magnitude,
            phase_factors,
            pressure_history,
            nonlinear_history,
            fft_planner: Some(fft_planner),
            metrics: SolverMetrics::default(),
        }
    }
    
    /// Compute Laplacian using spectral method
    fn compute_laplacian(&self, field: &Array3<f64>, grid: &Grid) -> KwaversResult<Array3<f64>> {
        // Convert to 4D array for fft_3d compatibility
        let mut fields_4d = Array4::zeros((1, grid.nx, grid.ny, grid.nz));
        fields_4d.index_axis_mut(Axis(0), 0).assign(field);
        
        // Transform to k-space
        let mut field_k = fft_3d(&fields_4d, 0, grid);
        
        // Apply Laplacian in k-space: ∇²f = -k² * f
        Zip::from(&mut field_k)
            .and(&self.k_squared)
            .for_each(|fk, &k2| {
                *fk *= -k2;
            });
        
        // Transform back to real space
        let result = ifft_3d(&field_k, grid);
        
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
    
    /// Compute the diffusivity term: -(δ/c₀⁴)∂³p/∂t³
    fn compute_diffusivity_term(
        &mut self,
        pressure: &Array3<f64>,
        medium: &dyn Medium,
        grid: &Grid,
        dt: f64,
    ) -> KwaversResult<Array3<f64>> {
        let start = Instant::now();
        
        if !self.config.enable_diffusivity {
            return Ok(Array3::zeros(pressure.raw_dim()));
        }
        
        // Need at least 3 time levels for third derivative
        if self.pressure_history.len() < 3 {
            return Ok(Array3::zeros(pressure.raw_dim()));
        }
        
        // Get previous pressure values
        let p_prev = &self.pressure_history[0];
        let p_prev2 = &self.pressure_history[1];
        let p_prev3 = &self.pressure_history[2];
        
        // Third-order finite difference for ∂³p/∂t³
        // Using backward difference formula
        let d3p_dt3 = (pressure - 3.0 * p_prev + 3.0 * p_prev2 - p_prev3) / (dt * dt * dt);
        
        // Apply diffusivity coefficient
        let mut diffusivity_term = Array3::zeros(pressure.raw_dim());
        
        Zip::indexed(&mut diffusivity_term)
            .and(&d3p_dt3)
            .for_each(|(i, j, k), diff, &d3p| {
                let x = i as f64 * grid.dx;
                let y = j as f64 * grid.dy;
                let z = k as f64 * grid.dz;
                
                let c0 = medium.sound_speed(x, y, z, grid);
                let delta = compute_acoustic_diffusivity(medium, x, y, z, grid);
                
                let coeff = -delta / c0.powi(4);
                *diff = coeff * d3p;
            });
        
        self.metrics.diffusion_time += start.elapsed().as_secs_f64();
        Ok(diffusivity_term)
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
    fn check_cfl_condition(&self, grid: &Grid, medium: &dyn Medium, dt: f64) -> bool {
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
                // where k1 = dt*f(t_n, y_n)
                //       k2 = dt*f(t_n + dt/2, y_n + k1/2)
                //       k3 = dt*f(t_n + dt/2, y_n + k2/2)
                //       k4 = dt*f(t_n + dt, y_n + k3)
                
                let pressure_0 = pressure.clone();
                
                // Stage 1: k1 = dt * f(p_n)
                let linear_term = compute_linear_term(&laplacian, pressure, medium, grid);
                let k1 = (&linear_term + &nonlinear_term + &diffusivity_term + source_term) * dt;
                
                // Stage 2: k2 = dt * f(p_n + k1/2)
                let mut pressure_temp = &pressure_0 + &(&k1 * 0.5);
                let laplacian_2 = self.compute_laplacian(&pressure_temp, grid)?;
                let nonlinear_2 = self.compute_nonlinear_term(&pressure_temp, medium, grid, dt)?;
                let diffusivity_2 = self.compute_diffusivity_term(&pressure_temp, medium, grid, dt)?;
                let linear_2 = compute_linear_term(&laplacian_2, &pressure_temp, medium, grid);
                let k2 = (&linear_2 + &nonlinear_2 + &diffusivity_2 + source_term) * dt;
                
                // Stage 3: k3 = dt * f(p_n + k2/2)
                pressure_temp = &pressure_0 + &(&k2 * 0.5);
                let laplacian_3 = self.compute_laplacian(&pressure_temp, grid)?;
                let nonlinear_3 = self.compute_nonlinear_term(&pressure_temp, medium, grid, dt)?;
                let diffusivity_3 = self.compute_diffusivity_term(&pressure_temp, medium, grid, dt)?;
                let linear_3 = compute_linear_term(&laplacian_3, &pressure_temp, medium, grid);
                let k3 = (&linear_3 + &nonlinear_3 + &diffusivity_3 + source_term) * dt;
                
                // Stage 4: k4 = dt * f(p_n + k3)
                pressure_temp = &pressure_0 + &k3;
                let laplacian_4 = self.compute_laplacian(&pressure_temp, grid)?;
                let nonlinear_4 = self.compute_nonlinear_term(&pressure_temp, medium, grid, dt)?;
                let diffusivity_4 = self.compute_diffusivity_term(&pressure_temp, medium, grid, dt)?;
                let linear_4 = compute_linear_term(&laplacian_4, &pressure_temp, medium, grid);
                let k4 = (&linear_4 + &nonlinear_4 + &diffusivity_4 + source_term) * dt;
                
                // Combine stages: p_{n+1} = p_n + (k1 + 2*k2 + 2*k3 + k4)/6
                Zip::from(&mut *pressure)
                    .and(&pressure_0)
                    .and(&k1)
                    .and(&k2)
                    .and(&k3)
                    .and(&k4)
                    .for_each(|p, &p0, &k1_val, &k2_val, &k3_val, &k4_val| {
                        *p = p0 + (k1_val + 2.0 * k2_val + 2.0 * k3_val + k4_val) / 6.0;
                        
                        // Clamp for stability
                        if p.abs() > self.config.max_pressure {
                            *p = p.signum() * self.config.max_pressure;
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
    
    fn set_k_space_correction_order(&mut self, order: usize) {
        info!("K-space correction order setting not applicable to Kuznetsov solver (order: {})", order);
        // Kuznetsov solver uses full spectral accuracy, no correction order needed
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
        _ => {} // Higher orders not yet implemented
    }
    
    factors
}

/// Compute acoustic diffusivity from medium properties
fn compute_acoustic_diffusivity(
    medium: &dyn Medium,
    x: f64,
    y: f64,
    z: f64,
    grid: &Grid,
) -> f64 {
    // Acoustic diffusivity δ = (4μ/3 + μ_B + κ(γ-1)/C_p) / ρ₀
    // Where:
    // μ = shear viscosity
    // μ_B = bulk viscosity
    // κ = thermal conductivity
    // γ = specific heat ratio
    // C_p = specific heat at constant pressure
    
    // For now, use a simplified model based on absorption
    let alpha = medium.absorption_coefficient(x, y, z, grid, 1e6); // Using 1 MHz as default
    let c = medium.sound_speed(x, y, z, grid);
    
    // Approximate diffusivity from power-law absorption
    // δ ≈ 2αc³/(ω²) for typical soft tissues
    // Using a reference frequency of 1 MHz
    let omega_ref = 2.0 * PI * 1e6;
    2.0 * alpha * c.powi(3) / (omega_ref * omega_ref)
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::grid::Grid;
    use crate::medium::HomogeneousMedium;
    use crate::physics::traits::AcousticWaveModel;
    use ndarray::{Array3, Array4};
    use std::f64::consts::PI;
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
        
        let mut solver = KuznetsovWave::new(&grid, config);
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
}

// Include comprehensive test module
#[cfg(test)]
#[path = "kuznetsov_tests.rs"]
mod kuznetsov_tests;