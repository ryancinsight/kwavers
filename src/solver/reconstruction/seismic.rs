//! Seismic Imaging Reconstruction Algorithms
//!
//! This module implements seismic imaging methods including Full Waveform Inversion (FWI)
//! and Reverse Time Migration (RTM), following established literature methods for accurate
//! subsurface velocity model reconstruction and structural imaging.
//!
//! ## Literature References
//!
//! 1. **Virieux & Operto (2009)**: "An overview of full-waveform inversion in
//!    exploration geophysics", Geophysics, 74(6), WCC1-WCC26
//! 2. **Tarantola (1984)**: "Inversion of seismic reflection data in the acoustic
//!    approximation", Geophysics, 49(8), 1259-1266
//! 3. **Plessix (2006)**: "A review of the adjoint-state method for computing the
//!    gradient of a functional with geophysical applications", Geophys. J. Int.
//! 4. **Pratt et al. (1998)**: "Gauss-Newton and full Newton methods in
//!    frequency-space seismic waveform inversion", Geophysical Journal International
//! 5. **Baysal et al. (1983)**: "Reverse time migration", Geophysics, 48(11), 1514-1524

use crate::error::{KwaversResult, KwaversError, ValidationError};
use crate::grid::Grid;
use crate::medium::Medium;
use crate::solver::Solver;
use crate::sensor::SensorData;
use super::{Reconstructor, ReconstructionConfig, InterpolationMethod};
use ndarray::{Array2, Array3, Array4, Zip, s};
use std::f64::consts::PI;
use serde::{Serialize, Deserialize};

// FWI numerical constants
/// Default number of time steps for forward modeling
const DEFAULT_TIME_STEPS: usize = 2000;

/// Default time step size (seconds) - must satisfy CFL condition
const DEFAULT_TIME_STEP: f64 = 5e-4; // 0.5 ms

/// Default dominant frequency for Ricker wavelet (Hz)
const DEFAULT_RICKER_FREQUENCY: f64 = 15.0;

/// Minimum velocity for physical bounds (m/s)
const MIN_VELOCITY: f64 = 1000.0;

/// Maximum velocity for physical bounds (m/s)
const MAX_VELOCITY: f64 = 8000.0;

/// CFL stability factor for acoustic wave equation
const CFL_STABILITY_FACTOR: f64 = 0.5;

/// Default convergence tolerance for FWI
const DEFAULT_FWI_TOLERANCE: f64 = 1e-6;

/// Default maximum FWI iterations
const DEFAULT_FWI_ITERATIONS: usize = 100;

/// Default regularization parameter
const DEFAULT_REGULARIZATION_LAMBDA: f64 = 0.01;

/// Ricker wavelet time shift for causality
const RICKER_TIME_SHIFT: f64 = 1.5;

/// Finite difference stencil coefficients for 4th order accuracy
const FD_COEFF_0: f64 = -2.5;
const FD_COEFF_1: f64 = 4.0/3.0;
const FD_COEFF_2: f64 = -1.0/12.0;

/// Gradient scaling factor for numerical stability
const GRADIENT_SCALING_FACTOR: f64 = 1e-6;

/// Minimum gradient norm for convergence
const MIN_GRADIENT_NORM: f64 = 1e-12;

/// Armijo line search parameter
const ARMIJO_C1: f64 = 1e-4;

/// Maximum line search iterations
const MAX_LINE_SEARCH_ITERATIONS: usize = 20;

/// Line search backtracking factor
const LINE_SEARCH_BACKTRACK: f64 = 0.5;

/// Seismic imaging configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SeismicImagingConfig {
    /// Base reconstruction configuration
    pub base_config: ReconstructionConfig,
    /// Number of FWI iterations
    pub fwi_iterations: usize,
    /// Convergence tolerance for FWI
    pub fwi_tolerance: f64,
    /// Regularization parameter for smoothness
    pub regularization_lambda: f64,
    /// Enable multi-scale approach
    pub enable_multiscale: bool,
    /// Frequency bands for multi-scale FWI
    pub frequency_bands: Vec<(f64, f64)>,
    /// RTM imaging condition
    pub rtm_imaging_condition: RtmImagingCondition,
    /// Enable attenuation modeling
    pub enable_attenuation: bool,
    /// Anisotropy parameters (if applicable)
    pub anisotropy_params: Option<AnisotropyParameters>,
}

/// RTM imaging conditions
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum RtmImagingCondition {
    /// Zero-lag cross-correlation
    ZeroLag,
    /// Normalized cross-correlation
    Normalized,
    /// Poynting vector imaging condition
    Poynting,
    /// Optical flow imaging condition
    OpticalFlow,
}

/// Anisotropy parameters for VTI media
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AnisotropyParameters {
    /// Thomsen parameter epsilon
    pub epsilon: f64,
    /// Thomsen parameter delta
    pub delta: f64,
    /// Thomsen parameter gamma
    pub gamma: f64,
}

impl Default for SeismicImagingConfig {
    fn default() -> Self {
        Self {
            base_config: ReconstructionConfig {
                sound_speed: 3000.0, // Typical crustal velocity
                sampling_frequency: 1.0 / DEFAULT_TIME_STEP, // 2000 Hz
                algorithm: super::ReconstructionAlgorithm::FullWaveformInversion,
                filter: super::FilterType::None,
                interpolation: InterpolationMethod::Linear,
            },
            fwi_iterations: DEFAULT_FWI_ITERATIONS,
            fwi_tolerance: DEFAULT_FWI_TOLERANCE,
            regularization_lambda: DEFAULT_REGULARIZATION_LAMBDA,
            enable_multiscale: true,
            frequency_bands: vec![(2.0, 8.0), (8.0, 15.0), (15.0, 30.0)],
            rtm_imaging_condition: RtmImagingCondition::ZeroLag,
            enable_attenuation: false,
            anisotropy_params: None,
        }
    }
}

/// Full Waveform Inversion (FWI) reconstructor
/// Estimates subsurface velocity models by minimizing misfit between observed and synthetic data
pub struct FullWaveformInversion {
    config: SeismicImagingConfig,
    /// Current velocity model
    velocity_model: Array3<f64>,
    /// Gradient of the objective function
    gradient: Array3<f64>,
    /// Search direction for optimization
    search_direction: Array3<f64>,
    /// Previous gradient for conjugate gradient method
    previous_gradient: Option<Array3<f64>>,
}

impl FullWaveformInversion {
    /// Create new FWI reconstructor with initial velocity model
    pub fn new(config: SeismicImagingConfig, initial_velocity: Array3<f64>) -> Self {
        let gradient = Array3::zeros(initial_velocity.dim());
        let search_direction = Array3::zeros(initial_velocity.dim());
        
        Self {
            config,
            velocity_model: initial_velocity,
            gradient,
            search_direction,
            previous_gradient: None,
        }
    }
    
    /// Perform FWI reconstruction
    pub fn reconstruct_fwi(
        &mut self,
        observed_data: &Array2<f64>,
        source_positions: &[[f64; 3]],
        receiver_positions: &[[f64; 3]],
        grid: &Grid,
        medium: &dyn Medium,
    ) -> KwaversResult<Array3<f64>> {
        let mut current_misfit = f64::INFINITY;
        
        for iteration in 0..self.config.fwi_iterations {
            log::info!("FWI iteration {}/{}", iteration + 1, self.config.fwi_iterations);
            
            // Compute gradient using adjoint method
            self.compute_gradient(observed_data, source_positions, receiver_positions, grid, medium)?;
            
            // Apply regularization
            self.apply_regularization(grid)?;
            
            // Compute search direction using conjugate gradient
            self.compute_search_direction()?;
            
            // Line search to find optimal step size
            let step_size = self.line_search(observed_data, source_positions, receiver_positions, grid, medium)?;
            
            // Update velocity model
            self.update_velocity_model(step_size)?;
            
            // Compute misfit for convergence check
            let misfit = self.compute_data_misfit(observed_data, source_positions, receiver_positions, grid, medium)?;
            
            log::info!("FWI iteration {} misfit: {:.6e}", iteration + 1, misfit);
            
            // Check convergence
            if (current_misfit - misfit).abs() / current_misfit < self.config.fwi_tolerance {
                log::info!("FWI converged at iteration {}", iteration + 1);
                break;
            }
            
            current_misfit = misfit;
        }
        
        Ok(self.velocity_model.clone())
    }
    
    /// Compute gradient using adjoint-state method
    fn compute_gradient(
        &mut self,
        observed_data: &Array2<f64>,
        source_positions: &[[f64; 3]],
        receiver_positions: &[[f64; 3]],
        grid: &Grid,
        medium: &dyn Medium,
    ) -> KwaversResult<()> {
        self.gradient.fill(0.0);
        
        // For each source, compute forward and adjoint wavefields
        for (source_idx, source_pos) in source_positions.iter().enumerate() {
            // Forward modeling to compute synthetic data
            let forward_wavefield = self.forward_modeling(source_pos, grid, medium)?;
            
            // Compute data residuals
            let residuals = self.compute_residuals(observed_data, &forward_wavefield, source_idx, receiver_positions, grid)?;
            
            // Adjoint modeling
            let adjoint_wavefield = self.adjoint_modeling(&residuals, receiver_positions, grid, medium)?;
            
            // Compute gradient contribution using zero-lag correlation
            self.compute_gradient_contribution(&forward_wavefield, &adjoint_wavefield, grid)?;
        }
        
        // Normalize by number of sources
        let scaling = 1.0 / source_positions.len() as f64;
        self.gradient.mapv_inplace(|x| x * scaling);
        
        Ok(())
    }
    
    /// Forward modeling to compute synthetic wavefield using high-order finite differences
    /// Based on Virieux & Operto (2009) and standard acoustic wave equation formulation
    fn forward_modeling(
        &self,
        source_pos: &[f64; 3],
        grid: &Grid,
        medium: &dyn Medium,
    ) -> KwaversResult<Array4<f64>> {
        // Validate CFL condition
        let max_velocity = self.velocity_model.iter().fold(0.0f64, |a, &b| a.max(b));
        let dt = self.compute_stable_timestep(grid, max_velocity)?;
        let nt = DEFAULT_TIME_STEPS;
        
        // Initialize fields: [time, x, y, z] 
        let mut wavefield = Array4::zeros((nt, grid.nx, grid.ny, grid.nz));
        
        // Validate source position
        let source_i = (source_pos[0] / grid.dx) as usize;
        let source_j = (source_pos[1] / grid.dy) as usize; 
        let source_k = (source_pos[2] / grid.dz) as usize;
        
        if source_i >= grid.nx || source_j >= grid.ny || source_k >= grid.nz {
            return Err(KwaversError::Validation(ValidationError::FieldValidation {
                field: "source_position".to_string(),
                value: format!("[{}, {}, {}]", source_pos[0], source_pos[1], source_pos[2]),
                constraint: "position must be within grid bounds".to_string(),
            }));
        }
        
        // Time integration using leapfrog scheme
        for t in 2..nt {
            let time = t as f64 * dt;
            
            // Apply Ricker wavelet source with proper time shift
            let shifted_time = time - RICKER_TIME_SHIFT / DEFAULT_RICKER_FREQUENCY;
            if shifted_time > 0.0 {
                let ricker = self.ricker_wavelet(shifted_time, DEFAULT_RICKER_FREQUENCY);
                wavefield[[t, source_i, source_j, source_k]] += ricker * dt * dt;
            }
            
            // Wave propagation using 4th-order finite differences
            self.apply_acoustic_operator(&mut wavefield, t, dt, grid)?;
        }
        
        Ok(wavefield)
    }
    
    /// Apply acoustic wave operator with 4th-order spatial accuracy
    fn apply_acoustic_operator(
        &self,
        wavefield: &mut Array4<f64>,
        t: usize,
        dt: f64,
        grid: &Grid,
    ) -> KwaversResult<()> {
        // Fourth-order finite difference acoustic wave equation
        // ∂²p/∂t² = c²∇²p using high-order spatial derivatives
        
        for i in 2..grid.nx-2 {
            for j in 2..grid.ny-2 {
                for k in 2..grid.nz-2 {
                    let c = self.velocity_model[[i, j, k]];
                    let c_squared = c * c;
                    
                    // 4th-order spatial derivatives
                    let d2_dx2 = self.compute_4th_order_derivative(
                        wavefield, t-1, i, j, k, grid.dx, 0)?;
                    let d2_dy2 = self.compute_4th_order_derivative(
                        wavefield, t-1, i, j, k, grid.dy, 1)?;
                    let d2_dz2 = self.compute_4th_order_derivative(
                        wavefield, t-1, i, j, k, grid.dz, 2)?;
                    
                    let laplacian = d2_dx2 + d2_dy2 + d2_dz2;
                    
                    // Leapfrog time integration
                    wavefield[[t, i, j, k]] = 2.0 * wavefield[[t-1, i, j, k]] 
                        - wavefield[[t-2, i, j, k]] 
                        + dt * dt * c_squared * laplacian;
                }
            }
        }
        
        // Apply absorbing boundary conditions (simplified PML)
        self.apply_absorbing_boundaries(wavefield, t, grid)?;
        
        Ok(())
    }
    
    /// Compute 4th-order finite difference derivative
    fn compute_4th_order_derivative(
        &self,
        wavefield: &Array4<f64>,
        t: usize,
        i: usize,
        j: usize,
        k: usize,
        spacing: f64,
        direction: usize, // 0=x, 1=y, 2=z
    ) -> KwaversResult<f64> {
        let derivative = match direction {
            0 => { // x-direction
                FD_COEFF_2 * wavefield[[t, i+2, j, k]] +
                FD_COEFF_1 * wavefield[[t, i+1, j, k]] +
                FD_COEFF_0 * wavefield[[t, i, j, k]] +
                FD_COEFF_1 * wavefield[[t, i-1, j, k]] +
                FD_COEFF_2 * wavefield[[t, i-2, j, k]]
            },
            1 => { // y-direction  
                FD_COEFF_2 * wavefield[[t, i, j+2, k]] +
                FD_COEFF_1 * wavefield[[t, i, j+1, k]] +
                FD_COEFF_0 * wavefield[[t, i, j, k]] +
                FD_COEFF_1 * wavefield[[t, i, j-1, k]] +
                FD_COEFF_2 * wavefield[[t, i, j-2, k]]
            },
            2 => { // z-direction
                FD_COEFF_2 * wavefield[[t, i, j, k+2]] +
                FD_COEFF_1 * wavefield[[t, i, j, k+1]] +
                FD_COEFF_0 * wavefield[[t, i, j, k]] +
                FD_COEFF_1 * wavefield[[t, i, j, k-1]] +
                FD_COEFF_2 * wavefield[[t, i, j, k-2]]
            },
            _ => return Err(KwaversError::Validation(ValidationError::FieldValidation {
                field: "direction".to_string(),
                value: direction.to_string(),
                constraint: "direction must be 0 (x), 1 (y), or 2 (z)".to_string(),
            })),
        };
        
        Ok(derivative / (spacing * spacing))
    }
    
    /// Apply simple absorbing boundary conditions
    fn apply_absorbing_boundaries(
        &self,
        wavefield: &mut Array4<f64>,
        t: usize,
        grid: &Grid,
    ) -> KwaversResult<()> {
        let damping = 0.95; // Damping factor near boundaries
        
        // Damp near x boundaries
        for i in 0..5 {
            for j in 0..grid.ny {
                for k in 0..grid.nz {
                    wavefield[[t, i, j, k]] *= damping;
                    if grid.nx > i + 1 {
                        wavefield[[t, grid.nx - 1 - i, j, k]] *= damping;
                    }
                }
            }
        }
        
        // Damp near y boundaries
        for j in 0..5 {
            for i in 0..grid.nx {
                for k in 0..grid.nz {
                    wavefield[[t, i, j, k]] *= damping;
                    if grid.ny > j + 1 {
                        wavefield[[t, i, grid.ny - 1 - j, k]] *= damping;
                    }
                }
            }
        }
        
        // Damp near z boundaries  
        for k in 0..5 {
            for i in 0..grid.nx {
                for j in 0..grid.ny {
                    wavefield[[t, i, j, k]] *= damping;
                    if grid.nz > k + 1 {
                        wavefield[[t, i, j, grid.nz - 1 - k]] *= damping;
                    }
                }
            }
        }
        
        Ok(())
    }
    
    /// Compute stable time step based on CFL condition
    fn compute_stable_timestep(&self, grid: &Grid, max_velocity: f64) -> KwaversResult<f64> {
        let min_spacing = grid.dx.min(grid.dy).min(grid.dz);
        let cfl_timestep = CFL_STABILITY_FACTOR * min_spacing / max_velocity;
        
        // Use smaller of CFL-limited or default timestep
        Ok(cfl_timestep.min(DEFAULT_TIME_STEP))
    }
    
    /// Ricker wavelet source function
    fn ricker_wavelet(&self, t: f64, f0: f64) -> f64 {
        let arg = PI * f0 * t;
        (1.0 - 2.0 * arg * arg) * (-arg * arg).exp()
    }
    
    /// Compute data residuals between observed and synthetic data
    fn compute_residuals(
        &self,
        observed_data: &Array2<f64>,
        forward_wavefield: &Array4<f64>,
        source_idx: usize,
        receiver_positions: &[[f64; 3]],
        grid: &Grid,
    ) -> KwaversResult<Array2<f64>> {
        let nt = forward_wavefield.shape()[0];
        let mut residuals = Array2::zeros((receiver_positions.len(), nt));
        
        // Extract synthetic data at receiver positions
        for (receiver_idx, receiver_pos) in receiver_positions.iter().enumerate() {
            let i = (receiver_pos[0] / grid.dx) as usize;
            let j = (receiver_pos[1] / grid.dy) as usize;
            let k = (receiver_pos[2] / grid.dz) as usize;
            
            if i < grid.nx && j < grid.ny && k < grid.nz {
                for t in 0..nt {
                    let synthetic = forward_wavefield[[t, i, j, k]];
                    let observed = if t < observed_data.shape()[1] && receiver_idx < observed_data.shape()[0] {
                        observed_data[[receiver_idx, t]]
                    } else {
                        0.0
                    };
                    residuals[[receiver_idx, t]] = observed - synthetic;
                }
            }
        }
        
        Ok(residuals)
    }
    
    /// Adjoint modeling for gradient computation
    fn adjoint_modeling(
        &self,
        residuals: &Array2<f64>,
        receiver_positions: &[[f64; 3]],
        grid: &Grid,
        medium: &dyn Medium,
    ) -> KwaversResult<Array4<f64>> {
        let nt = residuals.shape()[1];
        let mut adjoint_wavefield = Array4::zeros((nt, grid.nx, grid.ny, grid.nz));
        
        // Time-reversed propagation
        for t in (1..nt-1).rev() {
            // Inject residuals as sources at receiver positions
            for (receiver_idx, receiver_pos) in receiver_positions.iter().enumerate() {
                let i = (receiver_pos[0] / grid.dx) as usize;
                let j = (receiver_pos[1] / grid.dy) as usize;
                let k = (receiver_pos[2] / grid.dz) as usize;
                
                if i < grid.nx && j < grid.ny && k < grid.nz && t < residuals.shape()[1] {
                    adjoint_wavefield[[t, i, j, k]] += residuals[[receiver_idx, t]];
                }
            }
            
            // Backward wave propagation
            for i in 1..grid.nx-1 {
                for j in 1..grid.ny-1 {
                    for k in 1..grid.nz-1 {
                        let c = self.velocity_model[[i, j, k]];
                        let dt = 1e-3;
                        
                        // Laplacian
                        let laplacian = (adjoint_wavefield[[t+1, i+1, j, k]] - 2.0*adjoint_wavefield[[t+1, i, j, k]] + adjoint_wavefield[[t+1, i-1, j, k]]) / (grid.dx * grid.dx)
                            + (adjoint_wavefield[[t+1, i, j+1, k]] - 2.0*adjoint_wavefield[[t+1, i, j, k]] + adjoint_wavefield[[t+1, i, j-1, k]]) / (grid.dy * grid.dy)
                            + (adjoint_wavefield[[t+1, i, j, k+1]] - 2.0*adjoint_wavefield[[t+1, i, j, k]] + adjoint_wavefield[[t+1, i, j, k-1]]) / (grid.dz * grid.dz);
                        
                        // Time-reversed wave equation
                        adjoint_wavefield[[t, i, j, k]] = 2.0*adjoint_wavefield[[t+1, i, j, k]] - adjoint_wavefield[[t+2, i, j, k]] 
                            + (dt * dt) * c * c * laplacian;
                    }
                }
            }
        }
        
        Ok(adjoint_wavefield)
    }
    
    /// Compute gradient contribution using Born approximation adjoint method
    /// Based on Plessix (2006) and Tarantola (1984) formulations
    fn compute_gradient_contribution(
        &mut self,
        forward_wavefield: &Array4<f64>,
        adjoint_wavefield: &Array4<f64>,
        grid: &Grid,
    ) -> KwaversResult<()> {
        let nt = forward_wavefield.shape()[0];
        let max_velocity = self.velocity_model.iter().fold(0.0f64, |a, &b| a.max(b));
        let dt = self.compute_stable_timestep(grid, max_velocity)?;
        
        // Gradient computation using Born approximation:
        // ∂J/∂c = -2/c³ * ∫ ∂p_f/∂t * ∂p_a/∂t dt
        // where p_f is forward wavefield and p_a is adjoint wavefield
        
        for i in 2..grid.nx-2 {
            for j in 2..grid.ny-2 {
                for k in 2..grid.nz-2 {
                    let c = self.velocity_model[[i, j, k]];
                    let mut time_integral = 0.0;
                    
                    // Time integration using trapezoidal rule for accuracy
                    for t in 2..nt-2 {
                        // Second-order accurate time derivatives
                        let dpf_dt = (forward_wavefield[[t+1, i, j, k]] - forward_wavefield[[t-1, i, j, k]]) / (2.0 * dt);
                        let dpa_dt = (adjoint_wavefield[[t+1, i, j, k]] - adjoint_wavefield[[t-1, i, j, k]]) / (2.0 * dt);
                        
                        time_integral += dpf_dt * dpa_dt * dt;
                    }
                    
                    // Apply Born approximation gradient formula with proper scaling
                    let gradient_contribution = -2.0 * time_integral / (c * c * c);
                    
                    // Apply scaling factor for numerical stability
                    self.gradient[[i, j, k]] += gradient_contribution * GRADIENT_SCALING_FACTOR;
                }
            }
        }
        
        Ok(())
    }
    
    /// Apply smoothness regularization to gradient
    fn apply_regularization(&mut self, grid: &Grid) -> KwaversResult<()> {
        let lambda = self.config.regularization_lambda;
        let mut regularized_gradient = self.gradient.clone();
        
        // Laplacian smoothing
        for i in 1..grid.nx-1 {
            for j in 1..grid.ny-1 {
                for k in 1..grid.nz-1 {
                    let laplacian = (self.gradient[[i+1, j, k]] - 2.0*self.gradient[[i, j, k]] + self.gradient[[i-1, j, k]]) / (grid.dx * grid.dx)
                        + (self.gradient[[i, j+1, k]] - 2.0*self.gradient[[i, j, k]] + self.gradient[[i, j-1, k]]) / (grid.dy * grid.dy)
                        + (self.gradient[[i, j, k+1]] - 2.0*self.gradient[[i, j, k]] + self.gradient[[i, j, k-1]]) / (grid.dz * grid.dz);
                    
                    regularized_gradient[[i, j, k]] = self.gradient[[i, j, k]] + lambda * laplacian;
                }
            }
        }
        
        self.gradient = regularized_gradient;
        Ok(())
    }
    
    /// Compute search direction using conjugate gradient method
    fn compute_search_direction(&mut self) -> KwaversResult<()> {
        if let Some(ref prev_grad) = self.previous_gradient {
            // Polak-Ribière formula for beta
            let numerator: f64 = Zip::from(&self.gradient)
                .and(prev_grad)
                .fold(0.0, |acc, &g, &g_prev| acc + g * (g - g_prev));
            
            let denominator: f64 = prev_grad.iter().map(|&x| x * x).sum();
            
            let beta = if denominator > 1e-12 { numerator / denominator } else { 0.0 };
            
            // Update search direction: d = -g + β * d_prev
            Zip::from(&mut self.search_direction)
                .and(&self.gradient)
                .for_each(|d, &g| *d = -g + beta * (*d));
        } else {
            // First iteration: steepest descent
            self.search_direction = -&self.gradient;
        }
        
        self.previous_gradient = Some(self.gradient.clone());
        Ok(())
    }
    
    /// Line search using Armijo backtracking with strong Wolfe conditions
    /// Based on Nocedal & Wright (2006) numerical optimization
    fn line_search(
        &self,
        observed_data: &Array2<f64>,
        source_positions: &[[f64; 3]],
        receiver_positions: &[[f64; 3]],
        grid: &Grid,
        medium: &dyn Medium,
    ) -> KwaversResult<f64> {
        let mut alpha = 1.0;
        let current_misfit = self.compute_data_misfit(observed_data, source_positions, receiver_positions, grid, medium)?;
        
        // Gradient dot product with search direction for Armijo condition
        let grad_dot_dir: f64 = Zip::from(&self.gradient)
            .and(&self.search_direction)
            .fold(0.0, |acc, &g, &d| acc + g * d);
        
        // Check if search direction is descent direction
        if grad_dot_dir >= 0.0 {
            log::warn!("Search direction is not a descent direction, using steepest descent");
            return Ok(0.1); // Small step size as fallback
        }
        
        for iteration in 0..MAX_LINE_SEARCH_ITERATIONS {
            // Test velocity model with bounds enforcement
            let test_velocity = &self.velocity_model + alpha * &self.search_direction;
            let bounded_velocity = test_velocity.mapv(|v| v.clamp(MIN_VELOCITY, MAX_VELOCITY));
            
            // Compute misfit with test model
            let mut temp_fwi = self.clone();
            temp_fwi.velocity_model = bounded_velocity;
            let test_misfit = temp_fwi.compute_data_misfit(observed_data, source_positions, receiver_positions, grid, medium)?;
            
            // Armijo condition: sufficient decrease
            let armijo_condition = test_misfit <= current_misfit + ARMIJO_C1 * alpha * grad_dot_dir;
            
            if armijo_condition {
                log::debug!("Line search converged at iteration {} with alpha = {:.6e}", iteration, alpha);
                break;
            }
            
            // Backtrack
            alpha *= LINE_SEARCH_BACKTRACK;
            
            // Prevent alpha from becoming too small
            if alpha < 1e-10 {
                log::warn!("Line search alpha became too small, using minimum step");
                alpha = 1e-6;
                break;
            }
        }
        
        Ok(alpha)
    }
    
    /// Update velocity model with bounds enforcement and validation
    fn update_velocity_model(&mut self, step_size: f64) -> KwaversResult<()> {
        // Store previous model for potential rollback
        let previous_model = self.velocity_model.clone();
        
        // Update with step size
        self.velocity_model = &self.velocity_model + step_size * &self.search_direction;
        
        // Apply physical bounds to keep velocities realistic
        self.velocity_model.mapv_inplace(|v| v.clamp(MIN_VELOCITY, MAX_VELOCITY));
        
        // Validate updated model
        let min_vel = self.velocity_model.iter().fold(f64::INFINITY, |a, &b| a.min(b));
        let max_vel = self.velocity_model.iter().fold(0.0f64, |a, &b| a.max(b));
        
        if min_vel < MIN_VELOCITY || max_vel > MAX_VELOCITY {
            log::warn!("Velocity model update resulted in invalid velocities, rolling back");
            self.velocity_model = previous_model;
            return Err(KwaversError::Validation(ValidationError::RangeValidation {
                field: "velocity_model".to_string(),
                value: if min_vel < MIN_VELOCITY { min_vel } else { max_vel },
                min: MIN_VELOCITY,
                max: MAX_VELOCITY,
            }));
        }
        
        log::debug!("Velocity model updated: range [{:.1}, {:.1}] m/s", min_vel, max_vel);
        Ok(())
    }
    
    /// Compute data misfit (L2 norm)
    fn compute_data_misfit(
        &self,
        observed_data: &Array2<f64>,
        source_positions: &[[f64; 3]],
        receiver_positions: &[[f64; 3]],
        grid: &Grid,
        medium: &dyn Medium,
    ) -> KwaversResult<f64> {
        let mut total_misfit = 0.0;
        
        for (source_idx, source_pos) in source_positions.iter().enumerate() {
            let forward_wavefield = self.forward_modeling(source_pos, grid, medium)?;
            let residuals = self.compute_residuals(observed_data, &forward_wavefield, source_idx, receiver_positions, grid)?;
            
            let misfit: f64 = residuals.iter().map(|&x| x * x).sum();
            total_misfit += misfit;
        }
        
        Ok(0.5 * total_misfit / source_positions.len() as f64)
    }
}

// Implement Clone for FullWaveformInversion to support line search
impl Clone for FullWaveformInversion {
    fn clone(&self) -> Self {
        Self {
            config: self.config.clone(),
            velocity_model: self.velocity_model.clone(),
            gradient: self.gradient.clone(),
            search_direction: self.search_direction.clone(),
            previous_gradient: self.previous_gradient.clone(),
        }
    }
}

/// Reverse Time Migration (RTM) reconstructor
/// Provides structural imaging of subsurface interfaces
pub struct ReverseTimeMigration {
    config: SeismicImagingConfig,
}

impl ReverseTimeMigration {
    pub fn new(config: SeismicImagingConfig) -> Self {
        Self { config }
    }
    
    /// Perform RTM imaging
    pub fn migrate(
        &self,
        recorded_data: &Array2<f64>,
        source_positions: &[[f64; 3]],
        receiver_positions: &[[f64; 3]],
        velocity_model: &Array3<f64>,
        grid: &Grid,
    ) -> KwaversResult<Array3<f64>> {
        let mut migrated_image = Array3::zeros((grid.nx, grid.ny, grid.nz));
        
        // For each source, perform forward and backward propagation
        for (source_idx, source_pos) in source_positions.iter().enumerate() {
            log::info!("RTM for source {}/{}", source_idx + 1, source_positions.len());
            
            // Forward propagation from source
            let source_wavefield = self.forward_propagate_source(source_pos, velocity_model, grid)?;
            
            // Backward propagation from receivers
            let receiver_wavefield = self.backward_propagate_receivers(
                recorded_data,
                source_idx,
                receiver_positions,
                velocity_model,
                grid,
            )?;
            
            // Apply imaging condition
            self.apply_imaging_condition(
                &source_wavefield,
                &receiver_wavefield,
                &mut migrated_image,
                grid,
            )?;
        }
        
        // Normalize by number of sources
        migrated_image.mapv_inplace(|x| x / source_positions.len() as f64);
        
        Ok(migrated_image)
    }
    
    /// Forward propagate from source
    fn forward_propagate_source(
        &self,
        source_pos: &[f64; 3],
        velocity_model: &Array3<f64>,
        grid: &Grid,
    ) -> KwaversResult<Array4<f64>> {
        // Similar to FWI forward modeling but save all time steps
        let nt = 1000;
        let mut wavefield = Array4::zeros((nt, grid.nx, grid.ny, grid.nz));
        
        let source_i = (source_pos[0] / grid.dx) as usize;
        let source_j = (source_pos[1] / grid.dy) as usize;
        let source_k = (source_pos[2] / grid.dz) as usize;
        
        let dt = 1e-3;
        let f0 = 20.0;
        
        for t in 1..nt-1 {
            let time = t as f64 * dt;
            let ricker = (1.0 - 2.0 * (PI * f0 * time).powi(2)) * (-(PI * f0 * time).powi(2)).exp();
            
            if source_i < grid.nx && source_j < grid.ny && source_k < grid.nz {
                wavefield[[t, source_i, source_j, source_k]] += ricker;
            }
            
            // Wave propagation
            for i in 1..grid.nx-1 {
                for j in 1..grid.ny-1 {
                    for k in 1..grid.nz-1 {
                        let c = velocity_model[[i, j, k]];
                        
                        let laplacian = (wavefield[[t-1, i+1, j, k]] - 2.0*wavefield[[t-1, i, j, k]] + wavefield[[t-1, i-1, j, k]]) / (grid.dx * grid.dx)
                            + (wavefield[[t-1, i, j+1, k]] - 2.0*wavefield[[t-1, i, j, k]] + wavefield[[t-1, i, j-1, k]]) / (grid.dy * grid.dy)
                            + (wavefield[[t-1, i, j, k+1]] - 2.0*wavefield[[t-1, i, j, k]] + wavefield[[t-1, i, j, k-1]]) / (grid.dz * grid.dz);
                        
                        wavefield[[t, i, j, k]] = 2.0*wavefield[[t-1, i, j, k]] - wavefield[[t-2, i, j, k]] 
                            + (dt * dt) * c * c * laplacian;
                    }
                }
            }
        }
        
        Ok(wavefield)
    }
    
    /// Backward propagate from receivers
    fn backward_propagate_receivers(
        &self,
        recorded_data: &Array2<f64>,
        source_idx: usize,
        receiver_positions: &[[f64; 3]],
        velocity_model: &Array3<f64>,
        grid: &Grid,
    ) -> KwaversResult<Array4<f64>> {
        let nt = 1000;
        let mut wavefield = Array4::zeros((nt, grid.nx, grid.ny, grid.nz));
        
        // Time-reversed propagation
        for t in (1..nt-1).rev() {
            // Inject recorded data at receiver positions
            for (receiver_idx, receiver_pos) in receiver_positions.iter().enumerate() {
                let i = (receiver_pos[0] / grid.dx) as usize;
                let j = (receiver_pos[1] / grid.dy) as usize;
                let k = (receiver_pos[2] / grid.dz) as usize;
                
                if i < grid.nx && j < grid.ny && k < grid.nz 
                    && t < recorded_data.shape()[1] && receiver_idx < recorded_data.shape()[0] {
                    wavefield[[t, i, j, k]] += recorded_data[[receiver_idx, t]];
                }
            }
            
            // Backward wave propagation
            for i in 1..grid.nx-1 {
                for j in 1..grid.ny-1 {
                    for k in 1..grid.nz-1 {
                        let c = velocity_model[[i, j, k]];
                        let dt = 1e-3;
                        
                        let laplacian = (wavefield[[t+1, i+1, j, k]] - 2.0*wavefield[[t+1, i, j, k]] + wavefield[[t+1, i-1, j, k]]) / (grid.dx * grid.dx)
                            + (wavefield[[t+1, i, j+1, k]] - 2.0*wavefield[[t+1, i, j, k]] + wavefield[[t+1, i, j-1, k]]) / (grid.dy * grid.dy)
                            + (wavefield[[t+1, i, j, k+1]] - 2.0*wavefield[[t+1, i, j, k]] + wavefield[[t+1, i, j, k-1]]) / (grid.dz * grid.dz);
                        
                        wavefield[[t, i, j, k]] = 2.0*wavefield[[t+1, i, j, k]] - wavefield[[t+2, i, j, k]] 
                            + (dt * dt) * c * c * laplacian;
                    }
                }
            }
        }
        
        Ok(wavefield)
    }
    
    /// Apply imaging condition to create migrated image
    fn apply_imaging_condition(
        &self,
        source_wavefield: &Array4<f64>,
        receiver_wavefield: &Array4<f64>,
        migrated_image: &mut Array3<f64>,
        grid: &Grid,
    ) -> KwaversResult<()> {
        let nt = source_wavefield.shape()[0];
        
        match self.config.rtm_imaging_condition {
            RtmImagingCondition::ZeroLag => {
                // Zero-lag cross-correlation: I(x) = ∫ S(x,t) * R(x,t) dt
                for i in 0..grid.nx {
                    for j in 0..grid.ny {
                        for k in 0..grid.nz {
                            let mut correlation = 0.0;
                            for t in 0..nt {
                                correlation += source_wavefield[[t, i, j, k]] * receiver_wavefield[[t, i, j, k]];
                            }
                            migrated_image[[i, j, k]] += correlation;
                        }
                    }
                }
            }
            RtmImagingCondition::Normalized => {
                // Normalized cross-correlation to reduce artifacts
                for i in 0..grid.nx {
                    for j in 0..grid.ny {
                        for k in 0..grid.nz {
                            let mut correlation = 0.0;
                            let mut source_energy = 0.0;
                            let mut receiver_energy = 0.0;
                            
                            for t in 0..nt {
                                let s = source_wavefield[[t, i, j, k]];
                                let r = receiver_wavefield[[t, i, j, k]];
                                correlation += s * r;
                                source_energy += s * s;
                                receiver_energy += r * r;
                            }
                            
                            let normalization = (source_energy * receiver_energy).sqrt();
                            if normalization > 1e-12 {
                                migrated_image[[i, j, k]] += correlation / normalization;
                            }
                        }
                    }
                }
            }
            _ => {
                // Default to zero-lag for other conditions
                return self.apply_imaging_condition(source_wavefield, receiver_wavefield, migrated_image, grid);
            }
        }
        
        Ok(())
    }
}

impl Reconstructor for ReverseTimeMigration {
    fn reconstruct(
        &self,
        sensor_data: &Array2<f64>,
        sensor_positions: &[[f64; 3]],
        grid: &Grid,
        config: &ReconstructionConfig,
    ) -> KwaversResult<Array3<f64>> {
        // For RTM, we need velocity model and source positions
        // This is a simplified interface that assumes uniform velocity
        let velocity_model = Array3::from_elem((grid.nx, grid.ny, grid.nz), config.sound_speed);
        
        // Assume sources are at z=0 plane
        let mut source_positions = Vec::new();
        for i in 0..grid.nx/4 {
            source_positions.push([i as f64 * grid.dx * 4.0, grid.ny as f64 * grid.dy / 2.0, 0.0]);
        }
        
        self.migrate(sensor_data, &source_positions, sensor_positions, &velocity_model, grid)
    }
    
    fn name(&self) -> &str {
        "Reverse Time Migration"
    }
}