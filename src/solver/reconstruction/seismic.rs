//! Seismic Imaging Reconstruction Algorithms
//!
//! This module implements seismic imaging methods including Full Waveform Inversion (FWI)
//! and Reverse Time Migration (RTM), following similar principles to photoacoustic imaging
//! but adapted for seismic wave propagation and subsurface imaging.
//!
//! ## Literature References
//!
//! 1. **Virieux & Operto (2009)**: "An overview of full-waveform inversion in
//!    exploration geophysics", Geophysics, 74(6), WCC1-WCC26
//! 2. **Baysal et al. (1983)**: "Reverse time migration", Geophysics, 48(11), 1514-1524
//! 3. **Tarantola (1984)**: "Inversion of seismic reflection data in the acoustic
//!    approximation", Geophysics, 49(8), 1259-1266
//! 4. **Plessix (2006)**: "A review of the adjoint-state method for computing the
//!    gradient of a functional with geophysical applications", Geophys. J. Int.

use crate::error::{KwaversResult, KwaversError, ValidationError};
use crate::grid::Grid;
use crate::medium::Medium;
use crate::solver::Solver;
use crate::sensor::SensorData;
use super::{Reconstructor, ReconstructionConfig, InterpolationMethod};
use ndarray::{Array2, Array3, Array4, Zip, s};
use std::f64::consts::PI;
use serde::{Serialize, Deserialize};

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
                sampling_frequency: 100.0,
                algorithm: super::ReconstructionAlgorithm::BackProjection,
                filter: super::FilterType::None,
                interpolation: InterpolationMethod::Linear,
            },
            fwi_iterations: 50,
            fwi_tolerance: 1e-6,
            regularization_lambda: 0.01,
            enable_multiscale: true,
            frequency_bands: vec![(2.0, 5.0), (5.0, 10.0), (10.0, 20.0)],
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
    
    /// Forward modeling to compute synthetic wavefield
    fn forward_modeling(
        &self,
        source_pos: &[f64; 3],
        grid: &Grid,
        medium: &dyn Medium,
    ) -> KwaversResult<Array4<f64>> {
        // Initialize fields: [time, x, y, z]
        let nt = 1000; // Number of time steps
        let mut wavefield = Array4::zeros((nt, grid.nx, grid.ny, grid.nz));
        
        // Add point source at specified location
        let source_i = (source_pos[0] / grid.dx) as usize;
        let source_j = (source_pos[1] / grid.dy) as usize;
        let source_k = (source_pos[2] / grid.dz) as usize;
        
        // Simple Ricker wavelet source
        let dt = 1e-3; // 1 ms time step
        let f0 = 20.0; // 20 Hz dominant frequency
        
        for t in 1..nt-1 {
            let time = t as f64 * dt;
            let ricker = self.ricker_wavelet(time, f0);
            
            // Apply source term
            if source_i < grid.nx && source_j < grid.ny && source_k < grid.nz {
                wavefield[[t, source_i, source_j, source_k]] += ricker;
            }
            
            // Wave propagation using simple finite differences
            // This is a simplified acoustic wave equation: ∂²p/∂t² = c²∇²p
            for i in 1..grid.nx-1 {
                for j in 1..grid.ny-1 {
                    for k in 1..grid.nz-1 {
                        let x = i as f64 * grid.dx;
                        let y = j as f64 * grid.dy;
                        let z = k as f64 * grid.dz;
                        let c = self.velocity_model[[i, j, k]];
                        
                        // Laplacian using central differences
                        let laplacian = (wavefield[[t-1, i+1, j, k]] - 2.0*wavefield[[t-1, i, j, k]] + wavefield[[t-1, i-1, j, k]]) / (grid.dx * grid.dx)
                            + (wavefield[[t-1, i, j+1, k]] - 2.0*wavefield[[t-1, i, j, k]] + wavefield[[t-1, i, j-1, k]]) / (grid.dy * grid.dy)
                            + (wavefield[[t-1, i, j, k+1]] - 2.0*wavefield[[t-1, i, j, k]] + wavefield[[t-1, i, j, k-1]]) / (grid.dz * grid.dz);
                        
                        // Wave equation update
                        wavefield[[t, i, j, k]] = 2.0*wavefield[[t-1, i, j, k]] - wavefield[[t-2, i, j, k]] 
                            + (dt * dt) * c * c * laplacian;
                    }
                }
            }
        }
        
        Ok(wavefield)
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
    
    /// Compute gradient contribution using zero-lag cross-correlation
    fn compute_gradient_contribution(
        &mut self,
        forward_wavefield: &Array4<f64>,
        adjoint_wavefield: &Array4<f64>,
        grid: &Grid,
    ) -> KwaversResult<()> {
        let nt = forward_wavefield.shape()[0];
        
        // Gradient using the formula: ∇J = -2/c³ * ∫ ∂p_f/∂t * ∂p_a/∂t dt
        for i in 0..grid.nx {
            for j in 0..grid.ny {
                for k in 0..grid.nz {
                    let mut correlation = 0.0;
                    
                    for t in 1..nt-1 {
                        // Approximate time derivatives
                        let dpf_dt = forward_wavefield[[t+1, i, j, k]] - forward_wavefield[[t-1, i, j, k]];
                        let dpa_dt = adjoint_wavefield[[t+1, i, j, k]] - adjoint_wavefield[[t-1, i, j, k]];
                        
                        correlation += dpf_dt * dpa_dt;
                    }
                    
                    let c = self.velocity_model[[i, j, k]];
                    self.gradient[[i, j, k]] += -2.0 * correlation / (c * c * c);
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
    
    /// Line search to find optimal step size
    fn line_search(
        &self,
        observed_data: &Array2<f64>,
        source_positions: &[[f64; 3]],
        receiver_positions: &[[f64; 3]],
        grid: &Grid,
        medium: &dyn Medium,
    ) -> KwaversResult<f64> {
        // Simple backtracking line search
        let mut alpha = 1.0;
        let c1 = 1e-4; // Armijo constant
        let current_misfit = self.compute_data_misfit(observed_data, source_positions, receiver_positions, grid, medium)?;
        
        // Gradient dot product with search direction
        let grad_dot_dir: f64 = Zip::from(&self.gradient)
            .and(&self.search_direction)
            .fold(0.0, |acc, &g, &d| acc + g * d);
        
        for _ in 0..10 {
            // Test velocity model
            let test_velocity = &self.velocity_model + alpha * &self.search_direction;
            
            // Compute misfit with test model
            let mut temp_fwi = self.clone();
            temp_fwi.velocity_model = test_velocity;
            let test_misfit = temp_fwi.compute_data_misfit(observed_data, source_positions, receiver_positions, grid, medium)?;
            
            // Armijo condition
            if test_misfit <= current_misfit + c1 * alpha * grad_dot_dir {
                break;
            }
            
            alpha *= 0.5;
        }
        
        Ok(alpha)
    }
    
    /// Update velocity model
    fn update_velocity_model(&mut self, step_size: f64) -> KwaversResult<()> {
        self.velocity_model = &self.velocity_model + step_size * &self.search_direction;
        
        // Apply bounds to keep velocities physical
        self.velocity_model.mapv_inplace(|v| v.clamp(1000.0, 8000.0));
        
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