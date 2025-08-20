//! Full Waveform Inversion (FWI) implementation
//!
//! Based on:
//! - Virieux & Operto (2009): "An overview of full-waveform inversion"
//! - Tarantola (1984): "Inversion of seismic reflection data"
//! - Plessix (2006): "Adjoint-state method for gradient computation"

use ndarray::{Array2, Array3, Array4, Zip, s};
use crate::error::{KwaversResult, KwaversError};
use crate::grid::Grid;
use crate::medium::Medium;
use crate::solver::reconstruction::{Reconstructor, InterpolationMethod, ReconstructionConfig};
use super::config::SeismicImagingConfig;
use super::constants::*;
use super::wavelet::RickerWavelet;
use super::misfit::{MisfitFunction, MisfitType};
use super::fd_coeffs::{FD_COEFF_0, FD_COEFF_1, FD_COEFF_2};

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
    /// Misfit function calculator
    misfit_function: MisfitFunction,
}

impl FullWaveformInversion {
    /// Create new FWI reconstructor with initial velocity model
    pub fn new(config: SeismicImagingConfig, initial_velocity: Array3<f64>) -> Self {
        let gradient = Array3::zeros(initial_velocity.dim());
        let search_direction = Array3::zeros(initial_velocity.dim());
        let misfit_function = MisfitFunction::new(MisfitType::L2Norm);
        
        Self {
            config,
            velocity_model: initial_velocity,
            gradient,
            search_direction,
            previous_gradient: None,
            misfit_function,
        }
    }
    
    /// Perform FWI iteration
    /// Returns the misfit value and updates the velocity model
    pub fn iterate(
        &mut self,
        observed_data: &Array2<f64>,
        source_positions: &[(usize, usize, usize)],
        receiver_positions: &[(usize, usize, usize)],
        grid: &Grid,
    ) -> KwaversResult<f64> {
        // Step 1: Forward modeling with current velocity model
        let synthetic_data = self.forward_model(source_positions, receiver_positions, grid)?;
        
        // Step 2: Compute misfit
        let misfit = self.misfit_function.compute(observed_data, &synthetic_data)?;
        
        // Step 3: Compute adjoint source
        let adjoint_source = self.misfit_function.compute_adjoint_source(observed_data, &synthetic_data)?;
        
        // Step 4: Backward propagation to compute gradient
        self.compute_gradient(&adjoint_source, source_positions, receiver_positions, grid)?;
        
        // Step 5: Apply regularization
        self.apply_regularization()?;
        
        // Step 6: Update search direction (conjugate gradient)
        self.update_search_direction()?;
        
        // Step 7: Line search for optimal step length
        let step_length = self.line_search(
            observed_data,
            source_positions,
            receiver_positions,
            grid,
            misfit
        )?;
        
        // Step 8: Update velocity model
        Zip::from(&mut self.velocity_model)
            .and(&self.search_direction)
            .for_each(|v, &s| {
                *v -= step_length * s;
                // Apply physical bounds
                *v = v.clamp(MIN_VELOCITY, MAX_VELOCITY);
            });
        
        Ok(misfit)
    }
    
    /// Forward modeling: solve wave equation with current velocity model
    fn forward_model(
        &self,
        source_positions: &[(usize, usize, usize)],
        receiver_positions: &[(usize, usize, usize)],
        grid: &Grid,
    ) -> KwaversResult<Array2<f64>> {
        let n_sources = source_positions.len();
        let n_receivers = receiver_positions.len();
        let n_time_steps = DEFAULT_TIME_STEPS;
        
        // Initialize synthetic data array
        let mut synthetic_data = Array2::zeros((n_receivers, n_time_steps));
        
        // Create Ricker wavelet source
        let wavelet = RickerWavelet::new(DEFAULT_RICKER_FREQUENCY);
        let source_time_function = wavelet.generate_time_series(DEFAULT_TIME_STEP, n_time_steps);
        
        // For each source
        for (source_idx, &source_pos) in source_positions.iter().enumerate() {
            // Initialize wavefield
            let mut pressure = Array3::zeros((grid.nx, grid.ny, grid.nz));
            let mut pressure_old = Array3::zeros((grid.nx, grid.ny, grid.nz));
            
            // Time stepping
            for t in 0..n_time_steps {
                // Apply source
                pressure[source_pos] += source_time_function[t];
                
                // Update wavefield using finite differences
                self.update_wavefield(&mut pressure, &pressure_old, grid)?;
                
                // Record at receivers
                for (rec_idx, &rec_pos) in receiver_positions.iter().enumerate() {
                    synthetic_data[[rec_idx, t]] += pressure[rec_pos];
                }
                
                // Swap time levels
                std::mem::swap(&mut pressure, &mut pressure_old);
            }
        }
        
        // Average over sources
        synthetic_data /= n_sources as f64;
        
        Ok(synthetic_data)
    }
    
    /// Update wavefield using 4th-order finite differences
    fn update_wavefield(
        &self,
        pressure: &mut Array3<f64>,
        pressure_old: &Array3<f64>,
        grid: &Grid,
    ) -> KwaversResult<()> {
        let dt = DEFAULT_TIME_STEP;
        let (nx, ny, nz) = pressure.dim();
        
        // Compute Laplacian using 4th-order stencil
        let mut laplacian = Array3::zeros((nx, ny, nz));
        
        for i in 2..(nx-2) {
            for j in 2..(ny-2) {
                for k in 2..(nz-2) {
                    // X-direction
                    let d2p_dx2 = (FD_COEFF_2 * pressure[[i-2, j, k]]
                                + FD_COEFF_1 * pressure[[i-1, j, k]]
                                + FD_COEFF_0 * pressure[[i, j, k]]
                                + FD_COEFF_1 * pressure[[i+1, j, k]]
                                + FD_COEFF_2 * pressure[[i+2, j, k]]) / (grid.dx * grid.dx);
                    
                    // Y-direction
                    let d2p_dy2 = (FD_COEFF_2 * pressure[[i, j-2, k]]
                                + FD_COEFF_1 * pressure[[i, j-1, k]]
                                + FD_COEFF_0 * pressure[[i, j, k]]
                                + FD_COEFF_1 * pressure[[i, j+1, k]]
                                + FD_COEFF_2 * pressure[[i, j+2, k]]) / (grid.dy * grid.dy);
                    
                    // Z-direction
                    let d2p_dz2 = (FD_COEFF_2 * pressure[[i, j, k-2]]
                                + FD_COEFF_1 * pressure[[i, j, k-1]]
                                + FD_COEFF_0 * pressure[[i, j, k]]
                                + FD_COEFF_1 * pressure[[i, j, k+1]]
                                + FD_COEFF_2 * pressure[[i, j, k+2]]) / (grid.dz * grid.dz);
                    
                    laplacian[[i, j, k]] = d2p_dx2 + d2p_dy2 + d2p_dz2;
                }
            }
        }
        
        // Update pressure using wave equation
        Zip::from(pressure)
            .and(&*pressure_old)
            .and(&laplacian)
            .and(&self.velocity_model)
            .for_each(|p, &p_old, &lap, &vel| {
                let vel2_dt2 = vel * vel * dt * dt;
                *p = 2.0 * *p - p_old + vel2_dt2 * lap;
            });
        
        Ok(())
    }
    
    /// Compute gradient using adjoint-state method
    fn compute_gradient(
        &mut self,
        adjoint_source: &Array2<f64>,
        source_positions: &[(usize, usize, usize)],
        receiver_positions: &[(usize, usize, usize)],
        grid: &Grid,
    ) -> KwaversResult<()> {
        // Reset gradient
        self.gradient.fill(0.0);
        
        let n_time_steps = DEFAULT_TIME_STEPS;
        
        // Forward wavefield storage (simplified - in practice would use checkpointing)
        let mut forward_wavefield = Array4::zeros((n_time_steps, grid.nx, grid.ny, grid.nz));
        
        // Recompute forward wavefield and store
        let wavelet = RickerWavelet::new(DEFAULT_RICKER_FREQUENCY);
        let source_time_function = wavelet.generate_time_series(DEFAULT_TIME_STEP, n_time_steps);
        
        for &source_pos in source_positions {
            let mut pressure = Array3::zeros((grid.nx, grid.ny, grid.nz));
            let mut pressure_old = Array3::zeros((grid.nx, grid.ny, grid.nz));
            
            for t in 0..n_time_steps {
                pressure[source_pos] += source_time_function[t];
                self.update_wavefield(&mut pressure, &pressure_old, grid)?;
                forward_wavefield.slice_mut(s![t, .., .., ..]).assign(&pressure);
                std::mem::swap(&mut pressure, &mut pressure_old);
            }
        }
        
        // Backward propagation with adjoint source
        let mut adjoint_wavefield = Array3::zeros((grid.nx, grid.ny, grid.nz));
        let mut adjoint_old = Array3::zeros((grid.nx, grid.ny, grid.nz));
        
        // Time-reversed loop
        for t in (0..n_time_steps).rev() {
            // Apply adjoint sources at receiver positions
            for (rec_idx, &rec_pos) in receiver_positions.iter().enumerate() {
                adjoint_wavefield[rec_pos] += adjoint_source[[rec_idx, t]];
            }
            
            // Update adjoint wavefield
            self.update_wavefield(&mut adjoint_wavefield, &adjoint_old, grid)?;
            
            // Compute gradient: zero-lag correlation
            let forward_slice = forward_wavefield.slice(s![t, .., .., ..]);
            Zip::from(&mut self.gradient)
                .and(&forward_slice)
                .and(&adjoint_wavefield)
                .and(&self.velocity_model)
                .for_each(|g, &f, &a, &v| {
                    // Gradient of velocity with respect to misfit
                    *g += -2.0 * f * a / (v * v * v) * DEFAULT_TIME_STEP;
                });
            
            std::mem::swap(&mut adjoint_wavefield, &mut adjoint_old);
        }
        
        // Scale gradient
        self.gradient *= GRADIENT_SCALING_FACTOR;
        
        Ok(())
    }
    
    /// Apply Tikhonov regularization to gradient
    fn apply_regularization(&mut self) -> KwaversResult<()> {
        let lambda = self.config.regularization_lambda;
        
        if lambda > 0.0 {
            // Add smoothness constraint: λ * ∇²v
            let (nx, ny, nz) = self.gradient.dim();
            let mut regularization = Array3::zeros((nx, ny, nz));
            
            for i in 1..(nx-1) {
                for j in 1..(ny-1) {
                    for k in 1..(nz-1) {
                        // Discrete Laplacian
                        let lap = (self.velocity_model[[i+1, j, k]] 
                                + self.velocity_model[[i-1, j, k]]
                                + self.velocity_model[[i, j+1, k]]
                                + self.velocity_model[[i, j-1, k]]
                                + self.velocity_model[[i, j, k+1]]
                                + self.velocity_model[[i, j, k-1]]
                                - 6.0 * self.velocity_model[[i, j, k]]);
                        
                        regularization[[i, j, k]] = lambda * lap;
                    }
                }
            }
            
            self.gradient += &regularization;
        }
        
        Ok(())
    }
    
    /// Update search direction using conjugate gradient method
    fn update_search_direction(&mut self) -> KwaversResult<()> {
        if let Some(ref prev_grad) = self.previous_gradient {
            // Fletcher-Reeves formula
            let grad_norm_sq = self.gradient.mapv(|x| x * x).sum();
            let prev_norm_sq = prev_grad.mapv(|x| x * x).sum();
            
            if prev_norm_sq > MIN_GRADIENT_NORM {
                let beta = grad_norm_sq / prev_norm_sq;
                
                // Update search direction
                self.search_direction = &self.gradient + beta * &self.search_direction;
            } else {
                // Reset to steepest descent
                self.search_direction.assign(&self.gradient);
            }
        } else {
            // First iteration: use steepest descent
            self.search_direction.assign(&self.gradient);
        }
        
        // Store current gradient for next iteration
        self.previous_gradient = Some(self.gradient.clone());
        
        Ok(())
    }
    
    /// Armijo line search for optimal step length
    fn line_search(
        &self,
        observed_data: &Array2<f64>,
        source_positions: &[(usize, usize, usize)],
        receiver_positions: &[(usize, usize, usize)],
        grid: &Grid,
        current_misfit: f64,
    ) -> KwaversResult<f64> {
        let mut alpha = 1.0; // Initial step length
        let grad_dot_dir = Zip::from(&self.gradient)
            .and(&self.search_direction)
            .fold(0.0, |acc, &g, &d| acc + g * d);
        
        for _ in 0..MAX_LINE_SEARCH_ITERATIONS {
            // Test velocity model
            let mut test_velocity = self.velocity_model.clone();
            Zip::from(&mut test_velocity)
                .and(&self.search_direction)
                .for_each(|v, &s| {
                    *v -= alpha * s;
                    *v = v.clamp(MIN_VELOCITY, MAX_VELOCITY);
                });
            
            // Create temporary FWI with test velocity
            let mut test_fwi = Self::new(self.config.clone(), test_velocity);
            let synthetic_data = test_fwi.forward_model(source_positions, receiver_positions, grid)?;
            let test_misfit = self.misfit_function.compute(observed_data, &synthetic_data)?;
            
            // Armijo condition
            if test_misfit <= current_misfit - ARMIJO_C1 * alpha * grad_dot_dir {
                return Ok(alpha);
            }
            
            // Backtrack
            alpha *= LINE_SEARCH_BACKTRACK;
        }
        
        // Return small step if line search fails
        Ok(1e-6)
    }
    
    /// Get current velocity model
    pub fn get_velocity_model(&self) -> &Array3<f64> {
        &self.velocity_model
    }
    
    /// Get current gradient
    pub fn get_gradient(&self) -> &Array3<f64> {
        &self.gradient
    }
    
    /// Check convergence based on gradient norm
    pub fn is_converged(&self) -> bool {
        let grad_norm = self.gradient.mapv(|x| x * x).sum().sqrt();
        grad_norm < self.config.fwi_tolerance
    }
}

impl Reconstructor for FullWaveformInversion {
    fn name(&self) -> &str {
        "FullWaveformInversion"
    }

    fn reconstruct(
        &self,
        sensor_data: &Array2<f64>,
        sensor_positions: &[[f64; 3]],
        grid: &Grid,
        config: &ReconstructionConfig,
    ) -> KwaversResult<Array3<f64>> {
        // FWI requires mutable state for iterations
        // For now, return a copy of the current velocity model
        // TODO: Refactor to use interior mutability or separate iterator pattern
        Ok(self.velocity_model.clone())
    }
    
}