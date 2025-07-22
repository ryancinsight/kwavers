// src/physics/mechanics/acoustic_wave/nonlinear/optimized.rs
//! Optimized nonlinear wave implementation using zero-cost iterator abstractions

use crate::grid::Grid;
use crate::medium::Medium;
use crate::source::Source;
use crate::solver::PRESSURE_IDX;
use crate::utils::iterators::{GradientComputer, ChunkedProcessor};
use crate::utils::{fft_3d, ifft_3d};
use ndarray::{Array3, Array4, Axis, Zip};
use rayon::prelude::*;
use num_complex::Complex;
use log::{debug, warn, trace};
use std::time::Instant;

/// Optimized nonlinear wave solver using iterator patterns
#[derive(Debug, Clone)]
pub struct OptimizedNonlinearWave {
    // Performance metrics
    nonlinear_time: f64,
    fft_time: f64,
    source_time: f64,
    combination_time: f64,
    call_count: usize,
    
    // Physical model settings
    nonlinearity_scaling: f64,
    use_adaptive_timestep: bool,
    k_space_correction_order: usize,
    
    // Precomputed arrays
    k_squared: Option<Array3<f64>>,
    
    // Stability parameters
    max_pressure: f64,
    stability_threshold: f64,
    cfl_safety_factor: f64,
    clamp_gradients: bool,
    
    // Iterator optimization settings
    chunk_size: usize,
    use_chunked_processing: bool,
}

impl OptimizedNonlinearWave {
    /// Create a new optimized nonlinear wave solver
    pub fn new(grid: &Grid) -> Self {
        debug!("Initializing OptimizedNonlinearWave solver with iterator patterns");
        
        let k_squared = Some(grid.k_squared());
        let total_points = grid.nx * grid.ny * grid.nz;
        let chunk_size = if total_points > 1_000_000 {
            64 * 1024
        } else if total_points > 100_000 {
            16 * 1024
        } else {
            4 * 1024
        };
        
        Self {
            nonlinear_time: 0.0,
            fft_time: 0.0,
            source_time: 0.0,
            combination_time: 0.0,
            call_count: 0,
            nonlinearity_scaling: 1.0,
            use_adaptive_timestep: false,
            k_space_correction_order: 2,
            k_squared,
            max_pressure: 1e8,
            stability_threshold: 0.5,
            cfl_safety_factor: 0.8,
            clamp_gradients: true,
            chunk_size,
            use_chunked_processing: total_points > 10_000,
        }
    }
    
    /// Optimized source term calculation using iterator patterns
    fn compute_source_term_optimized(
        &self,
        source: &dyn Source,
        grid: &Grid,
        t: f64,
    ) -> Array3<f64> {
        let (nx, ny, nz) = (grid.nx, grid.ny, grid.nz);
        let mut src_term_array = Array3::<f64>::zeros((nx, ny, nz));
        
        Zip::indexed(&mut src_term_array)
            .for_each(|(i, j, k), src_val| {
                let x = i as f64 * grid.dx;
                let y = j as f64 * grid.dy;
                let z = k as f64 * grid.dz;
                *src_val = source.get_source_term(t, x, y, z, grid);
            });
        
        src_term_array
    }
    
    /// Optimized nonlinear term calculation using iterator patterns
    fn compute_nonlinear_term_optimized(
        &self,
        pressure_current: &Array3<f64>,
        pressure_prev: &Array3<f64>,
        grid: &Grid,
        medium: &dyn Medium,
        dt: f64,
    ) -> Array3<f64> {
        let (nx, ny, nz) = (grid.nx, grid.ny, grid.nz);
        let mut nonlinear_term = Array3::<f64>::zeros((nx, ny, nz));
        
        let min_grid_spacing = grid.dx.min(grid.dy).min(grid.dz);
        let max_gradient = if min_grid_spacing > 1e-9 { 
            self.max_pressure / min_grid_spacing 
        } else { 
            self.max_pressure 
        };
        
        // Use gradient computer for interior points
        let gradient_computer = GradientComputer::new(pressure_current.view());
        
        gradient_computer.compute_interior_gradients(
            grid.dx, grid.dy, grid.dz,
            |grad_x, grad_y, grad_z, i, j, k| {
                let x = i as f64 * grid.dx;
                let y = j as f64 * grid.dy;
                let z = k as f64 * grid.dz;
                
                let rho = medium.density(x, y, z, grid).max(1e-9);
                let c = medium.sound_speed(x, y, z, grid).max(1e-9);
                let b_a = medium.nonlinearity_coefficient(x, y, z, grid);
                let gradient_scale = dt / (2.0 * rho * c * c);
                
                let (grad_x_clamped, grad_y_clamped, grad_z_clamped) = if self.clamp_gradients {
                    (
                        grad_x.clamp(-max_gradient, max_gradient),
                        grad_y.clamp(-max_gradient, max_gradient),
                        grad_z.clamp(-max_gradient, max_gradient)
                    )
                } else {
                    (grad_x, grad_y, grad_z)
                };
                
                let grad_magnitude_sq = grad_x_clamped.powi(2) + grad_y_clamped.powi(2) + grad_z_clamped.powi(2);
                let grad_magnitude = grad_magnitude_sq.sqrt();
                let beta = b_a / (rho * c * c);
                let p_limited = pressure_current[[i, j, k]].clamp(-self.max_pressure, self.max_pressure);
                let nl_term_calc = -beta * self.nonlinearity_scaling * gradient_scale * p_limited * grad_magnitude;
                
                nonlinear_term[[i, j, k]] = if nl_term_calc.is_finite() {
                    nl_term_calc.clamp(-self.max_pressure, self.max_pressure)
                } else { 
                    0.0 
                };
            }
        );
        
        nonlinear_term
    }
}

// Implementation of the AcousticWaveModel trait
use crate::physics::traits::AcousticWaveModel;

impl AcousticWaveModel for OptimizedNonlinearWave {
    fn update_wave(
        &mut self,
        fields: &mut Array4<f64>,
        prev_pressure: &Array3<f64>,
        source: &dyn Source,
        grid: &Grid,
        medium: &dyn Medium,
        dt: f64,
        t: f64,
    ) {
        let start_total = Instant::now();
        self.call_count += 1;
        
        let pressure_at_start = fields.index_axis(Axis(0), PRESSURE_IDX).to_owned();
        
        let (nx, ny, nz) = (grid.nx, grid.ny, grid.nz);
        if nx == 0 || ny == 0 || nz == 0 {
            trace!("Wave update skipped for empty grid at t={}", t);
            return;
        }
        
        // Optimized source term calculation
        let start_source = Instant::now();
        let src_term_array = self.compute_source_term_optimized(source, grid, t);
        self.source_time += start_source.elapsed().as_secs_f64();
        
        // Optimized nonlinear term calculation
        let start_nonlinear = Instant::now();
        let nonlinear_term = self.compute_nonlinear_term_optimized(
            &pressure_at_start,
            prev_pressure,
            grid,
            medium,
            dt,
        );
        self.nonlinear_time += start_nonlinear.elapsed().as_secs_f64();
        
        // FFT processing (simplified for now)
        let start_fft = Instant::now();
        let p_fft = fft_3d(fields, PRESSURE_IDX, grid);
        let p_linear = ifft_3d(&p_fft, grid);
        self.fft_time += start_fft.elapsed().as_secs_f64();
        
        // Optimized field combination
        let start_combine = Instant::now();
        let mut p_output_view = fields.index_axis_mut(Axis(0), PRESSURE_IDX);
        
        Zip::from(&mut p_output_view)
            .and(&p_linear)
            .and(&nonlinear_term)
            .and(&src_term_array)
            .for_each(|p_out, &p_lin_val, &nl_val, &src_val| {
                *p_out = p_lin_val + nl_val + src_val;
            });
        
        self.combination_time += start_combine.elapsed().as_secs_f64();
        
        trace!("Optimized wave update for t={} completed in {:.3e} s", t, start_total.elapsed().as_secs_f64());
    }
    
    fn report_performance(&self) {
        if self.call_count == 0 {
            debug!("No calls to OptimizedNonlinearWave::update_wave yet");
            return;
        }
        
        let total_time = self.nonlinear_time + self.fft_time + self.source_time + self.combination_time;
        let avg_time = if self.call_count > 0 { total_time / self.call_count as f64 } else { 0.0 };
        
        debug!(
            "OptimizedNonlinearWave performance (avg over {} calls):",
            self.call_count
        );
        debug!("  Total time per call:   {:.3e} s", avg_time);
        debug!("  Chunk size used:       {}", self.chunk_size);
        debug!("  Chunked processing:    {}", self.use_chunked_processing);
        
        if total_time > 0.0 {
            debug!(
                "  Nonlinear term calc:    {:.3e} s ({:.1}%)",
                self.nonlinear_time / self.call_count as f64,
                100.0 * self.nonlinear_time / total_time
            );
            debug!(
                "  FFT operations:         {:.3e} s ({:.1}%)",
                self.fft_time / self.call_count as f64,
                100.0 * self.fft_time / total_time
            );
            debug!(
                "  Source application:     {:.3e} s ({:.1}%)",
                self.source_time / self.call_count as f64,
                100.0 * self.source_time / total_time
            );
            debug!(
                "  Field combination:      {:.3e} s ({:.1}%)",
                self.combination_time / self.call_count as f64,
                100.0 * self.combination_time / total_time
            );
        }
    }
    
    fn set_nonlinearity_scaling(&mut self, scaling: f64) {
        assert!(scaling > 0.0, "Nonlinearity scaling must be positive");
        self.nonlinearity_scaling = scaling;
    }
    
    fn set_k_space_correction_order(&mut self, order: usize) {
        self.k_space_correction_order = order;
    }
}
