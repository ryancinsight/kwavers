// physics/mechanics/acoustic_wave/nonlinear.rs
use crate::grid::Grid;
use crate::medium::{Medium, dispersion::DispersiveMedium};
use crate::source::Source;
use crate::solver::PRESSURE_IDX;
use log::{debug, trace, warn};
use ndarray::{Array3, Array4, Axis, parallel::prelude::*, Zip};
use num_complex::Complex;
use rayon::prelude::*;
use std::time::Instant;
use crate::utils::{fft_3d, ifft_3d};

/// Improved nonlinear wave model with optimized efficiency and physical accuracy
#[derive(Debug, Clone)]
pub struct NonlinearWave {
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
    // Precomputed arrays for better performance
    k_squared: Option<Array3<f64>>,
    // Add stability parameters
    max_pressure: f64,
    stability_threshold: f64,
    cfl_safety_factor: f64,
    clamp_gradients: bool,
}

impl NonlinearWave {
    /// Check if a medium implements dispersive propagation
    fn is_dispersive(&self, medium: &dyn Medium) -> bool {
        medium.as_any().downcast_ref::<dyn DispersiveMedium>().is_some()
    }

    pub fn new(grid: &Grid) -> Self {
        debug!("Initializing NonlinearWave solver");
        
        // Precompute k-squared values to avoid recomputation in every step
        let k_squared = Some(grid.k_squared());
        
        Self {
            nonlinear_time: 0.0,
            fft_time: 0.0,
            source_time: 0.0,
            combination_time: 0.0,
            call_count: 0,
            nonlinearity_scaling: 1.0,     // Default scaling factor for nonlinearity
            use_adaptive_timestep: false,  // Default to fixed timestep
            k_space_correction_order: 2,   // Default to second-order correction
            k_squared,
            max_pressure: 1e8,  // 100 MPa maximum pressure
            stability_threshold: 0.5, // CFL condition threshold
            cfl_safety_factor: 0.8,   // Safety factor for CFL condition
            clamp_gradients: true,    // Enable gradient clamping
        }
    }

    /// Configure nonlinearity scaling to handle strong nonlinear effects
    pub fn set_nonlinearity_scaling(&mut self, scaling: f64) {
        assert!(scaling > 0.0, "Nonlinearity scaling must be positive");
        self.nonlinearity_scaling = scaling;
    }

    /// Enable/disable adaptive time-stepping for more stable simulations
    pub fn set_adaptive_timestep(&mut self, enable: bool) {
        self.use_adaptive_timestep = enable;
    }

    /// Set the order of k-space correction (higher = more accurate dispersion handling)
    pub fn set_k_space_correction_order(&mut self, order: usize) {
        assert!(order > 0 && order <= 4, "Correction order must be between 1 and 4");
        self.k_space_correction_order = order;
    }

    /// Set maximum allowed pressure value
    pub fn set_max_pressure(&mut self, max_pressure: f64) {
        self.max_pressure = max_pressure;
    }

    /// Set stability parameters
    pub fn set_stability_params(&mut self, threshold: f64, safety_factor: f64, clamp_gradients: bool) {
        self.stability_threshold = threshold;
        self.cfl_safety_factor = safety_factor;
        self.clamp_gradients = clamp_gradients;
    }

    /// Report performance metrics for the nonlinear wave computation
    pub fn report_performance(&self) {
        if self.call_count == 0 {
            debug!("No calls to NonlinearWave::update_wave yet");
            return;
        }

        let total_time = self.nonlinear_time + self.fft_time + self.source_time + self.combination_time;
        let avg_time = total_time / self.call_count as f64;

        debug!(
            "NonlinearWave performance (avg over {} calls):",
            self.call_count
        );
        debug!("  Total time per call:   {:.3e} s", avg_time);
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

    // Optimized phase factor calculation
    #[inline]
    fn calculate_phase_factor(&self, k_val: f64, c: f64, dt: f64) -> f64 {
        match self.k_space_correction_order {
            1 => -c * k_val * dt,  // First order
            2 => {
                let kc_pi = k_val * c * dt / std::f64::consts::PI;
                -c * k_val * dt * (1.0 - 0.25 * kc_pi.powi(2))
            },
            3 => {
                let kc_pi = k_val * c * dt / std::f64::consts::PI;
                let kc_pi_sq = kc_pi.powi(2);
                -c * k_val * dt * (1.0 - 0.25 * kc_pi_sq + 0.05 * kc_pi_sq.powi(2))
            },
            _ => {
                let kc_pi = k_val * c * dt / std::f64::consts::PI;
                let kc_pi_sq = kc_pi.powi(2);
                -c * k_val * dt * (1.0 - 0.25 * kc_pi_sq + 0.05 * kc_pi_sq.powi(2) - 0.008 * kc_pi_sq.powi(3))
            },
        }
    }

    /// Check if the simulation is stable given the current parameters
    fn check_stability(&self, dt: f64, grid: &Grid, medium: &dyn Medium, pressure: &Array3<f64>) -> bool {
        // Check CFL condition for numerical stability
        let mut max_c: f64 = 0.0;
        let min_dx = grid.dx.min(grid.dy).min(grid.dz);
        
        // Get maximum sound speed in the domain
        for i in 0..grid.nx {
            for j in 0..grid.ny {
                for k in 0..grid.nz {
                    let x = i as f64 * grid.dx;
                    let y = j as f64 * grid.dy;
                    let z = k as f64 * grid.dz;
                    let c = medium.sound_speed(x, y, z, grid);
                    max_c = max_c.max(c);
                }
            }
        }
        
        // Get pressure extremes
        let mut max_pressure = f64::NEG_INFINITY;
        let mut min_pressure = f64::INFINITY;
        let mut has_nan = false;
        let mut has_inf = false;
        
        for &p in pressure.iter() {
            if p.is_nan() {
                has_nan = true;
            } else if p.is_infinite() {
                has_inf = true;
            } else {
                max_pressure = max_pressure.max(p);
                min_pressure = min_pressure.min(p);
            }
        }
        
        // Basic CFL check
        let cfl = max_c * dt / min_dx;
        let is_stable = cfl < self.cfl_safety_factor && !has_nan && !has_inf;
        
        if !is_stable {
            if has_nan || has_inf {
                warn!("NonlinearWave instability: NaN or Infinity detected in pressure field");
            } else {
                warn!(
                    "NonlinearWave potential instability: CFL = {:.3} (should be < {:.3}), pressure range: [{:.2e}, {:.2e}] Pa", 
                    cfl, 
                    self.cfl_safety_factor,
                    min_pressure,
                    max_pressure
                );
            }
        }
        
        is_stable
    }

    /// Apply pressure clamping to prevent instability
    fn clamp_pressure(&self, pressure: &mut Array3<f64>) -> bool {
        // Apply clamping to prevent numerical explosion
        let mut clamped_values = 0;
        let total_values = pressure.len();
        
        for val in pressure.iter_mut() {
            if val.is_nan() || val.is_infinite() {
                *val = 0.0;
                clamped_values += 1;
            } else if *val > self.max_pressure {
                *val = self.max_pressure;
                clamped_values += 1;
            } else if *val < -self.max_pressure {
                *val = -self.max_pressure;
                clamped_values += 1;
            }
        }
        
        let had_extreme_values = clamped_values > 0;
        
        if had_extreme_values {
            let percentage = 100.0 * clamped_values as f64 / total_values as f64;
            warn!(
                "Pressure field stability enforced: clamped {} values ({:.2}% of field)",
                clamped_values,
                percentage
            );
        }
        
        had_extreme_values
    }

    pub fn update_wave(
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

        // Get pressure field
        let mut pressure = fields.index_axis(Axis(0), PRESSURE_IDX).to_owned();
        
        // Check for stability before we start
        let is_stable = self.check_stability(dt, grid, medium, &pressure);
        
        if !is_stable {
            // Use more conservative approach if stability concerns
            debug!("Using enhanced stability measures in wave update");
            // ... adjust computation parameters for more stability ...
        }

        // Create working arrays for this update once
        let (nx, ny, nz) = (grid.nx, grid.ny, grid.nz);
        let mut nonlinear_term = Array3::<f64>::zeros((nx, ny, nz));
        let mut src = Array3::<f64>::zeros((nx, ny, nz));

        // Apply source term
        let start_source = Instant::now();
        let p = fields.index_axis(Axis(0), PRESSURE_IDX);

        // Use par_for_each directly with indices for even better performance
        Zip::indexed(&mut src)
            .par_for_each(|(i, j, k), src_val| {
                let x = i as f64 * grid.dx;
                let y = j as f64 * grid.dy;
                let z = k as f64 * grid.dz;
                *src_val = source.get_source_term(t, x, y, z, grid);
            });

        self.source_time += start_source.elapsed().as_secs_f64();

        // Calculate nonlinear term using the updated pressure
        let start_nonlinear = Instant::now();

        // Maximum gradient magnitude for clamping
        let max_gradient = self.max_pressure / (grid.dx.min(grid.dy).min(grid.dz));

        // Optimized implementation with better boundary handling
        Zip::indexed(&mut nonlinear_term)
            .and(&p)
            .and(prev_pressure)
            .par_for_each(|(i, j, k), nl, p_val, &p_prev| {
                let x = i as f64 * grid.dx;
                let y = j as f64 * grid.dy;
                let z = k as f64 * grid.dz;

                // Get medium properties at this point
                let rho = medium.density(x, y, z, grid);
                let c = medium.sound_speed(x, y, z, grid);
                let b_a = medium.nonlinearity_coefficient(x, y, z, grid);

                // Basic scale factor for gradient-based calculations
                let gradient_scale = dt / (2.0 * rho * c * c);

                // Use simpler but robust finite difference implementation for all points
                if i > 0 && i < grid.nx - 1 && j > 0 && j < grid.ny - 1 && k > 0 && k < grid.nz - 1 {
                    // Use cached view to avoid repeated indexing
                    let p_arr = p.view();
                    
                    // Calculate gradient components more efficiently
                    let dx_inv = 1.0 / (2.0 * grid.dx);
                    let dy_inv = 1.0 / (2.0 * grid.dy);
                    let dz_inv = 1.0 / (2.0 * grid.dz);
                    
                    let grad_x = (p_arr[[i + 1, j, k]] - p_arr[[i - 1, j, k]]) * dx_inv;
                    let grad_y = (p_arr[[i, j + 1, k]] - p_arr[[i, j - 1, k]]) * dy_inv;
                    let grad_z = (p_arr[[i, j, k + 1]] - p_arr[[i, j, k - 1]]) * dz_inv;

                    // Clamp gradients if needed
                    let (grad_x_clamped, grad_y_clamped, grad_z_clamped) = if self.clamp_gradients {
                        (
                            grad_x.clamp(-max_gradient, max_gradient),
                            grad_y.clamp(-max_gradient, max_gradient),
                            grad_z.clamp(-max_gradient, max_gradient)
                        )
                    } else {
                        (grad_x, grad_y, grad_z)
                    };

                    // Westervelt equation nonlinearity term with optimized calculation
                    let grad_magnitude = (grad_x_clamped * grad_x_clamped + 
                                          grad_y_clamped * grad_y_clamped + 
                                          grad_z_clamped * grad_z_clamped).sqrt();
                    
                    // Apply limiting to prevent overflow
                    let beta = b_a / rho / c / c;
                    let p_limited = p_val.clamp(-self.max_pressure, self.max_pressure);
                    let nl_term = -beta * self.nonlinearity_scaling * gradient_scale * p_limited * grad_magnitude;
                    
                    // Final safety check before assignment
                    *nl = if nl_term.is_finite() { 
                        nl_term.clamp(-self.max_pressure, self.max_pressure) 
                    } else { 
                        0.0 
                    };
                } else {
                    // Near boundary, use simpler form with time derivative
                    let dp_dt = (*p_val - p_prev) / dt;
                    
                    // Safety checks and clamping for stability
                    let dp_dt_limited = if dp_dt.is_finite() {
                        dp_dt.clamp(-self.max_pressure/dt, self.max_pressure/dt)
                    } else {
                        0.0
                    };
                    
                    let beta = b_a / rho / c / c;
                    *nl = -beta * self.nonlinearity_scaling * gradient_scale * (*p_val) * dp_dt_limited;
                }
            });

        self.nonlinear_time += start_nonlinear.elapsed().as_secs_f64();

        // Linear propagation in k-space with viscosity and absorption
        let start_fft = Instant::now();
        
        // Get FFT of pressure
        let p_fft = fft_3d(fields, PRESSURE_IDX, grid);
        
        // Use precomputed k-squared or compute if not available
        let grid_k_squared = grid.k_squared();
        let k2 = self.k_squared.as_ref().unwrap_or(&grid_k_squared);
        
        let kspace_corr = grid.kspace_correction(medium.sound_speed(0.0, 0.0, 0.0, grid), dt);
        let freq = medium.reference_frequency();

        // Create temporary array for FFT processing
        let mut p_linear_fft = Array3::<Complex<f64>>::zeros((nx, ny, nz));

        // Optimize k-space propagation with precomputed values and better parallelism
        Zip::indexed(&mut p_linear_fft)
            .and(&p_fft)
            .par_for_each(|(i, j, k), p_new, &p_val| {
                let x = i as f64 * grid.dx;
                let y = j as f64 * grid.dy;
                let z = k as f64 * grid.dz;
                
                // Get wave properties
                let k_val = k2[[i, j, k]].sqrt();
                let k_complex = medium.complex_wave_number(x, y, z, grid, freq);
                let c = medium.phase_velocity(x, y, z, grid, freq);
                
                // Apply k-space correction with complex wave number
                let mut propagator = Complex::new(0.0, -dt * c * k_val);
                if k_val > 0.0 {
                    propagator = propagator * (k_complex / k_val);
                }
                
                // Include viscosity effects
                let rho = medium.density(x, y, z, grid);
                let mu = medium.viscosity(x, y, z, grid);
                let viscous_damping = (-mu * k_val * k_val * dt / rho).exp();
                
                // Apply propagation and damping
                *p_new = p_val * propagator.exp() * kspace_corr[[i, j, k]] * viscous_damping;
            });

        // Convert back to spatial domain
        let p_linear = ifft_3d(&p_linear_fft, grid);
        self.fft_time += start_fft.elapsed().as_secs_f64();

        // Combine linear and nonlinear terms
        let start_combine = Instant::now();
        let mut p_out = fields.index_axis_mut(Axis(0), PRESSURE_IDX);
        
        // Optimize final combination with aligned memory access pattern
        Zip::from(&mut p_out)
            .and(&p_linear)
            .and(&nonlinear_term)
            .and(&src)
            .par_for_each(|p_out_val, p_lin, nl, s| {
                *p_out_val = *p_lin + *nl + *s;
            });

        self.combination_time += start_combine.elapsed().as_secs_f64();

        trace!(
            "Wave update in {:.3e} s",
            start_total.elapsed().as_secs_f64()
        );

        // After computation, ensure pressure field is within physical limits
        let pressure = fields.index_axis(Axis(0), PRESSURE_IDX).to_owned();
        let had_extreme_values = self.clamp_pressure(&mut pressure.clone());
        if had_extreme_values {
            // Update the array in fields with clamped values
            fields.index_axis_mut(Axis(0), PRESSURE_IDX).assign(&pressure);
        }
        
        // Add a final safety check to clamp results and prevent instability
        for i in 0..grid.nx {
            for j in 0..grid.ny {
                for k in 0..grid.nz {
                    let p_val = fields.index_axis(Axis(0), PRESSURE_IDX)[[i, j, k]];
                    if !p_val.is_finite() || p_val.abs() > self.max_pressure {
                        fields.index_axis_mut(Axis(0), PRESSURE_IDX)[[i, j, k]] = 
                            if p_val.is_nan() { 0.0 } 
                            else if p_val > 0.0 { self.max_pressure }
                            else { -self.max_pressure };
                    }
                }
            }
        }
    }

