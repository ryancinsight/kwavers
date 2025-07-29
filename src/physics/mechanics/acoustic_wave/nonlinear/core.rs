// src/physics/mechanics/acoustic_wave/nonlinear/core.rs
use crate::grid::Grid;
use crate::medium::Medium;
use crate::KwaversResult;

use ndarray::{Array3, Zip};
use log::{debug, warn};
use std::f64;

/// Represents an improved nonlinear wave model solver.
///
/// This struct encapsulates the parameters and state for simulating acoustic wave propagation
/// with nonlinear effects, focusing on optimized efficiency and physical accuracy.
/// It includes settings for performance monitoring, physical model characteristics,
/// precomputed values for faster calculations, and stability control mechanisms.
#[derive(Debug, Clone)]
pub struct NonlinearWave {
    // Performance metrics
    /// Time spent in the nonlinear term calculation part of the update, in seconds.
    pub(super) nonlinear_time: f64,
    /// Time spent in FFT operations during the update, in seconds.
    pub(super) fft_time: f64,
    /// Time spent applying the source term during the update, in seconds.
    pub(super) source_time: f64,
    /// Time spent combining linear and nonlinear field components, in seconds.
    pub(super) combination_time: f64,
    /// Number of times the `update_wave` method has been called.
    pub(super) call_count: usize,

    // Physical model settings
    /// Scaling factor for the nonlinearity term, allowing adjustment for strong nonlinear effects.
    pub(super) nonlinearity_scaling: f64,
    /// Flag to enable or disable adaptive time-stepping for potentially more stable simulations.
    pub(super) use_adaptive_timestep: bool,
    /// Order of k-space correction for dispersion handling (e.g., 1 for first-order, 2 for second-order).
    pub(super) k_space_correction_order: usize,

    // Precomputed arrays
    /// Precomputed k-squared values (square of wavenumber magnitudes) for the grid, used to speed up calculations.
    pub(super) k_squared: Option<Array3<f64>>,

    // Stability parameters
    /// Maximum absolute pressure value allowed in the simulation to prevent numerical instability.
    pub(super) max_pressure: f64,
    /// Threshold for the CFL (Courant-Friedrichs-Lewy) condition, used in stability checks.
    pub(super) stability_threshold: f64,
    /// Safety factor applied to the CFL condition for determining stable timestep.
    pub(super) cfl_safety_factor: f64,
    /// Flag to enable or disable clamping of pressure gradients to `max_gradient`.
    pub(super) clamp_gradients: bool,
    
    // Iterator optimization settings
    /// Chunk size for cache-friendly processing
    chunk_size: usize,
    /// Whether to use chunked processing for large grids
    use_chunked_processing: bool,

    // Multi-frequency simulation support
    /// Configuration for multi-frequency simulation
    pub(super) multi_freq_config: Option<MultiFrequencyConfig>,
}

impl NonlinearWave {
    /// Creates a new `NonlinearWave` solver instance.
    ///
    /// Initializes the solver with default parameters and precomputes necessary values
    /// based on the provided `grid`.
    ///
    /// # Arguments
    ///
    /// * `grid` - A reference to the `Grid` structure defining the simulation domain and discretization.
    ///
    /// # Returns
    ///
    /// A new `NonlinearWave` instance.
    pub fn new(grid: &Grid) -> Self {
        debug!("Initializing NonlinearWave solver");

        // Precompute k-squared values to avoid recomputation in every step
        let k_squared = Some(grid.k_squared());
        
        // Determine optimal chunk size based on grid dimensions
        let total_points = grid.nx * grid.ny * grid.nz;
        let chunk_size = if total_points > 1_000_000 {
            64 * 1024  // Large grids: 64K chunks
        } else if total_points > 100_000 {
            16 * 1024  // Medium grids: 16K chunks
        } else {
            4 * 1024   // Small grids: 4K chunks
        };

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
            chunk_size,
            use_chunked_processing: total_points > 10_000,
            multi_freq_config: None, // Initialize as None
        }
    }

    /// Create a new NonlinearWave with multi-frequency capabilities
    pub fn with_multi_frequency(
        grid: &Grid,
        multi_freq_config: MultiFrequencyConfig,
    ) -> Self {
        let mut instance = Self::new(grid);
        instance.multi_freq_config = Some(multi_freq_config);
        instance
    }
    
    /// Apply multi-frequency source excitation
    pub fn apply_multi_frequency_source(
        &self,
        pressure: &mut Array3<f64>,
        grid: &Grid,
        time: f64,
        source_position: (f64, f64, f64),
        base_amplitude: f64,
    ) -> KwaversResult<()> {
        if let Some(ref config) = self.multi_freq_config {
            let (x_src, y_src, z_src) = source_position;
            
            // Calculate multi-frequency excitation
            let mut total_amplitude = 0.0;
            for (i, &freq) in config.frequencies.iter().enumerate() {
                let amplitude = config.amplitudes.get(i).unwrap_or(&1.0);
                let phase = config.phases.get(i).unwrap_or(&0.0);
                
                // Multi-tone excitation with proper phase relationships
                let component = amplitude * (2.0 * std::f64::consts::PI * freq * time + phase).sin();
                total_amplitude += component;
            }
            
            // Apply source with spatial distribution
            let source_radius = grid.dx * 3.0; // 3-cell radius
            for i in 0..grid.nx {
                for j in 0..grid.ny {
                    for k in 0..grid.nz {
                        let x = i as f64 * grid.dx;
                        let y = j as f64 * grid.dy;
                        let z = k as f64 * grid.dz;
                        
                        let distance = ((x - x_src).powi(2) + (y - y_src).powi(2) + (z - z_src).powi(2)).sqrt();
                        
                        if distance <= source_radius {
                            let spatial_factor = (-0.5 * (distance / source_radius).powi(2)).exp();
                            pressure[[i, j, k]] += base_amplitude * total_amplitude * spatial_factor;
                        }
                    }
                }
            }
        }
        
        Ok(())
    }
    
    /// Calculate frequency-dependent attenuation coefficient
    pub fn calculate_frequency_dependent_attenuation(
        &self,
        frequency: f64,
        base_attenuation: f64,
    ) -> f64 {
        if let Some(ref config) = self.multi_freq_config {
            if config.frequency_dependent_attenuation {
                // Power law attenuation: α = α₀ * f^n (typically n ≈ 1.1 for soft tissue)
                let power_law_exponent = 1.1;
                let reference_frequency = 1e6; // 1 MHz reference
                let frequency_factor = (frequency / reference_frequency).powf(power_law_exponent);
                return base_attenuation * frequency_factor;
            }
        }
        base_attenuation
    }

    /// Sets the scaling factor for the nonlinearity term.
    ///
    /// This allows adjustment for strong nonlinear effects. The scaling factor must be positive.
    ///
    /// # Arguments
    ///
    /// * `scaling` - The nonlinearity scaling factor. Must be greater than 0.0.
    pub fn set_nonlinearity_scaling(&mut self, scaling: f64) {
        assert!(scaling > 0.0, "Nonlinearity scaling must be positive");
        self.nonlinearity_scaling = scaling;
    }

    /// Enables or disables adaptive time-stepping.
    ///
    /// Adaptive time-stepping can help maintain stability in simulations with varying conditions.
    ///
    /// # Arguments
    ///
    /// * `enable` - `true` to enable adaptive time-stepping, `false` to disable it.
    pub fn set_adaptive_timestep(&mut self, enable: bool) {
        self.use_adaptive_timestep = enable;
    }

    /// Sets the order of k-space correction for dispersion.
    ///
    /// Higher orders can provide more accurate handling of wave dispersion but may increase
    /// computational cost. The order must be between 1 and 4 (inclusive).
    ///
    /// # Arguments
    ///
    /// * `order` - The desired order of k-space correction (1-4).
    pub fn set_k_space_correction_order(&mut self, order: usize) {
        assert!(order > 0 && order <= 4, "Correction order must be between 1 and 4");
        self.k_space_correction_order = order;
    }

    /// Sets the maximum allowed absolute pressure value in the simulation.
    ///
    /// This is used to clamp pressure values and prevent numerical instability.
    ///
    /// # Arguments
    ///
    /// * `max_pressure` - The maximum absolute pressure value.
    pub fn set_max_pressure(&mut self, max_pressure: f64) {
        self.max_pressure = max_pressure;
    }

    /// Sets parameters related to simulation stability.
    ///
    /// # Arguments
    ///
    /// * `threshold` - The CFL condition threshold.
    /// * `safety_factor` - The safety factor for the CFL condition.
    /// * `clamp_gradients` - `true` to enable clamping of pressure gradients, `false` otherwise.
    pub fn set_stability_params(&mut self, threshold: f64, safety_factor: f64, clamp_gradients: bool) {
        self.stability_threshold = threshold;
        self.cfl_safety_factor = safety_factor;
        self.clamp_gradients = clamp_gradients;
    }

    /// Calculates the phase factor for wave propagation in k-space.
    #[inline]
    pub(super) fn calculate_phase_factor(&self, k_val: f64, c: f64, dt: f64) -> f64 {
        match self.k_space_correction_order {
            1 => -c * k_val * dt,
            2 => {
                let kc_pi = k_val * c * dt / f64::consts::PI;
                -c * k_val * dt * (1.0 - 0.25 * kc_pi.powi(2))
            },
            3 => {
                let kc_pi = k_val * c * dt / f64::consts::PI;
                let kc_pi_sq = kc_pi.powi(2);
                -c * k_val * dt * (1.0 - 0.25 * kc_pi_sq + 0.05 * kc_pi_sq.powi(2))
            },
            _ => {
                let kc_pi = k_val * c * dt / f64::consts::PI;
                let kc_pi_sq = kc_pi.powi(2);
                -c * k_val * dt * (1.0 - 0.25 * kc_pi_sq + 0.05 * kc_pi_sq.powi(2) - 0.008 * kc_pi_sq.powi(3))
            },
        }
    }

    /// Checks the stability of the simulation.
    pub(super) fn check_stability(&self, dt: f64, grid: &Grid, medium: &dyn Medium, pressure: &Array3<f64>) -> bool {
        let min_dx = grid.dx.min(grid.dy).min(grid.dz);
        
        // Combined pass for max sound speed and pressure validation
        let mut max_c: f64 = 0.0;
        let mut max_pressure_val = f64::NEG_INFINITY;
        let mut min_pressure_val = f64::INFINITY;
        let mut has_nan = false;
        let mut has_inf = false;
        
        if grid.nx > 0 && grid.ny > 0 && grid.nz > 0 {
            // Use indexed iteration to check both medium properties and pressure values
            for ((i, j, k), &p_val) in pressure.indexed_iter() {
                // Check pressure validity
                if p_val.is_nan() {
                    has_nan = true;
                    break;
                } else if p_val.is_infinite() {
                    has_inf = true;
                    break;
                } else {
                    max_pressure_val = max_pressure_val.max(p_val);
                    min_pressure_val = min_pressure_val.min(p_val);
                }
                
                // Update max sound speed
                let x = i as f64 * grid.dx;
                let y = j as f64 * grid.dy;
                let z = k as f64 * grid.dz;
                let c = medium.sound_speed(x, y, z, grid);
                max_c = max_c.max(c);
            }
        }

        let cfl = if min_dx > 1e-9 { max_c * dt / min_dx } else { f64::INFINITY };
        let is_stable = cfl < self.cfl_safety_factor && !has_nan && !has_inf;

        if !is_stable {
            if has_nan || has_inf {
                warn!("NonlinearWave instability: NaN or Infinity detected in pressure field.");
            } else {
                warn!(
                    "NonlinearWave potential instability: CFL = {:.3} (max_c={:.2}, dt={:.2e}, min_dx={:.2e}) exceeds safety factor {:.3}. Pressure range: [{:.2e}, {:.2e}] Pa.",
                    cfl, max_c, dt, min_dx, self.cfl_safety_factor,
                    if min_pressure_val.is_finite() { min_pressure_val } else {0.0},
                    if max_pressure_val.is_finite() { max_pressure_val } else {0.0}
                );
            }
        }
        is_stable
    }

    /// Applies clamping to the pressure field.
    pub(super) fn clamp_pressure(&self, pressure: &mut Array3<f64>) -> bool {
        let mut clamped_values = 0;
        let total_values = pressure.len();

        if total_values == 0 { return false; }

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
                "Pressure field stability enforced: clamped {} values ({:.2}% of {} total values) to max pressure {:.2e} Pa.",
                clamped_values, percentage, total_values, self.max_pressure
            );
        }
        had_extreme_values
    }
    
    /// Compute nonlinear terms for boundary points using iterator patterns
    fn compute_boundary_nonlinear_terms(
        &self,
        nonlinear_term: &mut Array3<f64>,
        pressure_current: &Array3<f64>,
        pressure_prev: &Array3<f64>,
        grid: &Grid,
        medium: &dyn Medium,
        dt: f64,
    ) {
        let (nx, ny, nz) = (grid.nx, grid.ny, grid.nz);
        let dp_dt_max_abs = if dt > 1e-9 { self.max_pressure / dt } else { self.max_pressure };
        
        // Process boundary points directly without collecting
        for i in 0..nx {
            for j in 0..ny {
                for k in 0..nz {
                    // Skip internal points
                    if i > 0 && i < nx-1 && 
                       j > 0 && j < ny-1 && 
                       k > 0 && k < nz-1 {
                        continue;
                    }
                    
                    let x = i as f64 * grid.dx;
                    let y = j as f64 * grid.dy;
                    let z = k as f64 * grid.dz;
                    
                    let rho = medium.density(x, y, z, grid).max(1e-9);
                    let c = medium.sound_speed(x, y, z, grid).max(1e-9);
                    let b_a = medium.nonlinearity_coefficient(x, y, z, grid);
                    let gradient_scale = dt / (2.0 * rho * c * c);
                    
                    let p_val_current = pressure_current[[i, j, k]];
                    let p_prev_val = pressure_prev[[i, j, k]];
                    let dp_dt = if dt > 1e-9 { (p_val_current - p_prev_val) / dt } else { 0.0 };
                    let dp_dt_limited = if dp_dt.is_finite() {
                        dp_dt.clamp(-dp_dt_max_abs, dp_dt_max_abs)
                    } else { 0.0 };
                    let beta = b_a / (rho * c * c);
                    
                    nonlinear_term[[i, j, k]] = -beta * self.nonlinearity_scaling * gradient_scale * p_val_current * dp_dt_limited;
                }
            }
        }
    }
}

// Move these imports before the test module
use crate::physics::traits::AcousticWaveModel;
use crate::source::Source;
use crate::solver::PRESSURE_IDX;
use crate::utils::{fft_3d, ifft_3d};
use log::{trace};
use ndarray::{Array4, Axis, ShapeBuilder};
use num_complex::Complex;
use std::time::Instant;

/// Represents the configuration for multi-frequency simulation
#[derive(Debug, Clone)]
pub struct MultiFrequencyConfig {
    /// Primary frequency components [Hz]
    pub frequencies: Vec<f64>,
    /// Relative amplitudes for each frequency [0.0-1.0]
    pub amplitudes: Vec<f64>,
    /// Phase relationships between frequencies [rad]
    pub phases: Vec<f64>,
    /// Enable frequency-dependent attenuation
    pub frequency_dependent_attenuation: bool,
    /// Enable harmonic generation
    pub enable_harmonics: bool,
}

impl Default for MultiFrequencyConfig {
    fn default() -> Self {
        Self {
            frequencies: vec![1e6], // Default 1 MHz
            amplitudes: vec![1.0],
            phases: vec![0.0],
            frequency_dependent_attenuation: true,
            enable_harmonics: false,
        }
    }
}

impl MultiFrequencyConfig {
    /// Create a new multi-frequency configuration
    pub fn new(frequencies: Vec<f64>) -> Self {
        let n = frequencies.len();
        Self {
            frequencies,
            amplitudes: vec![1.0; n],
            phases: vec![0.0; n],
            frequency_dependent_attenuation: true,
            enable_harmonics: false,
        }
    }
    
    /// Set relative amplitudes for each frequency component
    pub fn with_amplitudes(mut self, amplitudes: Vec<f64>) -> Result<Self, &'static str> {
        if amplitudes.len() != self.frequencies.len() {
            return Err("Number of amplitudes must match number of frequencies");
        }
        self.amplitudes = amplitudes;
        Ok(self)
    }
    
    /// Set phase relationships between frequency components
    pub fn with_phases(mut self, phases: Vec<f64>) -> Self {
        assert_eq!(phases.len(), self.frequencies.len(), 
                   "Number of phases must match number of frequencies");
        self.phases = phases;
        self
    }
    
    /// Enable frequency-dependent attenuation modeling
    pub fn with_frequency_dependent_attenuation(mut self, enable: bool) -> Self {
        self.frequency_dependent_attenuation = enable;
        self
    }
    
    /// Enable harmonic generation from nonlinear effects
    pub fn with_harmonics(mut self, enable: bool) -> Self {
        self.enable_harmonics = enable;
        self
    }
}

impl AcousticWaveModel for NonlinearWave {
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

        if !self.check_stability(dt, grid, medium, &pressure_at_start) {
            debug!("Potential instability detected at t={}. Enhanced stability measures might be active.", t);
        }

        let (nx, ny, nz) = (grid.nx, grid.ny, grid.nz);
        if nx == 0 || ny == 0 || nz == 0 {
            trace!("Wave update skipped for empty grid at t={}", t);
            return;
        }
        let mut nonlinear_term = Array3::<f64>::zeros((nx, ny, nz).f());
        let mut src_term_array = Array3::<f64>::zeros((nx, ny, nz).f());

        let start_source = Instant::now();
        Zip::indexed(&mut src_term_array)
            .for_each(|(i, j, k), src_val| {
                let x = i as f64 * grid.dx;
                let y = j as f64 * grid.dy;
                let z = k as f64 * grid.dz;
                *src_val = source.get_source_term(t, x, y, z, grid);
            });
        self.source_time += start_source.elapsed().as_secs_f64();

        let start_nonlinear = Instant::now();
        let min_grid_spacing = grid.dx.min(grid.dy).min(grid.dz);
        let max_gradient = if min_grid_spacing > 1e-9 { self.max_pressure / min_grid_spacing } else { self.max_pressure };

        // Use iterator pattern for gradient computation on interior points
        let dx_inv = 1.0 / (2.0 * grid.dx);
        let dy_inv = 1.0 / (2.0 * grid.dy);
        let dz_inv = 1.0 / (2.0 * grid.dz);
        
        // Process interior points with iterator patterns
        for i in 1..grid.nx-1 {
            for j in 1..grid.ny-1 {
                for k in 1..grid.nz-1 {
                    let x = i as f64 * grid.dx;
                    let y = j as f64 * grid.dy;
                    let z = k as f64 * grid.dz;
                    
                    // Compute gradients using central differences
                    let grad_x = (pressure_at_start[[i+1, j, k]] - pressure_at_start[[i-1, j, k]]) * dx_inv;
                    let grad_y = (pressure_at_start[[i, j+1, k]] - pressure_at_start[[i, j-1, k]]) * dy_inv;
                    let grad_z = (pressure_at_start[[i, j, k+1]] - pressure_at_start[[i, j, k-1]]) * dz_inv;
                    
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
                    let p_limited = pressure_at_start[[i, j, k]].clamp(-self.max_pressure, self.max_pressure);
                    let nl_term_calc = -beta * self.nonlinearity_scaling * gradient_scale * p_limited * grad_magnitude;
                    
                    nonlinear_term[[i, j, k]] = if nl_term_calc.is_finite() {
                        nl_term_calc.clamp(-self.max_pressure, self.max_pressure)
                    } else { 
                        0.0 
                    };
                }
            }
        }
        
        // Handle boundary points using traditional approach
        self.compute_boundary_nonlinear_terms(&mut nonlinear_term, &pressure_at_start, prev_pressure, grid, medium, dt);
        self.nonlinear_time += start_nonlinear.elapsed().as_secs_f64();

        let start_fft = Instant::now();
        let p_fft = fft_3d(fields, PRESSURE_IDX, grid);

        let k2_values = self.k_squared.as_ref().expect("k_squared should be initialized in new()");

        let kspace_corr_factor = grid.kspace_correction(medium.sound_speed(0.0, 0.0, 0.0, grid), dt);
        let ref_freq = medium.reference_frequency();

        let mut p_linear_fft = Array3::<Complex<f64>>::zeros((nx, ny, nz).f());

        Zip::indexed(&mut p_linear_fft)
            .and(&p_fft)
            .for_each(|idx, p_new_fft_val, p_old_fft_val_ref| {
                let (i,j,k) = idx;
                let p_old_fft_val = *p_old_fft_val_ref;
                let x = i as f64 * grid.dx;
                let y = j as f64 * grid.dy;
                let z = k as f64 * grid.dz;

                let c = medium.sound_speed(x, y, z, grid).max(1e-9);
                let mu = medium.viscosity(x, y, z, grid);
                let rho = medium.density(x, y, z, grid).max(1e-9);

                let k_val = k2_values[[i, j, k]].sqrt();
                let phase = self.calculate_phase_factor(k_val, c, dt);

                let viscous_damping_arg = -mu * k_val.powi(2) * dt / rho;
                let viscous_damping = if viscous_damping_arg.is_finite() { viscous_damping_arg.exp() } else { 1.0 };

                let absorption_damping_arg = -medium.absorption_coefficient(x, y, z, grid, ref_freq) * dt;
                let absorption_damping = if absorption_damping_arg.is_finite() { absorption_damping_arg.exp() } else { 1.0 };

                let phase_complex = Complex::new(phase.cos(), phase.sin());
                let decay = absorption_damping * viscous_damping;
                *p_new_fft_val = p_old_fft_val * phase_complex * kspace_corr_factor[[i, j, k]] * decay;
            });

        let p_linear = ifft_3d(&p_linear_fft, grid);
        self.fft_time += start_fft.elapsed().as_secs_f64();

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

        trace!( "Wave update for t={} completed in {:.3e} s", t, start_total.elapsed().as_secs_f64());

        let mut temp_pressure_to_clamp = fields.index_axis(Axis(0), PRESSURE_IDX).to_owned();
        if self.clamp_pressure(&mut temp_pressure_to_clamp) {
             fields.index_axis_mut(Axis(0), PRESSURE_IDX).assign(&temp_pressure_to_clamp);
        }

        let mut final_pressure_view_mut = fields.index_axis_mut(Axis(0), PRESSURE_IDX);
        for val in final_pressure_view_mut.iter_mut() {
            if !val.is_finite() {
                *val = 0.0;
            } else {
                *val = val.clamp(-self.max_pressure, self.max_pressure);
            }
        }
    }

    fn report_performance(&self) {
        if self.call_count == 0 {
            debug!("No calls to NonlinearWave::update_wave yet (via trait)");
            return;
        }

        let total_time = self.nonlinear_time + self.fft_time + self.source_time + self.combination_time;
        let avg_time = if self.call_count > 0 { total_time / self.call_count as f64 } else { 0.0 };

        debug!(
            "NonlinearWave performance with iterator patterns (avg over {} calls):",
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
        } else {
            debug!("  Detailed breakdown not available (total_time or call_count is zero leading to no meaningful percentages).");
            debug!("    Nonlinear term calc:    {:.3e} s", self.nonlinear_time / self.call_count.max(1) as f64);
            debug!("    FFT operations:         {:.3e} s", self.fft_time / self.call_count.max(1) as f64);
            debug!("    Source application:     {:.3e} s", self.source_time / self.call_count.max(1) as f64);
            debug!("    Field combination:      {:.3e} s", self.combination_time / self.call_count.max(1) as f64);
        }
    }

    fn set_nonlinearity_scaling(&mut self, scaling: f64) {
        self.set_nonlinearity_scaling(scaling);
    }

    fn set_k_space_correction_order(&mut self, order: usize) {
        self.set_k_space_correction_order(order);
    }
}
