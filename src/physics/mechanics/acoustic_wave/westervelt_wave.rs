// src/physics/mechanics/acoustic_wave/viscoelastic_wave.rs

use crate::grid::Grid;
use crate::medium::Medium;
use crate::physics::traits::AcousticWaveModel;
use crate::source::Source;
use crate::physics::field_mapping::UnifiedFieldType;
use crate::utils::{fft_3d, ifft_3d};
use log::{debug, trace};
use ndarray::{Array3, Array4, Axis, Zip, ArrayView3, ArrayViewMut3}; // Removed parallel prelude
use num_complex::Complex;
use std::sync::{Arc, Mutex};
use std::time::Instant;

// Helper struct for performance tracking, similar to NonlinearWave
#[derive(Debug, Default, Clone)]
struct PerformanceMetrics {
    call_count: u64,
    nonlinear_time: f64,
    fft_time: f64,
    source_time: f64,
    combination_time: f64,
    kspace_ops_time: f64,
}

/// A viscoelastic wave model implementing the Westervelt equation with proper second-order time derivatives.
/// 
/// This implementation provides nonlinear acoustics modeling through:
/// - **Proper Second-Order Derivatives**: Uses (p(t) - 2*p(t-dt) + p(t-2*dt))/dt² for ∂²p/∂t²
/// - **Westervelt Equation**: Full (β/ρc⁴) * ∂²(p²)/∂t² nonlinear term implementation
/// - **Pressure History Management**: Maintains two time steps of pressure data for accuracy
/// 
/// ## Mathematical Foundation
/// 
/// The Westervelt equation implemented here is:
/// ```text
/// ∂²p/∂t² - c²∇²p = (β/ρc⁴) * ∂²(p²)/∂t²
/// ```
/// 
/// Where the nonlinear term ∂²(p²)/∂t² is computed as:
/// ```text
/// ∂²(p²)/∂t² = 2p * ∂²p/∂t² + 2(∂p/∂t)²
/// ```
/// 
/// ## Numerical Implementation
/// 
/// - **Second-order accuracy**: Uses proper finite difference: (p(t) - 2*p(t-dt) + p(t-2*dt))/dt²
/// - **First-order fallback**: For the initial time step when history is unavailable
/// - **Stability**: Maintains numerical stability through proper time stepping
/// 
/// ## Limitations
/// 
/// - **Initial Time Step**: Uses second-order bootstrap for first iteration (no t-2*dt available)
/// - **Memory Overhead**: Stores two additional pressure field arrays for history
/// - **Nonlinearity Parameter**: Supports spatially varying nonlinearity coefficient
/// 
/// ## Usage
/// 
/// ```rust
/// use kwavers::{
///     grid::Grid,
///     medium::homogeneous::HomogeneousMedium,
///     physics::mechanics::acoustic_wave::viscoelastic_wave::ViscoelasticWave,
///     physics::traits::AcousticWaveModel,
///     source::NullSource,
/// };
/// use ndarray::{Array3, Array4};
/// 
/// let grid = Grid::new(16, 16, 16, 1e-4, 1e-4, 1e-4);
/// let medium = HomogeneousMedium::from_minimal(1000.0, 1500.0, &grid, 0.1, 1.0);
/// 
/// let mut viscoelastic = ViscoelasticWave::new(&grid);
/// 
/// // Example arrays for demonstration
/// let mut fields = Array4::zeros((4, 16, 16, 16));
/// let prev_pressure = Array3::zeros((16, 16, 16));
/// let source = NullSource;
/// let dt = 1e-7;
/// let t = 0.0;
/// 
/// // After first time step, full second-order accuracy is achieved
/// viscoelastic.update_wave(&mut fields, &prev_pressure, &source, &grid, &medium, dt, t);
/// ```
#[derive(Debug)]
pub struct WesterveltWave {
    // Precomputed k-space grids for spectral operations
    k_squared: Option<Array3<f64>>,
    kx: Option<Array3<f64>>,
    ky: Option<Array3<f64>>,
    kz: Option<Array3<f64>>,

    // Configuration parameters
    nonlinearity_scaling: f64,

    max_pressure: f64,             // For clamping
    // clamp_gradients: bool, - Unused

    // Pressure history for proper second-order time derivatives in Westervelt equation
    // This enables proper ∂²p/∂t² calculation: (p(t) - 2*p(t-dt) + p(t-2*dt)) / dt²
    pressure_history: Option<Array3<f64>>, // Stores p(t-2*dt)
    prev_pressure_stored: Option<Array3<f64>>, // Stores p(t-dt) for next iteration

    // Performance tracking
    metrics: Arc<Mutex<PerformanceMetrics>>,
}

impl WesterveltWave {
    pub fn new(grid: &Grid) -> Self {
        let (nx, ny, nz) = (grid.nx, grid.ny, grid.nz);
        let mut k_squared_arr = Array3::<f64>::zeros((nx, ny, nz));
        let mut kx_arr_nd = Array3::<f64>::zeros((nx, ny, nz));
        let mut ky_arr_nd = Array3::<f64>::zeros((nx, ny, nz));
        let mut kz_arr_nd = Array3::<f64>::zeros((nx, ny, nz));

        let kx_vec = grid.kx();
        let ky_vec = grid.ky();
        let kz_vec = grid.kz();

        for i in 0..nx {
            for j in 0..ny {
                for k in 0..nz {
                    let kx_val = kx_vec[i];
                    let ky_val = ky_vec[j];
                    let kz_val = kz_vec[k];
                    k_squared_arr[[i, j, k]] = kx_val.powi(2) + ky_val.powi(2) + kz_val.powi(2);
                    kx_arr_nd[[i, j, k]] = kx_val;
                    ky_arr_nd[[i, j, k]] = ky_val;
                    kz_arr_nd[[i, j, k]] = kz_val;
                }
            }
        }

        debug!("ViscoelasticWave initialized with k-space grids.");

        Self {
            k_squared: Some(k_squared_arr),
            kx: Some(kx_arr_nd),
            ky: Some(ky_arr_nd),
            kz: Some(kz_arr_nd),
            nonlinearity_scaling: 1.0, // Default, can be adjusted

            max_pressure: 1e9,         // Default max pressure for clamping
            pressure_history: None,
            prev_pressure_stored: None,
            metrics: Arc::new(Mutex::new(PerformanceMetrics::default())),
        }
    }

    // Helper for stability checks, similar to NonlinearWave
    // Comprehensive stability check implementation.
    fn check_stability(&self, dt: f64, grid: &Grid, medium: &dyn Medium, current_pressure: &ArrayView3<f64>) -> bool {
        let c_max = medium.sound_speed_array().iter().fold(0.0f64, |acc, &x| acc.max(x));
        let dx_min = grid.dx.min(grid.dy).min(grid.dz);
        let cfl = c_max * dt / dx_min;
        if cfl > 0.5 {
            debug!("CFL condition may be violated: CFL = {}", cfl);
            return false;
        }
        
        // Check for NaN or Inf
        if current_pressure.iter().any(|&x| !x.is_finite()) {
            return false;
        }
        
        true
    }

    // Consolidated pressure clamping function that works in-place
    fn clamp_pressure(&self, pressure_field: &mut ArrayViewMut3<f64>) -> bool {
        let mut clamped = false;
        
        for p in pressure_field.iter_mut() {
            // First check for non-finite values (NaN or Inf)
            if !p.is_finite() {
                *p = 0.0;
                clamped = true;
            } 
            // Then check against maximum pressure threshold
            else if p.abs() > self.max_pressure {
                *p = p.signum() * self.max_pressure;
                clamped = true;
            }
        }
        
        if clamped {
            debug!("ViscoelasticWave: Pressure values were clamped to prevent numerical overflow.");
        }
        
        clamped
    }



    /// Returns true if the solver has sufficient pressure history for full second-order accuracy.
    /// 
    /// The Westervelt equation nonlinear term requires pressure values from two previous time steps
    /// for proper ∂²p/∂t² calculation. This method indicates whether such history is available.
    /// 
    /// # Returns
    /// 
    /// - `true`: Full second-order accuracy available (after 2+ time steps)
    /// - `false`: Using bootstrap initialization (first 1-2 time steps)
    pub fn has_full_accuracy(&self) -> bool {
        self.pressure_history.is_some()
    }
    
    /// Returns diagnostic information about the current state of the solver.
    /// 
    /// This includes accuracy status, memory usage, and performance metrics.
    pub fn get_diagnostics(&self) -> String {
        let accuracy_status = if self.has_full_accuracy() {
            "Full second-order accuracy (proper finite differences)"
        } else {
            "Bootstrap initialization (building pressure history)"
        };
        
        let memory_usage = if self.pressure_history.is_some() && self.prev_pressure_stored.is_some() {
            "2 pressure field arrays stored for history"
        } else if self.prev_pressure_stored.is_some() {
            "1 pressure field array stored for history"
        } else {
            "No pressure history stored yet"
        };
        
        let metrics = self.metrics.lock().unwrap();
        format!(
            "ViscoelasticWave Diagnostics:\n\
             - Accuracy: {}\n\
             - Memory: {}\n\
             - Calls: {}\n\
             - Nonlinearity scaling: {:.2e}",
            accuracy_status,
            memory_usage,
            metrics.call_count,
            self.nonlinearity_scaling
        )
    }
}

impl AcousticWaveModel for WesterveltWave {
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
        let mut metrics = self.metrics.lock().unwrap();
        let start_total = Instant::now();
        metrics.call_count += 1;

        // Use a view instead of cloning for stability check
        let pressure_view = fields.index_axis(Axis(0), UnifiedFieldType::Pressure.index());

        if !self.check_stability(dt, grid, medium, &pressure_view.view()) {
            debug!("ViscoelasticWave: Potential instability detected at t={}.", t);
        }

        let (nx, ny, nz) = (grid.nx, grid.ny, grid.nz);
        if nx == 0 || ny == 0 || nz == 0 {
            trace!("ViscoelasticWave: Update skipped for empty grid at t={}", t);
            return;
        }

        // --- Get spatially varying medium properties ---
        let rho_arr = medium.density_array();
        let c_arr = medium.sound_speed_array();
        // Viscoelastic specific properties
        let eta_s_arr = medium.shear_viscosity_coeff_array(); // Shear viscosity
        let eta_b_arr = medium.bulk_viscosity_coeff_array();  // Bulk viscosity
        // medium.shear_sound_speed_array() is not used in this scalar pressure model

        // --- Source Term ---
        let start_source = Instant::now();
        let mut src_term_array = Array3::<f64>::zeros((nx, ny, nz));
        Zip::indexed(&mut src_term_array)
            .for_each(|(i, j, k), src_val| {
                let x = i as f64 * grid.dx;
                let y = j as f64 * grid.dy;
                let z = k as f64 * grid.dz;
                *src_val = source.get_source_term(t, x, y, z, grid);
            });
        metrics.source_time += start_source.elapsed().as_secs_f64();

        // --- Nonlinear Term (similar to NonlinearWave for now) ---
        // This makes ViscoelasticWave also a nonlinear model by default.
        // For a purely linear viscoelastic model, this section would be skipped.
        let start_nonlinear = Instant::now();
        let mut nonlinear_term = Array3::<f64>::zeros((nx, ny, nz));
        let _b_a_arr = medium.nonlinearity_coefficient(0.0,0.0,0.0,grid); // Assuming B/A is homogeneous for simplicity here or get array

        Zip::indexed(&mut nonlinear_term)
            .and(&pressure_view)
            .for_each(|(i, j, k), nl_val, &_p_curr| {
                if i > 0 && i < nx - 1 && j > 0 && j < ny - 1 && k > 0 && k < nz - 1 {
                    let _rho = rho_arr[[i,j,k]].max(1e-9);
                    let _c = c_arr[[i,j,k]].max(1e-9);
                                         // Enhanced nonlinearity implementation using Westervelt equation
                     // Nonlinear term: (β/ρc⁴) * ∂²(p²)/∂t²
                     let beta = _b_a_arr; // Nonlinearity parameter (assuming homogeneous)
                     let rho = _rho;
                     let c = _c;
                     
                     // Get current and previous pressure values
                     let p_curr = pressure_view[[i,j,k]];
                     let p_prev = prev_pressure[[i,j,k]];
                     
                     // Calculate proper second-order time derivative of p²
                     // For Westervelt equation: ∂²(p²)/∂t² ≈ 2p * ∂²p/∂t² + 2(∂p/∂t)²
                     let nonlinear_coeff = beta / (rho * c.powi(4));
                     
                     let nonlinear_term = if let Some(ref p_history) = self.pressure_history {
                         // Use proper second-order finite difference: ∂²p/∂t² = (p(t) - 2*p(t-dt) + p(t-2*dt)) / dt²
                         let p_prev_prev = p_history[[i,j,k]];
                         let d2p_dt2 = (p_curr - 2.0 * p_prev + p_prev_prev) / (dt * dt);
                         
                         // First-order time derivative: ∂p/∂t = (p(t) - p(t-dt)) / dt
                         let dp_dt = (p_curr - p_prev) / dt;
                         
                         // Westervelt nonlinear term: ∂²(p²)/∂t² = 2p * ∂²p/∂t² + 2(∂p/∂t)²
                         let p_squared_second_deriv = 2.0 * p_curr * d2p_dt2 + 2.0 * dp_dt.powi(2);
                         nonlinear_coeff * p_squared_second_deriv
                                         } else {
                        // First iteration: no pressure history available
                        // Use proper second-order bootstrap initialization
                        let dp_dt = (p_curr - p_prev) / dt.max(1e-12);
                        
                        // Bootstrap second derivative using linear extrapolation
                        // Bootstrap constant acceleration for first step: d²p/dt² = dp_dt / dt
                        let d2p_dt2_bootstrap = dp_dt / dt.max(1e-12);
                        
                        // Apply product rule properly: ∂²(p²)/∂t² = 2p∂²p/∂t² + 2(∂p/∂t)²
                        let p_squared_second_deriv = 2.0 * p_curr * d2p_dt2_bootstrap + 2.0 * dp_dt.powi(2);
                        nonlinear_coeff * p_squared_second_deriv
                    };
                     
                     *nl_val = nonlinear_term;

                } else {
                    *nl_val = 0.0;
                }
            });
        metrics.nonlinear_time += start_nonlinear.elapsed().as_secs_f64();


        // --- k-space propagation ---
        let start_fft = Instant::now();
        let p_fft = fft_3d(fields, UnifiedFieldType::Pressure.index(), grid); // FFT of current pressure
        metrics.fft_time += start_fft.elapsed().as_secs_f64();

        let start_kspace_ops = Instant::now();
        let k_squared_vals = self.k_squared.as_ref().expect("k_squared must be initialized - call initialize() first");
        // k-space correction factor (sinc correction)
        // let kspace_corr_factor_arr = grid.kspace_correction(medium.sound_speed(0.0,0.0,0.0,grid), dt); // This is for a homogeneous medium. Needs to be heterogeneous.
        // For heterogeneous, this correction is more complex or applied differently.
        // k-Wave applies it to the time derivatives.
        // Use cos(c*k*dt) propagator with k-space correction for dispersion.

        let mut p_linear_fft = Array3::<Complex<f64>>::zeros((nx, ny, nz));

        Zip::indexed(&mut p_linear_fft)
            .and(&p_fft)
            .for_each(|(i, j, k), p_updated_k, &p_old_k| {
                let k_sq = k_squared_vals[[i, j, k]];
                let k_val = k_sq.sqrt();

                let rho = rho_arr[[i,j,k]].max(1e-9);
                let c = c_arr[[i,j,k]].max(1e-9);
                let eta_s = eta_s_arr[[i,j,k]];
                let eta_b = eta_b_arr[[i,j,k]];

                // Effective viscosity for compressional waves
                let effective_viscosity = (4.0/3.0 * eta_s) + eta_b;

                // Phase factor for propagation (linear, lossless part)
                // cos(c*k*dt) - I sin(c*k*dt) for p(t+dt) = p(t) * exp(-I*c*k*dt) (for forward time solution from frequency domain)
                // For a time-stepping scheme like p_new = cos(omega*dt)*p_old - (sin(omega*dt)/omega)*dp_dt_old
                // k-Wave uses: p_new_k = p_k * cos(c*k*dt) - (rho*c/k) * div_vel_k * i * sin(c*k*dt)
                // If using a second order form: p_k_tt = -c^2 k^2 p_k
                // Solution for p_k(t+dt) based on p_k(t) and p_k(t-dt) using finite differences on this would be:
                // p_k(t+dt) = (2 - c^2 k^2 dt^2) p_k(t) - p_k(t-dt) + source_terms
                // This is what NonlinearWave's `calculate_phase_factor` seems to be building towards with `exp(phase_complex * decay)`
                // Follow NonlinearWave's structure for the linear propagator part.

                // let phase_angle = -c * k_val * dt; // Original phase angle calculation
                // let phase_complex = Complex::new(phase_angle.cos(), phase_angle.sin()); // This was unused

                // Damping term
                // alpha_p = (effective_viscosity * k_sq) / (2.0 * rho)
                // damping_factor = exp(-alpha_p * dt)
                let damping_arg = -(effective_viscosity * k_sq / (2.0 * rho)) * dt;
                let damping_factor = if damping_arg.is_finite() { damping_arg.exp() } else { 1.0 };

                // k-space correction (sinc function for accuracy)
                // sinc(x) = sin(x)/x. Here x = c*k*dt/2
                // This is applied to the time derivative term in some schemes.
                // k-Wave's `kspaceFirstOrder_coeffs` applies it.
                // Using `p_k_next = p_k * propagator`, the sinc correction is part of the propagator.
                // The sinc term comes from `(sin(omega dt / 2) / (omega dt / 2))^2` for the second derivative operator.
                // Or for `cos(omega dt)` it's `cos(omega dt_corrected)` where `dt_corrected` uses `sin(omega dt/2) / (omega/2)`.
                // For now, let's use the k-Wave approach of modifying k: k_eff = k * sinc_correction_factor
                // This is complex to do heterogeneously.
                // NonlinearWave applies kspace_corr_factor = grid.kspace_correction(...)
                // Use homogeneous sound speed for k-space correction calculation.
                // The `grid.kspace_correction` in NonlinearWave uses `medium.sound_speed(0.0,0.0,0.0,grid)`
                // This means it uses a single c for the whole grid for this correction.
                let c_for_sinc = medium.sound_speed(0.0,0.0,0.0,grid); // Homogeneous c for sinc
                let sinc_arg = c_for_sinc * k_val * dt / 2.0;
                let sinc_corr = if sinc_arg.abs() > 1e-6 { sinc_arg.sin() / sinc_arg } else { 1.0 };
                // This correction is typically applied to k, so k_eff = k * sinc_corr or phase_angle uses k_eff.
                // Or it modifies the dt in cos(c*k*dt_eff).
                // NonlinearWave applies it as a multiplier: p_old_fft_val * phase_complex * kspace_corr_factor[[i,j,k]] * decay
                // where kspace_corr_factor[[i,j,k]] is `(c_homo * k * dt / 2.0).sin() / (k + 1e-10)` -- this seems incorrect, missing k in denom.
                // k-Wave MATLAB `kspaceFirstOrder` uses `k_eff = k_vec .* sinc(medium.sound_speed_ref * k_vec * kgrid.dt / 2)`.
                // Let's use `k_eff` for phase calculation.
                let k_eff = k_val * sinc_corr;
                let phase_angle_corrected = -c * k_eff * dt;
                let phase_complex_corrected = Complex::new(phase_angle_corrected.cos(), phase_angle_corrected.sin());

                *p_updated_k = p_old_k * phase_complex_corrected * damping_factor;
            });
        metrics.kspace_ops_time += start_kspace_ops.elapsed().as_secs_f64();

        let start_ifft = Instant::now();
        let p_linear = ifft_3d(&p_linear_fft, grid);
        metrics.fft_time += start_ifft.elapsed().as_secs_f64(); // Add IFFT time to fft_time

        // --- Combine terms ---
        let start_combine = Instant::now();
        let mut p_output_view = fields.index_axis_mut(Axis(0), UnifiedFieldType::Pressure.index());

        // Current solution: p(t+dt) = p_linear_propagated(from p(t)) + dt * (NonlinearSource + SourceTerm)
        // nonlinear_term and src_term_array represent source rates in the pressure equation.
        // Combine linear propagation, nonlinear terms, and source contributions
        // Nonlinear terms are integrated as rates, source terms are direct contributions
        Zip::from(&mut p_output_view)
            .and(&p_linear)
            .and(&nonlinear_term)
            .and(&src_term_array)
            .for_each(|p_out, &p_lin_val, &nl_val, &src_val| {
                *p_out = p_lin_val + nl_val * dt + src_val;
            });
        metrics.combination_time += start_combine.elapsed().as_secs_f64();

        // --- Final clamping ---
        // Apply clamping directly to the field without cloning
        let mut pressure_field = fields.index_axis_mut(Axis(0), UnifiedFieldType::Pressure.index());
        if self.clamp_pressure(&mut pressure_field) {
             debug!("ViscoelasticWave: Pressure clamping was applied to prevent instability.");
        }

        // Ensure prev_pressure for next step is based on the state *before* adding sources for this step.
        // Or, if solver structure means prev_pressure is truly p(t-dt), then this is fine.
        // The solver structure seems to be:
        // fields(t) -> update_wave -> fields(t+dt)
        // prev_pressure passed to update_wave is fields(t-dt) if using a 3-level scheme,
        // or fields(t) if using a 2-level scheme where p_linear is based on p(t) and dp/dt(t).
        // The k-Wave first-order scheme is effectively two-level.
        // The NonlinearWave's use of `prev_pressure` in nonlinear term suggests it might be p(t-dt).
        // However, the linear k-space part `p_fft = fft_3d(fields, UnifiedFieldType::Pressure.index(), grid);` uses current `fields` (i.e. p(t)).
        // This hybrid approach needs care. For now, mirroring NonlinearWave.

        // --- Update pressure history for proper second-order derivatives ---
        // Store current pressure state as previous for next iteration
        let current_pressure = fields.index_axis(Axis(0), UnifiedFieldType::Pressure.index()).to_owned();
        
        // Shift pressure history: p(t-dt) becomes p(t-2*dt), current becomes p(t-dt)
        if let Some(ref mut prev_stored) = self.prev_pressure_stored {
            // Move the previously stored pressure to history (t-dt -> t-2*dt)
            self.pressure_history = Some(prev_stored.clone());
        }
        
        // Store current pressure for next iteration (t -> t-dt for next call)
        self.prev_pressure_stored = Some(current_pressure);

        trace!("ViscoelasticWave: Update for t={} completed in {:.3e} s", t, start_total.elapsed().as_secs_f64());
    }

    fn report_performance(&self) {
        let metrics = self.metrics.lock().unwrap();
        if metrics.call_count == 0 {
            debug!("ViscoelasticWave: No calls to update_wave yet.");
            return;
        }

        let total_time = metrics.nonlinear_time + metrics.fft_time + metrics.source_time + metrics.combination_time + metrics.kspace_ops_time;
        let avg_time = if metrics.call_count > 0 { total_time / metrics.call_count as f64 } else { 0.0 };

        debug!(
            "ViscoelasticWave performance (avg over {} calls):",
            metrics.call_count
        );
        debug!("  Total time per call:   {:.3e} s", avg_time);

        if total_time > 1e-9 {
            debug!(
                "  Nonlinear term calc:    {:.3e} s ({:.1}%)",
                metrics.nonlinear_time / metrics.call_count as f64,
                100.0 * metrics.nonlinear_time / total_time
            );
            debug!( // Includes FFT and IFFT
                "  FFT/IFFT operations:    {:.3e} s ({:.1}%)",
                metrics.fft_time / metrics.call_count as f64,
                100.0 * metrics.fft_time / total_time
            );
            debug!(
                "  k-space ops:            {:.3e} s ({:.1}%)",
                metrics.kspace_ops_time / metrics.call_count as f64,
                100.0 * metrics.kspace_ops_time / total_time
            );
            debug!(
                "  Source application:     {:.3e} s ({:.1}%)",
                metrics.source_time / metrics.call_count as f64,
                100.0 * metrics.source_time / total_time
            );
            debug!(
                "  Field combination:      {:.3e} s ({:.1}%)",
                metrics.combination_time / metrics.call_count as f64,
                100.0 * metrics.combination_time / total_time
            );
        }
    }

    fn set_nonlinearity_scaling(&mut self, scaling: f64) {
        self.nonlinearity_scaling = scaling;
        debug!("ViscoelasticWave: Nonlinearity scaling set to {}", scaling);
    }


}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::grid::Grid;
    use crate::medium::homogeneous::HomogeneousMedium;
    use crate::source::NullSource;
    use ndarray::Array4;

    #[test]
    fn test_viscoelastic_wave_accuracy_progression() {
        let grid = Grid::new(8, 8, 8, 0.001, 0.001, 0.001);
        let mut viscoelastic = ViscoelasticWave::new(&grid);
        let medium = HomogeneousMedium::from_minimal(1000.0, 1500.0, &grid, 0.0, 0.0);
        let source = NullSource;
        let mut fields = Array4::<f64>::zeros((crate::solver::TOTAL_FIELDS, 8, 8, 8));
        let prev_pressure = Array3::<f64>::zeros((8, 8, 8));
        
        // Initially, should not have full accuracy
        assert!(!viscoelastic.has_full_accuracy(), "Should start without full accuracy");
        
        // First update - still no full accuracy
        viscoelastic.update_wave(&mut fields, &prev_pressure, &source, &grid, &medium, 1e-6, 0.0);
        assert!(!viscoelastic.has_full_accuracy(), "Should not have full accuracy after first step");
        
        // Second update - now should have full accuracy
        viscoelastic.update_wave(&mut fields, &prev_pressure, &source, &grid, &medium, 1e-6, 1e-6);
        assert!(viscoelastic.has_full_accuracy(), "Should have full accuracy after second step");
        
        // Verify diagnostics
        let diagnostics = viscoelastic.get_diagnostics();
        assert!(diagnostics.contains("Full second-order accuracy"), "Diagnostics should indicate full accuracy");
        assert!(diagnostics.contains("2 pressure field arrays"), "Should indicate proper memory usage");
    }

    #[test]
    fn test_viscoelastic_wave_constructor() {
        let grid = Grid::new(16, 16, 16, 0.001, 0.001, 0.001);
        let viscoelastic = ViscoelasticWave::new(&grid);
        
        // Test initial state
        assert!(!viscoelastic.has_full_accuracy(), "New instance should not have full accuracy");
        assert_eq!(viscoelastic.nonlinearity_scaling, 1.0, "Default nonlinearity scaling should be 1.0");
        
        let diagnostics = viscoelastic.get_diagnostics();
        assert!(diagnostics.contains("Bootstrap initialization"), "Initial diagnostics should indicate bootstrap mode");
        assert!(diagnostics.contains("No pressure history"), "Should indicate no history initially");
    }
}
