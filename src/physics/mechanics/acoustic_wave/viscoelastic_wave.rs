// src/physics/mechanics/acoustic_wave/viscoelastic_wave.rs

use crate::grid::Grid;
use crate::medium::Medium;
use crate::physics::traits::AcousticWaveModel;
use crate::source::Source;
use crate::solver::PRESSURE_IDX; // Assuming pressure is still the primary field
use crate::utils::{fft_3d, ifft_3d};
use log::{debug, trace, warn};
use ndarray::{Array3, Array4, Axis, Zip}; // Removed parallel prelude
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

#[derive(Debug)]
pub struct ViscoelasticWave {
    // Precomputed k-space grids (kx, ky, kz, k_squared)
    // Similar to NonlinearWave, these can be stored in OnceLock<Array3<f64>> or similar
    // For simplicity, let's assume they are passed or computed if needed.
    // NonlinearWave uses self.k_squared: Option<Array3<f64>>
    k_squared: Option<Array3<f64>>,
    // kx: Option<Array3<f64>>, // For gradient calculation if needed - Unused
    // ky: Option<Array3<f64>>, - Unused
    // kz: Option<Array3<f64>>, - Unused

    // Configuration parameters
    nonlinearity_scaling: f64,
    k_space_correction_order: usize, // May not be used if we follow NonlinearWave's sinc correction
    max_pressure: f64,             // For clamping
    // clamp_gradients: bool, - Unused

    // Performance tracking
    metrics: Arc<Mutex<PerformanceMetrics>>,
}

impl ViscoelasticWave {
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
            // kx: Some(kx_arr_nd), // Field commented out
            // ky: Some(ky_arr_nd), // Field commented out
            // kz: Some(kz_arr_nd), // Field commented out
            nonlinearity_scaling: 1.0, // Default, can be adjusted
            k_space_correction_order: 1, // Default
            max_pressure: 1e9,         // Default max pressure for clamping
            // clamp_gradients: false,     // Field commented out
            metrics: Arc::new(Mutex::new(PerformanceMetrics::default())),
        }
    }

    // Helper for stability checks, similar to NonlinearWave
    // This is a simplified version for now.
    fn check_stability(&self, dt: f64, grid: &Grid, medium: &dyn Medium, _current_pressure: &Array3<f64>) -> bool {
        let c_max = medium.sound_speed_array().iter().fold(0.0f64, |acc, &x| acc.max(x));
        let dx_min = grid.dx.min(grid.dy).min(grid.dz);
        let cfl = c_max * dt / dx_min;
        if cfl > 1.0 { // Courant number condition for simple FDTD, k-space methods have different criteria
            warn!(
                "CFL condition potentially violated: c_max={}, dt={}, dx_min={}, CFL={}",
                c_max, dt, dx_min, cfl
            );
            return false;
        }
        true
    }

    // Placeholder for gradient clamping logic if used from NonlinearWave
    fn clamp_pressure(&self, pressure_field: &mut Array3<f64>) -> bool {
        let mut clamped = false;
        for val in pressure_field.iter_mut() {
            if !val.is_finite() {
                *val = 0.0;
                clamped = true;
            } else if val.abs() > self.max_pressure {
                *val = val.signum() * self.max_pressure;
                clamped = true;
            }
        }
        if clamped {
            debug!("Pressure field clamped due to non-finite values or exceeding max_pressure.");
        }
        clamped
    }

    // Placeholder for phase factor calculation from NonlinearWave
    //  fn calculate_phase_factor(&self, k_val: f64, c_val: f64, dt: f64) -> f64 {
    //     // This is a simplified phase factor; k-Wave uses a k-space correction (sinc term)
    //     // which is often applied differently. For a simple linear acoustic solver:
    //     -c_val * k_val * dt // This results in exp(-j*c*k*dt) for forward time
    // }

}

impl AcousticWaveModel for ViscoelasticWave {
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

        let pressure_at_start = fields.index_axis(Axis(0), PRESSURE_IDX).to_owned();

        if !self.check_stability(dt, grid, medium, &pressure_at_start) {
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
        // medium.shear_sound_speed_array() is not used in this simplified scalar model

        // --- Source Term ---
        let start_source = Instant::now();
        let mut src_term_array = Array3::<f64>::zeros((nx, ny, nz));
        Zip::indexed(&mut src_term_array)
            .par_for_each(|(i, j, k), src_val| {
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
        let b_a_arr = medium.nonlinearity_coefficient(0.0,0.0,0.0,grid); // Assuming B/A is homogeneous for simplicity here or get array

        Zip::indexed(&mut nonlinear_term)
            .and(&pressure_at_start)
            .par_for_each(|(i, j, k), nl_val, &p_curr| {
                if i > 0 && i < nx - 1 && j > 0 && j < ny - 1 && k > 0 && k < nz - 1 {
                    let rho = rho_arr[[i,j,k]].max(1e-9);
                    let c = c_arr[[i,j,k]].max(1e-9);
                    // Using a simplified nonlinearity term from k-Wave (beta * d/dt (p^2/2))
                    // Or Westervelt: (beta / (rho * c^4)) * p * d^2p/dt^2 (approx)
                    // Or (B/A / (2 * rho_0 * c_0^4)) * d/dt (p^2)
                    // For now, let's use a simplified form as in NonlinearWave, which needs gradients.
                    // This requires kx, ky, kz or finite differences.
                    // For simplicity, let's use the d/dt(p^2) form if nonlinearity were active.
                    // let p_prev = prev_pressure[[i,j,k]];
                    // let dp_dt_approx = (p_curr - p_prev) / dt.max(1e-9);
                    // let beta_val = b_a_arr / (rho * c * c);

                    // For Viscoelastic, we are currently assuming a linear model in this part of the code.
                    *nl_val = 0.0;

                } else {
                    *nl_val = 0.0;
                }
            });
        metrics.nonlinear_time += start_nonlinear.elapsed().as_secs_f64();


        // --- k-space propagation ---
        let start_fft = Instant::now();
        let p_fft = fft_3d(fields, PRESSURE_IDX, grid); // FFT of current pressure
        metrics.fft_time += start_fft.elapsed().as_secs_f64();

        let start_kspace_ops = Instant::now();
        let k_squared_vals = self.k_squared.as_ref().expect("k_squared not initialized");
        // k-space correction factor (sinc correction)
        // let kspace_corr_factor_arr = grid.kspace_correction(medium.sound_speed(0.0,0.0,0.0,grid), dt); // This is for a homogeneous medium. Needs to be heterogeneous.
        // For heterogeneous, this correction is more complex or applied differently.
        // k-Wave applies it to the time derivatives.
        // Let's use the simpler cos(c*k*dt) and handle correction via k_eff if needed.

        let mut p_linear_fft = Array3::<Complex<f64>>::zeros((nx, ny, nz));

        Zip::indexed(&mut p_linear_fft)
            .and(&p_fft)
            .par_for_each(|(i, j, k), p_new_k, &p_old_k| {
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
                // Let's follow NonlinearWave's structure for the linear propagator part.

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
                // If we use the simpler `p_k_next = p_k * propagator`, the sinc correction is part of the propagator.
                // The sinc term comes from `(sin(omega dt / 2) / (omega dt / 2))^2` for the second derivative operator.
                // Or for `cos(omega dt)` it's `cos(omega dt_corrected)` where `dt_corrected` uses `sin(omega dt/2) / (omega/2)`.
                // For now, let's use the k-Wave approach of modifying k: k_eff = k * sinc_correction_factor
                // This is complex to do heterogeneously.
                // NonlinearWave applies kspace_corr_factor = grid.kspace_correction(...)
                // Let's assume a homogeneous c for this correction for now, or ignore it if it's too complex.
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

                *p_new_k = p_old_k * phase_complex_corrected * damping_factor;
            });
        metrics.kspace_ops_time += start_kspace_ops.elapsed().as_secs_f64();

        let start_ifft = Instant::now();
        let p_linear = ifft_3d(&p_linear_fft, grid);
        metrics.fft_time += start_ifft.elapsed().as_secs_f64(); // Add IFFT time to fft_time

        // --- Combine terms ---
        let start_combine = Instant::now();
        let mut p_output_view = fields.index_axis_mut(Axis(0), PRESSURE_IDX);

        // Current solution: p(t+dt) = p_linear_propagated(from p(t)) + dt * (NonlinearSource + SourceTerm)
        // This assumes nonlinear_term and src_term_array are source rates.
        // If they are direct pressure contributions: p(t+dt) = p_linear + NL + Src
        // Let's assume they are direct contributions for now, matching NonlinearWave structure.
        Zip::from(&mut p_output_view)
            .and(&p_linear)
            .and(&nonlinear_term)
            .and(&src_term_array)
            .par_for_each(|p_out, &p_lin_val, &nl_val, &src_val| {
                *p_out = p_lin_val + nl_val + src_val; // dt scaling might be needed for NL and Src if they are rates
            });
        metrics.combination_time += start_combine.elapsed().as_secs_f64();

        // --- Final clamping ---
        let mut temp_pressure_to_clamp = fields.index_axis(Axis(0), PRESSURE_IDX).to_owned();
        if self.clamp_pressure(&mut temp_pressure_to_clamp) {
             fields.index_axis_mut(Axis(0), PRESSURE_IDX).assign(&temp_pressure_to_clamp);
        }

        // Ensure prev_pressure for next step is based on the state *before* adding sources for this step.
        // Or, if solver structure means prev_pressure is truly p(t-dt), then this is fine.
        // The solver structure seems to be:
        // fields(t) -> update_wave -> fields(t+dt)
        // prev_pressure passed to update_wave is fields(t-dt) if using a 3-level scheme,
        // or fields(t) if using a 2-level scheme where p_linear is based on p(t) and dp/dt(t).
        // The k-Wave first-order scheme is effectively two-level.
        // The NonlinearWave's use of `prev_pressure` in nonlinear term suggests it might be p(t-dt).
        // However, the linear k-space part `p_fft = fft_3d(fields, PRESSURE_IDX, grid);` uses current `fields` (i.e. p(t)).
        // This hybrid approach needs care. For now, mirroring NonlinearWave.

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

    fn set_k_space_correction_order(&mut self, order: usize) {
        self.k_space_correction_order = order;
        // Note: current implementation uses a specific sinc correction, order not directly used in the same way.
        debug!("ViscoelasticWave: k-space correction order set to {} (may have limited effect in current impl)", order);
    }
}
