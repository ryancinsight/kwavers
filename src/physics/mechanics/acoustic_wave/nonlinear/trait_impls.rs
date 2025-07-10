// src/physics/mechanics/acoustic_wave/nonlinear/trait_impls.rs
use super::config::NonlinearWave;
use crate::grid::Grid;
use crate::medium::Medium;
use crate::physics::traits::AcousticWaveModel;
use crate::source::Source;
use crate::solver::PRESSURE_IDX;
use crate::utils::{fft_3d, ifft_3d};
use log::{debug, trace};
use ndarray::{Array3, Array4, Axis, parallel::prelude::*, Zip, ShapeBuilder};
use num_complex::Complex;
use std::time::Instant;

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

        // Assuming check_stability is an inherent method of NonlinearWave
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
            .par_for_each(|(i, j, k), src_val| {
                let x = i as f64 * grid.dx;
                let y = j as f64 * grid.dy;
                let z = k as f64 * grid.dz;
                *src_val = source.get_source_term(t, x, y, z, grid);
            });
        self.source_time += start_source.elapsed().as_secs_f64();

        let start_nonlinear = Instant::now();
        let min_grid_spacing = grid.dx.min(grid.dy).min(grid.dz);
        let max_gradient = if min_grid_spacing > 1e-9 { self.max_pressure / min_grid_spacing } else { self.max_pressure };

        let p_current_view = pressure_at_start.view();

        Zip::indexed(&mut nonlinear_term)
            .and(&p_current_view)
            .and(prev_pressure)
            .par_for_each(|idx, nl_val, p_val_current_ref, p_prev_val_ref| {
                let (i, j, k) = idx;
                let p_val_current = *p_val_current_ref;
                let p_prev_val = *p_prev_val_ref;
                let x = i as f64 * grid.dx;
                let y = j as f64 * grid.dy;
                let z = k as f64 * grid.dz;

                let rho = medium.density(x, y, z, grid).max(1e-9);
                let c = medium.sound_speed(x, y, z, grid).max(1e-9);
                let b_a = medium.nonlinearity_coefficient(x, y, z, grid);
                let gradient_scale = dt / (2.0 * rho * c * c);

                if i > 0 && i < grid.nx - 1 && j > 0 && j < grid.ny - 1 && k > 0 && k < grid.nz - 1 {
                    let dx_inv = if grid.dx > 1e-9 { 1.0 / (2.0 * grid.dx) } else {0.0};
                    let dy_inv = if grid.dy > 1e-9 { 1.0 / (2.0 * grid.dy) } else {0.0};
                    let dz_inv = if grid.dz > 1e-9 { 1.0 / (2.0 * grid.dz) } else {0.0};

                    let grad_x = (p_current_view[[i + 1, j, k]] - p_current_view[[i - 1, j, k]]) * dx_inv;
                    let grad_y = (p_current_view[[i, j + 1, k]] - p_current_view[[i, j - 1, k]]) * dy_inv;
                    let grad_z = (p_current_view[[i, j, k + 1]] - p_current_view[[i, j, k - 1]]) * dz_inv;

                    let (grad_x_clamped, grad_y_clamped, grad_z_clamped) = if self.clamp_gradients { // Assumes clamp_gradients is a field on NonlinearWave
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
                    let p_limited = p_val_current.clamp(-self.max_pressure, self.max_pressure); // Assumes max_pressure is a field
                    let nl_term_calc = -beta * self.nonlinearity_scaling * gradient_scale * p_limited * grad_magnitude; // Assumes nonlinearity_scaling is a field

                    *nl_val = if nl_term_calc.is_finite() {
                        nl_term_calc.clamp(-self.max_pressure, self.max_pressure)
                    } else { 0.0 };
                } else {
                    let dp_dt = if dt > 1e-9 { (p_val_current - p_prev_val) / dt } else { 0.0 };
                    let dp_dt_max_abs = if dt > 1e-9 { self.max_pressure / dt } else { self.max_pressure };
                    let dp_dt_limited = if dp_dt.is_finite() {
                        dp_dt.clamp(-dp_dt_max_abs, dp_dt_max_abs)
                    } else { 0.0 };
                    let beta = b_a / (rho * c * c);
                    *nl_val = -beta * self.nonlinearity_scaling * gradient_scale * p_val_current * dp_dt_limited;
                }
            });
        self.nonlinear_time += start_nonlinear.elapsed().as_secs_f64();

        let start_fft = Instant::now();
        let p_fft = fft_3d(fields, PRESSURE_IDX, grid);

        let k2_values = self.k_squared.as_ref().expect("k_squared should be initialized in new()");

        let kspace_corr_factor = grid.kspace_correction(medium.sound_speed(0.0, 0.0, 0.0, grid), dt);
        let ref_freq = medium.reference_frequency();

        let mut p_linear_fft = Array3::<Complex<f64>>::zeros((nx, ny, nz).f());

        Zip::indexed(&mut p_linear_fft)
            .and(&p_fft)
            .par_for_each(|idx, p_new_fft_val, p_old_fft_val_ref| {
                let (i,j,k) = idx;
                let p_old_fft_val = *p_old_fft_val_ref;
                let x = i as f64 * grid.dx;
                let y = j as f64 * grid.dy;
                let z = k as f64 * grid.dz;

                let c = medium.sound_speed(x, y, z, grid).max(1e-9);
                let mu = medium.viscosity(x, y, z, grid);
                let rho = medium.density(x, y, z, grid).max(1e-9);

                let k_val = k2_values[[i, j, k]].sqrt();
                // Assuming calculate_phase_factor is an inherent method
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
            .par_for_each(|p_out, &p_lin_val, &nl_val, &src_val| {
                *p_out = p_lin_val + nl_val + src_val;
            });
        self.combination_time += start_combine.elapsed().as_secs_f64();

        trace!( "Wave update for t={} completed in {:.3e} s", t, start_total.elapsed().as_secs_f64());

        let mut temp_pressure_to_clamp = fields.index_axis(Axis(0), PRESSURE_IDX).to_owned();
        // Assuming clamp_pressure is an inherent method
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

    // report_performance will be added next
    fn report_performance(&self) {
        // Content from src/physics/mechanics/acoustic_wave/nonlinear/performance.rs will go here
        if self.call_count == 0 {
            debug!("No calls to NonlinearWave::update_wave yet (via trait)");
            return;
        }

        let total_time = self.nonlinear_time + self.fft_time + self.source_time + self.combination_time;
        let avg_time = if self.call_count > 0 { total_time / self.call_count as f64 } else { 0.0 };

        debug!(
            "NonlinearWave performance (via trait) (avg over {} calls):",
            self.call_count
        );
        debug!("  Total time per call:   {:.3e} s", avg_time);

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
        // Call the inherent method from NonlinearWave (defined in config.rs)
        self.set_nonlinearity_scaling(scaling);
    }

    fn set_k_space_correction_order(&mut self, order: usize) {
        // Call the inherent method from NonlinearWave (defined in config.rs)
        self.set_k_space_correction_order(order);
    }
}
