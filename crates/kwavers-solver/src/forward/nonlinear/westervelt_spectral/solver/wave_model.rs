use kwavers_core::error::KwaversResult;
use kwavers_field::mapping::UnifiedFieldType;
use kwavers_grid::Grid;
use kwavers_medium::Medium;
use kwavers_physics::traits::AcousticWaveModel;
use kwavers_source::Source;
use moirai_parallel::{enumerate_mut_with, Adaptive};
use leto::{
    Array3,
    Array4,
};
use std::time::Instant;

use super::super::nonlinear::{compute_nonlinear_term_into, compute_viscoelastic_term_into};
use super::super::spectral::{compute_laplacian_spectral, compute_laplacian_spectral_into};
use super::{compute_laplacian_fd, WesterveltWave};

/// Advance the spectral Westervelt model with borrowed pressure history.
///
/// The leapfrog recurrence requires `p[n]`, `p[n-1]`, and one writable
/// `p[n+1]` buffer. `WesterveltWave::pressure_buffers_for_step` proves these
/// buffers are disjoint by matching the ring-buffer permutation, so the hot
/// path does not clone pressure volumes before evaluating the RHS.
impl AcousticWaveModel for WesterveltWave {
    fn update_wave(
        &mut self,
        fields: &mut Array4<f64>,
        _prev_pressure: &Array3<f64>,
        source: &dyn Source,
        grid: &Grid,
        medium: &dyn Medium,
        dt: f64,
        t: f64,
    ) -> KwaversResult<()> {
        {
            let mut metrics = self.metrics.lock().unwrap();
            metrics.increment_calls();
        }

        let (nx, ny, nz) = (grid.nx, grid.ny, grid.nz);
        if nx == 0 || ny == 0 || nz == 0 {
            return Ok(());
        }

        // Seed the leapfrog history on the first step: previous = current = p(t=0)
        // (zero initial velocity, the standard IVP start). Without this the
        // previous buffer is zero on step 0, injecting a spurious velocity kick.
        if self.current_step == 0 {
            let initial_pressure = fields
                .index_axis::<3>(0, UnifiedFieldType::Pressure.index())
                .expect("invariant: pressure field index within field stack");
            self.initialize_buffers(initial_pressure);
        }

        if !self.check_stability(dt, grid, medium) {
            log::debug!("WesterveltWave: Potential instability at t={}", t);
        }

        let pressure_field = fields
            .index_axis::<3>(0, UnifiedFieldType::Pressure.index())
            .expect("invariant: pressure field index within field stack");
        self.pressure_buffers[self.buffer_indices[1]].assign(&pressure_field);

        let (next_idx, curr_idx, prev_idx) = (
            self.buffer_indices[0],
            self.buffer_indices[1],
            self.buffer_indices[2],
        );
        debug_assert_ne!(next_idx, curr_idx);
        debug_assert_ne!(next_idx, prev_idx);
        debug_assert_ne!(curr_idx, prev_idx);

        let pressure_current = &self.pressure_buffers[curr_idx];
        // Authoritative history is the solver's internal ring buffer (seeded on
        // the first step), matching the Kuznetsov solver. An external
        // `prev_pressure` is intentionally ignored: a constant caller-supplied
        // buffer (e.g. zeros every step) would degenerate the leapfrog
        // `p_next = 2·p_curr − p_prev + …` into `2·p_curr + …`, which diverges.
        let pressure_previous = &self.pressure_buffers[prev_idx];

        let c_arr = medium.sound_speed_array();

        // Compute Laplacian via spectral method with pre-allocated scratch buffers.
        let start = Instant::now();
        let has_spectral_scratch = self.k_squared.is_some()
            && self.fft_scratch.is_some()
            && self.laplacian_scratch.is_some();
        if has_spectral_scratch {
            let k_sq = self.k_squared.as_ref().unwrap();
            let fft_s = self.fft_scratch.as_mut().unwrap();
            let lap_s = self.laplacian_scratch.as_mut().unwrap();
            compute_laplacian_spectral_into(pressure_current, k_sq, fft_s, lap_s);
        }
        let laplacian_owned: Option<Array3<f64>> = if !has_spectral_scratch {
            if let Some(k_sq) = &self.k_squared {
                Some(compute_laplacian_spectral(pressure_current, k_sq))
            } else {
                Some(compute_laplacian_fd(pressure_current, grid))
            }
        } else {
            None
        };
        {
            let mut metrics = self.metrics.lock().unwrap();
            metrics.record_kspace(start.elapsed());
        }

        // Nonlinear term
        let start = Instant::now();
        compute_nonlinear_term_into(
            &mut self.nonlinear_scratch,
            pressure_current,
            pressure_previous,
            None,
            medium,
            grid,
            dt,
        );
        let nonlinearity_scaling = self.nonlinearity_scaling;
        self.nonlinear_scratch
            .iter_mut()
            .for_each(|v| *v *= nonlinearity_scaling);
        {
            let mut metrics = self.metrics.lock().unwrap();
            metrics.record_nonlinear(start.elapsed());
        }

        // Viscoelastic damping term
        let start = Instant::now();
        compute_viscoelastic_term_into(
            &mut self.damping_scratch,
            pressure_current,
            pressure_previous,
            medium,
            grid,
            dt,
        );
        if self.damping_scaling != 1.0 {
            let damping_scaling = self.damping_scaling;
            self.damping_scratch
                .iter_mut()
                .for_each(|v| *v *= damping_scaling);
        }
        {
            let mut metrics = self.metrics.lock().unwrap();
            metrics.record_nonlinear(start.elapsed());
        }

        // Source term
        let start = Instant::now();
        source.create_mask_into(grid, &mut self.source_mask_scratch);
        let src_amplitude = source.amplitude(t);
        {
            let mut metrics = self.metrics.lock().unwrap();
            metrics.record_source(start.elapsed());
        }

        // Leapfrog update via Moirai parallel iteration.
        // Disjoint field borrows: laplacian_scratch vs pressure_buffers[next_idx].
        let start = Instant::now();
        let laplacian_slice: &[f64] = if has_spectral_scratch {
            self.laplacian_scratch
                .as_ref()
                .unwrap()
                .as_slice()
                .expect("laplacian_scratch must be contiguous")
        } else {
            laplacian_owned
                .as_ref()
                .unwrap()
                .as_slice()
                .expect("laplacian_owned must be contiguous")
        };
        let (pressure_next, pressure_current, pressure_previous_buffer) =
            WesterveltWave::pressure_buffers_for_step(
                &mut self.pressure_buffers,
                self.buffer_indices,
            );
        let pressure_previous = pressure_previous_buffer;
        let nonlinear_slice = self
            .nonlinear_scratch
            .as_slice()
            .expect("nonlinear_scratch must be contiguous");
        let damping_slice = self
            .damping_scratch
            .as_slice()
            .expect("damping_scratch must be contiguous");
        let source_slice = self
            .source_mask_scratch
            .as_slice()
            .expect("source_mask_scratch must be contiguous");
        let pressure_current_slice = pressure_current
            .as_slice()
            .expect("pressure_current must be contiguous");
        let pressure_previous_slice = pressure_previous
            .as_slice()
            .expect("pressure_previous must be contiguous");
        let sound_speed_slice = c_arr
            .as_slice()
            .expect("sound_speed_array must be contiguous");
        let dt2 = dt * dt;
        let pressure_next_slice = pressure_next
            .as_slice_mut()
            .expect("pressure_next must be contiguous");
        enumerate_mut_with::<Adaptive, _, _>(pressure_next_slice, |idx, p_next| {
            let p_curr = pressure_current_slice[idx];
            let p_prev = pressure_previous_slice[idx];
            let c = sound_speed_slice[idx];
            let lap = laplacian_slice[idx];
            let nl = nonlinear_slice[idx];
            let damp = damping_slice[idx];
            let src = source_slice[idx] * src_amplitude;

            let c2 = c * c;
            let update = dt2 * (c2.mul_add(lap, nl) + damp + src);
            *p_next = 2.0f64.mul_add(p_curr, -p_prev) + update;
        });

        fields
            .index_axis_mut::<3>(0, UnifiedFieldType::Pressure.index())
            .expect("invariant: pressure field index within field stack")
            .assign(pressure_next);
        {
            let mut metrics = self.metrics.lock().unwrap();
            metrics.record_combination(start.elapsed());
        }

        self.buffer_indices.rotate_right(1);
        self.current_step += 1;
        self.current_time += dt;
        self.check_conservation_laws();

        Ok(())
    }

    fn report_performance(&self) {
        let metrics = self.metrics.lock().unwrap();
        metrics.summary();
    }

    fn set_nonlinearity_scaling(&mut self, scaling: f64) {
        self.nonlinearity_scaling = scaling;
    }
}

impl WesterveltWave {
    /// Scale the viscoelastic damping term. Set to `0.0` for a lossless linear
    /// run (matching the lossless modes of the FDTD/PSTD/Kuznetsov solvers).
    pub fn set_damping_scaling(&mut self, scaling: f64) {
        self.damping_scaling = scaling;
    }
}
