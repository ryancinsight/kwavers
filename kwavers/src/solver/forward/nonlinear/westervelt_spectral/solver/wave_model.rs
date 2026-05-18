use crate::core::error::KwaversResult;
use crate::domain::field::mapping::UnifiedFieldType;
use crate::domain::grid::Grid;
use crate::domain::medium::Medium;
use crate::domain::source::Source;
use crate::physics::traits::AcousticWaveModel;
use ndarray::{Array3, Array4, Axis};
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
        prev_pressure: &Array3<f64>,
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

        if self.buffer_indices[1] == self.buffer_indices[2] {
            let initial_pressure = fields.index_axis(Axis(0), UnifiedFieldType::Pressure.index());
            self.initialize_buffers(initial_pressure);
        }

        if !self.check_stability(dt, grid, medium) {
            log::debug!("WesterveltWave: Potential instability at t={}", t);
        }

        let pressure_field = fields.index_axis(Axis(0), UnifiedFieldType::Pressure.index());
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
        let pressure_previous = if prev_pressure.dim() == pressure_current.dim() {
            prev_pressure
        } else {
            &self.pressure_buffers[prev_idx]
        };

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
            compute_laplacian_spectral_into(&pressure_current, k_sq, fft_s, lap_s);
        }
        let laplacian_owned: Option<Array3<f64>> = if !has_spectral_scratch {
            if let Some(k_sq) = &self.k_squared {
                Some(compute_laplacian_spectral(&pressure_current, k_sq))
            } else {
                Some(compute_laplacian_fd(&pressure_current, grid))
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
            &pressure_current,
            &pressure_previous,
            None,
            medium,
            grid,
            dt,
        );
        self.nonlinear_scratch *= self.nonlinearity_scaling;
        {
            let mut metrics = self.metrics.lock().unwrap();
            metrics.record_nonlinear(start.elapsed());
        }

        // Viscoelastic damping term
        let start = Instant::now();
        compute_viscoelastic_term_into(
            &mut self.damping_scratch,
            &pressure_current,
            &pressure_previous,
            medium,
            grid,
            dt,
        );
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

        // Leapfrog update via rayon parallel iteration.
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
        let pressure_previous = if prev_pressure.dim() == pressure_current.dim() {
            prev_pressure
        } else {
            pressure_previous_buffer
        };
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
        use rayon::prelude::*;
        let dt2 = dt * dt;
        pressure_next
            .as_slice_mut()
            .unwrap()
            .par_iter_mut()
            .enumerate()
            .for_each(|(idx, p_next)| {
                let p_curr = pressure_current.as_slice().unwrap()[idx];
                let p_prev = pressure_previous.as_slice().unwrap()[idx];
                let c = c_arr.as_slice().unwrap()[idx];
                let lap = laplacian_slice[idx];
                let nl = nonlinear_slice[idx];
                let damp = damping_slice[idx];
                let src = source_slice[idx] * src_amplitude;

                let c2 = c * c;
                let update = dt2 * (c2.mul_add(lap, nl) + damp + src);
                *p_next = 2.0f64.mul_add(p_curr, -p_prev) + update;
            });

        fields
            .index_axis_mut(Axis(0), UnifiedFieldType::Pressure.index())
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
