use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

use super::GpuPstdSession;
use eunomia::FloatElement;
use kwavers_source::{velocity_source_scale, SourceMode};

impl GpuPstdSession {
    pub(crate) fn rebuild_source_sensor_indices(
        &mut self,
        mask_arr: leto::ArrayView3<'_, f64>,
    ) -> PyResult<()> {
        if mask_arr.shape() != [self.nx, self.ny, self.nz] {
            return Err(PyValueError::new_err(format!(
                "mask shape {:?} must match session grid ({}, {}, {})",
                mask_arr.shape(),
                self.nx,
                self.ny,
                self.nz
            )));
        }

        self.vel_x_indices.clear();
        for ix in 0..self.nx {
            for iy in 0..self.ny {
                for iz in 0..self.nz {
                    if mask_arr[[ix, iy, iz]] != 0.0 {
                        let flat = ix * self.ny * self.nz + iy * self.nz + iz;
                        self.vel_x_indices.push(flat as u32);
                    }
                }
            }
        }

        self.sensor_indices.clear();
        for iz in 0..self.nz {
            for iy in 0..self.ny {
                for ix in 0..self.nx {
                    if mask_arr[[ix, iy, iz]] != 0.0 {
                        let flat = ix * self.ny * self.nz + iy * self.nz + iz;
                        self.sensor_indices.push(flat as u32);
                    }
                }
            }
        }

        Ok(())
    }

    pub(crate) fn update_velocity_signal_rows(
        &mut self,
        sig_arr: leto::ArrayView2<'_, f64>,
    ) -> PyResult<()> {
        let time_steps = self.time_steps;
        let signal_shape = sig_arr.shape();
        if signal_shape.len() != 2 {
            return Err(PyValueError::new_err(format!(
                "ux_signals must be 2D, got shape {:?}",
                signal_shape
            )));
        }

        let n_vel = self.vel_x_indices.len();
        let n_sig_srcs = signal_shape[0];
        let n_sig_cols = signal_shape[1].min(time_steps);

        if n_vel > 0 && n_sig_srcs == 0 {
            return Err(PyValueError::new_err(
                "ux_signals must contain at least one source row for a non-empty mask",
            ));
        }

        // Scaled once here with the shared 2·c₀·Δt/Δx helper — the same
        // quotient the CPU solver and the Rust runner apply, so the session
        // agrees with both without normalisation. The session carries no
        // source mode; it upholds no-correction semantics (the run passes
        // `velocity_source_correction: false`), matching AdditiveNoCorrection.
        if !(self.dt.is_finite() && self.dt > 0.0) {
            return Err(PyValueError::new_err(format!(
                "GPU session velocity upload requires finite positive dt; got {}",
                self.dt
            )));
        }
        self.vel_x_signals.clear();
        self.vel_x_signals.resize(n_vel * time_steps, 0.0f32);
        for src_idx in 0..n_vel {
            let flat = usize::try_from(self.vel_x_indices[src_idx]).map_err(|_| {
                PyValueError::new_err("GPU session source index cannot address host memory")
            })?;
            let local_sound_speed = *self.c0_flat.get(flat).ok_or_else(|| {
                PyValueError::new_err(format!(
                    "GPU session source index {flat} exceeds the sound-speed field"
                ))
            })?;
            if !(local_sound_speed.is_finite() && local_sound_speed > 0.0) {
                return Err(PyValueError::new_err(format!(
                    "GPU session sound speed must be finite and positive at flat index {flat}; got {local_sound_speed}"
                )));
            }
            let source_scale = velocity_source_scale(
                f64::from(local_sound_speed),
                self.dt,
                self.dx,
                SourceMode::AdditiveNoCorrection,
            );
            let sig_row = src_idx.min(n_sig_srcs.saturating_sub(1));
            for step in 0..n_sig_cols {
                let scaled = sig_arr[[sig_row, step]] * source_scale;
                if !(scaled.is_finite() && scaled.abs() <= f64::from(f32::MAX)) {
                    return Err(PyValueError::new_err(format!(
                        "GPU session velocity source {src_idx} step {step} is not representable as finite f32"
                    )));
                }
                self.vel_x_signals[src_idx * time_steps + step] = f32::from_f64(scaled);
            }
        }

        Ok(())
    }
}
