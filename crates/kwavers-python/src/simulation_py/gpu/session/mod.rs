mod construction;
mod control;
mod run;

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

/// Persistent GPU PSTD session for efficient B-mode scan-line loops.
///
/// Creating a new `GpuPstdSolver` per scan line is expensive (~500 ms) because
/// wgpu must compile ~13 WGSL compute pipelines from scratch.  `GpuPstdSession`
/// creates the solver **once** and re-uses compiled pipelines.  Between scan
/// lines you only re-upload the medium arrays via `run_scan_line()`.
#[cfg_attr(not(feature = "gpu"), allow(dead_code))]
#[pyclass(unsendable)]
pub struct GpuPstdSession {
    #[cfg(feature = "gpu")]
    pub(crate) solver: kwavers_gpu::pstd_gpu::GpuPstdSolver,

    pub(crate) nx: usize,
    pub(crate) ny: usize,
    pub(crate) nz: usize,

    pub(crate) bon_a_flat: Vec<f32>,
    pub(crate) absorb_tau_flat: Vec<f32>,
    pub(crate) absorb_eta_flat: Vec<f32>,
    pub(crate) has_absorption: bool,

    pub(crate) time_steps: usize,

    pub(crate) sensor_indices: Vec<u32>,
    pub(crate) vel_x_indices: Vec<u32>,
    pub(crate) vel_x_signals: Vec<f32>,
    pub(crate) last_medium_upload_ns: u64,
    pub(crate) last_medium_variable_upload_ns: u64,
    pub(crate) last_medium_static_upload_ns: u64,
    pub(crate) last_solver_run_ns: u64,
    pub(crate) last_materialize_ns: u64,
    pub(crate) last_total_ns: u64,
}

impl GpuPstdSession {
    pub(crate) fn rebuild_source_sensor_indices(
        &mut self,
        mask_arr: ndarray::ArrayView3<'_, f64>,
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
        sig_arr: ndarray::ArrayView2<'_, f64>,
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

        self.vel_x_signals.clear();
        self.vel_x_signals.resize(n_vel * time_steps, 0.0f32);
        for src_idx in 0..n_vel {
            let sig_row = src_idx.min(n_sig_srcs.saturating_sub(1));
            for step in 0..n_sig_cols {
                self.vel_x_signals[src_idx * time_steps + step] = sig_arr[[sig_row, step]] as f32;
            }
        }

        Ok(())
    }
}
