use numpy::PyArray2;
use numpy::PyReadonlyArray3;
// Only referenced by the `#[cfg(not(feature = "gpu"))]` "feature not enabled" branches.
#[cfg(not(feature = "gpu"))]
use pyo3::exceptions::PyRuntimeError;
use pyo3::prelude::*;

use super::GpuPstdSession;

#[pymethods]
impl GpuPstdSession {
    /// Run one scan line with updated medium (sound_speed, density).
    pub fn run_scan_line<'py>(
        &mut self,
        _py: Python<'py>,
        _sound_speed: PyReadonlyArray3<f64>,
        _density: PyReadonlyArray3<f64>,
    ) -> PyResult<Bound<'py, PyArray2<f64>>> {
        #[cfg(not(feature = "gpu"))]
        {
            Err(PyRuntimeError::new_err("GPU feature not enabled"))
        }

        #[cfg(feature = "gpu")]
        {
            use std::borrow::Cow;

            let total_t0 = std::time::Instant::now();
            let ss_arr = _sound_speed.as_array();
            let rho_arr = _density.as_array();

            let c0_flat: Cow<'_, [f64]> = match ss_arr.as_slice() {
                Some(slice) => Cow::Borrowed(slice),
                None => Cow::Owned(ss_arr.iter().copied().collect()),
            };
            let rho0_flat: Cow<'_, [f64]> = match rho_arr.as_slice() {
                Some(slice) => Cow::Borrowed(slice),
                None => Cow::Owned(rho_arr.iter().copied().collect()),
            };

            let upload_t0 = std::time::Instant::now();
            self.solver
                .update_medium_variable(c0_flat.as_ref(), rho0_flat.as_ref());
            let medium_upload_ns = upload_t0.elapsed().as_nanos() as u64;

            let result = self.run_scan_line_cached(_py);
            self.last_medium_variable_upload_ns = medium_upload_ns;
            self.last_medium_static_upload_ns = 0;
            self.last_medium_upload_ns = medium_upload_ns;
            self.last_total_ns = total_t0.elapsed().as_nanos() as u64;
            result
        }
    }

    /// Run one scan line using the currently resident medium buffers.
    pub fn run_scan_line_cached<'py>(
        &mut self,
        _py: Python<'py>,
    ) -> PyResult<Bound<'py, PyArray2<f64>>> {
        #[cfg(not(feature = "gpu"))]
        {
            Err(PyRuntimeError::new_err("GPU feature not enabled"))
        }

        #[cfg(feature = "gpu")]
        {
            use crate::breast_fwi_bindings::complex_compat::leto2_to_nd2;
            use kwavers_gpu::pstd_gpu::{PstdOutputRequest, PstdRunInputs};

            let total_t0 = std::time::Instant::now();
            self.last_medium_upload_ns = 0;
            self.last_medium_variable_upload_ns = 0;
            self.last_medium_static_upload_ns = 0;
            let time_steps = self.time_steps;

            let solver_t0 = std::time::Instant::now();
            let result = self.solver.run(PstdRunInputs {
                sensor_indices: &self.sensor_indices,
                source_indices: &[],
                source_signals: &[],
                pressure_source_correction: false,
                vel_x_indices: &self.vel_x_indices,
                vel_x_signals: &self.vel_x_signals,
                velocity_source_correction: false,
                output_request: PstdOutputRequest::sensor_traces(),
            });
            self.last_solver_run_ns = solver_t0.elapsed().as_nanos() as u64;

            let materialize_t0 = std::time::Instant::now();
            let n_sensors = self.sensor_indices.len();
            let out_flat: Vec<f64> = result.sensor_data.iter().map(|&v| v as f64).collect();
            let out = leto::Array2::from_shape_vec((n_sensors, time_steps), out_flat)
                .expect("sensor_data shape mismatch");
            self.last_materialize_ns = materialize_t0.elapsed().as_nanos() as u64;
            if self.last_medium_upload_ns == 0 {
                self.last_total_ns = total_t0.elapsed().as_nanos() as u64;
            }

            Ok(PyArray2::from_owned_array(_py, leto2_to_nd2(out)))
        }
    }
}
