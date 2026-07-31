use numpy::{PyReadonlyArray2, PyReadonlyArray3};
// Only referenced by the `#[cfg(not(feature = "gpu"))]` "feature not enabled" branches.
#[cfg(not(feature = "gpu"))]
use pyo3::exceptions::PyRuntimeError;
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyTuple};

use super::GpuPstdSession;
use crate::breast_fwi_bindings::complex_compat::{nd_to_leto2, nd_to_leto3};

#[pymethods]
impl GpuPstdSession {
    /// Set the source and sensor mask for all scan lines (constant per session).
    pub fn set_source_sensor(
        &mut self,
        _py: Python<'_>,
        mask: PyReadonlyArray3<f64>,
        ux_signals: PyReadonlyArray2<f64>,
    ) -> PyResult<()> {
        let mask_leto = nd_to_leto3(mask.as_array().to_owned());
        let sig_leto = nd_to_leto2(ux_signals.as_array().to_owned());
        self.rebuild_source_sensor_indices(mask_leto.view())?;
        self.update_velocity_signal_rows(sig_leto.view())
    }

    /// Cache the source/sensor mask when the geometry is invariant across runs.
    pub fn set_source_sensor_mask(
        &mut self,
        _py: Python<'_>,
        mask: PyReadonlyArray3<f64>,
    ) -> PyResult<()> {
        let mask_leto = nd_to_leto3(mask.as_array().to_owned());
        self.rebuild_source_sensor_indices(mask_leto.view())
    }

    /// Update only the x-velocity source signals for a previously cached mask.
    pub fn set_velocity_signals(
        &mut self,
        _py: Python<'_>,
        ux_signals: PyReadonlyArray2<f64>,
    ) -> PyResult<()> {
        let sig_leto = nd_to_leto2(ux_signals.as_array().to_owned());
        self.update_velocity_signal_rows(sig_leto.view())
    }

    /// Disable the k-space source correction (sets source_kappa = 1 everywhere).
    pub fn disable_source_correction(&self, _py: Python<'_>) -> PyResult<()> {
        #[cfg(feature = "gpu")]
        {
            self.solver.disable_source_correction();
            Ok(())
        }
        #[cfg(not(feature = "gpu"))]
        {
            Err(PyRuntimeError::new_err(
                "GpuPstdSession requires the 'gpu' feature.",
            ))
        }
    }

    /// Return the timing profile from the most recent scan-line execution.
    #[getter]
    pub fn last_run_profile<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let profile = PyDict::new(py);
        profile.set_item("medium_upload_ns", self.last_medium_upload_ns)?;
        profile.set_item(
            "medium_variable_upload_ns",
            self.last_medium_variable_upload_ns,
        )?;
        profile.set_item("medium_static_upload_ns", self.last_medium_static_upload_ns)?;
        profile.set_item("solver_run_ns", self.last_solver_run_ns)?;
        profile.set_item("materialize_ns", self.last_materialize_ns)?;
        profile.set_item("total_ns", self.last_total_ns)?;
        profile.set_item("n_sensors", self.sensor_indices.len())?;
        profile.set_item("n_velocity_sources", self.vel_x_indices.len())?;
        Ok(profile)
    }

    /// Return the most recent scan-line timing profile as a compact tuple.
    #[getter]
    pub fn last_run_profile_ns<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyTuple>> {
        PyTuple::new(
            py,
            [
                self.last_medium_upload_ns,
                self.last_medium_variable_upload_ns,
                self.last_medium_static_upload_ns,
                self.last_solver_run_ns,
                self.last_materialize_ns,
                self.last_total_ns,
            ],
        )
    }
}
