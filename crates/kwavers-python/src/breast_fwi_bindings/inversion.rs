//! PyO3 bindings: `FrequencyObservation` class and top-level FWI functions
//! (`ali_2025_breast_fwi_frequency_sweep_hz`, `simulate_breast_fwi_frequency_observation`,
//! `snap_breast_fwi_array_to_grid`, `invert_breast_fwi`).

use eunomia::Complex64;
use kwavers_diagnostics::reconstruction::breast_ust_fwi::{
    reconstruct_breast_ust_sound_speed_volume, snap_multi_row_ring_array_to_grid,
};
use kwavers_physics::acoustics::imaging::modalities::ultrasound::frequency_domain_fwi::ali_2025_frequency_sweep_hz;
use kwavers_solver::inverse::fwi::frequency_domain::{
    simulate_frequency_observation, FrequencyObservation,
};
use numpy::{PyArray1, PyArray2, PyReadonlyArray2, PyReadonlyArray3};
use pyo3::prelude::*;
use pyo3::types::PyDict;

use super::array_config::PyMultiRowRingArray;
use super::fwi_config::PyFrequencyDomainFwiConfig;
use super::helpers::{kwavers_to_py, observations_from_stack};

use crate::array_utils::{
    leto2_to_pyarray2, leto3_to_pyarray3, pyarray2_to_leto2, pyarray3_to_leto3, vec_to_pyarray1,
};

#[pyclass(name = "FrequencyObservation", from_py_object)]
#[derive(Clone)]
pub struct PyFrequencyObservation {
    pub(super) inner: FrequencyObservation,
}

#[pymethods]
impl PyFrequencyObservation {
    #[new]
    pub fn new(
        frequency_hz: f64,
        observed_pressure: PyReadonlyArray2<Complex64>,
    ) -> PyResult<Self> {
        Ok(Self {
            inner: FrequencyObservation::new(frequency_hz, pyarray2_to_leto2(&observed_pressure)?),
        })
    }

    #[getter]
    pub fn frequency_hz(&self) -> f64 {
        self.inner.frequency_hz
    }

    #[getter]
    pub fn observed_pressure<'py>(&self, py: Python<'py>) -> PyResult<Py<PyArray2<Complex64>>> {
        leto2_to_pyarray2(py, self.inner.observed_pressure.clone())
    }
}

#[pyfunction]
pub fn ali_2025_breast_fwi_frequency_sweep_hz<'py>(py: Python<'py>) -> Py<PyArray1<f64>> {
    vec_to_pyarray1(py, ali_2025_frequency_sweep_hz())
}

#[pyfunction]
pub fn simulate_breast_fwi_frequency_observation<'py>(
    py: Python<'py>,
    sound_speed_m_s: PyReadonlyArray3<'py, f64>,
    array: &PyMultiRowRingArray,
    frequency_hz: f64,
    config: &PyFrequencyDomainFwiConfig,
) -> PyResult<Py<PyArray2<Complex64>>> {
    let sound_speed = pyarray3_to_leto3(&sound_speed_m_s)?;
    let pressure = py
        .detach(|| {
            simulate_frequency_observation(&sound_speed, &array.inner, frequency_hz, &config.inner)
        })
        .map_err(kwavers_to_py)?;
    leto2_to_pyarray2(py, pressure)
}

#[pyfunction]
pub fn snap_breast_fwi_array_to_grid(
    array: &PyMultiRowRingArray,
    dimensions: (usize, usize, usize),
    spacing_m: f64,
) -> PyResult<PyMultiRowRingArray> {
    Ok(PyMultiRowRingArray {
        inner: snap_multi_row_ring_array_to_grid(&array.inner, dimensions, spacing_m)
            .map_err(kwavers_to_py)?,
    })
}

#[pyfunction]
pub fn invert_breast_fwi<'py>(
    py: Python<'py>,
    frequencies_hz: Vec<f64>,
    observed_pressure: PyReadonlyArray3<'py, Complex64>,
    array: &PyMultiRowRingArray,
    initial_sound_speed_m_s: PyReadonlyArray3<'py, f64>,
    config: &PyFrequencyDomainFwiConfig,
) -> PyResult<Bound<'py, PyDict>> {
    let pressure = observed_pressure.as_array().to_owned();
    let initial = pyarray3_to_leto3(&initial_sound_speed_m_s)?;
    let observations = observations_from_stack(&frequencies_hz, pressure)?;
    let result = py
        .detach(|| {
            reconstruct_breast_ust_sound_speed_volume(
                &observations,
                &array.inner,
                &initial,
                &config.inner,
            )
        })
        .map_err(kwavers_to_py)?;

    let out = PyDict::new(py);
    out.set_item(
        "sound_speed_m_s",
        leto3_to_pyarray3(py, result.sound_speed_m_s)?,
    )?;
    out.set_item(
        "objective_history",
        vec_to_pyarray1(py, result.objective_history),
    )?;
    out.set_item("frequencies_used", result.frequencies_used)?;
    out.set_item("transmissions_used", result.transmissions_used)?;
    out.set_item("receivers_used", result.receivers_used)?;
    out.set_item("model_family", result.model_family)?;
    out.set_item("solver_model_family", result.solver_model_family)?;
    Ok(out)
}
