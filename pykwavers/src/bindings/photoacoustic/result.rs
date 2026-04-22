use numpy::{PyArray1, PyArray2, PyArray3};
use pyo3::prelude::*;

#[pyclass]
pub struct PhotoacousticRunResult {
    pub(crate) optical_fluence: Py<PyArray3<f64>>,
    pub(crate) initial_pressure: Py<PyArray3<f64>>,
    pub(crate) reconstruction: Py<PyArray3<f64>>,
    pub(crate) sensor_data: Py<PyArray2<f64>>,
    pub(crate) time_points: Py<PyArray1<f64>>,
    #[pyo3(get)]
    pub optical_model: String,
    #[pyo3(get)]
    pub wavelength_nm: f64,
    #[pyo3(get)]
    pub stress_confined: bool,
    #[pyo3(get)]
    pub thermal_confined: bool,
    #[pyo3(get)]
    pub total_optical_energy: f64,
    #[pyo3(get)]
    pub max_initial_pressure: f64,
    #[pyo3(get)]
    pub relative_pressure_balance_error: f64,
}

#[pymethods]
impl PhotoacousticRunResult {
    #[getter]
    fn optical_fluence<'py>(&self, py: Python<'py>) -> Py<PyAny> {
        self.optical_fluence.clone_ref(py).into_any()
    }

    #[getter]
    fn initial_pressure<'py>(&self, py: Python<'py>) -> Py<PyAny> {
        self.initial_pressure.clone_ref(py).into_any()
    }

    #[getter]
    fn reconstruction<'py>(&self, py: Python<'py>) -> Py<PyAny> {
        self.reconstruction.clone_ref(py).into_any()
    }

    #[getter]
    fn sensor_data<'py>(&self, py: Python<'py>) -> Py<PyAny> {
        self.sensor_data.clone_ref(py).into_any()
    }

    #[getter]
    fn time_points<'py>(&self, py: Python<'py>) -> Py<PyAny> {
        self.time_points.clone_ref(py).into_any()
    }
}
