use kwavers::simulation::PhotoacousticRunner as KwaversPhotoacousticRunner;
use ndarray::Array1;
use numpy::{PyArray1, PyArray2, PyArray3};
use pyo3::prelude::*;

use super::parity::parity_references;
use super::result::PhotoacousticRunResult;
use super::scenario::PhotoacousticScenario;
use crate::bindings::common::kwavers_error_to_py;

#[pyclass]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PhotoacousticOpticalModel {
    Diffusion,
    MonteCarlo,
}

#[pymethods]
impl PhotoacousticOpticalModel {
    fn __repr__(&self) -> String {
        match self {
            Self::Diffusion => "PhotoacousticOpticalModel.Diffusion".to_string(),
            Self::MonteCarlo => "PhotoacousticOpticalModel.MonteCarlo".to_string(),
        }
    }
}

#[pyclass]
#[derive(Default)]
pub struct PhotoacousticRunner {
    inner: KwaversPhotoacousticRunner,
}

#[pymethods]
impl PhotoacousticRunner {
    #[new]
    fn new() -> Self {
        Self::default()
    }

    fn run<'py>(
        &self,
        py: Python<'py>,
        scenario: &PhotoacousticScenario,
    ) -> PyResult<PhotoacousticRunResult> {
        let (simulation, report) = self
            .inner
            .run(&scenario.inner)
            .map_err(kwavers_error_to_py)?;

        let time_points =
            PyArray1::from_owned_array(py, Array1::from_vec(simulation.time_points)).into();
        let optical_fluence = PyArray3::from_owned_array(py, simulation.optical_fluence).into();
        let initial_pressure =
            PyArray3::from_owned_array(py, simulation.initial_pressure.pressure).into();
        let reconstruction = PyArray3::from_owned_array(py, simulation.reconstruction).into();
        let sensor_data = PyArray2::from_owned_array(py, simulation.signals.sensor_data).into();

        Ok(PhotoacousticRunResult {
            optical_fluence,
            initial_pressure,
            reconstruction,
            sensor_data,
            time_points,
            optical_model: report.optical_model,
            wavelength_nm: report.wavelength_nm,
            stress_confined: report.stress_confined,
            thermal_confined: report.thermal_confined,
            total_optical_energy: report.total_optical_energy,
            max_initial_pressure: report.max_initial_pressure,
            relative_pressure_balance_error: report.relative_pressure_balance_error,
        })
    }

    #[staticmethod]
    fn parity_references() -> Vec<String> {
        parity_references()
            .iter()
            .map(|entry| (*entry).to_string())
            .collect()
    }
}
