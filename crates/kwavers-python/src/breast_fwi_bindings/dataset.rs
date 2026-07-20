//! PyO3 wrapper for PSTD breast-FWI dataset generation.

use super::complex_compat::{leto3_to_nd3, nd_to_leto3};
use super::helpers::kwavers_to_py;
use super::PyMultiRowRingArray;
use kwavers_diagnostics::reconstruction::breast_ust_fwi::{
    generate_breast_ust_pstd_frequency_dataset, BreastUstPstdDatasetConfig,
};
use numpy::ndarray::Array1;
use numpy::{PyReadonlyArray3, ToPyArray};
use pyo3::prelude::*;
use pyo3::types::PyDict;

#[pyclass(name = "BreastFwiPstdDatasetConfig", from_py_object)]
#[derive(Clone, Copy)]
pub struct PyBreastFwiPstdDatasetConfig {
    pub(super) inner: BreastUstPstdDatasetConfig,
}

#[pymethods]
impl PyBreastFwiPstdDatasetConfig {
    #[new]
    #[pyo3(signature = (
        spacing_m = 1.0e-3,
        time_step_s = 1.0e-7,
        cycles_per_frequency = 4,
        frequency_bin_cycles = 1,
        source_amplitude_pa = 1.0e3,
        density_kg_m3 = 1000.0,
        cpml_thickness_cells = 8
    ))]
    pub fn new(
        spacing_m: f64,
        time_step_s: f64,
        cycles_per_frequency: usize,
        frequency_bin_cycles: usize,
        source_amplitude_pa: f64,
        density_kg_m3: f64,
        cpml_thickness_cells: usize,
    ) -> Self {
        Self {
            inner: BreastUstPstdDatasetConfig {
                spacing_m,
                time_step_s,
                cycles_per_frequency,
                frequency_bin_cycles,
                source_amplitude_pa,
                density_kg_m3,
                cpml_thickness_cells,
            },
        }
    }

    #[getter]
    pub fn spacing_m(&self) -> f64 {
        self.inner.spacing_m
    }

    #[getter]
    pub fn time_step_s(&self) -> f64 {
        self.inner.time_step_s
    }

    #[getter]
    pub fn cycles_per_frequency(&self) -> usize {
        self.inner.cycles_per_frequency
    }

    #[getter]
    pub fn frequency_bin_cycles(&self) -> usize {
        self.inner.frequency_bin_cycles
    }

    #[getter]
    pub fn source_amplitude_pa(&self) -> f64 {
        self.inner.source_amplitude_pa
    }

    #[getter]
    pub fn density_kg_m3(&self) -> f64 {
        self.inner.density_kg_m3
    }

    #[getter]
    pub fn cpml_thickness_cells(&self) -> usize {
        self.inner.cpml_thickness_cells
    }
}

#[pyfunction]
pub fn generate_breast_fwi_pstd_dataset<'py>(
    py: Python<'py>,
    sound_speed_m_s: PyReadonlyArray3<'py, f64>,
    array: &PyMultiRowRingArray,
    frequencies_hz: Vec<f64>,
    config: &PyBreastFwiPstdDatasetConfig,
) -> PyResult<Bound<'py, PyDict>> {
    let sound_speed = nd_to_leto3(sound_speed_m_s.as_array().to_owned());
    let dataset = py
        .detach(|| {
            generate_breast_ust_pstd_frequency_dataset(
                &sound_speed,
                &array.inner,
                &frequencies_hz,
                config.inner,
            )
        })
        .map_err(kwavers_to_py)?;

    let out = PyDict::new(py);
    out.set_item(
        "frequencies_hz",
        Array1::from(dataset.frequencies_hz.clone()).to_pyarray(py),
    )?;
    out.set_item(
        "observed_pressure",
        leto3_to_nd3(dataset.observed_pressure).to_pyarray(py),
    )?;
    out.set_item("transmissions", dataset.transmissions)?;
    out.set_item("receivers", dataset.receivers)?;
    out.set_item("time_steps_per_frequency", dataset.time_steps_per_frequency)?;
    out.set_item(
        "frequency_bin_start_steps_per_frequency",
        dataset.frequency_bin_start_steps_per_frequency,
    )?;
    out.set_item("model_family", dataset.model_family)?;
    Ok(out)
}
