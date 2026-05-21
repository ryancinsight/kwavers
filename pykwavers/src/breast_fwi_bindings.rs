//! PyO3 bindings for Ali et al. 2025 breast UST frequency-domain FWI.
//!
//! The binding layer performs only Python/Rust data conversion. Geometry and
//! paper identities stay in `physics`, numerical propagation stays in `solver`,
//! and clinical image metadata stays in `clinical::imaging::reconstruction`.

mod dataset;
mod phantom;

use kwavers::clinical::imaging::reconstruction::breast_ust_fwi::{
    reconstruct_breast_ust_sound_speed_volume, snap_multi_row_ring_array_to_grid,
};
use kwavers::physics::acoustics::imaging::modalities::ultrasound::frequency_domain_fwi::{
    ali_2025_frequency_sweep_hz, MultiRowRingArray,
};
use kwavers::solver::inverse::fwi::frequency_domain::{
    simulate_frequency_observation, AbsorbingBoundary, Config, DenseConvergentBornOperator,
    FrequencyObservation, HelmholtzForwardOperator, SingleScatterBornOperator,
    SpectralConvergentBornOperator,
};
use std::sync::Arc;
use ndarray::{s, Array1, Array2, Array3};
use num_complex::Complex64;
use numpy::{IntoPyArray, PyArray1, PyArray2, PyReadonlyArray2, PyReadonlyArray3};
use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyModule};

pub use dataset::{generate_breast_fwi_pstd_dataset, PyBreastFwiPstdDatasetConfig};
pub use phantom::load_ali_2025_breast_fwi_phantom;

#[pyclass(name = "MultiRowRingArray")]
#[derive(Clone)]
pub struct PyMultiRowRingArray {
    inner: MultiRowRingArray,
}

#[pymethods]
impl PyMultiRowRingArray {
    #[new]
    pub fn new(
        circumferential_elements: usize,
        rows: usize,
        diameter_m: f64,
        row_spacing_m: f64,
    ) -> PyResult<Self> {
        Ok(Self {
            inner: MultiRowRingArray::new(
                circumferential_elements,
                rows,
                diameter_m,
                row_spacing_m,
            )
            .map_err(kwavers_to_py)?,
        })
    }

    #[staticmethod]
    pub fn ali_2025() -> PyResult<Self> {
        Ok(Self {
            inner: MultiRowRingArray::ali_2025().map_err(kwavers_to_py)?,
        })
    }

    #[getter]
    pub fn circumferential_elements(&self) -> usize {
        self.inner.circumferential_elements()
    }

    #[getter]
    pub fn rows(&self) -> usize {
        self.inner.rows()
    }

    #[getter]
    pub fn diameter_m(&self) -> f64 {
        self.inner.diameter_m()
    }

    #[getter]
    pub fn row_spacing_m(&self) -> f64 {
        self.inner.row_spacing_m()
    }

    #[getter]
    pub fn element_count(&self) -> usize {
        self.inner.element_count()
    }

    pub fn elements<'py>(&self, py: Python<'py>) -> Py<PyArray2<f64>> {
        points_to_array(self.inner.elements())
            .into_pyarray(py)
            .into()
    }

    pub fn cylindrical_source<'py>(
        &self,
        py: Python<'py>,
        transmit_index: usize,
    ) -> Py<PyArray2<f64>> {
        points_to_array(&self.inner.cylindrical_source(transmit_index))
            .into_pyarray(py)
            .into()
    }
}

#[pyclass(name = "FrequencyDomainFwiConfig")]
#[derive(Clone)]
pub struct PyFrequencyDomainFwiConfig {
    inner: Config,
}

#[pymethods]
impl PyFrequencyDomainFwiConfig {
    #[new]
    #[pyo3(signature = (
        reference_sound_speed_m_s = 1500.0,
        spacing_m = 1.0e-3,
        iterations = 5,
        initial_step_s_per_m = 2.0e-6,
        min_sound_speed_m_s = 1400.0,
        max_sound_speed_m_s = 1600.0,
        estimate_source_scaling = true,
        tikhonov_weight = 0.0,
        propagation_model = "single_scatter_born",
        cbs_iterations = 64,
        cbs_relative_tolerance = 1.0e-10,
        absorbing_boundary = "disabled",
        absorbing_thickness_cells = 1,
        absorbing_strength_nepers = 1.5,
        absorbing_order = 2
    ))]
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        reference_sound_speed_m_s: f64,
        spacing_m: f64,
        iterations: usize,
        initial_step_s_per_m: f64,
        min_sound_speed_m_s: f64,
        max_sound_speed_m_s: f64,
        estimate_source_scaling: bool,
        tikhonov_weight: f64,
        propagation_model: &str,
        cbs_iterations: usize,
        cbs_relative_tolerance: f64,
        absorbing_boundary: &str,
        absorbing_thickness_cells: usize,
        absorbing_strength_nepers: f64,
        absorbing_order: u32,
    ) -> PyResult<Self> {
        let forward_operator = parse_forward_operator(
            propagation_model,
            cbs_iterations,
            cbs_relative_tolerance,
            absorbing_boundary,
            absorbing_thickness_cells,
            absorbing_strength_nepers,
            absorbing_order,
        )?;
        Ok(Self {
            inner: Config {
                reference_sound_speed_m_s,
                spacing_m,
                iterations,
                initial_step_s_per_m,
                min_sound_speed_m_s,
                max_sound_speed_m_s,
                estimate_source_scaling,
                tikhonov_weight,
                forward_operator,
            },
        })
    }

    #[staticmethod]
    pub fn single_scatter() -> Self {
        Self {
            inner: Config::default(),
        }
    }

    #[staticmethod]
    #[pyo3(signature = (iterations = 64, relative_tolerance = 1.0e-10))]
    pub fn dense_convergent_born(iterations: usize, relative_tolerance: f64) -> Self {
        Self {
            inner: Config {
                forward_operator: Arc::new(DenseConvergentBornOperator {
                    iterations,
                    relative_tolerance,
                }),
                ..Config::default()
            },
        }
    }

    #[staticmethod]
    #[pyo3(signature = (
        iterations = 64,
        relative_tolerance = 1.0e-10,
        absorbing_thickness_cells = 1,
        absorbing_strength_nepers = 1.5,
        absorbing_order = 2
    ))]
    pub fn spectral_convergent_born(
        iterations: usize,
        relative_tolerance: f64,
        absorbing_thickness_cells: usize,
        absorbing_strength_nepers: f64,
        absorbing_order: u32,
    ) -> PyResult<Self> {
        Ok(Self {
            inner: Config {
                forward_operator: Arc::new(SpectralConvergentBornOperator {
                    iterations,
                    relative_tolerance,
                    absorbing_boundary: AbsorbingBoundary::polynomial(
                        absorbing_thickness_cells,
                        absorbing_strength_nepers,
                        absorbing_order,
                    )
                    .map_err(kwavers_to_py)?,
                }),
                ..Config::default()
            },
        })
    }

    #[getter]
    pub fn reference_sound_speed_m_s(&self) -> f64 {
        self.inner.reference_sound_speed_m_s
    }

    #[getter]
    pub fn spacing_m(&self) -> f64 {
        self.inner.spacing_m
    }

    #[getter]
    pub fn iterations(&self) -> usize {
        self.inner.iterations
    }

    #[getter]
    pub fn propagation_model(&self) -> &'static str {
        self.inner.forward_operator.model_id()
    }
}

#[pyclass(name = "FrequencyObservation")]
#[derive(Clone)]
pub struct PyFrequencyObservation {
    inner: FrequencyObservation,
}

#[pymethods]
impl PyFrequencyObservation {
    #[new]
    pub fn new(frequency_hz: f64, observed_pressure: PyReadonlyArray2<Complex64>) -> Self {
        Self {
            inner: FrequencyObservation::new(frequency_hz, observed_pressure.as_array().to_owned()),
        }
    }

    #[getter]
    pub fn frequency_hz(&self) -> f64 {
        self.inner.frequency_hz
    }

    #[getter]
    pub fn observed_pressure<'py>(&self, py: Python<'py>) -> Py<PyArray2<Complex64>> {
        self.inner.observed_pressure.clone().into_pyarray(py).into()
    }
}

#[pyfunction]
pub fn ali_2025_breast_fwi_frequency_sweep_hz<'py>(py: Python<'py>) -> Py<PyArray1<f64>> {
    Array1::from(ali_2025_frequency_sweep_hz())
        .into_pyarray(py)
        .into()
}

#[pyfunction]
pub fn simulate_breast_fwi_frequency_observation<'py>(
    py: Python<'py>,
    sound_speed_m_s: PyReadonlyArray3<'py, f64>,
    array: &PyMultiRowRingArray,
    frequency_hz: f64,
    config: &PyFrequencyDomainFwiConfig,
) -> PyResult<Py<PyArray2<Complex64>>> {
    let sound_speed = sound_speed_m_s.as_array().to_owned();
    let pressure = py
        .detach(|| {
            simulate_frequency_observation(&sound_speed, &array.inner, frequency_hz, &config.inner)
        })
        .map_err(kwavers_to_py)?;
    Ok(pressure.into_pyarray(py).into())
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
    let initial = initial_sound_speed_m_s.as_array().to_owned();
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
    out.set_item("sound_speed_m_s", result.sound_speed_m_s.into_pyarray(py))?;
    out.set_item(
        "objective_history",
        Array1::from(result.objective_history).into_pyarray(py),
    )?;
    out.set_item("frequencies_used", result.frequencies_used)?;
    out.set_item("transmissions_used", result.transmissions_used)?;
    out.set_item("receivers_used", result.receivers_used)?;
    out.set_item("model_family", result.model_family)?;
    out.set_item("solver_model_family", result.solver_model_family)?;
    Ok(out)
}

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyMultiRowRingArray>()?;
    m.add_class::<PyFrequencyDomainFwiConfig>()?;
    m.add_class::<PyFrequencyObservation>()?;
    m.add_class::<PyBreastFwiPstdDatasetConfig>()?;
    m.add_function(wrap_pyfunction!(ali_2025_breast_fwi_frequency_sweep_hz, m)?)?;
    m.add_function(wrap_pyfunction!(
        simulate_breast_fwi_frequency_observation,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(snap_breast_fwi_array_to_grid, m)?)?;
    m.add_function(wrap_pyfunction!(load_ali_2025_breast_fwi_phantom, m)?)?;
    m.add_function(wrap_pyfunction!(generate_breast_fwi_pstd_dataset, m)?)?;
    m.add_function(wrap_pyfunction!(invert_breast_fwi, m)?)?;
    Ok(())
}

fn parse_forward_operator(
    propagation_model: &str,
    cbs_iterations: usize,
    cbs_relative_tolerance: f64,
    absorbing_boundary: &str,
    absorbing_thickness_cells: usize,
    absorbing_strength_nepers: f64,
    absorbing_order: u32,
) -> PyResult<Arc<dyn HelmholtzForwardOperator>> {
    match propagation_model {
        "single_scatter_born" => Ok(Arc::new(SingleScatterBornOperator)),
        "dense_convergent_born" => Ok(Arc::new(DenseConvergentBornOperator {
            iterations: cbs_iterations,
            relative_tolerance: cbs_relative_tolerance,
        })),
        "spectral_convergent_born" => Ok(Arc::new(SpectralConvergentBornOperator {
            iterations: cbs_iterations,
            relative_tolerance: cbs_relative_tolerance,
            absorbing_boundary: parse_absorbing_boundary(
                absorbing_boundary,
                absorbing_thickness_cells,
                absorbing_strength_nepers,
                absorbing_order,
            )?,
        })),
        other => Err(PyValueError::new_err(format!(
            "unknown breast FWI propagation_model '{other}'"
        ))),
    }
}

fn parse_absorbing_boundary(
    absorbing_boundary: &str,
    thickness_cells: usize,
    strength_nepers: f64,
    order: u32,
) -> PyResult<AbsorbingBoundary> {
    match absorbing_boundary {
        "disabled" => Ok(AbsorbingBoundary::disabled()),
        "polynomial" => AbsorbingBoundary::polynomial(thickness_cells, strength_nepers, order)
            .map_err(kwavers_to_py),
        other => Err(PyValueError::new_err(format!(
            "unknown breast FWI absorbing_boundary '{other}'"
        ))),
    }
}

fn points_to_array(
    points: &[kwavers::physics::acoustics::imaging::modalities::ultrasound::frequency_domain_fwi::RingPoint],
) -> Array2<f64> {
    Array2::from_shape_fn((points.len(), 3), |(row, col)| match col {
        0 => points[row].x_m,
        1 => points[row].y_m,
        _ => points[row].z_m,
    })
}

fn observations_from_stack(
    frequencies_hz: &[f64],
    observed_pressure: Array3<Complex64>,
) -> PyResult<Vec<FrequencyObservation>> {
    let (frequency_count, _, _) = observed_pressure.dim();
    if frequencies_hz.len() != frequency_count {
        return Err(PyValueError::new_err(format!(
            "frequencies_hz length {} must match observed_pressure first dimension {}",
            frequencies_hz.len(),
            frequency_count
        )));
    }
    Ok(frequencies_hz
        .iter()
        .enumerate()
        .map(|(index, &frequency_hz)| {
            FrequencyObservation::new(
                frequency_hz,
                observed_pressure.slice(s![index, .., ..]).to_owned(),
            )
        })
        .collect())
}

fn kwavers_to_py(err: kwavers::core::error::KwaversError) -> PyErr {
    PyRuntimeError::new_err(format!("kwavers breast FWI failed: {err}"))
}
