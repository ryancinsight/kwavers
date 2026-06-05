//! PyO3 binding: `FrequencyDomainFwiConfig` Python class.

use kwavers_solver::inverse::fwi::frequency_domain::{
    Config, DenseConvergentBornOperator, PstdSpectralConvergentBornOperator,
    PstdTemporalTransferConfig, SpectralConvergentBornOperator,
};
use pyo3::prelude::*;
use std::sync::Arc;

use super::helpers::{absorbing_boundary_from_thickness, make_config, parse_forward_operator};

#[pyclass(name = "FrequencyDomainFwiConfig")]
#[derive(Clone)]
pub struct PyFrequencyDomainFwiConfig {
    pub(super) inner: Config,
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
        absorbing_order = 2,
        pstd_time_step_s = 1.0e-7,
        pstd_source_amplitude_pa = 1.0e3,
        pstd_cycles_per_frequency = 4,
        pstd_frequency_bin_cycles = 1
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
        pstd_time_step_s: f64,
        pstd_source_amplitude_pa: f64,
        pstd_cycles_per_frequency: usize,
        pstd_frequency_bin_cycles: usize,
    ) -> PyResult<Self> {
        let forward_operator = parse_forward_operator(
            propagation_model,
            cbs_iterations,
            cbs_relative_tolerance,
            absorbing_boundary,
            absorbing_thickness_cells,
            absorbing_strength_nepers,
            absorbing_order,
            pstd_time_step_s,
            pstd_source_amplitude_pa,
            pstd_cycles_per_frequency,
            pstd_frequency_bin_cycles,
        )?;
        Ok(Self {
            inner: make_config(
                reference_sound_speed_m_s,
                spacing_m,
                iterations,
                initial_step_s_per_m,
                min_sound_speed_m_s,
                max_sound_speed_m_s,
                estimate_source_scaling,
                tikhonov_weight,
                forward_operator,
            ),
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
                    absorbing_boundary: absorbing_boundary_from_thickness(
                        absorbing_thickness_cells,
                        absorbing_strength_nepers,
                        absorbing_order,
                    )?,
                }),
                ..Config::default()
            },
        })
    }

    #[staticmethod]
    #[pyo3(signature = (
        iterations = 64,
        relative_tolerance = 1.0e-10,
        time_step_s = 1.0e-7,
        source_amplitude_pa = 1.0e3,
        cycles_per_frequency = 4,
        frequency_bin_cycles = 1,
        absorbing_thickness_cells = 1,
        absorbing_strength_nepers = 1.5,
        absorbing_order = 2
    ))]
    pub fn pstd_spectral_convergent_born(
        iterations: usize,
        relative_tolerance: f64,
        time_step_s: f64,
        source_amplitude_pa: f64,
        cycles_per_frequency: usize,
        frequency_bin_cycles: usize,
        absorbing_thickness_cells: usize,
        absorbing_strength_nepers: f64,
        absorbing_order: u32,
    ) -> PyResult<Self> {
        Ok(Self {
            inner: Config {
                forward_operator: Arc::new(PstdSpectralConvergentBornOperator {
                    iterations,
                    relative_tolerance,
                    time_step_s,
                    temporal_transfer: Some(PstdTemporalTransferConfig {
                        source_amplitude_pa,
                        cycles_per_frequency,
                        frequency_bin_cycles,
                    }),
                    absorbing_boundary: absorbing_boundary_from_thickness(
                        absorbing_thickness_cells,
                        absorbing_strength_nepers,
                        absorbing_order,
                    )?,
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
