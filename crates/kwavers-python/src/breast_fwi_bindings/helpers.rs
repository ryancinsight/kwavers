//! Internal helpers: operator parsing, boundary parsing, array conversion,
//! observation stack construction, and error conversion for breast FWI bindings.

use kwavers_solver::inverse::fwi::frequency_domain::{
    AbsorbingBoundary, Config, DenseConvergentBornOperator, FrequencyObservation,
    HelmholtzForwardOperator, PstdFiniteWindowBornOperator,
    PstdFiniteWindowBornSecondOrderOperator, PstdSpectralConvergentBornOperator,
    PstdTemporalTransferConfig, SingleScatterBornOperator, SpectralConvergentBornOperator,
};
use kwavers_transducer::transducers::ElementPosition;
use ndarray::{s, Array2, Array3};
use num_complex::Complex64;
use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use std::sync::Arc;

pub(super) fn parse_forward_operator(
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
        "pstd_spectral_convergent_born" => Ok(Arc::new(PstdSpectralConvergentBornOperator {
            iterations: cbs_iterations,
            relative_tolerance: cbs_relative_tolerance,
            time_step_s: pstd_time_step_s,
            temporal_transfer: Some(PstdTemporalTransferConfig {
                source_amplitude_pa: pstd_source_amplitude_pa,
                cycles_per_frequency: pstd_cycles_per_frequency,
                frequency_bin_cycles: pstd_frequency_bin_cycles,
            }),
            absorbing_boundary: parse_absorbing_boundary(
                absorbing_boundary,
                absorbing_thickness_cells,
                absorbing_strength_nepers,
                absorbing_order,
            )?,
        })),
        "pstd_finite_window_born" => Ok(Arc::new(PstdFiniteWindowBornOperator {
            time_step_s: pstd_time_step_s,
            source_amplitude_pa: pstd_source_amplitude_pa,
            cycles_per_frequency: pstd_cycles_per_frequency,
            frequency_bin_cycles: pstd_frequency_bin_cycles,
        })),
        "pstd_finite_window_born_second_order" => {
            Ok(Arc::new(PstdFiniteWindowBornSecondOrderOperator {
                time_step_s: pstd_time_step_s,
                source_amplitude_pa: pstd_source_amplitude_pa,
                cycles_per_frequency: pstd_cycles_per_frequency,
                frequency_bin_cycles: pstd_frequency_bin_cycles,
            }))
        }
        other => Err(PyValueError::new_err(format!(
            "unknown breast FWI propagation_model '{other}'"
        ))),
    }
}

pub(super) fn parse_absorbing_boundary(
    absorbing_boundary: &str,
    thickness_cells: usize,
    strength_nepers: f64,
    order: u32,
) -> PyResult<AbsorbingBoundary> {
    match absorbing_boundary {
        "disabled" => Ok(AbsorbingBoundary::disabled()),
        "polynomial" if thickness_cells == 0 => Ok(AbsorbingBoundary::disabled()),
        "polynomial" => AbsorbingBoundary::polynomial(thickness_cells, strength_nepers, order)
            .map_err(kwavers_to_py),
        other => Err(PyValueError::new_err(format!(
            "unknown breast FWI absorbing_boundary '{other}'"
        ))),
    }
}

pub(super) fn absorbing_boundary_from_thickness(
    thickness_cells: usize,
    strength_nepers: f64,
    order: u32,
) -> PyResult<AbsorbingBoundary> {
    if thickness_cells == 0 {
        Ok(AbsorbingBoundary::disabled())
    } else {
        AbsorbingBoundary::polynomial(thickness_cells, strength_nepers, order)
            .map_err(kwavers_to_py)
    }
}

pub(super) fn points_to_array(points: &[ElementPosition]) -> Array2<f64> {
    Array2::from_shape_fn((points.len(), 3), |(row, col)| match col {
        0 => points[row].x_m,
        1 => points[row].y_m,
        _ => points[row].z_m,
    })
}

pub(super) fn observations_from_stack(
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

pub(super) fn kwavers_to_py(err: kwavers_core::error::KwaversError) -> PyErr {
    PyRuntimeError::new_err(format!("kwavers breast FWI failed: {err}"))
}

pub(super) fn make_config(
    reference_sound_speed_m_s: f64,
    spacing_m: f64,
    iterations: usize,
    initial_step_s_per_m: f64,
    min_sound_speed_m_s: f64,
    max_sound_speed_m_s: f64,
    estimate_source_scaling: bool,
    tikhonov_weight: f64,
    forward_operator: Arc<dyn HelmholtzForwardOperator>,
) -> Config {
    Config {
        reference_sound_speed_m_s,
        spacing_m,
        iterations,
        initial_step_s_per_m,
        min_sound_speed_m_s,
        max_sound_speed_m_s,
        estimate_source_scaling,
        tikhonov_weight,
        forward_operator,
    }
}
