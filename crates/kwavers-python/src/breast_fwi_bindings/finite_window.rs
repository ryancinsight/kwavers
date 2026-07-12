//! PyO3 conversion surface for finite-window PSTD Born prediction.

use super::complex_compat::{leto2_to_nd2, nd_to_leto3};
use super::helpers::kwavers_to_py;
use super::PyMultiRowRingArray;
use eunomia::Complex64;
use kwavers_solver::inverse::fwi::frequency_domain::{
    simulate_pstd_finite_window_born_observation,
    simulate_pstd_finite_window_born_second_order_observation, PstdFiniteWindowBornConfig,
};
use numpy::{PyArray2, PyReadonlyArray3, ToPyArray};
use pyo3::prelude::*;
use pyo3::types::PyModule;

#[pyfunction]
#[pyo3(signature = (
    sound_speed_m_s,
    array,
    frequency_hz,
    reference_sound_speed_m_s = 1500.0,
    spacing_m = 1.0e-3,
    time_step_s = 1.0e-7,
    source_amplitude_pa = 1.0e3,
    cycles_per_frequency = 4,
    frequency_bin_cycles = 1,
    transmissions = None
))]
#[allow(clippy::too_many_arguments)]
pub fn simulate_breast_fwi_pstd_finite_window_born_observation<'py>(
    py: Python<'py>,
    sound_speed_m_s: PyReadonlyArray3<'py, f64>,
    array: &PyMultiRowRingArray,
    frequency_hz: f64,
    reference_sound_speed_m_s: f64,
    spacing_m: f64,
    time_step_s: f64,
    source_amplitude_pa: f64,
    cycles_per_frequency: usize,
    frequency_bin_cycles: usize,
    transmissions: Option<usize>,
) -> PyResult<Py<PyArray2<Complex64>>> {
    let sound_speed = nd_to_leto3(sound_speed_m_s.as_array().to_owned());
    let transmissions = transmissions.unwrap_or_else(|| array.inner.circumferential_elements());
    let config = PstdFiniteWindowBornConfig {
        reference_sound_speed_m_s,
        spacing_m,
        time_step_s,
        source_amplitude_pa,
        cycles_per_frequency,
        frequency_bin_cycles,
    };
    let pressure = py
        .detach(|| {
            simulate_pstd_finite_window_born_observation(
                &sound_speed,
                &array.inner,
                frequency_hz,
                config,
                transmissions,
            )
        })
        .map_err(kwavers_to_py)?;
    Ok(leto2_to_nd2(pressure).to_pyarray(py).into())
}

#[pyfunction]
#[pyo3(signature = (
    sound_speed_m_s,
    array,
    frequency_hz,
    reference_sound_speed_m_s = 1500.0,
    spacing_m = 1.0e-3,
    time_step_s = 1.0e-7,
    source_amplitude_pa = 1.0e3,
    cycles_per_frequency = 4,
    frequency_bin_cycles = 1,
    transmissions = None
))]
#[allow(clippy::too_many_arguments)]
pub fn simulate_breast_fwi_pstd_finite_window_born_second_order_observation<'py>(
    py: Python<'py>,
    sound_speed_m_s: PyReadonlyArray3<'py, f64>,
    array: &PyMultiRowRingArray,
    frequency_hz: f64,
    reference_sound_speed_m_s: f64,
    spacing_m: f64,
    time_step_s: f64,
    source_amplitude_pa: f64,
    cycles_per_frequency: usize,
    frequency_bin_cycles: usize,
    transmissions: Option<usize>,
) -> PyResult<Py<PyArray2<Complex64>>> {
    let sound_speed = nd_to_leto3(sound_speed_m_s.as_array().to_owned());
    let transmissions = transmissions.unwrap_or_else(|| array.inner.circumferential_elements());
    let config = PstdFiniteWindowBornConfig {
        reference_sound_speed_m_s,
        spacing_m,
        time_step_s,
        source_amplitude_pa,
        cycles_per_frequency,
        frequency_bin_cycles,
    };
    let pressure = py
        .detach(|| {
            simulate_pstd_finite_window_born_second_order_observation(
                &sound_speed,
                &array.inner,
                frequency_hz,
                config,
                transmissions,
            )
        })
        .map_err(kwavers_to_py)?;
    Ok(leto2_to_nd2(pressure).to_pyarray(py).into())
}

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(
        simulate_breast_fwi_pstd_finite_window_born_observation,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        simulate_breast_fwi_pstd_finite_window_born_second_order_observation,
        m
    )?)?;
    Ok(())
}
