//! PyO3 conversion surface for finite-window PSTD Born prediction.

use super::PyMultiRowRingArray;
use super::helpers::kwavers_to_py;
use kwavers::solver::inverse::fwi::frequency_domain::{
    simulate_pstd_finite_window_born_observation, PstdFiniteWindowBornConfig,
};
use num_complex::Complex64;
use numpy::{IntoPyArray, PyArray2, PyReadonlyArray3};
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
    let sound_speed = sound_speed_m_s.as_array().to_owned();
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
    Ok(pressure.into_pyarray(py).into())
}

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(
        simulate_breast_fwi_pstd_finite_window_born_observation,
        m
    )?)?;
    Ok(())
}
