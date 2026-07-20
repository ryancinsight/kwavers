//! PyO3 wrappers for Doppler and vector-flow imaging helpers.

use kwavers_analysis::signal_processing::doppler::{
    continuous_wave_vector_flow_fixture as core_continuous_wave_vector_flow_fixture, VectorVelocity,
};
use kwavers_physics::analytical::imaging::{self, ContrastAgentDopplerConfig};
use numpy::ndarray::Array2;
use numpy::{PyArray1, PyReadonlyArray1, ToPyArray};
use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::PyDict;

/// Compute the Doppler frequency shift.
///
/// f_d = 2 * f0 * v * cos(theta) / c
///
/// Args:
///     v_m_s: Scatterer velocity [m/s].
///     theta_rad: Angle between beam and velocity vector [rad].
///     f0_hz: Transmit centre frequency [Hz].
///     c: Sound speed [m/s].
///
/// Returns:
///     Doppler shift [Hz].
#[pyfunction]
#[pyo3(signature = (v_m_s, theta_rad, f0_hz, c))]
pub fn doppler_frequency_shift(v_m_s: f64, theta_rad: f64, f0_hz: f64, c: f64) -> PyResult<f64> {
    Ok(imaging::doppler_frequency_shift(v_m_s, theta_rad, f0_hz, c))
}

/// Compute contrast-agent Doppler IQ, spectrum, and Kasai estimate.
///
/// Returns a dictionary with slow-time IQ arrays, shifted velocity axis,
/// shifted spectrum power, true Doppler shift, Kasai-estimated Doppler shift,
/// estimated velocity, and pulsed-wave Nyquist velocity.
#[pyfunction]
#[pyo3(signature = (
    n_ensemble,
    fft_multiplier,
    prf_hz,
    velocity_m_s,
    theta_rad,
    f0_hz,
    sound_speed_m_s,
    amplitude
))]
#[allow(clippy::too_many_arguments)]
pub fn contrast_agent_doppler_spectrum<'py>(
    py: Python<'py>,
    n_ensemble: usize,
    fft_multiplier: usize,
    prf_hz: f64,
    velocity_m_s: f64,
    theta_rad: f64,
    f0_hz: f64,
    sound_speed_m_s: f64,
    amplitude: f64,
) -> PyResult<Bound<'py, PyDict>> {
    let spectrum = imaging::contrast_agent_doppler_spectrum(ContrastAgentDopplerConfig {
        n_ensemble,
        fft_multiplier,
        prf_hz,
        velocity_m_s,
        theta_rad,
        f0_hz,
        sound_speed_m_s,
        amplitude,
    })
    .map_err(PyValueError::new_err)?;

    let out = PyDict::new(py);
    out.set_item("slow_time_s", spectrum.slow_time_s.to_pyarray(py))?;
    out.set_item("iq_real", spectrum.iq_real.to_pyarray(py))?;
    out.set_item("iq_imag", spectrum.iq_imag.to_pyarray(py))?;
    out.set_item("velocity_m_s", spectrum.velocity_m_s.to_pyarray(py))?;
    out.set_item("power", spectrum.power.to_pyarray(py))?;
    out.set_item("doppler_shift_hz", spectrum.doppler_shift_hz)?;
    out.set_item("estimated_shift_hz", spectrum.estimated_shift_hz)?;
    out.set_item("estimated_velocity_m_s", spectrum.estimated_velocity_m_s)?;
    out.set_item("nyquist_velocity_m_s", spectrum.nyquist_velocity_m_s)?;
    Ok(out)
}

/// Compute the Chapter 5 continuous-wave and vector-flow Doppler fixture.
///
/// Returns CW spectrum arrays, pulsed-wave Nyquist velocity, beam geometry,
/// beam-projected velocities, and the recovered 2-D velocity vector.
#[pyfunction]
#[pyo3(signature = (
    center_frequency_hz=2.5e6,
    sampling_rate_hz=20e6,
    baseband_rate_hz=100e3,
    jet_velocity_m_s=2.0,
    pulsed_prf_hz=5e3,
    true_velocity_x_m_s=0.35,
    true_velocity_z_m_s=0.55,
    beam_angles_rad=None,
    sound_speed_m_s=1540.0,
    n_baseband_bins=2048
))]
#[allow(clippy::too_many_arguments)]
pub fn continuous_wave_vector_flow_fixture<'py>(
    py: Python<'py>,
    center_frequency_hz: f64,
    sampling_rate_hz: f64,
    baseband_rate_hz: f64,
    jet_velocity_m_s: f64,
    pulsed_prf_hz: f64,
    true_velocity_x_m_s: f64,
    true_velocity_z_m_s: f64,
    beam_angles_rad: Option<PyReadonlyArray1<'_, f64>>,
    sound_speed_m_s: f64,
    n_baseband_bins: usize,
) -> PyResult<Bound<'py, PyDict>> {
    let angles: Vec<f64> = if let Some(angles) = beam_angles_rad {
        angles
            .as_slice()
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))?
            .to_vec()
    } else {
        vec![25.0_f64.to_radians(), (-25.0_f64).to_radians()]
    };

    let fixture = py
        .detach(|| {
            core_continuous_wave_vector_flow_fixture(
                center_frequency_hz,
                sampling_rate_hz,
                baseband_rate_hz,
                jet_velocity_m_s,
                pulsed_prf_hz,
                sound_speed_m_s,
                n_baseband_bins,
                VectorVelocity {
                    vx: true_velocity_x_m_s,
                    vz: true_velocity_z_m_s,
                },
                &angles,
            )
        })
        .map_err(|e| PyValueError::new_err(e.to_string()))?;

    let beam_direction_flat: Vec<f64> = fixture
        .beam_directions
        .iter()
        .flat_map(|direction| direction.iter().copied())
        .collect();
    let cw_velocity_m_s = PyArray1::from_iter(py, fixture.cw_velocity_m_s.iter().copied());
    let cw_power = PyArray1::from_iter(py, fixture.cw_power.iter().copied());
    let beam_angles_rad = PyArray1::from_vec(py, fixture.beam_angles_rad);
    let projected_velocity_m_s = PyArray1::from_vec(py, fixture.projected_velocity_m_s);
    let out = PyDict::new(py);
    out.set_item("cw_velocity_m_s", cw_velocity_m_s)?;
    out.set_item("cw_power", cw_power)?;
    out.set_item(
        "pulsed_wave_nyquist_velocity_m_s",
        fixture.pulsed_wave_nyquist_velocity_m_s,
    )?;
    out.set_item("beam_angles_rad", beam_angles_rad)?;
    out.set_item(
        "beam_directions",
        Array2::from_shape_vec((fixture.beam_directions.len(), 2), beam_direction_flat)
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))?
            .to_pyarray(py),
    )?;
    out.set_item(
        "projected_velocity_m_s",
        projected_velocity_m_s,
    )?;
    out.set_item(
        "true_velocity_m_s",
        [fixture.true_velocity_m_s.vx, fixture.true_velocity_m_s.vz].to_pyarray(py),
    )?;
    out.set_item(
        "recovered_velocity_m_s",
        [
            fixture.recovered_velocity_m_s.vx,
            fixture.recovered_velocity_m_s.vz,
        ]
        .to_pyarray(py),
    )?;
    out.set_item("vector_error_m_s", fixture.vector_error_m_s)?;
    Ok(out)
}
