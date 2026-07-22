//! Deterministic Doppler book fixtures built from production Doppler kernels.

use super::continuous_wave::{ContinuousWaveDoppler, CwDopplerConfig};
use super::vector_flow::{VectorFlowEstimator, VectorVelocity};
use kwavers_core::constants::numerical::TWO_PI;
use kwavers_core::error::{KwaversError, KwaversResult};
use leto::Array1;

/// Continuous-wave spectrum and vector-flow recovery fixture.
#[derive(Debug, Clone)]
pub struct ContinuousWaveVectorFlowFixture {
    /// Signed CW velocity axis [m/s].
    pub cw_velocity_m_s: Array1<f64>,
    /// CW spectral power.
    pub cw_power: Array1<f64>,
    /// Pulsed-wave Nyquist velocity for the comparison PRF [m/s].
    pub pulsed_wave_nyquist_velocity_m_s: f64,
    /// Beam angles from the axial axis `rad`.
    pub beam_angles_rad: Vec<f64>,
    /// Normalized beam directions `(d_x, d_z)`.
    pub beam_directions: Vec<[f64; 2]>,
    /// Doppler-measured velocity projections along each beam [m/s].
    pub projected_velocity_m_s: Vec<f64>,
    /// True 2-D velocity vector used by the fixture [m/s].
    pub true_velocity_m_s: VectorVelocity,
    /// Recovered 2-D velocity vector [m/s].
    pub recovered_velocity_m_s: VectorVelocity,
    /// Euclidean recovery error [m/s].
    pub vector_error_m_s: f64,
}

/// Build the Chapter 5 CW/vector-flow fixture from Rust Doppler primitives.
///
/// The fixture synthesizes a real CW received tone for the high-velocity jet,
/// computes the two-sided CW spectrum with [`ContinuousWaveDoppler`], and
/// recovers a 2-D velocity vector with [`VectorFlowEstimator`].
///
/// # Errors
/// Returns `KwaversError::InvalidInput` for non-finite or nonpositive
/// frequencies/rates/sound speed, zero baseband bins, non-finite velocities, or
/// invalid/collinear beam angles.
#[allow(clippy::too_many_arguments)]
pub fn continuous_wave_vector_flow_fixture(
    center_frequency_hz: f64,
    sampling_rate_hz: f64,
    baseband_rate_hz: f64,
    jet_velocity_m_s: f64,
    pulsed_prf_hz: f64,
    sound_speed_m_s: f64,
    n_baseband_bins: usize,
    true_velocity_m_s: VectorVelocity,
    beam_angles_rad: &[f64],
) -> KwaversResult<ContinuousWaveVectorFlowFixture> {
    validate_positive_finite("center_frequency_hz", center_frequency_hz)?;
    validate_positive_finite("sampling_rate_hz", sampling_rate_hz)?;
    validate_positive_finite("baseband_rate_hz", baseband_rate_hz)?;
    validate_positive_finite("pulsed_prf_hz", pulsed_prf_hz)?;
    validate_positive_finite("sound_speed_m_s", sound_speed_m_s)?;
    if n_baseband_bins == 0 {
        return Err(KwaversError::InvalidInput(
            "n_baseband_bins must be greater than zero".to_owned(),
        ));
    }
    if !jet_velocity_m_s.is_finite()
        || !true_velocity_m_s.vx.is_finite()
        || !true_velocity_m_s.vz.is_finite()
    {
        return Err(KwaversError::InvalidInput(
            "fixture velocities must be finite".to_owned(),
        ));
    }
    if beam_angles_rad.iter().any(|angle| !angle.is_finite()) {
        return Err(KwaversError::InvalidInput(
            "beam angles must be finite".to_owned(),
        ));
    }

    let mut config = CwDopplerConfig::new(center_frequency_hz, sampling_rate_hz, 0.0);
    config.baseband_rate = baseband_rate_hz;
    config.sound_speed = sound_speed_m_s;
    let decimation = config.decimation();
    let n_samples = decimation
        .checked_mul(n_baseband_bins)
        .ok_or_else(|| KwaversError::InvalidInput("RF sample count overflows usize".to_owned()))?;
    let doppler_shift_hz = 2.0 * jet_velocity_m_s * center_frequency_hz / sound_speed_m_s;
    let received_frequency_hz = center_frequency_hz + doppler_shift_hz;
    let rf: Vec<f64> = (0..n_samples)
        .map(|sample| {
            let t_s = sample as f64 / sampling_rate_hz;
            (TWO_PI * received_frequency_hz * t_s).cos()
        })
        .collect();
    let spectrum = ContinuousWaveDoppler::new(config).spectrum(&rf)?;

    let beam_directions: Vec<[f64; 2]> = beam_angles_rad
        .iter()
        .map(|angle| [angle.sin(), angle.cos()])
        .collect();
    let projected_velocity_m_s: Vec<f64> = beam_directions
        .iter()
        .map(|direction| true_velocity_m_s.vx * direction[0] + true_velocity_m_s.vz * direction[1])
        .collect();
    let recovered_velocity_m_s =
        VectorFlowEstimator::new(&beam_directions)?.estimate(&projected_velocity_m_s)?;
    let vector_error_m_s = (recovered_velocity_m_s.vx - true_velocity_m_s.vx)
        .hypot(recovered_velocity_m_s.vz - true_velocity_m_s.vz);

    Ok(ContinuousWaveVectorFlowFixture {
        cw_velocity_m_s: spectrum.velocity,
        cw_power: spectrum.power,
        pulsed_wave_nyquist_velocity_m_s: pulsed_prf_hz * sound_speed_m_s
            / (4.0 * center_frequency_hz),
        beam_angles_rad: beam_angles_rad.to_vec(),
        beam_directions,
        projected_velocity_m_s,
        true_velocity_m_s,
        recovered_velocity_m_s,
        vector_error_m_s,
    })
}

fn validate_positive_finite(name: &str, value: f64) -> KwaversResult<()> {
    if value.is_finite() && value > 0.0 {
        return Ok(());
    }
    Err(KwaversError::InvalidInput(format!(
        "{name} must be finite and greater than zero"
    )))
}
