use super::contrast_agent_doppler_spectrum;
use super::doppler_frequency_shift;
use super::ContrastAgentDopplerConfig;
use kwavers_core::constants::fundamental::SOUND_SPEED_WATER_SIM;
use kwavers_core::constants::numerical::MHZ_TO_HZ;
use std::f64::consts::PI;

#[test]
fn doppler_towards_transducer() {
    // θ = 0 → maximum shift
    let df = doppler_frequency_shift(1.0, 0.0, MHZ_TO_HZ, SOUND_SPEED_WATER_SIM);
    assert!((df - 2.0 * MHZ_TO_HZ / SOUND_SPEED_WATER_SIM).abs() < 1e-6);
}

#[test]
fn doppler_perpendicular_is_zero() {
    let df = doppler_frequency_shift(1.0, PI / 2.0, MHZ_TO_HZ, SOUND_SPEED_WATER_SIM);
    assert!(df.abs() < 1e-10);
}

#[test]
fn contrast_agent_doppler_spectrum_recovers_velocity() {
    let spectrum = contrast_agent_doppler_spectrum(ContrastAgentDopplerConfig {
        n_ensemble: 128,
        fft_multiplier: 4,
        prf_hz: 10_000.0,
        velocity_m_s: 0.3,
        theta_rad: 60_f64.to_radians(),
        f0_hz: 5.0 * MHZ_TO_HZ,
        sound_speed_m_s: 1540.0,
        amplitude: 0.02,
    })
    .unwrap();

    assert_eq!(spectrum.slow_time_s.len(), 128);
    assert_eq!(spectrum.iq_real.len(), 128);
    assert_eq!(spectrum.iq_imag.len(), 128);
    assert_eq!(spectrum.velocity_m_s.len(), 512);
    assert_eq!(spectrum.power.len(), 512);
    assert!((spectrum.doppler_shift_hz - 974.025_974_025_974).abs() < 1.0e-9);
    assert!((spectrum.estimated_velocity_m_s - 0.3).abs() < 1.0e-12);
    assert!((spectrum.nyquist_velocity_m_s - 1.54).abs() < 1.0e-12);

    let peak_idx = spectrum
        .power
        .iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.total_cmp(b))
        .map(|(idx, _)| idx)
        .unwrap();
    assert!((spectrum.velocity_m_s[peak_idx] - 0.300_781_25).abs() < 1.0e-12);
}

#[test]
fn contrast_agent_doppler_spectrum_rejects_invalid_inputs() {
    assert!(contrast_agent_doppler_spectrum(ContrastAgentDopplerConfig {
        n_ensemble: 1,
        fft_multiplier: 4,
        prf_hz: 10_000.0,
        velocity_m_s: 0.3,
        theta_rad: 0.0,
        f0_hz: 5.0 * MHZ_TO_HZ,
        sound_speed_m_s: 1540.0,
        amplitude: 0.02,
    })
    .unwrap_err()
    .contains("n_ensemble"));
    assert!(contrast_agent_doppler_spectrum(ContrastAgentDopplerConfig {
        n_ensemble: 128,
        fft_multiplier: 4,
        prf_hz: 10_000.0,
        velocity_m_s: 0.3,
        theta_rad: PI / 2.0,
        f0_hz: 5.0 * MHZ_TO_HZ,
        sound_speed_m_s: 1540.0,
        amplitude: 0.02,
    })
    .unwrap_err()
    .contains("cos(theta_rad)"));
}
