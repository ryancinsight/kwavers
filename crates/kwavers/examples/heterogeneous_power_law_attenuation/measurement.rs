use std::f64::consts::TAU;

use anyhow::{anyhow, Result};
use leto::Array3;

use super::configuration::{
    ANALYSIS_FREQUENCIES, C0, DX, GATE_HALF_STEPS, N, PULSE_WIDTH_S, SENSOR_FAR, SENSOR_NEAR,
    SOURCE_INDEX,
};
use super::propagation::run_pulse;

/// Single-frequency DFT magnitude of one direct arrival.
///
/// The rectangular gate contains the dispersively broadened pulse without
/// weighting the near and far arrivals differently. A Hann taper biases the
/// recovered attenuation low by 8–19 percent (KW-SOL-072).
fn windowed_magnitude(trace: &[f64], sensor_index: usize, frequency_hz: f64, dt: f64) -> f64 {
    let emission = (3.0 * PULSE_WIDTH_S / dt).round() as usize;
    let arrival = emission + ((sensor_index - SOURCE_INDEX) as f64 * DX / C0 / dt).round() as usize;
    let lo = arrival.saturating_sub(GATE_HALF_STEPS);
    let hi = (arrival + GATE_HALF_STEPS).min(trace.len());

    let (mut re, mut im) = (0.0_f64, 0.0_f64);
    for (offset, &value) in trace[lo..hi].iter().enumerate() {
        let phase = TAU * frequency_hz * (lo + offset) as f64 * dt;
        re += value * phase.cos();
        im -= value * phase.sin();
    }
    re.hypot(im)
}

pub(crate) fn sensor_spectra(near: &[f64], far: &[f64], dt: f64) -> Result<Vec<(f64, f64)>> {
    ANALYSIS_FREQUENCIES
        .iter()
        .map(|&frequency_hz| {
            let near_amplitude = windowed_magnitude(near, SENSOR_NEAR, frequency_hz, dt);
            let far_amplitude = windowed_magnitude(far, SENSOR_FAR, frequency_hz, dt);
            if near_amplitude <= 0.0 || far_amplitude <= 0.0 {
                return Err(anyhow!("no spectral energy at {frequency_hz:e} Hz"));
            }
            Ok((near_amplitude, far_amplitude))
        })
        .collect()
}

/// Recover attenuation by a lossless-reference-normalized sensor ratio.
pub(crate) fn measure_attenuation(
    run: &[(f64, f64)],
    reference: &[(f64, f64)],
) -> Result<Vec<(f64, f64)>> {
    let separation = (SENSOR_FAR - SENSOR_NEAR) as f64 * DX;
    ANALYSIS_FREQUENCIES
        .iter()
        .zip(run)
        .zip(reference)
        .map(|((&frequency_hz, &(near, far)), &(near_ref, far_ref))| {
            let ratio = (far / near) / (far_ref / near_ref);
            if !ratio.is_finite() || ratio <= 0.0 {
                return Err(anyhow!("degenerate spectral ratio at {frequency_hz:e} Hz"));
            }
            Ok((frequency_hz, -ratio.ln() / separation))
        })
        .collect()
}

/// Lossless run of the identical geometry.
pub(crate) fn reference_spectra(dt: f64) -> Result<Vec<(f64, f64)>> {
    let alpha = Array3::<f64>::zeros([N, 1, 1]);
    let gamma = Array3::<f64>::from_elem([N, 1, 1], 1.0);
    let (near, far) = run_pulse(&alpha, &gamma, dt)?;
    sensor_spectra(&near, &far, dt)
}
