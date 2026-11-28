use super::KUJI_PROBE_BAND_HZ;
use apollo::{fft_1d_slice_half, ifft_1d_slice_half_into};
use kwavers_core::error::{KwaversError, KwaversResult};

/// High-pass realisation for the Kuji filter. The paper specifies
/// only passband edge, −60 dB stopband and 0.15·fc transition
/// (§2.7); the realisation must be declared with any result.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum KujiFilterShape {
    /// Zero-phase raised-cosine ramp: 1e-3 (−60 dB) below 0.85·fc,
    /// raised-cosine over the transition, 1 at/above fc (D8/B1).
    KujiRaisedCosine,
    /// Brick-wall at fc (sensitivity only; not the paper's shape).
    BrickWall,
}

/// Kuji high-pass amplitude response at frequency `f_hz` for
/// passband edge `fc_hz`.
#[must_use]
pub fn kuji_hp_response(f_hz: f64, fc_hz: f64, shape: KujiFilterShape) -> f64 {
    if fc_hz <= 0.0 {
        return 1.0;
    }
    match shape {
        KujiFilterShape::BrickWall => {
            if f_hz >= fc_hz {
                1.0
            } else {
                0.0
            }
        }
        KujiFilterShape::KujiRaisedCosine => {
            let lo = 0.85 * fc_hz;
            if f_hz < lo {
                1e-3
            } else if f_hz >= fc_hz {
                1.0
            } else {
                let u = (f_hz - lo) / (0.15 * fc_hz);
                1e-3 + (1.0 - 1e-3) * 0.5 * (1.0 - (std::f64::consts::PI * u).cos())
            }
        }
    }
}

/// Zero-phase FFT filtering of one trace with the Kuji high-pass
/// (and, when `probe_band` is set, the declared probe band).
///
/// # Errors
/// [`KwaversError::InvalidInput`] on empty input or non-positive dt.
pub fn kuji_filter_trace(
    signal: &[f64],
    dt_s: f64,
    fc_hz: f64,
    shape: KujiFilterShape,
    probe_band: bool,
) -> KwaversResult<Vec<f64>> {
    let n = signal.len();
    if n == 0 || !dt_s.is_finite() || dt_s <= 0.0 {
        return Err(KwaversError::InvalidInput(
            "kuji_metric: trace must be non-empty and dt > 0".to_owned(),
        ));
    }
    // Half-spectrum FFT (n/2+1 bins in standard order). The full
    // apollo FFT uses a packed split layout for real input, which
    // is not the DFT bin layout this response is defined on.
    let mut spectrum = fft_1d_slice_half(signal);
    let df = 1.0 / (n as f64 * dt_s);
    for (i, val) in spectrum.iter_mut().enumerate() {
        let f = i as f64 * df;
        let mut resp = kuji_hp_response(f, fc_hz, shape);
        if probe_band && (f < KUJI_PROBE_BAND_HZ.0 || f > KUJI_PROBE_BAND_HZ.1) {
            resp = 0.0;
        }
        *val *= resp;
    }
    let mut out = vec![0.0_f64; n];
    ifft_1d_slice_half_into(&mut spectrum, &mut out);
    Ok(out)
}
