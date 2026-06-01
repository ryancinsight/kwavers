//! Legacy analytic-signal complex baseband snapshot extraction.
//!
//! Prefer windowed snapshots via `SnapshotSelection` for MVDR/MUSIC pipelines.

use super::config::BasebandSnapshotConfig;
use crate::core::constants::numerical::TWO_PI;
use crate::core::error::{KwaversError, KwaversResult};
use crate::math::fft::{fft_1d_array, ifft_1d_complex, Complex64};
use ndarray::{Array2, Array3};

/// Convert a real-valued time series into its analytic signal via Hilbert transform.
///
/// Returns `z[n] = x[n] + j * H{x}[n]`.
///
/// # Errors
/// - Rejects empty input.
///
/// # Implementation
/// FFT-based Hilbert transform:
/// - Forward FFT of real input (embedded as complex)
/// - Multiply FFT bins by the "analytic signal" multiplier:
///   - DC: keep (×1)
///   - Nyquist (even N): keep (×1)
///   - positive frequencies: ×2
///   - negative frequencies: ×0
/// - Inverse FFT and scale by 1/N (Apollo's transforms are unnormalized)
fn analytic_signal_hilbert(signal: &[f64]) -> KwaversResult<Vec<Complex64>> {
    let n = signal.len();
    if n == 0 {
        return Err(KwaversError::InvalidInput(
            "analytic_signal_hilbert: signal must be non-empty".to_owned(),
        ));
    }

    let mut spectrum = fft_1d_array(&ndarray::Array1::from_vec(signal.to_vec()));

    // Apply analytic-signal (Hilbert/one-sided) multiplier.
    //
    // Indexing conventions:
    // - For N even: positive bins are k=1..(N/2-1), Nyquist is k=N/2, negatives are k=(N/2+1)..(N-1).
    // - For N odd:  positive bins are k=1..((N-1)/2), negatives are k=((N+1)/2)..(N-1).
    if n >= 2 {
        let half = n / 2;

        if n.is_multiple_of(2) {
            for v in spectrum.iter_mut().take(half).skip(1) {
                *v *= 2.0;
            }
            for v in spectrum.iter_mut().skip(half + 1) {
                *v = Complex64::new(0.0, 0.0);
            }
        } else {
            for v in spectrum.iter_mut().take(half + 1).skip(1) {
                *v *= 2.0;
            }
            for v in spectrum.iter_mut().skip(half + 1) {
                *v = Complex64::new(0.0, 0.0);
            }
        }
    }

    Ok(ifft_1d_complex(&spectrum).to_vec())
}

/// Downconvert an analytic (complex) signal to complex baseband with center frequency `f0`.
///
/// Computes: `y[n] = z[n] * exp(-j 2π f0 (n / fs))`.
///
/// # Errors
/// - Rejects invalid frequencies.
fn downconvert_to_baseband(
    analytic: &[Complex64],
    sampling_frequency_hz: f64,
    center_frequency_hz: f64,
) -> KwaversResult<Vec<Complex64>> {
    if analytic.is_empty() {
        return Err(KwaversError::InvalidInput(
            "downconvert_to_baseband: analytic must be non-empty".to_owned(),
        ));
    }
    if !sampling_frequency_hz.is_finite() || sampling_frequency_hz <= 0.0 {
        return Err(KwaversError::InvalidInput(
            "downconvert_to_baseband: sampling_frequency_hz must be finite and > 0".to_owned(),
        ));
    }
    if !center_frequency_hz.is_finite() || center_frequency_hz <= 0.0 {
        return Err(KwaversError::InvalidInput(
            "downconvert_to_baseband: center_frequency_hz must be finite and > 0".to_owned(),
        ));
    }

    let n = analytic.len();
    let omega = -TWO_PI * center_frequency_hz;
    let inv_fs = 1.0 / sampling_frequency_hz;

    let mut out = Vec::with_capacity(n);
    for (idx, &z) in analytic.iter().enumerate() {
        let t = (idx as f64) * inv_fs;
        let rot = Complex64::new(0.0, omega * t).exp();
        out.push(z * rot);
    }

    Ok(out)
}

/// Extract **legacy** complex baseband snapshots from sensor time series.
///
/// This uses analytic signal (Hilbert) + downconversion and then samples the complex baseband at
/// stride `snapshot_step_samples`.
///
/// # Recommendation
/// For advanced MVDR/MUSIC, prefer windowed snapshots (`extract_narrowband_snapshots` with
/// `SnapshotSelection::{Auto,Explicit}`), which aligns with standard STFT-bin snapshot models.
///
/// # Parameters
/// - `sensor_data`: real-valued array shaped `(n_sensors, 1, n_samples)`.
/// - `cfg`: baseband and snapshot config.
///
/// # Returns
/// An `Array2<Complex64>` of shape `(n_sensors, n_snapshots)` where each column is a snapshot
/// vector `x_k` across sensors. Snapshots are taken at indices:
///
/// `t_k = k * cfg.snapshot_step_samples`, for `k = 0..n_snapshots`.
///
/// # Errors
/// - Invalid shapes (channels != 1, n_sensors == 0, n_samples == 0)
/// - Invalid config
pub fn extract_complex_baseband_snapshots(
    sensor_data: &Array3<f64>,
    cfg: &BasebandSnapshotConfig,
) -> KwaversResult<Array2<Complex64>> {
    cfg.validate()?;

    let (n_sensors, channels, n_samples) = sensor_data.dim();
    if channels != 1 {
        return Err(KwaversError::InvalidInput(format!(
            "extract_complex_baseband_snapshots expects sensor_data shape (n_sensors, 1, n_samples); got channels={channels}"
        )));
    }
    if n_sensors == 0 || n_samples == 0 {
        return Err(KwaversError::InvalidInput(
            "extract_complex_baseband_snapshots requires n_sensors > 0 and n_samples > 0"
                .to_owned(),
        ));
    }

    let step = cfg.snapshot_step_samples;
    if step == 0 {
        return Err(KwaversError::InvalidInput(
            "extract_complex_baseband_snapshots: snapshot_step_samples must be >= 1".to_owned(),
        ));
    }

    let n_snapshots = ((n_samples - 1) / step) + 1;

    let mut snapshots = Array2::<Complex64>::zeros((n_sensors, n_snapshots));

    for s in 0..n_sensors {
        let mut x = Vec::with_capacity(n_samples);
        for t in 0..n_samples {
            x.push(sensor_data[(s, 0, t)]);
        }

        let analytic = analytic_signal_hilbert(&x)?;
        let baseband = downconvert_to_baseband(
            &analytic,
            cfg.sampling_frequency_hz,
            cfg.center_frequency_hz,
        )?;

        for k in 0..n_snapshots {
            let t_idx = k * step;
            snapshots[(s, k)] = baseband[t_idx];
        }
    }

    Ok(snapshots)
}

#[cfg(test)]
mod tests {
    use super::super::config::BasebandSnapshotConfig;
    use super::{
        analytic_signal_hilbert, downconvert_to_baseband, extract_complex_baseband_snapshots,
    };
    use crate::math::fft::Complex64;
    use approx::assert_abs_diff_eq;
    use ndarray::Array3;

    #[test]
    fn analytic_signal_of_cos_has_unit_magnitude_envelope_for_tone() {
        let fs = 1024.0;
        let f = 64.0;
        let n = 1024usize;

        let omega = 2.0 * std::f64::consts::PI * f;

        let mut x = Vec::with_capacity(n);
        for t in 0..n {
            let ts = (t as f64) / fs;
            x.push((omega * ts).cos());
        }

        let z = analytic_signal_hilbert(&x).expect("analytic");
        for v in z.iter().take(64) {
            assert_abs_diff_eq!(v.norm(), 1.0, epsilon = 5e-2);
        }
    }

    #[test]
    fn downconversion_moves_tone_to_dc() {
        let fs = 1024.0;
        let f0 = 64.0;
        let n = 1024usize;

        let omega = 2.0 * std::f64::consts::PI * f0;

        let mut x = Vec::with_capacity(n);
        for t in 0..n {
            let ts = (t as f64) / fs;
            x.push((omega * ts).cos());
        }

        let z = analytic_signal_hilbert(&x).expect("analytic");
        let bb = downconvert_to_baseband(&z, fs, f0).expect("bb");

        let mean = bb.iter().fold(Complex64::new(0.0, 0.0), |acc, &v| acc + v) * (1.0 / n as f64);
        assert!(mean.norm() > 0.5);
    }

    #[test]
    fn snapshot_extraction_shapes_match() {
        let n_sensors = 3usize;
        let n_samples = 128usize;
        let mut data = Array3::<f64>::zeros((n_sensors, 1, n_samples));
        for s in 0..n_sensors {
            for t in 0..n_samples {
                data[(s, 0, t)] = (s as f64) + (t as f64) * 1e-3;
            }
        }

        let cfg = BasebandSnapshotConfig {
            sampling_frequency_hz: 1_000.0,
            center_frequency_hz: 100.0,
            snapshot_step_samples: 4,
        };

        let snaps = extract_complex_baseband_snapshots(&data, &cfg).expect("snaps");
        assert_eq!(snaps.nrows(), n_sensors);
        assert!(snaps.ncols() > 0);
    }

    #[test]
    fn rejects_invalid_shape() {
        let data = Array3::<f64>::zeros((2, 2, 16));
        let cfg = BasebandSnapshotConfig::default();
        let err = extract_complex_baseband_snapshots(&data, &cfg).expect_err("must fail");
        assert!(err.to_string().contains("channels"));
    }
}
