use super::types::{SnapshotMethod, SnapshotSelection, StftBinConfig};
use crate::core::error::{KwaversError, KwaversResult};
use crate::math::fft::fft_1d_array;
use ndarray::{Array2, Array3};
use num_complex::Complex64;

/// Extract complex snapshots according to a selection policy.
///
/// # Inputs
/// - `sensor_data` shape: `(n_sensors, 1, n_samples)`
/// - `selection`: explicit or auto selection
///
/// # Output
/// `(n_sensors, n_snapshots)` complex snapshots.
pub fn extract_windowed_snapshots(
    sensor_data: &Array3<f64>,
    selection: &SnapshotSelection,
) -> KwaversResult<Array2<Complex64>> {
    let (n_sensors, channels, n_samples) = sensor_data.dim();
    if channels != 1 {
        return Err(KwaversError::InvalidInput(format!(
            "extract_windowed_snapshots expects sensor_data shape (n_sensors, 1, n_samples); got channels={channels}"
        )));
    }
    if n_sensors == 0 || n_samples == 0 {
        return Err(KwaversError::InvalidInput(
            "extract_windowed_snapshots requires n_sensors > 0 and n_samples > 0".to_string(),
        ));
    }

    let method = selection.resolve(n_samples)?;
    match method {
        SnapshotMethod::StftBin(cfg) => extract_stft_bin_snapshots(sensor_data, &cfg),
    }
}

/// Extract STFT-bin snapshots at `cfg.frequency_hz`.
///
/// Each frame produces one complex snapshot across sensors.
pub fn extract_stft_bin_snapshots(
    sensor_data: &Array3<f64>,
    cfg: &StftBinConfig,
) -> KwaversResult<Array2<Complex64>> {
    cfg.validate()?;

    let (n_sensors, channels, n_samples) = sensor_data.dim();
    if channels != 1 {
        return Err(KwaversError::InvalidInput(format!(
            "extract_stft_bin_snapshots expects sensor_data shape (n_sensors, 1, n_samples); got channels={channels}"
        )));
    }
    if n_sensors == 0 || n_samples == 0 {
        return Err(KwaversError::InvalidInput(
            "extract_stft_bin_snapshots requires n_sensors > 0 and n_samples > 0".to_string(),
        ));
    }
    if cfg.frame_len_samples > n_samples {
        return Err(KwaversError::InvalidInput(format!(
            "extract_stft_bin_snapshots: frame_len_samples ({}) must be <= n_samples ({n_samples})",
            cfg.frame_len_samples
        )));
    }

    let n = cfg.frame_len_samples;
    let hop = cfg.hop_len_samples;
    let bin = cfg.bin_index();

    let n_frames = ((n_samples - n) / hop) + 1;

    let window = cfg.window.build(n);

    let mut out = Array2::<Complex64>::zeros((n_sensors, n_frames));
    let mut frame = ndarray::Array1::<f64>::zeros(n);

    for s in 0..n_sensors {
        for frame_idx in 0..n_frames {
            let start = frame_idx * hop;
            let end = start + n;

            debug_assert!(end <= n_samples);

            let mean = if cfg.remove_mean {
                let mut acc = 0.0;
                for t in start..end {
                    acc += sensor_data[(s, 0, t)];
                }
                acc / (n as f64)
            } else {
                0.0
            };

            for i in 0..n {
                let x = sensor_data[(s, 0, start + i)] - mean;
                frame[i] = x * window[i];
            }

            let spectrum = fft_1d_array(&frame);
            out[(s, frame_idx)] = spectrum[bin];
        }
    }

    Ok(out)
}
