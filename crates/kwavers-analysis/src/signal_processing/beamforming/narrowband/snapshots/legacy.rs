//! Legacy analytic-signal complex baseband snapshot extraction.
//!
//! Prefer windowed snapshots via `SnapshotSelection` for MVDR/MUSIC pipelines.
//! The RF-to-I/Q operation itself is owned by
//! [`crate::signal_processing::baseband::demodulate_rf_to_iq`]; this module
//! only selects decimated narrowband snapshots from that canonical record.

use super::config::BasebandSnapshotConfig;
use crate::signal_processing::baseband::{demodulate_rf_to_iq, IqDemodulationConfig};
use eunomia::Complex64;
use kwavers_core::error::{KwaversError, KwaversResult};
use leto::{Array2, Array3};

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

    let [n_sensors, channels, n_samples] = sensor_data.shape();
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

    let rf = Array2::from_shape_vec(
        [n_sensors, n_samples],
        (0..n_sensors)
            .flat_map(|sensor| (0..n_samples).map(move |sample| sensor_data[[sensor, 0, sample]]))
            .collect(),
    )
    .map_err(|error| KwaversError::InvalidInput(format!("legacy baseband RF shape: {error}")))?;
    let baseband = demodulate_rf_to_iq(
        rf.view(),
        IqDemodulationConfig::new(cfg.sampling_frequency_hz, cfg.center_frequency_hz)?,
    )?;
    let mut snapshots =
        Array2::<Complex64>::from_elem((n_sensors, n_snapshots), Complex64::default());
    for sensor in 0..n_sensors {
        for k in 0..n_snapshots {
            let t_idx = k * step;
            snapshots[[sensor, k]] = baseband[[sensor, t_idx]];
        }
    }

    Ok(snapshots)
}

#[cfg(test)]
mod tests {
    use super::super::config::BasebandSnapshotConfig;
    use super::extract_complex_baseband_snapshots;
    use leto::Array3;

    #[test]
    fn snapshot_extraction_shapes_match() {
        let n_sensors = 3usize;
        let n_samples = 128usize;
        let mut data = Array3::<f64>::zeros((n_sensors, 1, n_samples));
        for s in 0..n_sensors {
            for t in 0..n_samples {
                data[[s, 0, t]] = (s as f64) + (t as f64) * 1e-3;
            }
        }

        let cfg = BasebandSnapshotConfig {
            sampling_frequency_hz: 1_000.0,
            center_frequency_hz: 100.0,
            snapshot_step_samples: 4,
        };

        let snaps = extract_complex_baseband_snapshots(&data, &cfg).expect("snaps");
        assert_eq!(snaps.shape()[0], n_sensors);
        assert!(snaps.shape()[1] > 0);
    }

    #[test]
    fn rejects_invalid_shape() {
        let data = Array3::<f64>::zeros((2, 2, 16));
        let cfg = BasebandSnapshotConfig::default();
        let err = extract_complex_baseband_snapshots(&data, &cfg).expect_err("must fail");
        assert!(err.to_string().contains("channels"));
    }
}
