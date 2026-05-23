use super::extraction::{extract_stft_bin_snapshots, extract_windowed_snapshots};
use super::types::{
    SnapshotMethod, SnapshotScenario, SnapshotSelection, StftBinConfig, WindowFunction,
};
use crate::core::constants::fundamental::SOUND_SPEED_WATER_SIM;
use approx::assert_relative_eq;
use ndarray::Array3;

fn sensor_positions_m() -> Vec<[f64; 3]> {
    vec![
        [-0.015, 0.0, 0.0],
        [-0.005, 0.0, 0.0],
        [0.005, 0.0, 0.0],
        [0.015, 0.0, 0.0],
    ]
}

fn euclidean_distance_m(a: [f64; 3], b: [f64; 3]) -> f64 {
    let dx = a[0] - b[0];
    let dy = a[1] - b[1];
    let dz = a[2] - b[2];
    (dx * dx + dy * dy + dz * dz).sqrt()
}

fn tof_s(sensor_pos: [f64; 3], source_pos: [f64; 3], sound_speed: f64) -> f64 {
    euclidean_distance_m(sensor_pos, source_pos) / sound_speed
}

fn synth_narrowband_sensor_data(
    sensor_positions: &[[f64; 3]],
    true_source: [f64; 3],
    sound_speed: f64,
    frequency_hz: f64,
    sampling_frequency_hz: f64,
    n_samples: usize,
) -> Array3<f64> {
    let n_sensors = sensor_positions.len();
    let mut data = Array3::<f64>::zeros((n_sensors, 1, n_samples));
    let omega = 2.0 * std::f64::consts::PI * frequency_hz;

    for (i, &pos) in sensor_positions.iter().enumerate() {
        let tau = tof_s(pos, true_source, sound_speed);
        for t in 0..n_samples {
            let time_s = (t as f64) / sampling_frequency_hz;
            data[(i, 0, t)] = (omega * (time_s - tau)).cos();
        }
    }

    data
}

#[test]
fn stft_bin_snapshots_shape_is_correct() {
    let sound_speed = SOUND_SPEED_WATER_SIM;
    let sampling_frequency_hz = 2_000_000.0;
    let frequency_hz = 200_000.0;
    let n_samples = 4096;

    let sensors = sensor_positions_m();
    let true_source = [0.0, 0.01, 0.02];

    let data = synth_narrowband_sensor_data(
        &sensors,
        true_source,
        sound_speed,
        frequency_hz,
        sampling_frequency_hz,
        n_samples,
    );

    let cfg = StftBinConfig {
        sampling_frequency_hz,
        frequency_hz,
        frame_len_samples: 512,
        hop_len_samples: 128,
        window: WindowFunction::Hann,
        remove_mean: true,
    };

    let snaps = extract_stft_bin_snapshots(&data, &cfg).expect("snapshots");
    assert_eq!(snaps.nrows(), sensors.len());
    assert!(snaps.ncols() > 0);
}

#[test]
fn auto_selection_is_deterministic_and_valid() {
    let n_samples = 4096;
    let sel = SnapshotSelection::Auto(SnapshotScenario {
        frequency_hz: 200_000.0,
        sampling_frequency_hz: 2_000_000.0,
        fractional_bandwidth: Some(0.05),
        prefer_robustness: true,
        prefer_time_resolution: false,
    });

    let m1 = sel.resolve(n_samples).expect("method");
    let m2 = sel.resolve(n_samples).expect("method");
    assert_eq!(m1, m2);

    match m1 {
        SnapshotMethod::StftBin(cfg) => {
            cfg.validate().expect("cfg valid");
            assert!(cfg.frame_len_samples <= n_samples);
        }
    }
}

#[test]
fn stft_bin_picks_correct_bin_for_exact_tone() {
    let fs = 1024.0;
    let n = 256usize;
    let k = 16usize;
    let f = (k as f64) * fs / (n as f64);

    let n_samples = 1024usize;
    let n_sensors = 2usize;
    let mut data = Array3::<f64>::zeros((n_sensors, 1, n_samples));
    for s in 0..n_sensors {
        for t in 0..n_samples {
            let ts = (t as f64) / fs;
            data[(s, 0, t)] = (2.0 * std::f64::consts::PI * f * ts).cos();
        }
    }

    let cfg = StftBinConfig {
        sampling_frequency_hz: fs,
        frequency_hz: f,
        frame_len_samples: n,
        hop_len_samples: n,
        window: WindowFunction::Rectangular,
        remove_mean: true,
    };

    let snaps = extract_stft_bin_snapshots(&data, &cfg).expect("snaps");
    let bin = cfg.bin_index();

    let mags: Vec<f64> = (0..snaps.ncols()).map(|i| snaps[(0, i)].norm()).collect();
    let mean = mags.iter().copied().sum::<f64>() / mags.len().max(1) as f64;

    assert!(mean > 1.0, "expected non-trivial bin magnitude at k={bin}");

    if mags.len() >= 2 {
        let max = mags.iter().copied().fold(0.0, f64::max);
        let min = mags.iter().copied().fold(f64::INFINITY, f64::min);
        assert_relative_eq!(max / min, 1.0, epsilon = 1e-2);
    }
}

// suppress unused import warning
const _: fn() = || {
    let _: fn(&Array3<f64>, &SnapshotSelection) -> _ = extract_windowed_snapshots;
};
