#![deny(missing_docs)]
//! Windowed snapshot extraction for narrowband adaptive array processing (MVDR/Capon, MUSIC, ESMV).
//!
//! # Literature alignment
//! Most narrowband adaptive beamforming literature assumes complex snapshots `x_k ∈ ℂ^M` and a
//! Hermitian covariance `R = (1/K) ∑ x_k x_kᴴ`. Practically, `x_k` is typically formed from
//! windowed blocks of data (time snapshots) or from **STFT/FFT bin** values at the narrowband
//! frequency of interest.
//!
//! This module provides **windowed STFT-bin snapshots** as an SSOT primitive.
//!
//! # Data model
//! Input sensor time series is shaped `(n_sensors, 1, n_samples)`.
//! Output snapshots are shaped `(n_sensors, n_snapshots)`, where each column is a complex snapshot
//! vector across sensors.
//!
//! # Key design constraint: scenario-driven auto selection
//! You can select a snapshot method explicitly, or ask the system to automatically choose a
//! literature-consistent method depending on the scenario.
//!
//! # No error masking
//! - Validation is strict (no silent clamp/ceil on indices).
//! - Auto-selection is deterministic and documented.
//!
//! # Implementation note
//! - STFT bin snapshotting is implemented via windowed FFT per sensor per frame. This is O(M * K * N log N).
//! - RustFFT is un-normalized; we scale inverse transforms where applicable (not used here). For bin
//!   extraction, we keep raw FFT bin values; any global scaling cancels in MVDR spectrum ratios,
//!   but relative scaling across snapshots matters. Windowing is applied consistently across sensors.

use crate::core::error::{KwaversError, KwaversResult};
use ndarray::{Array2, Array3};
use num_complex::Complex64;
use rustfft::FftPlanner;

/// Window function for windowed snapshot extraction.
///
/// # Literature note
/// Hann is the most common default for STFT-like processing; it reduces spectral leakage.
/// Rectangular is useful for exact periodic signals aligned to the window.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WindowFunction {
    /// Rectangular window (no taper).
    Rectangular,
    /// Hann window.
    Hann,
}

impl WindowFunction {
    fn coefficient(self, n: usize, len: usize) -> f64 {
        match self {
            Self::Rectangular => 1.0,
            Self::Hann => {
                // Hann: w[n] = 0.5 * (1 - cos(2π n/(N-1)))  for N>1
                if len <= 1 {
                    1.0
                } else {
                    let denom = (len - 1) as f64;
                    0.5 * (1.0 - (2.0 * std::f64::consts::PI * (n as f64) / denom).cos())
                }
            }
        }
    }

    fn build(self, len: usize) -> Vec<f64> {
        (0..len).map(|n| self.coefficient(n, len)).collect()
    }
}

/// How to form complex snapshots for narrowband processing.
///
/// Naming is field-jargon aligned and intentionally avoids ambiguous "baseband" wording where
/// it matters: this type is about snapshot *formation*, not steering.
#[derive(Debug, Clone, PartialEq)]
pub enum SnapshotMethod {
    /// Windowed STFT/FFT bin snapshots:
    /// - Segment the time series into overlapping (or non-overlapping) frames.
    /// - Apply window function to each frame.
    /// - FFT the frame and take the complex value at the bin nearest `frequency_hz`.
    ///
    /// Each frame produces one snapshot. This is the standard "frequency-bin snapshot" model.
    StftBin(StftBinConfig),
}

/// Scenario metadata used for automatic snapshot-method selection.
///
/// The intent is to provide enough information to make a deterministic, literature-consistent choice
/// without introducing silent behavior changes.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SnapshotScenario {
    /// The signal-of-interest center frequency (Hz). Must be finite and > 0.
    pub frequency_hz: f64,
    /// Sampling frequency (Hz). Must be finite and > 0.
    pub sampling_frequency_hz: f64,
    /// Expected fractional bandwidth (dimensionless), e.g. 0.05 for 5%.
    ///
    /// If unknown, set `None` and the selector will use conservative defaults.
    pub fractional_bandwidth: Option<f64>,
    /// If true, prefer robustness against leakage / slight frequency mismatch (STFT bin is preferred).
    pub prefer_robustness: bool,
    /// If true, prefer maximum time resolution (shorter windows / larger hop).
    pub prefer_time_resolution: bool,
}

impl SnapshotScenario {
    /// Validate scenario invariants.
    pub fn validate(&self) -> KwaversResult<()> {
        if !self.frequency_hz.is_finite() || self.frequency_hz <= 0.0 {
            return Err(KwaversError::InvalidInput(
                "SnapshotScenario: frequency_hz must be finite and > 0".to_string(),
            ));
        }
        if !self.sampling_frequency_hz.is_finite() || self.sampling_frequency_hz <= 0.0 {
            return Err(KwaversError::InvalidInput(
                "SnapshotScenario: sampling_frequency_hz must be finite and > 0".to_string(),
            ));
        }
        if self.frequency_hz >= 0.5 * self.sampling_frequency_hz {
            return Err(KwaversError::InvalidInput(
                "SnapshotScenario: frequency_hz must be < Nyquist (fs/2)".to_string(),
            ));
        }
        if let Some(bw) = self.fractional_bandwidth {
            if !bw.is_finite() || bw <= 0.0 || bw >= 1.0 {
                return Err(KwaversError::InvalidInput(
                    "SnapshotScenario: fractional_bandwidth must be finite and in (0,1) when provided"
                        .to_string(),
                ));
            }
        }
        Ok(())
    }
}

/// Configuration for STFT-bin snapshot extraction.
#[derive(Debug, Clone, PartialEq)]
pub struct StftBinConfig {
    /// Sampling frequency (Hz). Must be finite and > 0.
    pub sampling_frequency_hz: f64,
    /// Target frequency (Hz). Must be finite and > 0 and < Nyquist.
    pub frequency_hz: f64,
    /// Frame length in samples (>= 2).
    pub frame_len_samples: usize,
    /// Hop length in samples (>= 1 and <= frame_len_samples).
    pub hop_len_samples: usize,
    /// Window function applied to each frame.
    pub window: WindowFunction,
    /// If true, subtract per-frame mean (DC removal) before window+FFT.
    ///
    /// This is often useful for baseband/low-frequency leakage. For narrowband bin snapshots,
    /// DC bias can contaminate adjacent bins via window sidelobes.
    pub remove_mean: bool,
}

impl StftBinConfig {
    /// Validate invariants.
    pub fn validate(&self) -> KwaversResult<()> {
        if !self.sampling_frequency_hz.is_finite() || self.sampling_frequency_hz <= 0.0 {
            return Err(KwaversError::InvalidInput(
                "StftBinConfig: sampling_frequency_hz must be finite and > 0".to_string(),
            ));
        }
        if !self.frequency_hz.is_finite() || self.frequency_hz <= 0.0 {
            return Err(KwaversError::InvalidInput(
                "StftBinConfig: frequency_hz must be finite and > 0".to_string(),
            ));
        }
        if self.frequency_hz >= 0.5 * self.sampling_frequency_hz {
            return Err(KwaversError::InvalidInput(
                "StftBinConfig: frequency_hz must be < Nyquist (fs/2)".to_string(),
            ));
        }
        if self.frame_len_samples < 2 {
            return Err(KwaversError::InvalidInput(
                "StftBinConfig: frame_len_samples must be >= 2".to_string(),
            ));
        }
        if self.hop_len_samples == 0 || self.hop_len_samples > self.frame_len_samples {
            return Err(KwaversError::InvalidInput(
                "StftBinConfig: hop_len_samples must be in [1, frame_len_samples]".to_string(),
            ));
        }
        Ok(())
    }

    /// Resolve the FFT bin index for the target frequency.
    ///
    /// We choose the nearest bin to `frequency_hz` in a length-N FFT, i.e.:
    /// `k = round(f * N / fs)`, with `k ∈ [0, N-1]`.
    ///
    /// # Invariant
    /// Since `frequency_hz < fs/2`, the nearest bin will be in the nonnegative-frequency half.
    pub fn bin_index(&self) -> usize {
        let n = self.frame_len_samples as f64;
        let k = (self.frequency_hz * n / self.sampling_frequency_hz).round();
        let k_usize = if k < 0.0 { 0 } else { k as usize };
        k_usize.min(self.frame_len_samples - 1)
    }
}

impl Default for StftBinConfig {
    fn default() -> Self {
        Self {
            sampling_frequency_hz: 1_000_000.0,
            frequency_hz: 200_000.0,
            frame_len_samples: 512,
            hop_len_samples: 128,
            window: WindowFunction::Hann,
            remove_mean: true,
        }
    }
}

/// Policy enum for selecting a snapshot method.
///
/// - `Explicit` uses exactly the provided method.
/// - `Auto` chooses deterministically based on a scenario (and may still expose chosen config).
#[derive(Debug, Clone, PartialEq)]
pub enum SnapshotSelection {
    /// Use an explicitly provided method.
    Explicit(SnapshotMethod),
    /// Choose the best available method based on the scenario.
    Auto(SnapshotScenario),
}

impl SnapshotSelection {
    /// Resolve to a concrete snapshot method.
    ///
    /// # Determinism
    /// This function is deterministic for a given scenario.
    ///
    /// # Auto-selection rule (current best practice)
    /// Always selects `StftBin` with:
    /// - Hann window (robust leakage control),
    /// - frame length chosen to give adequate frequency resolution,
    /// - hop set based on time-resolution preference.
    ///
    /// This is the most robust general-purpose narrowband choice without requiring an explicit
    /// bandpass/decimator pipeline.
    pub fn resolve(&self, n_samples: usize) -> KwaversResult<SnapshotMethod> {
        match self {
            Self::Explicit(m) => Ok(m.clone()),
            Self::Auto(s) => {
                s.validate()?;

                // Choose a frame length based on desired frequency resolution.
                // For narrowband processing, we want bin width Δf = fs/N to be small relative
                // to expected bandwidth and/or to tolerate slight mismatch.
                //
                // Heuristic:
                // - If fractional bandwidth is known: target Δf ≤ (bw * f0) / 8
                // - Else: target Δf ≤ f0 / 64  (gives 64 bins per center frequency span)
                let fs = s.sampling_frequency_hz;
                let f0 = s.frequency_hz;

                let target_df = if let Some(frac_bw) = s.fractional_bandwidth {
                    (frac_bw * f0) / 8.0
                } else {
                    f0 / 64.0
                };

                // N >= fs/target_df.
                let mut n = (fs / target_df).ceil() as usize;

                // Clamp N to a valid range and to available samples.
                // - Minimum 64 to avoid degenerate resolution.
                // - Maximum is n_samples (cannot frame longer than the signal).
                // - Prefer power-of-two for FFT efficiency, but do not exceed n_samples.
                n = n.max(64).min(n_samples.max(2));

                let n_pow2 = n.next_power_of_two();
                let n = if n_pow2 <= n_samples { n_pow2 } else { n };

                // Hop selection:
                // - If prefer_time_resolution: higher overlap (smaller hop)
                // - Else: moderate overlap
                let hop = if s.prefer_time_resolution {
                    (n / 8).max(1)
                } else if s.prefer_robustness {
                    (n / 4).max(1)
                } else {
                    (n / 2).max(1)
                };

                let cfg = StftBinConfig {
                    sampling_frequency_hz: fs,
                    frequency_hz: f0,
                    frame_len_samples: n,
                    hop_len_samples: hop.min(n),
                    window: WindowFunction::Hann,
                    remove_mean: true,
                };
                cfg.validate()?;

                Ok(SnapshotMethod::StftBin(cfg))
            }
        }
    }
}

/// Extract complex snapshots according to a selection policy.
///
/// # Inputs
/// - `sensor_data` shape: `(n_sensors, 1, n_samples)`
/// - `selection`: explicit or auto selection
///
/// # Output
/// `(n_sensors, n_snapshots)` complex snapshots.
///
/// # Errors
/// - invalid shapes
/// - invalid selection/config
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

/// Extract STFT-bin snapshots (frequency-bin snapshots) at `cfg.frequency_hz`.
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

    // Number of frames with exact indexing:
    // start positions: 0, hop, 2hop, ... with start + n <= n_samples
    let n_frames = ((n_samples - n) / hop) + 1;

    let window = cfg.window.build(n);

    // Prepare FFT plan once (reused per sensor per frame).
    let mut planner = FftPlanner::<f64>::new();
    let fft = planner.plan_fft_forward(n);

    let mut out = Array2::<Complex64>::zeros((n_sensors, n_frames));

    // Temporary buffers to minimize allocations inside loops.
    let mut frame: Vec<Complex64> = vec![Complex64::new(0.0, 0.0); n];

    for s in 0..n_sensors {
        for frame_idx in 0..n_frames {
            let start = frame_idx * hop;
            let end = start + n;

            // Safety: by construction end <= n_samples
            debug_assert!(end <= n_samples);

            // Mean removal (optional)
            let mean = if cfg.remove_mean {
                let mut acc = 0.0;
                for t in start..end {
                    acc += sensor_data[(s, 0, t)];
                }
                acc / (n as f64)
            } else {
                0.0
            };

            // Window + pack into complex buffer
            for i in 0..n {
                let x = sensor_data[(s, 0, start + i)] - mean;
                frame[i] = Complex64::new(x * window[i], 0.0);
            }

            // FFT in-place
            fft.process(&mut frame);

            // Take bin value as snapshot component
            out[(s, frame_idx)] = frame[bin];
        }
    }

    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

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
        let sound_speed = 1500.0;
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
        // For an exact tone aligned to integer FFT bins, the chosen bin should dominate.
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
            hop_len_samples: n, // non-overlap
            window: WindowFunction::Rectangular,
            remove_mean: true,
        };

        let snaps = extract_stft_bin_snapshots(&data, &cfg).expect("snaps");
        let bin = cfg.bin_index();

        // Verify that the snapshot's magnitude is stable across frames and non-trivial.
        let mags: Vec<f64> = (0..snaps.ncols()).map(|i| snaps[(0, i)].norm()).collect();
        let mean = mags.iter().copied().sum::<f64>() / mags.len().max(1) as f64;

        assert!(mean > 1.0, "expected non-trivial bin magnitude at k={bin}");

        // Relative variation should be small for a stationary tone with rectangular/no overlap.
        if mags.len() >= 2 {
            let max = mags.iter().copied().fold(0.0, f64::max);
            let min = mags.iter().copied().fold(f64::INFINITY, f64::min);
            assert_relative_eq!(max / min, 1.0, epsilon = 1e-2);
        }
    }
}
