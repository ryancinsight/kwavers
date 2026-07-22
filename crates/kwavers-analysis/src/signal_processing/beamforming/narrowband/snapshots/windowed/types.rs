use kwavers_core::constants::numerical::MHZ_TO_HZ;
use kwavers_core::constants::numerical::TWO_PI;
use kwavers_core::error::{KwaversError, KwaversResult};

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
    pub(super) fn coefficient(self, n: usize, len: usize) -> f64 {
        match self {
            Self::Rectangular => 1.0,
            Self::Hann => {
                if len <= 1 {
                    1.0
                } else {
                    let denom = (len - 1) as f64;
                    0.5 * (1.0 - (TWO_PI * (n as f64) / denom).cos())
                }
            }
        }
    }

    pub(super) fn build(self, len: usize) -> Vec<f64> {
        (0..len).map(|n| self.coefficient(n, len)).collect()
    }
}

/// How to form complex snapshots for narrowband processing.
#[derive(Debug, Clone, PartialEq)]
pub enum SnapshotMethod {
    /// Windowed STFT/FFT bin snapshots.
    StftBin(StftBinConfig),
}

/// Scenario metadata used for automatic snapshot-method selection.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SnapshotScenario {
    /// The signal-of-interest center frequency (Hz). Must be finite and > 0.
    pub frequency_hz: f64,
    /// Sampling frequency (Hz). Must be finite and > 0.
    pub sampling_frequency_hz: f64,
    /// Expected fractional bandwidth (dimensionless), e.g. 0.05 for 5%.
    pub fractional_bandwidth: Option<f64>,
    /// If true, prefer robustness against leakage / slight frequency mismatch.
    pub prefer_robustness: bool,
    /// If true, prefer maximum time resolution (shorter windows / larger hop).
    pub prefer_time_resolution: bool,
}

impl SnapshotScenario {
    /// Validate scenario invariants.
    /// # Errors
    /// - Returns `KwaversError::InvalidInput` if the precondition for invalid or out-of-range input parameters is violated.
    ///
    pub fn validate(&self) -> KwaversResult<()> {
        if !self.frequency_hz.is_finite() || self.frequency_hz <= 0.0 {
            return Err(KwaversError::InvalidInput(
                "SnapshotScenario: frequency_hz must be finite and > 0".to_owned(),
            ));
        }
        if !self.sampling_frequency_hz.is_finite() || self.sampling_frequency_hz <= 0.0 {
            return Err(KwaversError::InvalidInput(
                "SnapshotScenario: sampling_frequency_hz must be finite and > 0".to_owned(),
            ));
        }
        if self.frequency_hz >= 0.5 * self.sampling_frequency_hz {
            return Err(KwaversError::InvalidInput(
                "SnapshotScenario: frequency_hz must be < Nyquist (fs/2)".to_owned(),
            ));
        }
        if let Some(bw) = self.fractional_bandwidth {
            if !bw.is_finite() || bw <= 0.0 || bw >= 1.0 {
                return Err(KwaversError::InvalidInput(
                    "SnapshotScenario: fractional_bandwidth must be finite and in (0,1) when provided".to_owned(),
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
    pub remove_mean: bool,
}

impl StftBinConfig {
    /// Validate invariants.
    /// # Errors
    /// - Returns `KwaversError::InvalidInput` if the precondition for invalid or out-of-range input parameters is violated.
    ///
    pub fn validate(&self) -> KwaversResult<()> {
        if !self.sampling_frequency_hz.is_finite() || self.sampling_frequency_hz <= 0.0 {
            return Err(KwaversError::InvalidInput(
                "StftBinConfig: sampling_frequency_hz must be finite and > 0".to_owned(),
            ));
        }
        if !self.frequency_hz.is_finite() || self.frequency_hz <= 0.0 {
            return Err(KwaversError::InvalidInput(
                "StftBinConfig: frequency_hz must be finite and > 0".to_owned(),
            ));
        }
        if self.frequency_hz >= 0.5 * self.sampling_frequency_hz {
            return Err(KwaversError::InvalidInput(
                "StftBinConfig: frequency_hz must be < Nyquist (fs/2)".to_owned(),
            ));
        }
        if self.frame_len_samples < 2 {
            return Err(KwaversError::InvalidInput(
                "StftBinConfig: frame_len_samples must be >= 2".to_owned(),
            ));
        }
        if self.hop_len_samples == 0 || self.hop_len_samples > self.frame_len_samples {
            return Err(KwaversError::InvalidInput(
                "StftBinConfig: hop_len_samples must be in [1, frame_len_samples]".to_owned(),
            ));
        }
        Ok(())
    }

    /// Resolve the FFT bin index for the target frequency.
    ///
    /// `k = round(f * N / fs)`, with `k ∈ [0, N-1]`.
    #[must_use]
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
            sampling_frequency_hz: MHZ_TO_HZ,
            frequency_hz: 200_000.0,
            frame_len_samples: 512,
            hop_len_samples: 128,
            window: WindowFunction::Hann,
            remove_mean: true,
        }
    }
}

/// Policy enum for selecting a snapshot method.
#[derive(Debug, Clone, PartialEq)]
pub enum SnapshotSelection {
    /// Use an explicitly provided method.
    Explicit(SnapshotMethod),
    /// Choose the best available method based on the scenario.
    Auto(SnapshotScenario),
}

impl SnapshotSelection {
    /// Resolve to a concrete snapshot method (deterministic for a given scenario).
    /// # Errors
    /// - Propagates any `KwaversError` returned by called functions.
    ///
    pub fn resolve(&self, n_samples: usize) -> KwaversResult<SnapshotMethod> {
        match self {
            Self::Explicit(m) => Ok(m.clone()),
            Self::Auto(s) => {
                s.validate()?;

                let fs = s.sampling_frequency_hz;
                let f0 = s.frequency_hz;

                let target_df = if let Some(frac_bw) = s.fractional_bandwidth {
                    (frac_bw * f0) / 8.0
                } else {
                    f0 / 64.0
                };

                let mut n = (fs / target_df).ceil() as usize;
                n = n.max(64).min(n_samples.max(2));

                let n_pow2 = n.next_power_of_two();
                let n = if n_pow2 <= n_samples { n_pow2 } else { n };

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
