use serde::{Deserialize, Serialize};

/// Methods for separating clutter and blood subspaces
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum SubspaceSeparationMethod {
    /// Fixed rank cutoff (e.g., first N eigenmodes are clutter)
    FixedRank { clutter_rank: usize },
    /// Adaptive threshold based on eigenvalue decay
    AdaptiveThreshold { decay_factor: f64 },
    /// CBR-based automatic selection: the smallest rank whose estimated
    /// clutter-to-residual power ratio reaches `target_cbr_db`.
    ///
    /// `CBR(r)` is non-decreasing in `r` while residual power remains:
    /// raising `r` moves one mode's floored power from the residual sum
    /// into the clutter sum. The first rank at or above the target is
    /// therefore the smallest clutter subspace whose residual-clutter
    /// fraction `1/CBR` already falls below `1/target`: any smaller rank
    /// leaves more clutter than specified, any larger rank keeps removing
    /// modes past the point the target is met. This is eigenvalue-based
    /// order selection in the sense of Yu and Lovstakken's review of
    /// eigen-based clutter filter design (IEEE Trans. Ultrason.
    /// Ferroelectr. Freq. Control, vol. 57, no. 5, pp. 1096-1111, May
    /// 2010), whose design scenarios are parameterized by the
    /// clutter-to-blood ratio itself; eigenvalue-difference/ratio
    /// thresholding for order selection is laid out in Sec. III of the
    /// precursor by Lovstakken, Yu and Torp (Proc. IEEE Ultrason. Symp.
    /// 2007, pp. 2429-2432).
    ///
    /// `target_cbr_db` is a power ratio in dB (`10 * log10`), converted with
    /// `/ 10`, matching `AdaptiveFilter::current_cbr_db`.
    /// A larger target removes more modes; a target above the maximum
    /// achievable ratio falls back to `n/2`, as before.
    CbrBased { target_cbr_db: f64 },
}

/// Methods for estimating clutter-to-blood ratio
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum CbrEstimationMethod {
    /// Sum of eigenvalues (simple, fast)
    EigenvalueSum,
    /// Power ratio between subspaces
    PowerRatio,
    /// Maximum likelihood estimation
    MaximumLikelihood,
}

/// Configuration for adaptive clutter filter
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AdaptiveFilterConfig {
    /// Method for clutter/blood subspace separation
    pub separation_method: SubspaceSeparationMethod,
    /// CBR estimation method
    pub cbr_estimation: CbrEstimationMethod,
    /// Minimum eigenvalue threshold (relative to maximum eigenvalue)
    pub noise_floor_threshold: f64,
    /// Enable temporal smoothing of CBR estimates across ensembles
    pub temporal_smoothing: bool,
    /// Smoothing window size (only used if temporal_smoothing = true)
    pub smoothing_window: usize,
}

impl Default for AdaptiveFilterConfig {
    fn default() -> Self {
        Self {
            separation_method: SubspaceSeparationMethod::AdaptiveThreshold { decay_factor: 0.1 },
            cbr_estimation: CbrEstimationMethod::EigenvalueSum,
            noise_floor_threshold: 1e-6,
            temporal_smoothing: false,
            smoothing_window: 3,
        }
    }
}
