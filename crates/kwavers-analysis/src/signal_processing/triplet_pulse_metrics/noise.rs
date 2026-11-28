use super::filter::{kuji_filter_trace, KujiFilterShape};
use kwavers_core::error::{KwaversError, KwaversResult};
use leto::{Array2, ArrayView2};

/// Pure-noise asymptote of the mean-of-ratios reduction statistic
/// (Build 15 / WS-C): with noise-only channel RF the reduction
/// reads ≈ 2.40 linear, independent of the RF content. Any
/// measured reduction within a few percent of this value carries
/// essentially no RF information.
pub const KUJI_PURE_NOISE_REDUCTION_ASYMPTOTE: f64 = 2.40;

/// In-band SNR headroom of the Kuji experiment's own noise floor
/// [dB, power] relative to the tissue 1P echo in the HP band
/// (WS-C bound from Kuji Figs. 8/9: the effective floor sits
/// ≈ 33 dB below the tissue echo and ~10 dB below the measured
/// in-band 3P residual, so the paper's ~31 dB tissue reduction is
/// physics-limited, not floor-limited).
pub const KUJI_EXPERIMENT_INBAND_SNR_DB: f64 = 33.0;

/// Declared physical noise floor for the corrected Kuji metric.
///
/// # The failure this prevents (Build 15 / WS-C)
///
/// Build 14's B1 added white noise at `σ = 1e-3 × max|channel RF|`
/// — nominally "−60 dB". But `max|RF|` is the direct / slab-top
/// arrival, while the ROI in-band tissue echo is only
/// `3.5 × 10⁻⁶ × max|RF|`: the imposed DAS-level in-band SNR was
/// ≈ **−23 dB** (noise ≈ 14× the signal), and B1's headline
/// reductions (2.44–2.50) sat on the statistic's pure-noise
/// asymptote ([`KUJI_PURE_NOISE_REDUCTION_ASYMPTOTE`], 2.40) at
/// every PNP — they carried essentially no RF information.
/// Removing the noise on the same RF, the tissue reduction is
/// 99.7 linear (40.0 dB) at 0.26 MPa / fc 3.0 MHz: the imposed
/// floor accounted for ≥ 100 % of the tissue-side gap at low PNP.
///
/// The invariant is therefore the **in-band SNR relative to the
/// tissue-echo level** — never a raw σ relative to `max|RF|`.
/// In B1's normalisation, the floor that reproduces the paper's
/// tissue target is `σ* ≈ 4.8 × 10⁻⁶ × max|RF|` (≈ 200× below
/// B1's σ), and the experiment's own floor
/// ([`KUJI_EXPERIMENT_INBAND_SNR_DB`]) leaves ≈ 33 dB of in-band
/// headroom.
///
/// The default for every metric entry point in this module remains
/// **no added noise** (the pre-Build-15 behaviour); a floor is
/// applied only when the caller constructs one of these and calls
/// [`add_declared_noise_to_rf`] on the channel RF before
/// beamforming.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct DeclaredNoiseFloor {
    /// In-band SNR [dB, power] of the added noise relative to the
    /// tissue-echo level supplied to
    /// [`add_declared_noise_to_rf`]. Positive = noise below the
    /// echo. The experiment's floor corresponds to ≈ +33 dB;
    /// B1's accidental floor was ≈ −23 dB at DAS level.
    pub inband_snr_db: f64,
    /// Seed of the deterministic noise stream (SplitMix64 +
    /// Box–Muller, local to this module; no new dependency).
    /// Use a distinct seed per phase record so the three phases
    /// carry independent noise, as a physical receiver does.
    pub seed: u64,
}

/// Per-channel in-band noise RMS [same units as the RF] for a
/// declared in-band SNR relative to the tissue-echo in-band RMS:
/// `σ = echo_rms · 10^(−SNR/20)`.
#[must_use]
pub fn inband_noise_sigma(tissue_echo_rms_inband: f64, inband_snr_db: f64) -> f64 {
    tissue_echo_rms_inband * 10f64.powf(-inband_snr_db / 20.0)
}

/// Deterministic Gaussian stream (SplitMix64 + Box–Muller).
struct NoiseStream {
    state: u64,
    spare: Option<f64>,
}

impl NoiseStream {
    fn next_word(&mut self) -> u64 {
        self.state = self.state.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    fn next_unit(&mut self) -> f64 {
        // 53-bit uniform on (0, 1].
        let v = (self.next_word() >> 11) as f64 / (1u64 << 53) as f64;
        1.0 - v
    }

    fn next_gaussian(&mut self) -> f64 {
        if let Some(s) = self.spare.take() {
            return s;
        }
        let u1 = self.next_unit();
        let u2 = self.next_unit();
        let r = (-2.0 * u1.ln()).sqrt();
        let theta = 2.0 * std::f64::consts::PI * u2;
        self.spare = Some(r * theta.sin());
        r * theta.cos()
    }
}

/// Add a [`DeclaredNoiseFloor`] to one phase's channel RF, before
/// beamforming (spec step 4's "caller's declared noise model").
///
/// White Gaussian noise is generated per channel, passed through
/// the same Kuji high-pass (+ optional probe band) the metric
/// applies, and scaled so its **in-band RMS equals
/// [`inband_noise_sigma`]** for the caller-supplied tissue-echo
/// in-band RMS — the SNR is therefore a property of the band the
/// metric reads, not of the trace maximum. The input RF is not
/// modified; the noised copy is returned.
///
/// # Errors
/// [`KwaversError::InvalidInput`] on empty RF or a non-positive
/// tissue-echo level; propagates filter errors.
pub fn add_declared_noise_to_rf(
    rf: ArrayView2<'_, f64>,
    dt_s: f64,
    fc_hz: f64,
    shape: KujiFilterShape,
    probe_band: bool,
    tissue_echo_rms_inband: f64,
    floor: &DeclaredNoiseFloor,
) -> KwaversResult<Array2<f64>> {
    let [n_rows, n_samples] = rf.shape();
    if n_rows == 0 || n_samples == 0 || tissue_echo_rms_inband <= 0.0 {
        return Err(KwaversError::InvalidInput(
            "kuji_metric: noise floor needs non-empty RF and a positive tissue-echo level"
                .to_owned(),
        ));
    }
    let sigma = inband_noise_sigma(tissue_echo_rms_inband, floor.inband_snr_db);
    let mut out = Array2::zeros([n_rows, n_samples]);
    for e in 0..n_rows {
        let mut stream = NoiseStream {
            state: floor.seed ^ (e as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15),
            spare: None,
        };
        let white: Vec<f64> = (0..n_samples).map(|_| stream.next_gaussian()).collect();
        let banded = kuji_filter_trace(&white, dt_s, fc_hz, shape, probe_band)?;
        let rms = (banded.iter().map(|v| v * v).sum::<f64>() / n_samples as f64).sqrt();
        if rms <= 0.0 {
            return Err(KwaversError::InvalidInput(
                "kuji_metric: band-limited noise stream has zero RMS".to_owned(),
            ));
        }
        let scale = sigma / rms;
        for t in 0..n_samples {
            out[[e, t]] = rf[[e, t]] + scale * banded[t];
        }
    }
    Ok(out)
}
