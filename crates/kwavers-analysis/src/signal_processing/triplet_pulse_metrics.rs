//! Corrected Kuji 2024 and Iwasaki 2018 triplet-pulse metrics
//! (Build 14, KW-IMG-047; spec: `B14_AUDIT.md` §e).
//!
//! This module supersedes the proxy contrast in
//! [`super::triplet_pulse`] for any claim about the papers' own
//! quantities. The proxy (B-mode ROI-mean brightness ratio on
//! envelope-before-DAS images) remains for prototype parity only.
//!
//! # Kuji 2024 metric (corrected spec, audit §e steps 1–10)
//!
//! 1. Signals: per-phase channel RF; 1P = phase 0°; 3P = mean of
//!    0°/120°/240° per channel.
//! 2. Filter: high-pass on the RF **before** beamforming. Passband
//!    edge `fc` = attenuation-start frequency; stopband −60 dB;
//!    transition 0.15·fc immediately below `fc`. Realisation declared:
//!    zero-phase raised-cosine ([`KujiFilterShape::KujiRaisedCosine`],
//!    the D8/B1 shape) or brick-wall (sensitivity). The comparison
//!    sweep is **only** 2.3–3.7 MHz in 0.1 MHz steps
//!    ([`KUJI_FC_GRID_HZ`]). No low-pass; an optional declared probe
//!    receive band 1.5–4.3 MHz ([`KUJI_PROBE_BAND_HZ`]) may be applied
//!    per channel before beamforming.
//! 3. Beamforming: **coherent** delay-and-sum of the filtered RF,
//!    envelope (Hilbert magnitude) **after** the aperture sum. The
//!    envelope is never taken before the sum (that was the prototype
//!    divergence, audit §b #6).
//! 4. Per-pixel reduction `R = |1P| / |3P|`, linear, no dB, **no
//!    artificial floor**. A physical noise floor, when needed, is the
//!    caller's declared noise model added to the RF — this module
//!    adds none by default; see [`DeclaredNoiseFloor`] for the
//!    declared model and the failure it prevents.
//! 5. ROI values: tissue reduction = mean of R over the ROI in the
//!    **before** frame; cavitation reduction = mean of R over the
//!    same ROI in the **after** frame (linear-domain averaging).
//! 6. Contrast `= 20·log10(tissue / cavitation)` dB.
//! 7. Ensemble: reductions are averaged over ≥ 5 independent
//!    realisations before the contrast is formed (paper Fig. 12
//!    method); per-realisation contrasts are also reported as
//!    mean ± sample SD. Single-realisation results carry no
//!    peak/no-peak verdict (spec step 8).
//! 8. Peak test (spec step 10): the optimal cutoff per PNP is the
//!    argmax of contrast over the step-2 grid; the 0.26 MPa peak
//!    claim is evaluated only on that optimal trajectory.
//!
//! # Validation status (Build 14)
//!
//! * The reference Python implementation (`b14/metric.py`) reproduces
//!   the audit's seed-11 legacy numbers exactly (≤ 0.02 dB); this
//!   Rust component implements the same corrected statistic and is
//!   unit-tested here on synthetic signals with analytically known
//!   reductions.
//! * B1 (corrected metric, ensemble n = 5, coherent DAS + probe
//!   band): simulated contrast ≈ 0 dB at every PNP and cutoff —
//!   tissue and cavitation reductions are both ≈ 2.4–2.7 linear, so
//!   Kuji's 0.26 MPa peak is **absent** in this model and the
//!   absolute-magnitude gap to the paper's ~10–40 dB scale is an
//!   open physics item (KW-IMG-050), not a metric error.
//! * Caveat (B3): coherent DAS forms the tissue image only on
//!   discrete-speckle media; on the prototype's smooth
//!   volume-scattering tissue model it aberrates (KW-IMG-051).
//!
//! # Iwasaki 2018 statistic (corrected spec, audit §e)
//!
//! Coherent phase sum → coherent DAS → envelope (this module's DAS;
//! no band filter beyond a declared probe band). Numerator =
//! spatiotemporal maximum envelope over the full image and all
//! frames; denominator = mean envelope over depth 70–100 mm ×
//! width ±16 mm in the same frame set; contrast =
//! `20·log10(num / den)` per realisation, reported mean ± SD over
//! N = 10. The paper does not state whether the background average
//! is linear or dB-brightness; both are provided
//! ([`IwasakiBackgroundAveraging`]) and the choice must be declared.

use super::triplet_pulse::TripletPulseGeometry;
use apollo::{fft_1d_slice_half, ifft_1d_slice_half_into};
use kwavers_core::error::{KwaversError, KwaversResult};
use kwavers_signal::analytic::instantaneous_envelope;
use leto::{Array1, Array2, ArrayView2};

/// Kuji comparison sweep: passband edges 2.3–3.7 MHz in 0.1 MHz
/// steps (audit spec step 2 — the only grid comparable to Kuji
/// Figs. 12–14).
pub const KUJI_FC_GRID_HZ: [f64; 15] = [
    2.3e6, 2.4e6, 2.5e6, 2.6e6, 2.7e6, 2.8e6, 2.9e6, 3.0e6, 3.1e6, 3.2e6, 3.3e6, 3.4e6, 3.5e6,
    3.6e6, 3.7e6,
];

/// Kuji probe (UST-52105) receive band [Hz] (paper §2.5). Declared
/// model of the physical band; applied per channel before
/// beamforming when enabled.
pub const KUJI_PROBE_BAND_HZ: (f64, f64) = (1.5e6, 4.3e6);

/// Kuji imaging PNP set [Pa] (paper §3.1).
pub const KUJI_PNP_PA: [f64; 4] = [0.12e6, 0.26e6, 0.72e6, 1.23e6];

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

/// Rectangular region of interest (metres, inclusive bounds).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RectRoi {
    /// Lateral lower bound [m].
    pub x_lo_m: f64,
    /// Lateral upper bound [m].
    pub x_hi_m: f64,
    /// Depth lower bound [m].
    pub y_lo_m: f64,
    /// Depth upper bound [m].
    pub y_hi_m: f64,
}

impl RectRoi {
    /// Kuji's paper ROI: depth 62–68 mm × width −3…+3 mm (§2.7).
    #[must_use]
    pub const fn kuji_paper() -> Self {
        Self {
            x_lo_m: -3.0e-3,
            x_hi_m: 3.0e-3,
            y_lo_m: 62.0e-3,
            y_hi_m: 68.0e-3,
        }
    }

    /// Iwasaki background region: depth 70–100 mm × width ±16 mm.
    #[must_use]
    pub const fn iwasaki_background() -> Self {
        Self {
            x_lo_m: -16.0e-3,
            x_hi_m: 16.0e-3,
            y_lo_m: 70.0e-3,
            y_hi_m: 100.0e-3,
        }
    }

    /// Whether `(x, y)` is inside the ROI.
    #[must_use]
    pub fn contains(&self, x_m: f64, y_m: f64) -> bool {
        x_m >= self.x_lo_m && x_m <= self.x_hi_m && y_m >= self.y_lo_m && y_m <= self.y_hi_m
    }
}

/// Coherent DAS envelope image of (already filtered) channel RF.
///
/// For each pixel the aperture RF traces are delay-and-summed
/// (nearest sample, divided by the full element count), a window of
/// `±half_win` samples around the focal alignment is Hilbert
/// enveloped, and the centre sample is the pixel value — the
/// envelope is taken **after** the aperture sum (spec step 3).
///
/// # Errors
/// [`KwaversError::InvalidInput`] on shape/geometry mismatch, empty
/// grid, or non-finite input.
pub fn coherent_das_envelope(
    rf: ArrayView2<'_, f64>,
    geometry: &TripletPulseGeometry,
    xs_m: &[f64],
    ys_m: &[f64],
    half_win: usize,
) -> KwaversResult<Array2<f64>> {
    let [n_rows, n_samples] = rf.shape();
    if n_rows == 0 || n_samples == 0 || n_rows != geometry.n_elem() {
        return Err(KwaversError::InvalidInput(format!(
            "kuji_metric: RF rows {n_rows} must equal geometry elements {} and be non-empty",
            geometry.n_elem()
        )));
    }
    if xs_m.is_empty() || ys_m.is_empty() || !rf.iter().all(|v| v.is_finite()) {
        return Err(KwaversError::InvalidInput(
            "kuji_metric: grid must be non-empty and RF finite".to_owned(),
        ));
    }
    let n_elem = geometry.n_elem() as f64;
    let win = 2 * half_win + 1;
    let mut image = Array2::zeros([ys_m.len(), xs_m.len()]);
    for (iy, &y) in ys_m.iter().enumerate() {
        for (ix, &x) in xs_m.iter().enumerate() {
            let d_tx = geometry.transmit_delay_s(x, y) * geometry.sound_speed_m_s;
            let mut summed = vec![0.0_f64; win];
            for (e, &x_el) in geometry.element_x_m.iter().enumerate() {
                let dx = x - x_el;
                let dy = y - geometry.element_depth_m;
                let d_rx = (dx * dx + dy * dy).sqrt();
                let centre =
                    ((d_tx + d_rx) / geometry.sound_speed_m_s / geometry.dt_s).round() as i64;
                for (k, slot) in summed.iter_mut().enumerate() {
                    let idx = centre - half_win as i64 + k as i64;
                    if idx >= 0 && (idx as usize) < n_samples {
                        *slot += rf[[e, idx as usize]];
                    }
                }
            }
            for slot in &mut summed {
                *slot /= n_elem;
            }
            let trace = Array1::from_shape_vec([win], summed).map_err(|e| {
                KwaversError::InvalidInput(format!("kuji_metric: window copy failed: {e}"))
            })?;
            let env = instantaneous_envelope(&trace);
            image[[iy, ix]] = env[half_win];
        }
    }
    Ok(image)
}

/// Channel RF of the three transmit phases of one acquisition.
pub struct PhaseRf<'a> {
    /// RF of the 0° phase, one row per element.
    pub ph0: ArrayView2<'a, f64>,
    /// RF of the 120° phase, same shape as `ph0`.
    pub ph120: ArrayView2<'a, f64>,
    /// RF of the 240° phase, same shape as `ph0`.
    pub ph240: ArrayView2<'a, f64>,
}

/// Declared Kuji high-pass applied to every channel before
/// beamforming (spec step 2).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct KujiFilter {
    /// Passband edge `fc` [Hz].
    pub fc_hz: f64,
    /// Realisation of the high-pass.
    pub shape: KujiFilterShape,
    /// Whether the declared probe receive band is also applied.
    pub probe_band: bool,
}

/// 1P and 3P envelope images from the three phase RF records at one
/// cutoff: combine per channel, Kuji-filter each channel, coherent
/// DAS (spec steps 1–3).
///
/// # Errors
/// Propagates shape/filter/DAS validation errors.
pub fn kuji_images(
    phases: PhaseRf<'_>,
    geometry: &TripletPulseGeometry,
    xs_m: &[f64],
    ys_m: &[f64],
    filter: KujiFilter,
    half_win: usize,
) -> KwaversResult<(Array2<f64>, Array2<f64>)> {
    let PhaseRf { ph0, ph120, ph240 } = phases;
    let KujiFilter {
        fc_hz,
        shape,
        probe_band,
    } = filter;
    let shape0 = ph0.shape();
    if ph120.shape() != shape0 || ph240.shape() != shape0 {
        return Err(KwaversError::InvalidInput(
            "kuji_metric: phase RF shapes must match".to_owned(),
        ));
    }
    let [n_elem, n_samples] = shape0;
    let mut one = Array2::zeros([n_elem, n_samples]);
    let mut three = Array2::zeros([n_elem, n_samples]);
    for e in 0..n_elem {
        let row0: Vec<f64> = (0..n_samples).map(|t| ph0[[e, t]]).collect();
        let row3: Vec<f64> = (0..n_samples)
            .map(|t| (ph0[[e, t]] + ph120[[e, t]] + ph240[[e, t]]) / 3.0)
            .collect();
        let f0 = kuji_filter_trace(&row0, geometry.dt_s, fc_hz, shape, probe_band)?;
        let f3 = kuji_filter_trace(&row3, geometry.dt_s, fc_hz, shape, probe_band)?;
        for t in 0..n_samples {
            one[[e, t]] = f0[t];
            three[[e, t]] = f3[t];
        }
    }
    Ok((
        coherent_das_envelope(one.view(), geometry, xs_m, ys_m, half_win)?,
        coherent_das_envelope(three.view(), geometry, xs_m, ys_m, half_win)?,
    ))
}

/// Mean of the per-pixel linear reduction `R = |1P|/|3P|` over an
/// ROI (spec steps 4–6, linear domain). No floor is applied.
///
/// # Errors
/// [`KwaversError::InvalidInput`] on grid mismatch or empty ROI.
pub fn roi_mean_reduction(
    img_1p: ArrayView2<'_, f64>,
    img_3p: ArrayView2<'_, f64>,
    xs_m: &[f64],
    ys_m: &[f64],
    roi: &RectRoi,
) -> KwaversResult<f64> {
    if img_1p.shape() != [ys_m.len(), xs_m.len()] || img_3p.shape() != img_1p.shape() {
        return Err(KwaversError::InvalidInput(
            "kuji_metric: image shapes must match the grid".to_owned(),
        ));
    }
    let mut sum = 0.0;
    let mut count = 0_usize;
    for (iy, &y) in ys_m.iter().enumerate() {
        for (ix, &x) in xs_m.iter().enumerate() {
            if roi.contains(x, y) {
                sum += img_1p[[iy, ix]] / img_3p[[iy, ix]];
                count += 1;
            }
        }
    }
    if count == 0 {
        return Err(KwaversError::InvalidInput(
            "kuji_metric: ROI contains no pixel".to_owned(),
        ));
    }
    Ok(sum / count as f64)
}

/// One realisation's Kuji reductions and contrast at one cutoff.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct KujiReductions {
    /// Mean linear reduction in the before (tissue) frame.
    pub red_tissue: f64,
    /// Mean linear reduction in the after (cavitation) frame.
    pub red_cavitation: f64,
    /// `20·log10(red_tissue / red_cavitation)` [dB].
    pub contrast_db: f64,
}

impl KujiReductions {
    /// Build from the two reductions; contrast formed in dB only
    /// here (spec step 7).
    #[must_use]
    pub fn new(red_tissue: f64, red_cavitation: f64) -> Self {
        Self {
            red_tissue,
            red_cavitation,
            contrast_db: 20.0 * (red_tissue / red_cavitation).log10(),
        }
    }
}

/// Kuji contrast from before/after 1P and 3P envelope images.
///
/// # Errors
/// Propagates [`roi_mean_reduction`] errors.
pub fn kuji_contrast_from_images(
    before_1p: ArrayView2<'_, f64>,
    before_3p: ArrayView2<'_, f64>,
    after_1p: ArrayView2<'_, f64>,
    after_3p: ArrayView2<'_, f64>,
    xs_m: &[f64],
    ys_m: &[f64],
    roi: &RectRoi,
) -> KwaversResult<KujiReductions> {
    let red_t = roi_mean_reduction(before_1p, before_3p, xs_m, ys_m, roi)?;
    let red_c = roi_mean_reduction(after_1p, after_3p, xs_m, ys_m, roi)?;
    Ok(KujiReductions::new(red_t, red_c))
}

/// Ensemble reduction of per-realisation results (spec steps 7–8):
/// reductions averaged over realisations first (paper Fig. 12
/// method), contrast from the averaged reductions; per-realisation
/// contrasts summarised as mean ± sample SD alongside.
#[derive(Debug, Clone, PartialEq)]
pub struct KujiEnsemble {
    /// Number of realisations.
    pub n_realizations: usize,
    /// Ensemble-mean tissue reduction.
    pub mean_red_tissue: f64,
    /// Ensemble-mean cavitation reduction.
    pub mean_red_cavitation: f64,
    /// Contrast from the ensemble-averaged reductions [dB].
    pub contrast_db: f64,
    /// Mean of the per-realisation contrasts [dB].
    pub per_realization_mean_db: f64,
    /// Sample SD (n−1) of the per-realisation contrasts [dB];
    /// 0 for a single realisation (which carries no verdict).
    pub per_realization_sd_db: f64,
}

/// Average per-realisation [`KujiReductions`] into an ensemble.
///
/// # Errors
/// [`KwaversError::InvalidInput`] on an empty slice.
pub fn kuji_ensemble(results: &[KujiReductions]) -> KwaversResult<KujiEnsemble> {
    if results.is_empty() {
        return Err(KwaversError::InvalidInput(
            "kuji_metric: ensemble needs at least one realisation".to_owned(),
        ));
    }
    let n = results.len() as f64;
    let mean_t = results.iter().map(|r| r.red_tissue).sum::<f64>() / n;
    let mean_c = results.iter().map(|r| r.red_cavitation).sum::<f64>() / n;
    let mean_db = results.iter().map(|r| r.contrast_db).sum::<f64>() / n;
    let sd_db = if results.len() > 1 {
        let var = results
            .iter()
            .map(|r| (r.contrast_db - mean_db).powi(2))
            .sum::<f64>()
            / (n - 1.0);
        var.sqrt()
    } else {
        0.0
    };
    Ok(KujiEnsemble {
        n_realizations: results.len(),
        mean_red_tissue: mean_t,
        mean_red_cavitation: mean_c,
        contrast_db: 20.0 * (mean_t / mean_c).log10(),
        per_realization_mean_db: mean_db,
        per_realization_sd_db: sd_db,
    })
}

/// Optimal cutoff (argmax contrast) over a sweep — one PNP's entry
/// of the optimal trajectory (spec step 10).
///
/// # Errors
/// [`KwaversError::InvalidInput`] on an empty sweep.
pub fn optimal_cutoff(sweep: &[(f64, KujiEnsemble)]) -> KwaversResult<(f64, f64)> {
    sweep
        .iter()
        .max_by(|a, b| a.1.contrast_db.total_cmp(&b.1.contrast_db))
        .map(|(fc, ens)| (*fc, ens.contrast_db))
        .ok_or_else(|| KwaversError::InvalidInput("kuji_metric: empty sweep".to_owned()))
}

/// How Iwasaki's background brightness is averaged (the paper does
/// not resolve this; declare the choice).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum IwasakiBackgroundAveraging {
    /// Arithmetic mean of the linear envelope.
    Linear,
    /// Mean of the dB brightness (geometric mean in linear units).
    DbBrightness,
}

/// Iwasaki 2018 contrast for one method: spatiotemporal maximum
/// envelope over all frames ÷ background mean over `background`
/// in the same frames (corrected spec, audit §e).
///
/// # Errors
/// [`KwaversError::InvalidInput`] on no frames, grid mismatch, or
/// an empty background region.
pub fn iwasaki_contrast(
    frames: &[Array2<f64>],
    xs_m: &[f64],
    ys_m: &[f64],
    background: &RectRoi,
    averaging: IwasakiBackgroundAveraging,
) -> KwaversResult<f64> {
    if frames.is_empty() {
        return Err(KwaversError::InvalidInput(
            "iwasaki_metric: at least one frame is required".to_owned(),
        ));
    }
    let mut max_all = f64::NEG_INFINITY;
    let mut bg_sum = 0.0;
    let mut bg_count = 0_usize;
    for frame in frames {
        if frame.shape() != [ys_m.len(), xs_m.len()] {
            return Err(KwaversError::InvalidInput(
                "iwasaki_metric: frame shape must match the grid".to_owned(),
            ));
        }
        for (iy, &y) in ys_m.iter().enumerate() {
            for (ix, &x) in xs_m.iter().enumerate() {
                let v = frame[[iy, ix]];
                if v > max_all {
                    max_all = v;
                }
                if background.contains(x, y) {
                    bg_sum += match averaging {
                        IwasakiBackgroundAveraging::Linear => v,
                        IwasakiBackgroundAveraging::DbBrightness => 20.0 * v.log10(),
                    };
                    bg_count += 1;
                }
            }
        }
    }
    if bg_count == 0 {
        return Err(KwaversError::InvalidInput(
            "iwasaki_metric: background region contains no pixel".to_owned(),
        ));
    }
    let bg_mean = bg_sum / bg_count as f64;
    let denominator = match averaging {
        IwasakiBackgroundAveraging::Linear => bg_mean,
        IwasakiBackgroundAveraging::DbBrightness => 10f64.powf(bg_mean / 20.0),
    };
    Ok(20.0 * (max_all / denominator).log10())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn hp_response_matches_kuji_shape() {
        let fc = 2.5e6;
        assert_eq!(
            kuji_hp_response(fc, fc, KujiFilterShape::KujiRaisedCosine),
            1.0
        );
        assert_eq!(
            kuji_hp_response(0.85 * fc, fc, KujiFilterShape::KujiRaisedCosine),
            1e-3
        );
        assert_eq!(
            kuji_hp_response(0.5 * fc, fc, KujiFilterShape::KujiRaisedCosine),
            1e-3
        );
        // Transition midpoint u = 0.5: 1e-3 + (1 - 1e-3) * 0.5.
        let mid = kuji_hp_response(0.925 * fc, fc, KujiFilterShape::KujiRaisedCosine);
        assert!((mid - 0.5005).abs() < 1e-12, "mid {mid}");
        assert_eq!(
            kuji_hp_response(0.9 * fc, fc, KujiFilterShape::BrickWall),
            0.0
        );
        assert_eq!(KUJI_FC_GRID_HZ.len(), 15);
        assert_eq!(KUJI_FC_GRID_HZ[0], 2.3e6);
        assert_eq!(KUJI_FC_GRID_HZ[14], 3.7e6);
    }

    #[test]
    fn filter_trace_suppresses_below_stopband_and_passes_passband() {
        // Bin-centred tones: n = 10000 at dt = 20 ns gives df = 5 kHz,
        // so 2 MHz (bin 400) and 3 MHz (bin 600) are exactly periodic
        // in the window — no leakage to confuse the response check.
        let dt = 20e-9;
        let n = 10000;
        let low: Vec<f64> = (0..n)
            .map(|i| (2.0 * std::f64::consts::PI * 2.0e6 * i as f64 * dt).sin())
            .collect();
        let out =
            kuji_filter_trace(&low, dt, 2.5e6, KujiFilterShape::KujiRaisedCosine, false).unwrap();
        let amp = out.iter().fold(0.0_f64, |a, &b| a.max(b.abs()));
        assert!(amp < 2e-3, "2 MHz amplitude after HP: {amp}");
        let high: Vec<f64> = (0..n)
            .map(|i| (2.0 * std::f64::consts::PI * 3.0e6 * i as f64 * dt).sin())
            .collect();
        let out =
            kuji_filter_trace(&high, dt, 2.5e6, KujiFilterShape::KujiRaisedCosine, false).unwrap();
        let amp = out.iter().fold(0.0_f64, |a, &b| a.max(b.abs()));
        assert!((amp - 1.0).abs() < 0.02, "3 MHz amplitude after HP: {amp}");
    }

    fn impulse_rf(geom: &TripletPulseGeometry, sx: f64, sy: f64, scale: f64) -> Array2<f64> {
        let n_samples = 1800;
        let mut rf = Array2::zeros([geom.n_elem(), n_samples]);
        for (e, &x_el) in geom.element_x_m.iter().enumerate() {
            let d_tx = geom.transmit_delay_s(sx, sy);
            let dx = sx - x_el;
            let dy = sy - geom.element_depth_m;
            let d_rx = (dx * dx + dy * dy).sqrt() / geom.sound_speed_m_s;
            let idx = ((d_tx + d_rx) / geom.dt_s).round() as usize;
            rf[[e, idx]] = scale;
        }
        rf
    }

    #[test]
    fn coherent_das_envelope_localizes_impulse() {
        let g = TripletPulseGeometry::stage1().unwrap();
        let rf = impulse_rf(&g, 19.0e-3, 20.0e-3, 1.0);
        let xs: Vec<f64> = (0..57).map(|i| 5.0e-3 + i as f64 * 0.5e-3).collect();
        let ys: Vec<f64> = (0..49).map(|i| 12.0e-3 + i as f64 * 0.5e-3).collect();
        let img = coherent_das_envelope(rf.view(), &g, &xs, &ys, 8).unwrap();
        // Source pixel: x = 19.0 mm is index 28, y = 20 mm is index 16.
        assert!((img[[16, 28]] - 1.0).abs() < 0.05, "peak {}", img[[16, 28]]);
    }

    #[test]
    fn known_ratio_images_give_known_contrast() {
        // img1 = 4 * img3 before, img1 = 2 * img3 after, uniformly:
        // reductions 4 and 2, contrast = 20*log10(2) = 6.0206 dB.
        let xs = vec![0.0, 1.0e-3];
        let ys = vec![0.0, 1.0e-3];
        let roi = RectRoi {
            x_lo_m: -1.0e-3,
            x_hi_m: 2.0e-3,
            y_lo_m: -1.0e-3,
            y_hi_m: 2.0e-3,
        };
        let b3 = Array2::from_elem([2, 2], 0.5);
        let b1 = Array2::from_elem([2, 2], 2.0);
        let a3 = Array2::from_elem([2, 2], 0.25);
        let a1 = Array2::from_elem([2, 2], 0.5);
        let r =
            kuji_contrast_from_images(b1.view(), b3.view(), a1.view(), a3.view(), &xs, &ys, &roi)
                .unwrap();
        assert_eq!(r.red_tissue, 4.0);
        assert_eq!(r.red_cavitation, 2.0);
        assert!((r.contrast_db - 20.0 * 2.0_f64.log10()).abs() < 1e-12);
    }

    #[test]
    fn ensemble_averages_reductions_before_contrast() {
        let results = [KujiReductions::new(4.0, 2.0), KujiReductions::new(2.0, 2.0)];
        let ens = kuji_ensemble(&results).unwrap();
        assert_eq!(ens.mean_red_tissue, 3.0);
        assert_eq!(ens.mean_red_cavitation, 2.0);
        assert!((ens.contrast_db - 20.0 * 1.5_f64.log10()).abs() < 1e-12);
        // Per-realisation contrasts 6.0206 and 0 dB: mean 3.0103,
        // sample SD 4.2568.
        assert!((ens.per_realization_mean_db - 10.0 * 2.0_f64.log10()).abs() < 1e-9);
        assert!((ens.per_realization_sd_db - 4.2568).abs() < 1e-3);
        let lower = kuji_ensemble(&[KujiReductions::new(2.0, 2.0)]).unwrap();
        let (fc, c) = optimal_cutoff(&[(2.5e6, ens), (3.0e6, lower)]).unwrap();
        assert_eq!(fc, 2.5e6);
        assert!(c > 0.0);
    }

    #[test]
    fn iwasaki_contrast_known_values() {
        let xs = vec![0.0, 1.0e-3, 2.0e-3];
        let ys = vec![0.0, 1.0e-3, 2.0e-3];
        let bg = RectRoi {
            x_lo_m: -1.0e-3,
            x_hi_m: 3.0e-3,
            y_lo_m: -1.0e-3,
            y_hi_m: 3.0e-3,
        };
        let mut frame = Array2::from_elem([3, 3], 1.0);
        frame[[1, 1]] = 10.0;
        // Background mean includes the peak pixel: (8*1 + 10)/9 = 2.
        let c = iwasaki_contrast(
            std::slice::from_ref(&frame),
            &xs,
            &ys,
            &bg,
            IwasakiBackgroundAveraging::Linear,
        )
        .unwrap();
        assert!((c - 20.0 * 5.0_f64.log10()).abs() < 1e-12, "contrast {c}");
        // Spatiotemporal max: a later frame's larger peak wins.
        let mut frame2 = Array2::from_elem([3, 3], 1.0);
        frame2[[0, 0]] = 20.0;
        let c2 = iwasaki_contrast(
            &[frame.clone(), frame2],
            &xs,
            &ys,
            &bg,
            IwasakiBackgroundAveraging::Linear,
        )
        .unwrap();
        assert!(
            c2 > c,
            "spatiotemporal max must raise contrast: {c2} vs {c}"
        );
    }

    #[test]
    fn noise_sigma_is_parameterised_by_inband_snr_not_trace_max() {
        // WS-C arithmetic: σ = echo_rms · 10^(−SNR/20).
        assert_eq!(inband_noise_sigma(2.0, 0.0), 2.0);
        let s33 = inband_noise_sigma(1.0, KUJI_EXPERIMENT_INBAND_SNR_DB);
        assert!((s33 - 10f64.powf(-33.0 / 20.0)).abs() < 1e-15, "σ33 {s33}");
        // B1's accidental DAS-level floor was ≈ −23 dB: the noise
        // RMS is then ≈ 14× the echo it was meant to perturb.
        let s_neg23 = inband_noise_sigma(1.0, -23.0);
        assert!((s_neg23 - 14.13).abs() < 0.01, "σ(−23 dB) {s_neg23}");
        // Monotone: more headroom, less noise.
        assert!(inband_noise_sigma(1.0, 40.0) < s33);
    }

    /// WS-C pure-noise asymptote: noise-only channel RF through the
    /// full metric pipeline (declared floor → filter → coherent
    /// DAS → ROI reduction) reads ≈ 2.40 linear — the value B1's
    /// floored headline reductions (2.44–2.50) were pinned to.
    /// Tolerance ±0.6: the asymptote is a statistic of the DAS
    /// noise field (finite grid/ROI realisation), not an identity.
    #[test]
    fn noise_only_rf_reads_the_pure_noise_asymptote() {
        let g = TripletPulseGeometry::stage1().unwrap();
        let n_samples = 1800;
        let zeros = Array2::zeros([g.n_elem(), n_samples]);
        let echo_rms = 1.0;
        let mk_phase = |seed: u64| {
            add_declared_noise_to_rf(
                zeros.view(),
                g.dt_s,
                3.0e6,
                KujiFilterShape::KujiRaisedCosine,
                true,
                echo_rms,
                &DeclaredNoiseFloor {
                    inband_snr_db: 0.0,
                    seed,
                },
            )
            .unwrap()
        };
        let ph0 = mk_phase(21_000);
        let ph120 = mk_phase(21_001);
        let ph240 = mk_phase(21_002);
        let xs: Vec<f64> = (0..9).map(|i| 15.0e-3 + i as f64 * 1.0e-3).collect();
        let ys: Vec<f64> = (0..9).map(|i| 16.0e-3 + i as f64 * 1.0e-3).collect();
        let (img1, img3) = kuji_images(
            PhaseRf {
                ph0: ph0.view(),
                ph120: ph120.view(),
                ph240: ph240.view(),
            },
            &g,
            &xs,
            &ys,
            KujiFilter {
                fc_hz: 3.0e6,
                shape: KujiFilterShape::KujiRaisedCosine,
                probe_band: true,
            },
            8,
        )
        .unwrap();
        let roi = RectRoi {
            x_lo_m: 14.0e-3,
            x_hi_m: 24.0e-3,
            y_lo_m: 15.0e-3,
            y_hi_m: 25.0e-3,
        };
        let red = roi_mean_reduction(img1.view(), img3.view(), &xs, &ys, &roi).unwrap();
        assert!(
            (red - KUJI_PURE_NOISE_REDUCTION_ASYMPTOTE).abs() < 0.6,
            "noise-only reduction {red} vs asymptote {KUJI_PURE_NOISE_REDUCTION_ASYMPTOTE}"
        );
    }
}
