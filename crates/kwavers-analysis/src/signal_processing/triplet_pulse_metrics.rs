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

mod filter;
mod noise;

pub use filter::{kuji_filter_trace, kuji_hp_response, KujiFilterShape};
pub use noise::{
    add_declared_noise_to_rf, inband_noise_sigma, DeclaredNoiseFloor,
    KUJI_EXPERIMENT_INBAND_SNR_DB, KUJI_PURE_NOISE_REDUCTION_ASYMPTOTE,
};

use super::triplet_pulse::TripletPulseGeometry;
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
mod tests;
