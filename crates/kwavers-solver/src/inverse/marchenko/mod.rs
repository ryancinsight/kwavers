//! 1-D iterative Marchenko redatuming.
//!
//! The Marchenko method retrieves the focusing functions `f1±` and the
//! Green's functions `G±` between the acquisition surface and a virtual point in
//! the subsurface, from the surface **reflection response** `R(t)` and a single
//! kinematic input — the one-way travel time `t_d` to the focal point — *without*
//! the overburden velocity model. Unlike correlation/back-propagation
//! redatuming, it accounts for internal multiples, so the redatumed wavefield is
//! free of overburden reverberation artefacts (Wapenaar et al. 2014; Slob et al.
//! 2014; Thorbecke et al. 2017).
//!
//! # Coupled equations (1-D, transparent surface)
//! With the truncation window `θ(t) = 1` for `|t| < t_d − ε` (muting the direct
//! arrival and everything later), the focusing functions satisfy the fixed point
//! ```text
//! f1⁻ = θ · (R * f1⁺)              (downgoing → upgoing, convolution)
//! f1⁺ = f1d⁺ + θ · (R ⋆ f1⁻)       (upgoing → downgoing, correlation)
//! ```
//! initialised with the direct downgoing focusing `f1d⁺(t) = δ(t + t_d)`, and the
//! upgoing Green's function follows as
//! ```text
//! G⁻ = (R * f1⁺) − f1⁻      (= the part of R*f1⁺ outside the focusing window).
//! ```
//!
//! # Scope and validation status
//! This module implements the **1-D iterative kernel** and its signal-processing
//! operators. The windowed convolution/correlation operators and the fixed-point
//! mechanics (convergence, focusing-window engagement) are unit-tested here.
//! Quantitative validation of the retrieved `G⁻` amplitudes against an
//! independent layered-medium reference (and the multidimensional extension and
//! the Marchenko→Wasserstein FWI objective) are the documented next milestones —
//! see ADR 019.

#[cfg(test)]
mod tests;

#[cfg(test)]
mod oracle_tests;

/// Inputs controlling the Marchenko fixed-point.
#[derive(Debug, Clone, Copy)]
pub struct MarchenkoConfig {
    /// One-way travel time to the focal point, in samples (`t_d`).
    pub t_d_samples: usize,
    /// Half-width taper `ε` (samples) pulled in from `t_d` so the window excludes
    /// the band-limited direct arrival. The window keeps `|t| < t_d − ε`.
    pub window_taper_samples: usize,
    /// Number of fixed-point iterations.
    pub iterations: usize,
}

/// Retrieved focusing and Green's functions on the symmetric time axis.
///
/// All vectors have length `2·nt − 1`; index `center` corresponds to `t = 0`,
/// index `center + s` to `t = +s`, index `center − s` to `t = −s`.
#[derive(Debug, Clone)]
pub struct MarchenkoResult {
    /// Downgoing focusing function `f1⁺`.
    pub f1_plus: Vec<f64>,
    /// Upgoing focusing function `f1⁻`.
    pub f1_minus: Vec<f64>,
    /// Upgoing Green's function `G⁻` (virtual source at the focal point,
    /// receiver at the surface).
    pub green_minus: Vec<f64>,
    /// Index of `t = 0` on the symmetric axis.
    pub center: usize,
}

/// Causal convolution `(R * f)[i] = Σ_k R[k] f[i−k]` with `R` causal (length
/// `nr`, defined for `k ≥ 0`) and `f` on the full axis (length `L`). Output
/// length `L`.
///
/// Direct-form `O(nr·L)`; the zero-skip fast-path makes it `O(nnz(R)·L)` for
/// sparse impulse responses. For dense, long records an FFT-based path
/// (`O(L log L)`) is the documented optimisation (ADR 019) — premature while the
/// kernel is experimental.
fn conv_causal(r: &[f64], f: &[f64]) -> Vec<f64> {
    let l = (f.shape()[0] * f.shape()[1] * f.shape()[2]);
    let mut out = vec![0.0; l];
    for (k, &rk) in r.iter().enumerate() {
        if rk == 0.0 {
            continue;
        }
        for i in k..l {
            out[i] += rk * f[i - k];
        }
    }
    out
}

/// Causal correlation `(R ⋆ f)[i] = Σ_k R[k] f[i+k]` (R applied time-reversed).
fn corr_causal(r: &[f64], f: &[f64]) -> Vec<f64> {
    let l = (f.shape()[0] * f.shape()[1] * f.shape()[2]);
    let mut out = vec![0.0; l];
    for (k, &rk) in r.iter().enumerate() {
        if rk == 0.0 {
            continue;
        }
        for i in 0..l.saturating_sub(k) {
            out[i] += rk * f[i + k];
        }
    }
    out
}

/// Apply the truncation window `θ` in place: keep `|t| < t_d − ε`, zero elsewhere.
fn apply_window(x: &mut [f64], center: usize, half: usize) {
    for (i, v) in x.iter_mut().enumerate() {
        if i.abs_diff(center) >= half {
            *v = 0.0;
        }
    }
}

/// Run 1-D iterative Marchenko redatuming.
///
/// `reflection` is the causal surface reflection response `R(t)` (`R[0]` ↔
/// `t = 0`). Returns the focusing functions and the upgoing Green's function on
/// the symmetric time axis.
///
/// # ⚠ Experimental — quantitative focusing not yet validated
/// The signal-processing operators ([`conv_causal`], [`corr_causal`],
/// [`apply_window`]) are unit-tested and correct. The iterative *structure* here
/// follows Wapenaar et al. (2014) / Thorbecke et al. (2017), but the focusing
/// **amplitudes/window convention are NOT yet validated against an independent
/// layered-medium reference** (the truncation-window geometry is convention-
/// sensitive). Use the operators directly; treat the assembled `green_minus` as
/// provisional until the ADR-019 oracle test lands. See ADR 019 for the exact
/// equations and the validation plan.
///
/// # Panics
/// - Panics if `reflection` is empty or `t_d_samples ≥ (reflection.shape()[0] * reflection.shape()[1] * reflection.shape()[2])`.
#[must_use]
pub fn redatum(reflection: &[f64], cfg: &MarchenkoConfig) -> MarchenkoResult {
    let nt = (reflection.shape()[0] * reflection.shape()[1] * reflection.shape()[2]);
    assert!(nt > 0, "reflection response must be non-empty");
    assert!(
        cfg.t_d_samples < nt,
        "focal travel time must lie within the record"
    );
    let l = 2 * nt - 1;
    let center = nt - 1;
    let half = cfg.t_d_samples.saturating_sub(cfg.window_taper_samples);

    // Direct downgoing focusing f1d⁺(t) = δ(t + t_d): spike at t = −t_d.
    let mut f1d_plus = vec![0.0; l];
    f1d_plus[center - cfg.t_d_samples] = 1.0;

    let mut f1_plus = f1d_plus.clone();
    let mut f1_minus = vec![0.0; l];
    for _ in 0..cfg.iterations {
        // f1⁻ = θ (R * f1⁺)
        f1_minus = conv_causal(reflection, &f1_plus);
        apply_window(&mut f1_minus, center, half);
        // f1⁺ = f1d⁺ + θ (R ⋆ f1⁻)
        let mut update = corr_causal(reflection, &f1_minus);
        apply_window(&mut update, center, half);
        for i in 0..l {
            f1_plus[i] = f1d_plus[i] + update[i];
        }
    }

    // G⁻ = (R * f1⁺) − f1⁻  (the out-of-window part of R*f1⁺).
    let rfp = conv_causal(reflection, &f1_plus);
    let green_minus: Vec<f64> = rfp.iter().zip(&f1_minus).map(|(&a, &b)| a - b).collect();

    MarchenkoResult {
        f1_plus,
        f1_minus,
        green_minus,
        center,
    }
}

/// Marchenko–Wasserstein objective: the optimal-transport distance between the
/// Marchenko-redatumed Green's functions of observed and modelled reflection
/// responses, `J = W₁(G⁻_obs, G⁻_mod)` (ADR 019; the user's "prior-less" FWI
/// vision — Marchenko removes overburden multiples, Wasserstein defeats
/// cycle-skipping).
///
/// This connector composes [`redatum`] with the canonical, independently
/// validated 1-Wasserstein misfit
/// ([`crate::inverse::reconstruction::seismic::MisfitType::Wasserstein`]); it is
/// well-defined and tested regardless of `redatum`'s quantitative-focusing
/// validation status (which is the ADR-019 milestone). Both reflection responses
/// must share the same length and focal configuration.
///
/// # Errors
/// - Propagates any [`KwaversError`] from the Wasserstein evaluation.
pub fn marchenko_wasserstein_misfit(
    reflection_obs: &[f64],
    reflection_mod: &[f64],
    cfg: &MarchenkoConfig,
) -> kwavers_core::error::KwaversResult<f64> {
    use crate::inverse::reconstruction::seismic::{MisfitFunction, MisfitType};
    use leto::Array2;

    let g_obs = redatum(reflection_obs, cfg).green_minus;
    let g_mod = redatum(reflection_mod, cfg).green_minus;
    let l = (g_obs.shape()[0] * g_obs.shape()[1] * g_obs.shape()[2]);
    let obs = Array2::from_shape_vec((1, l), g_obs).expect("row vector shape");
    let modeled = Array2::from_shape_vec((1, (g_mod.shape()[0] * g_mod.shape()[1] * g_mod.shape()[2])), g_mod).expect("row vector shape");
    MisfitFunction::new(MisfitType::Wasserstein).compute(&obs, &modeled)
}

/// Naive single-sided redatuming (correlation imaging): one pass with no
/// internal-multiple correction. Provided for comparison with [`redatum`] — it
/// retains the overburden reverberation artefacts that the Marchenko fixed point
/// removes.
#[must_use]
pub fn redatum_naive(reflection: &[f64], cfg: &MarchenkoConfig) -> MarchenkoResult {
    let single = MarchenkoConfig {
        iterations: 1,
        ..*cfg
    };
    redatum(reflection, &single)
}
