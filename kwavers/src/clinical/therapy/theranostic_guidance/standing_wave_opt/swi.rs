//! Standing Wave Index (SWI) computation via windowed spectral detection.
//!
//! # Theory
//!
//! For a 1D intensity profile `I(x) = |p(x, y_focus)|²` along the propagation
//! axis, a single-reflection standing wave has the Fourier decomposition:
//!
//! ```text
//! I(x) = I₀ [1 + |Γ|² + 2|Γ| cos(2kx + arg Γ)]
//! ```
//!
//! The DC component is `I₀(1 + |Γ|²)` and the standing-wave component at
//! spatial frequency `k_sw = 2k = 4πf/c` has amplitude `I₀ |Γ|`.  Therefore:
//!
//! ```text
//! SWI = 2 |DFT{I}[k_sw]| / DFT{I}[0]  ≈  2|Γ|  for  |Γ| ≪ 1
//! ```
//!
//! This gives SWI = 0 for a pure travelling wave and SWI = 1 for a perfect
//! standing wave (|Γ| = 1), matching the voltage standing-wave ratio (VSWR).
//!
//! DFT is evaluated via explicit summation over a ±15% window around the
//! expected standing-wave bin, avoiding a full FFT dependency for a single
//! frequency component.

use ndarray::{Array2, Zip};

use super::config::StandingWaveOptConfig;
use crate::core::constants::numerical::TWO_PI;

// ---------------------------------------------------------------------------
// Complex field superposition
// ---------------------------------------------------------------------------

/// Reconstruct the complex pressure field from Green's function superposition.
///
/// Implements the Born/RTM imaging operator:
/// `p(x,y; φ) = Σ_i exp(iφ_i) G_i(x,y)`.
///
/// Returns `(p_re, p_im)` each of shape `(nx, ny)`.
pub(super) fn superpose(
    g_re: &[Array2<f64>],
    g_im: &[Array2<f64>],
    phases: &[f64],
) -> (Array2<f64>, Array2<f64>) {
    let nx = g_re[0].nrows();
    let ny = g_re[0].ncols();
    let mut p_re = Array2::<f64>::zeros((nx, ny));
    let mut p_im = Array2::<f64>::zeros((nx, ny));
    for ((&phi, gre), gim) in phases.iter().zip(g_re).zip(g_im) {
        let c = phi.cos();
        let s = phi.sin();
        Zip::from(&mut p_re)
            .and(&mut p_im)
            .and(gre)
            .and(gim)
            .for_each(|pr, pi, &gr, &gi| {
                *pr += c * gr - s * gi;
                *pi += s * gr + c * gi;
            });
    }
    (p_re, p_im)
}

// ---------------------------------------------------------------------------
// SWI spectral detection
// ---------------------------------------------------------------------------

/// Windowed DFT power at bin index `k` (0-based, real-valued input).
fn dft_amplitude(x: &[f64], k: f64) -> f64 {
    let n = x.len() as f64;
    let omega = TWO_PI * k / n;
    let (re, im) = x.iter().enumerate().fold((0.0, 0.0), |(re, im), (j, &v)| {
        (
            re + v * (omega * j as f64).cos(),
            im - v * (omega * j as f64).sin(),
        )
    });
    (re * re + im * im).sqrt()
}

/// Compute the SWI from a 1D intensity profile along the propagation axis.
///
/// Searches a ±15% frequency window around the expected standing-wave bin
/// `k_sw = N / (c / (2 f₀ Δx))` for robustness against numerical dispersion
/// and domain-truncation effects.
pub(super) fn swi_from_profile(intensity: &[f64], config: &StandingWaveOptConfig) -> f64 {
    let n = intensity.len();
    if n < 8 {
        return 0.0;
    }
    let dc: f64 = intensity.iter().sum();
    if dc <= 0.0 {
        return 0.0;
    }
    // Standing-wave fringe period = λ/2 = c/(2f) in physical units.
    // In grid cells: period_cells = c_ref / (2 f₀ dx).
    let period_cells = config.c_ref_m_s / (2.0 * config.frequency_hz * config.dx_m);
    let k_center = n as f64 / period_cells;
    let window = (k_center * 0.15).max(1.0);
    let k_lo = ((k_center - window).round() as usize).max(1);
    let k_hi = ((k_center + window).round() as usize).min(n / 2);
    let sw_amp = (k_lo..=k_hi)
        .map(|k| dft_amplitude(intensity, k as f64))
        .fold(f64::NEG_INFINITY, f64::max);
    // SWI = 2 |DFT{I}[k_sw]| / DFT{I}[0] (DC = unnormalised sum)
    (2.0 * sw_amp / dc).clamp(0.0, 1.0)
}

/// Extract the focal-axis intensity profile and compute SWI.
///
/// Averages `|p(x,y)|²` over a lateral band of half-width
/// `swi_axis_half_width` around `focus_y`, for `x ∈ [source_x, layer_x_start)`.
pub(super) fn compute_swi(
    p_re: &Array2<f64>,
    p_im: &Array2<f64>,
    config: &StandingWaveOptConfig,
) -> f64 {
    let profile = axial_intensity_profile(p_re, p_im, config);
    swi_from_profile(&profile, config)
}

/// Peak `|p|` in a `focal_radius_cells` neighbourhood of the focal point.
pub(super) fn compute_focal_pressure(
    p_re: &Array2<f64>,
    p_im: &Array2<f64>,
    config: &StandingWaveOptConfig,
) -> f64 {
    let nx = config.nx;
    let ny = config.ny;
    let r = config.focal_radius_cells;
    let x0 = config.focus_x.saturating_sub(r);
    let x1 = (config.focus_x + r + 1).min(nx);
    let y0 = config.focus_y.saturating_sub(r);
    let y1 = (config.focus_y + r + 1).min(ny);
    let mut peak = 0.0_f64;
    for xi in x0..x1 {
        for yi in y0..y1 {
            let mag = (p_re[[xi, yi]].powi(2) + p_im[[xi, yi]].powi(2)).sqrt();
            if mag > peak {
                peak = mag;
            }
        }
    }
    peak
}

/// Focal-axis intensity profile `I(x) = ⟨|p(x,·)|²⟩_y` for export/visualisation.
///
/// Averages laterally over `y ∈ [focus_y − half_width, focus_y + half_width]`.
pub(crate) fn axial_intensity_profile(
    p_re: &Array2<f64>,
    p_im: &Array2<f64>,
    config: &StandingWaveOptConfig,
) -> Vec<f64> {
    let ny = config.ny;
    let hw = config.swi_axis_half_width;
    let y0 = config.focus_y.saturating_sub(hw);
    let y1 = (config.focus_y + hw + 1).min(ny);
    let n_y = (y1 - y0).max(1) as f64;
    let x_range = config.source_x..config.layer_x_start.min(config.nx);
    x_range
        .map(|xi| {
            let sum: f64 = (y0..y1)
                .map(|yi| p_re[[xi, yi]].powi(2) + p_im[[xi, yi]].powi(2))
                .sum();
            sum / n_y
        })
        .collect()
}
