//! Oracle validation of 1-D Marchenko `redatum` against the (independently
//! validated) self-adjoint acoustic engine (ADR 016).
//!
//! A 1-D layered medium with a transparent (sponge) surface is forward-modelled
//! to produce both the surface reflection response `R` and the true upgoing
//! Green's function `G⁻` from a virtual source at the focal depth. Marchenko
//! redatuming of `R` must reproduce the true `G⁻` (up to the direct-transmission
//! amplitude, hence the scale-invariant comparison) and do so *better* than naive
//! single-sided redatuming, which retains overburden multiples.

use super::{redatum, redatum_naive, MarchenkoConfig};
use crate::inverse::fwi::time_domain::{FwiEngine, FwiGeometry, FwiProcessor};
use crate::inverse::seismic::parameters::FwiParameters;
use kwavers_grid::Grid;
use kwavers_source::{GridSource, SourceMode};
use ndarray::{Array2, Array3};

const DX: f64 = 1e-3;
const DT: f64 = 2e-7;
const RHO: f64 = 1000.0;
const SPONGE: usize = 16;
const WAVELET_LEN: usize = 24;

/// Two-sided quadratic edge sponge (transparent surface + absorbing bottom).
fn sponge(nx: usize, b_max: f64) -> Array3<f64> {
    let mut b = Array3::zeros((nx, 1, 1));
    for i in 0..nx {
        let from_edge = i.min(nx - 1 - i);
        if from_edge < SPONGE {
            let d = (SPONGE - from_edge) as f64 / SPONGE as f64;
            b[[i, 0, 0]] = b_max * d * d;
        }
    }
    b
}

/// Short band-limited wavelet.
fn wavelet(nt: usize) -> Array2<f64> {
    let mut s = Array2::zeros((1, nt));
    for t in 0..WAVELET_LEN.min(nt) {
        let phase = (t as f64) * 0.35;
        s[[0, t]] = (-phase * phase * 0.2).exp() * (2.0 * phase).sin();
    }
    s
}

/// Record the surface trace for a source at `src_cell`, receiver at `recv_cell`,
/// through the layered velocity `c`, with a transparent/absorbing sponge.
fn engine_trace(c: &[f64], src_cell: usize, recv_cell: usize, nt: usize) -> Vec<f64> {
    let nx = c.len();
    let grid = Grid::new(nx, 1, 1, DX, DX, DX).expect("grid");
    let mut model = Array3::zeros((nx, 1, 1));
    for (i, &ci) in c.iter().enumerate() {
        model[[i, 0, 0]] = ci;
    }
    let mut p_mask = Array3::zeros((nx, 1, 1));
    p_mask[[src_cell, 0, 0]] = 1.0;
    let source = GridSource {
        p0: None,
        u0: None,
        p_mask: Some(p_mask),
        p_signal: Some(wavelet(nt)),
        p_mode: SourceMode::Additive,
        u_mask: None,
        u_signal: None,
        u_mode: SourceMode::default(),
    };
    let mut sensor_mask = Array3::from_elem((nx, 1, 1), false);
    sensor_mask[[recv_cell, 0, 0]] = true;
    let geometry = FwiGeometry::new(source, sensor_mask);
    let params = FwiParameters {
        nt,
        dt: DT,
        frequency: 5e5,
        ..FwiParameters::default()
    };
    let c_max = c.iter().copied().fold(0.0_f64, f64::max);
    let processor = FwiProcessor::new(params)
        .with_engine(FwiEngine::SecondOrderSelfAdjoint)
        .with_self_adjoint_damping(sponge(nx, 4.0 / (RHO * c_max * SPONGE as f64 * DX)))
        .expect("damping");
    let synth = processor
        .generate_synthetic_data(&model, &geometry, &grid)
        .expect("forward");
    (0..nt).map(|t| synth[[0, t]]).collect()
}

/// Scale-invariant cosine similarity of two traces over `range`.
fn cosine(a: &[f64], b: &[f64], lo: usize, hi: usize) -> f64 {
    let mut dot = 0.0;
    let mut na = 0.0;
    let mut nb = 0.0;
    for i in lo..hi {
        dot += a[i] * b[i];
        na += a[i] * a[i];
        nb += b[i] * b[i];
    }
    if na <= 0.0 || nb <= 0.0 {
        return 0.0;
    }
    dot / (na.sqrt() * nb.sqrt())
}

/// **Validation harness / acceptance target for `redatum` (currently IGNORED).**
///
/// This is the oracle test that will retire the experimental status of
/// [`super::redatum`]. It builds `R` and the true `G⁻` from the validated SA
/// engine and requires Marchenko redatuming to reproduce `G⁻`.
///
/// Current empirical status (2026-06-08): `corr(Marchenko, true) ≈ corr(naive,
/// true) ≈ 0.14` — the iteration does not yet engage (coda ≈ 0 ⇒ Marchenko ==
/// naive) and the redatumed trace does not match the oracle. Identified work
/// before un-ignoring (ADR 019): (a) the truncation-window geometry (a symmetric
/// `|t| < t_d−ε` window collapses the coda update to zero); (b) the
/// convolution/correlation convention of the `f1⁺` update; (c) the time
/// referencing/alignment of `G⁻` on the symmetric axis vs the surface recording;
/// (d) amplitude normalisation by the direct transmission `T_d`. The harness is
/// kept (and compiled) so the fix is a matter of correcting the kernel until this
/// asserts.
/// # Panics
/// - Panics on any assertion failure or solve error.
#[test]
#[ignore = "redatum convention not yet reference-validated; see ADR 019 (corr≈0.14)"]
fn redatum_matches_engine_green_function() {
    let nx = 140usize;
    let s0 = SPONGE + 8; // acquisition surface (just inside the transparent edge)
    let zi = 60usize; // overburden interface
    let zf = 92usize; // focal depth
    let nt = 600usize;

    // Layered sound speed: c0 overburden, c1 between interfaces, c2 below focal.
    let (c0, c1, c2) = (1500.0, 1750.0, 2050.0);
    let mut c = vec![c0; nx];
    for ci in c.iter_mut().take(zf).skip(zi) {
        *ci = c1;
    }
    for ci in c.iter_mut().skip(zf) {
        *ci = c2;
    }

    // One-way travel time s0 → zf in record samples.
    let t_d_seconds = (zi - s0) as f64 * DX / c0 + (zf - zi) as f64 * DX / c1;
    let t_d = (t_d_seconds / DT).round() as usize;

    // Reflection response R: source & receiver at the surface; mute the direct
    // source wavelet (early samples) to isolate the upgoing reflections.
    let mut r = engine_trace(&c, s0, s0, nt);
    for v in r.iter_mut().take(WAVELET_LEN + 8) {
        *v = 0.0;
    }
    // True upgoing Green's function: virtual source at the focal depth.
    let g_true = engine_trace(&c, zf, s0, nt);

    let cfg = MarchenkoConfig {
        t_d_samples: t_d,
        window_taper_samples: WAVELET_LEN,
        iterations: 12,
    };
    let res = redatum(&r, &cfg);
    let naive = redatum_naive(&r, &cfg);

    // Compare on the symmetric axis post-focal window (t ≥ t_d), where G⁻ lives.
    // res arrays have length 2*nt-1, centre = nt-1; map true (causal, length nt)
    // onto that axis at indices centre..centre+nt.
    let center = res.center;
    let mut g_true_axis = vec![0.0; res.green_minus.len()];
    for t in 0..nt {
        g_true_axis[center + t] = g_true[t];
    }
    let lo = center + t_d.saturating_sub(WAVELET_LEN);
    let hi = (center + nt).min(res.green_minus.len());

    let corr_march = cosine(&res.green_minus, &g_true_axis, lo, hi);
    let corr_naive = cosine(&naive.green_minus, &g_true_axis, lo, hi);
    eprintln!(
        "t_d={t_d}  corr(Marchenko, true)={corr_march:.4}  corr(naive, true)={corr_naive:.4}"
    );

    assert!(
        corr_march > 0.85,
        "Marchenko G⁻ must match the engine's true G⁻ (cosine > 0.85); got {corr_march:.4}"
    );
    assert!(
        corr_march >= corr_naive - 1e-6,
        "Marchenko must be no worse than naive redatuming; \
         Marchenko={corr_march:.4}, naive={corr_naive:.4}"
    );
}
