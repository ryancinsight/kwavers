//! Tests for the 1-D Marchenko kernel.
//!
//! The signal-processing operators are validated exactly. The assembled
//! `redatum` Green's function is exercised for runs-and-is-finite / non-trivial
//! behaviour only — its quantitative focusing is validated against a
//! layered-medium reference as a separate milestone (ADR 019), so no
//! amplitude/multiple-removal correctness is asserted here.

use super::{conv_causal, corr_causal, marchenko_wasserstein_misfit, redatum, MarchenkoConfig};

/// `conv_causal` and `corr_causal` match hand-computed sums on small arrays.
/// # Panics
/// - Panics on any assertion failure.
#[test]
fn operators_match_reference() {
    // R causal length 3; f on a length-7 axis.
    let r = [1.0, 0.5, -0.25];
    let f = [0.0, 0.0, 1.0, 0.0, 2.0, 0.0, 0.0];

    // conv: out[i] = Σ_k r[k] f[i-k].
    let c = conv_causal(&r, &f);
    // i=2: r0 f2 = 1; i=3: r1 f2 = 0.5; i=4: r0 f4 + r2 f2 = 2 + (-0.25) = 1.75;
    // i=5: r1 f4 = 1.0; i=6: r2 f4 = -0.5.
    assert!((c[2] - 1.0).abs() < 1e-12);
    assert!((c[3] - 0.5).abs() < 1e-12);
    assert!((c[4] - 1.75).abs() < 1e-12);
    assert!((c[5] - 1.0).abs() < 1e-12);
    assert!((c[6] + 0.5).abs() < 1e-12);

    // corr: out[i] = Σ_k r[k] f[i+k].
    let cc = corr_causal(&r, &f);
    // i=2: r0 f2 + r2 f4 = 1 + (-0.25)(2) = 0.5; i=0: r2 f2 = -0.25;
    // i=4: r0 f4 = 2.0.
    assert!((cc[2] - 0.5).abs() < 1e-12);
    assert!((cc[0] + 0.25).abs() < 1e-12);
    assert!((cc[4] - 2.0).abs() < 1e-12);
}

/// `conv_causal` is linear (a basic algebraic invariant the redatuming relies on).
/// # Panics
/// - Panics on any assertion failure.
#[test]
fn conv_is_linear() {
    let r = [0.3, -0.2, 0.1];
    let a = [0.0, 1.0, 0.0, 0.5, 0.0, 0.0];
    let b = [0.0, 0.0, 2.0, 0.0, -1.0, 0.0];
    let sum: Vec<f64> = a.iter().zip(&b).map(|(x, y)| x + y).collect();
    let lhs = conv_causal(&r, &sum);
    let ca = conv_causal(&r, &a);
    let cb = conv_causal(&r, &b);
    for i in 0..(lhs.len()) {
        assert!((lhs[i] - (ca[i] + cb[i])).abs() < 1e-12);
    }
}

/// `redatum` runs and assembles a finite Green's function whose energy lies in
/// the post-focal (`t ≥ t_d`) region (the transmitted response), on a synthetic
/// reflection response. (Quantitative focusing is validated separately — ADR 019.)
/// # Panics
/// - Panics on any assertion failure.
#[test]
fn redatum_runs_and_greens_function_is_finite() {
    let nt = 80;
    let mut r = vec![0.0; nt];
    r[16] = 0.4; // primary at the focal two-way time
    r[48] = -0.1; // a later (internal-multiple-like) event
    let cfg = MarchenkoConfig {
        t_d_samples: 8,
        window_taper_samples: 2,
        iterations: 5,
    };
    let res = redatum(&r, &cfg);

    assert!(
        res.green_minus.iter().all(|v| v.is_finite()) && res.f1_plus.iter().all(|v| v.is_finite()),
        "outputs must be finite"
    );
    // Energy of G⁻ must lie at/after the focal time (the upgoing transmitted
    // wavefield), not before it.
    let pre: f64 = res.green_minus[..res.center].iter().map(|v| v * v).sum();
    let post: f64 = res.green_minus[res.center..].iter().map(|v| v * v).sum();
    assert!(
        post > 0.0 && pre <= post,
        "Green's-function energy must be post-focal: pre={pre:e}, post={post:e}"
    );
}

/// The Marchenko–Wasserstein objective vanishes for identical reflection data and
/// is strictly positive when the modelled response differs — the well-posed
/// behaviour of a misfit functional (independent of redatum's quantitative
/// validation).
/// # Panics
/// - Panics on any assertion failure.
#[test]
fn marchenko_wasserstein_objective_is_well_posed() {
    let nt = 80;
    let mut r_obs = vec![0.0; nt];
    r_obs[16] = 0.4;
    r_obs[40] = 0.5;
    r_obs[56] = -0.08;
    let cfg = MarchenkoConfig {
        t_d_samples: 8,
        window_taper_samples: 2,
        iterations: 5,
    };

    // Identical data ⇒ zero objective.
    let self_dist = marchenko_wasserstein_misfit(&r_obs, &r_obs, &cfg).expect("self");
    assert!(self_dist < 1e-9, "self-distance must vanish: {self_dist:e}");

    // A different modelled response ⇒ strictly positive objective.
    let mut r_mod = r_obs.clone();
    r_mod[40] = 0.3; // weaker focal-depth reflector
    let dist = marchenko_wasserstein_misfit(&r_obs, &r_mod, &cfg).expect("dist");
    assert!(
        dist > 1e-6,
        "differing data must give a positive objective: {dist:e}"
    );
}
