use super::*;
use leto::Array3;
use num_complex::Complex64;

// ─── HighPass: exact mathematical properties ─────────────────────────────

/// Constant ensemble after HighPass is identically zero.
///
/// For ensemble [c, c, …, c] (N copies):
///   mean = c
///   filtered[n] = c − c = 0 for every n.
#[test]
fn wall_filter_highpass_constant_ensemble_outputs_zero() {
    let cfg = WallFilterConfig {
        filter_type: WallFilterType::HighPass,
        prf: 4e3,
    };
    let wf = WallFilter::new(cfg);
    let c = Complex64::new(3.5, -2.1);
    // shape: (ensemble=4, depths=3, beams=2)
    let iq = Array3::from_elem((4, 3, 2), c);
    let out = wf.apply(&iq.view()).unwrap();
    for v in out.iter() {
        assert!(
            v.norm() < 1e-12,
            "HighPass on constant ensemble: expected 0+0i, got {v}"
        );
    }
}

/// Alternating ensemble [+A, −A, +A, −A] has mean = 0, so HighPass preserves it exactly.
///
/// mean = (A − A + A − A) / 4 = 0
/// filtered[n] = s[n] − 0 = s[n]
#[test]
fn wall_filter_highpass_zero_mean_ensemble_is_unchanged() {
    let cfg = WallFilterConfig {
        filter_type: WallFilterType::HighPass,
        prf: 4e3,
    };
    let wf = WallFilter::new(cfg);
    let a = Complex64::new(1.0, 0.5);
    // (ensemble=4, depths=2, beams=2): alternating +a / −a
    let mut iq = Array3::zeros((4, 2, 2));
    for depth in 0..2 {
        for beam in 0..2 {
            iq[[0, depth, beam]] = a;
            iq[[1, depth, beam]] = -a;
            iq[[2, depth, beam]] = a;
            iq[[3, depth, beam]] = -a;
        }
    }
    let out = wf.apply(&iq.view()).unwrap();
    for (in_val, out_val) in iq.iter().zip(out.iter()) {
        assert!(
            (in_val - out_val).norm() < 1e-12,
            "HighPass on zero-mean ensemble: expected {in_val}, got {out_val}"
        );
    }
}

/// After HighPass the ensemble sum at every (depth, beam) is zero.
///
/// Algebraic identity: Σ(xₙ − mean) = Σxₙ − N · mean = 0.
/// Holds for any input, including non-uniform complex values.
#[test]
fn wall_filter_highpass_ensemble_sum_is_zero_for_arbitrary_input() {
    let cfg = WallFilterConfig {
        filter_type: WallFilterType::HighPass,
        prf: 4e3,
    };
    let wf = WallFilter::new(cfg);
    let ensemble_size = 6;
    let n_depths = 3;
    let n_beams = 2;
    let mut iq = Array3::zeros((ensemble_size, n_depths, n_beams));
    // Non-uniform values to ensure a nontrivial mean.
    for n in 0..ensemble_size {
        for d in 0..n_depths {
            for b in 0..n_beams {
                iq[[n, d, b]] = Complex64::new((n + d + b) as f64, (n * 2) as f64);
            }
        }
    }
    let out = wf.apply(&iq.view()).unwrap();
    for depth in 0..n_depths {
        for beam in 0..n_beams {
            let sum: Complex64 = (0..ensemble_size).map(|n| out[[n, depth, beam]]).sum();
            assert!(
                sum.norm() < 1e-10,
                "ensemble sum at ({depth},{beam}) = {sum:.2e}, expected 0"
            );
        }
    }
}

/// Polynomial order-2 filter zeroes a constant ensemble.
///
/// The constant signal lies in the polynomial subspace span{1, t, t²},
/// so the residual after orthogonal projection is exactly zero.
#[test]
fn wall_filter_polynomial_constant_ensemble_outputs_zero() {
    let cfg = WallFilterConfig {
        filter_type: WallFilterType::Polynomial { order: 2 },
        prf: 4e3,
    };
    let wf = WallFilter::new(cfg);
    let c = Complex64::new(-7.0, 4.2);
    let iq = Array3::from_elem((5, 2, 3), c);
    let out = wf.apply(&iq.view()).unwrap();
    for v in out.iter() {
        assert!(
            v.norm() < 1e-10,
            "Polynomial filter on constant ensemble: expected 0+0i, got {v}"
        );
    }
}

/// Polynomial order-1 filter zeroes a linear ramp ensemble.
///
/// A linear signal x[n] = a + b·t lies in span{1, t}, so order-1 polynomial
/// regression removes it exactly. This validates that the polynomial filter
/// actually uses the `order` parameter rather than reducing to DC removal.
#[test]
fn wall_filter_polynomial_linear_ramp_outputs_zero() {
    let cfg = WallFilterConfig {
        filter_type: WallFilterType::Polynomial { order: 1 },
        prf: 4e3,
    };
    let wf = WallFilter::new(cfg);
    let ensemble = 6;
    let mut iq = Array3::<Complex64>::zeros((ensemble, 1, 1));
    for n in 0..ensemble {
        // x[n] = (2.0 + 3.0·n) + i·(−1.0 + 0.5·n)
        iq[[n, 0, 0]] = Complex64::new(2.0 + 3.0 * n as f64, -1.0 + 0.5 * n as f64);
    }
    let out = wf.apply(&iq.view()).unwrap();
    for v in out.iter() {
        assert!(
            v.norm() < 1e-10,
            "Polynomial order-1 on linear ramp: expected 0+0i, got {v}"
        );
    }
}

/// Polynomial order-2 filter zeroes a quadratic ensemble.
///
/// A quadratic signal x[n] = a + b·t + c·t² lies in span{1, t, t²}, so
/// order-2 polynomial regression removes it exactly. Validates that the
/// projector handles each order correctly.
#[test]
fn wall_filter_polynomial_quadratic_outputs_zero() {
    let cfg = WallFilterConfig {
        filter_type: WallFilterType::Polynomial { order: 2 },
        prf: 4e3,
    };
    let wf = WallFilter::new(cfg);
    let ensemble = 8;
    let mut iq = Array3::<Complex64>::zeros((ensemble, 1, 1));
    for n in 0..ensemble {
        let nf = n as f64;
        iq[[n, 0, 0]] = Complex64::new(1.0 + 2.0 * nf + 0.5 * nf * nf, -2.0 - nf + 0.25 * nf * nf);
    }
    let out = wf.apply(&iq.view()).unwrap();
    for v in out.iter() {
        assert!(
            v.norm() < 1e-10,
            "Polynomial order-2 on quadratic: expected 0+0i, got {v}"
        );
    }
}

/// Polynomial order-1 filter does NOT zero a quadratic ensemble.
///
/// A quadratic signal is not in span{1, t}; the residual after order-1
/// regression must be non-zero. This validates that the polynomial filter
/// is genuinely order-dependent (not collapsing to a higher-order projection).
#[test]
fn wall_filter_polynomial_order1_leaves_quadratic_residual() {
    let cfg = WallFilterConfig {
        filter_type: WallFilterType::Polynomial { order: 1 },
        prf: 4e3,
    };
    let wf = WallFilter::new(cfg);
    let ensemble = 8;
    let mut iq = Array3::<Complex64>::zeros((ensemble, 1, 1));
    for n in 0..ensemble {
        let nf = n as f64;
        iq[[n, 0, 0]] = Complex64::new(nf * nf, 0.0);
    }
    let out = wf.apply(&iq.view()).unwrap();
    let total_energy: f64 = out.iter().map(|v| v.norm_sqr()).sum();
    assert!(
        total_energy > 1e-3,
        "Order-1 on quadratic should leave residual energy; got {total_energy}"
    );
}

/// IIR high-pass: constant DC input produces a transient that decays to zero.
///
/// For x[n] = c and one-pole HPF y[n] = α(y[n-1] + x[n] - x[n-1]) with
/// y[-1] = x[-1] = 0:
///   y[0] = α·c
///   y[1] = α²·c
///   y[n] = α^(n+1)·c
/// The steady-state response to DC is zero, but the transient is non-zero
/// — this is the correct high-pass behavior (Oppenheim & Schafer §8.3).
#[test]
fn wall_filter_iir_dc_input_decays_geometrically() {
    let prf = 4.0e3_f64;
    let cutoff = 100.0_f64;
    let cfg = WallFilterConfig {
        filter_type: WallFilterType::IIR {
            cutoff_frequency: cutoff,
        },
        prf,
    };
    let wf = WallFilter::new(cfg);
    let c = Complex64::new(2.0, -1.0);
    let ensemble = 8;
    let iq = Array3::from_elem((ensemble, 1, 1), c);
    let out = wf.apply(&iq.view()).unwrap();

    let alpha = (-2.0 * std::f64::consts::PI * cutoff / prf).exp();
    for n in 0..ensemble {
        let expected = alpha.powi((n as i32) + 1) * c;
        let actual = out[[n, 0, 0]];
        assert!(
            (actual - expected).norm() < 1e-12,
            "IIR DC transient sample {n}: expected {expected}, got {actual}"
        );
    }
}

/// IIR high-pass: alternating Nyquist-frequency input passes through with gain.
///
/// For x[n] = (-1)^n · c, the difference x[n] − x[n-1] alternates with
/// magnitude 2|c|, producing a response near the HPF passband.
#[test]
fn wall_filter_iir_alternating_input_is_passed() {
    let prf = 4.0e3_f64;
    let cfg = WallFilterConfig {
        filter_type: WallFilterType::IIR {
            cutoff_frequency: 100.0,
        },
        prf,
    };
    let wf = WallFilter::new(cfg);
    let a = Complex64::new(1.0, 0.0);
    let ensemble = 16;
    let mut iq = Array3::<Complex64>::zeros((ensemble, 1, 1));
    for n in 0..ensemble {
        iq[[n, 0, 0]] = if n.is_multiple_of(2) { a } else { -a };
    }
    let out = wf.apply(&iq.view()).unwrap();
    let dc_input_energy: f64 = iq.iter().map(|v| v.norm_sqr()).sum();
    let out_energy: f64 = out.iter().map(|v| v.norm_sqr()).sum();
    // For a Nyquist-frequency input through a HPF the steady-state gain
    // is large (>0.5 of input energy). The transient samples may be even
    // larger because of the leading edge.
    assert!(
        out_energy > 0.5 * dc_input_energy,
        "IIR should pass Nyquist input: in={dc_input_energy:.3}, out={out_energy:.3}"
    );
}
