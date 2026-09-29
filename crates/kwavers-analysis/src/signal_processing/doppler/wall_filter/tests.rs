use super::*;
use eunomia::Complex64;
use leto::Array3;

// ─── HighPass: exact mathematical properties ─────────────────────────────

/// Constant ensemble after HighPass is identically zero.
///
/// For ensemble [c, c, …, c] (N copies):
///   mean = c
///   filtered`N` = c − c = 0 for every n.
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
/// filtered`N` = s`N` − 0 = s`N`
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
                "ensemble sum norm at ({depth},{beam}) = {:.2e}, expected 0",
                sum.norm()
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
/// A linear signal x`N` = a + b·t lies in span{1, t}, so order-1 polynomial
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
/// A quadratic signal x`N` = a + b·t + c·t² lies in span{1, t, t²}, so
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
/// For x`N` = c and one-pole HPF y`N` = α(y[n-1] + x`N` - x[n-1]) with
/// y[-1] = x[-1] = 0:
///   y[0] = α·c
///   y[1] = α²·c
///   y`N` = α^(n+1)·c
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
/// For x`N` = (-1)^n · c, the difference x`N` − x[n-1] alternates with
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

// ─── ATLAS-ARCH-008: flat-array conversion characterization ──────────────
//
// polynomial_basis/polynomial_projector/project_out moved from nested
// Vec<Vec<f64>> to a contiguous leto::Array2<f64> (row-major, `[[row,
// col]]` indexed). The arithmetic and its order are unchanged — only the
// storage/indexing shape changed — so these tests pin the converted
// functions' outputs against independently computed oracles (not against
// the pre-conversion code itself, which used the identical operation
// sequence and would only prove the diff, not the math).

/// Vandermonde basis columns equal `t^i` computed via an independent route
/// (`f64::powi`, repeated squaring) rather than the implementation's
/// iterative-multiply accumulation. Each route accumulates O(i) relative
/// rounding error (Higham, *Accuracy and Stability of Numerical
/// Algorithms*, ch. 3: each IEEE-754 multiply has relative error <= 1 ULP),
/// so for i <= 3 the two routes agree to within a small constant number of
/// ULPs.
#[test]
fn polynomial_basis_matches_closed_form_power() {
    let ensemble_size = 7;
    let order = 3;
    let basis = polynomial_basis(ensemble_size, order);
    let denom = (ensemble_size - 1) as f64;
    for n in 0..ensemble_size {
        let t = n as f64 / denom;
        for i in 0..=order {
            let expected = t.powi(i32::try_from(i).unwrap());
            let actual = basis[[n, i]];
            let tol = 2.0 * (i as f64 + 1.0) * f64::EPSILON * expected.abs().max(1.0);
            assert!(
                (actual - expected).abs() <= tol,
                "basis[[{n},{i}]]: expected {expected}, got {actual} (tol {tol})"
            );
        }
    }
}

/// `projector` is the algebraic inverse of the Gram matrix it was built
/// from: `projector @ Gram(basis) == I`. The Gram matrix here is
/// recomputed independently of `polynomial_projector`'s internal one, so
/// this checks the actual inversion, not a tautology against the function's
/// own intermediate state.
#[test]
fn polynomial_projector_inverts_gram_matrix() {
    let ensemble_size = 9;
    let order = 3;
    let k = order + 1;
    let basis = polynomial_basis(ensemble_size, order);
    let projector = polynomial_projector(&basis);

    for i in 0..k {
        for l in 0..k {
            let mut acc = 0.0_f64;
            for j in 0..k {
                let gram_jl: f64 = (0..ensemble_size)
                    .map(|t| basis[[t, j]] * basis[[t, l]])
                    .sum();
                acc += projector[[i, j]] * gram_jl;
            }
            let expected = if i == l { 1.0 } else { 0.0 };
            // Gaussian elimination with partial pivoting on a k=4
            // well-conditioned Vandermonde Gram matrix (entries scale with
            // ensemble_size) accumulates O(k) rounding steps; bound the
            // residual generously against the matrix scale.
            let tol = 1e2 * f64::EPSILON * ensemble_size as f64;
            assert!(
                (acc - expected).abs() <= tol,
                "(projector * gram)[{i}][{l}] = {acc}, expected {expected} (tol {tol})"
            );
        }
    }
}

/// Order-0 (`k=1`) degenerates to mean-removal: `Gram = [[n]]`, so the
/// projector is exactly `[[1/n]]` — a single IEEE-754 division, bit-exact.
#[test]
fn polynomial_projector_order_zero_is_reciprocal_count() {
    let ensemble_size = 5;
    let basis = polynomial_basis(ensemble_size, 0);
    let projector = polynomial_projector(&basis);
    let expected = 1.0 / ensemble_size as f64;
    assert_eq!(
        projector[[0, 0]],
        expected,
        "order-0 projector must equal 1/n exactly"
    );
}

/// `project_out`'s Array2-indexed hot loop (`basis[[t,i]]`,
/// `projector[[i,j]]`) matches a closed-form 2x2 normal-equations solve —
/// an oracle independent of `polynomial_projector`'s Gaussian-elimination
/// code path. The signal is quadratic in `t` (not linear), so an order-1
/// fit leaves a genuine, non-zero residual to compare.
#[test]
fn project_out_matches_closed_form_linear_least_squares() {
    let ensemble_size = 5;
    let order = 1;
    let basis = polynomial_basis(ensemble_size, order);
    let projector = polynomial_projector(&basis);

    let denom = (ensemble_size - 1) as f64;
    let ts: Vec<f64> = (0..ensemble_size).map(|n| n as f64 / denom).collect();
    let signal: Vec<Complex64> = ts
        .iter()
        .map(|&t| Complex64::new(1.0 + 2.0 * t * t, 0.5 - t * t * t))
        .collect();

    // Closed-form 2x2 normal-equations solve: coeffs = (VᵀV)⁻¹ Vᵀx.
    let s0 = ensemble_size as f64;
    let s1: f64 = ts.iter().sum();
    let s2: f64 = ts.iter().map(|t| t * t).sum();
    let det = s0 * s2 - s1 * s1;
    let vt_x0: Complex64 = signal.iter().copied().sum();
    let vt_x1: Complex64 = signal.iter().zip(&ts).map(|(s, &t)| *s * t).sum();
    let a0 = (s2 * vt_x0 - s1 * vt_x1) / det;
    let a1 = (s0 * vt_x1 - s1 * vt_x0) / det;

    let residual = project_out(&signal, &basis, &projector);
    for (n, (&t, r)) in ts.iter().zip(residual.iter()).enumerate() {
        let expected = signal[n] - (a0 + a1 * t);
        assert!(
            (r - expected).norm() < 1e-9,
            "project_out[{n}]: expected {expected}, got {r}"
        );
    }
}

/// `polynomial_basis` is bit-for-bit identical to the pre-conversion
/// `Vec<Vec<f64>>` algorithm (same iterative `pow *= t` accumulation, same
/// order) reimplemented here directly from the ATLAS-ARCH-008 commit
/// message's description of the prior code, not merely close under an
/// epsilon. Only the storage container changed.
#[test]
fn polynomial_basis_is_bitwise_identical_to_pre_conversion_algorithm() {
    let ensemble_size: usize = 6;
    let order: usize = 2;
    let k = order + 1;
    let denom = (ensemble_size.saturating_sub(1)).max(1) as f64;

    // Pre-conversion algorithm, reproduced verbatim into Vec<Vec<f64>>.
    let reference: Vec<Vec<f64>> = (0..ensemble_size)
        .map(|n| {
            let t = n as f64 / denom;
            let mut row = Vec::with_capacity(k);
            let mut pow = 1.0;
            for _ in 0..k {
                row.push(pow);
                pow *= t;
            }
            row
        })
        .collect();

    let basis = polynomial_basis(ensemble_size, order);
    for n in 0..ensemble_size {
        for i in 0..k {
            assert_eq!(
                basis[[n, i]],
                reference[n][i],
                "basis[[{n},{i}]] must be bit-identical to the pre-conversion value"
            );
        }
    }
}
