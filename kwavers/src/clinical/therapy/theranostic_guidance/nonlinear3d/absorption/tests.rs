//! Unit tests for the fractional-Laplacian absorption operator: coefficient
//! magnitudes against Treeby-Cox 2010 Eq. 11, zero-attenuation short-circuit,
//! and the discrete-adjoint transpose identity `⟨A·v, w⟩ = ⟨v, Aᵀ·w⟩`.

use super::{AbsorptionBuilder, FractionalLaplacianAbsorption};

fn build_homogeneous_operator(
    n: usize,
    spacing_m: f64,
    dt_s: f64,
    c0: f64,
    alpha0_np_per_m_at_1mhz: f64,
    y: f64,
) -> FractionalLaplacianAbsorption {
    let cells = n * n * n;
    let speed = vec![c0; cells];
    let alpha = vec![alpha0_np_per_m_at_1mhz; cells];
    let y_field = vec![y; cells];
    FractionalLaplacianAbsorption::maybe_new(AbsorptionBuilder {
        n,
        spacing_m,
        dt_s,
        speed_m_s: &speed,
        attenuation_np_per_m_mhz: &alpha,
        attenuation_power_law_y: &y_field,
    })
    .expect("non-zero α₀ must yield an operator")
}

/// Construction sanity: τ and η have the expected sign and order of
/// magnitude for cortical bone (α₀ ≈ 150 Np/m at 1 MHz, y = 2) at
/// CFL-typical dt and water-like c.
#[test]
fn fractional_laplacian_absorption_builder_matches_treeby_cox_2010_coefficients() {
    let n = 8;
    let spacing_m = 1.0e-4;
    let c0 = 1500.0;
    let dt_s = 0.4 * spacing_m / (c0 * 3.0_f64.sqrt());
    let alpha0 = 149.7; // skull at 1 MHz, Np/m
    let y = 2.0;
    let op = build_homogeneous_operator(n, spacing_m, dt_s, c0, alpha0, y);

    // The FDTD wave-equation form (Treeby-Cox 2010 Eq. 11 multiplied
    // by c²) gives τ_FDTD = 2·α₀_ω·c^(y+1) with α₀_ω = α₀_f / ω_ref^y.
    let omega_ref = std::f64::consts::TAU * 1.0e6;
    let alpha0_omega = alpha0 / omega_ref.powf(y);
    let expected_tau = 2.0 * alpha0_omega * c0.powf(y + 1.0);
    let expected_dt_tau = dt_s * expected_tau;
    for &dt_tau in &op.dt_tau {
        let rel_err = (dt_tau - expected_dt_tau).abs() / expected_dt_tau.abs().max(1.0e-30);
        assert!(
            rel_err < 1.0e-9,
            "dt·τ mismatch: expected {expected_dt_tau}, got {dt_tau}",
        );
    }
}

/// `maybe_new` short-circuits to `None` when α₀ is identically zero
/// (the loss-free baseline). This preserves zero-cost behaviour in
/// the forward and adjoint paths.
#[test]
fn maybe_new_returns_none_for_zero_attenuation() {
    let n = 4;
    let cells = n * n * n;
    let speed = vec![1500.0; cells];
    let alpha = vec![0.0; cells];
    let y_field = vec![1.05; cells];
    let op = FractionalLaplacianAbsorption::maybe_new(AbsorptionBuilder {
        n,
        spacing_m: 1.0e-4,
        dt_s: 1.0e-8,
        speed_m_s: &speed,
        attenuation_np_per_m_mhz: &alpha,
        attenuation_power_law_y: &y_field,
    });
    assert!(op.is_none(), "α₀ ≡ 0 must short-circuit to None");
}

/// Self-adjointness sanity: `apply_transpose` is the transpose of the
/// `apply`-induced Jacobian. We verify `⟨A·v, w⟩ = ⟨v, Aᵀ·w⟩` on a
/// pair of random fields using two independent zero-init runs of
/// `apply` and `apply_transpose` so the prev-step cache does not
/// interfere with the inner products.
#[test]
fn apply_transpose_is_jacobian_transpose() {
    let n = 8;
    let cells = n * n * n;
    let spacing_m = 1.0e-4;
    let c0 = 1500.0;
    let dt_s = 0.4 * spacing_m / (c0 * 3.0_f64.sqrt());
    let alpha0 = 5.8; // soft-tissue Np/m at 1 MHz
    let y = 1.05;

    let mut op = build_homogeneous_operator(n, spacing_m, dt_s, c0, alpha0, y);

    // Random-ish probe fields (deterministic): use sin/cos products
    // with non-trivial phases to populate all frequencies.
    let v_curr: Vec<f64> = (0..cells)
        .map(|i| {
            let x = (i / (n * n)) as f64;
            let y = ((i / n) % n) as f64;
            let z = (i % n) as f64;
            (x * 0.31 + y * 0.71 + z * 1.13).sin()
        })
        .collect();
    let v_prev: Vec<f64> = (0..cells)
        .map(|i| {
            let x = (i / (n * n)) as f64;
            let y = ((i / n) % n) as f64;
            let z = (i % n) as f64;
            (x * 0.59 + y * 0.23 + z * 0.87).cos()
        })
        .collect();
    let w: Vec<f64> = (0..cells)
        .map(|i| {
            let x = (i / (n * n)) as f64;
            let y = ((i / n) % n) as f64;
            let z = (i % n) as f64;
            (x * 0.47 - y * 0.97 + z * 0.41).sin()
        })
        .collect();

    // Forward: compute Δp = apply(v_curr, v_prev, 0).
    let mut next = vec![0.0; cells];
    op.apply(&v_curr, &v_prev, &mut next);
    // `next` now equals the Jacobian-vector product J·[v_curr, v_prev]^T.

    let mut adj_curr = vec![0.0; cells];
    let mut adj_prev = vec![0.0; cells];
    op.reset(); // ensure transpose is independent of forward cache
    op.apply_transpose(&w, &mut adj_curr, &mut adj_prev);

    let lhs = next.iter().zip(w.iter()).map(|(a, b)| a * b).sum::<f64>();
    let rhs = v_curr
        .iter()
        .zip(adj_curr.iter())
        .map(|(a, b)| a * b)
        .sum::<f64>()
        + v_prev
            .iter()
            .zip(adj_prev.iter())
            .map(|(a, b)| a * b)
            .sum::<f64>();
    let rel_err = (lhs - rhs).abs() / lhs.abs().max(rhs.abs()).max(1.0e-30);
    assert!(
        rel_err < 1.0e-9,
        "transpose identity violated: ⟨Av, w⟩={lhs}, ⟨v, Aᵀw⟩={rhs}, rel_err={rel_err}"
    );
}
