use super::super::ViscoacousticGrid;
use super::super::ViscoacousticMemorySolver;
use super::ARMS;
use super::M_INF;
use super::RHO;
use kwavers_core::error::KwaversError;
use kwavers_core::test_support::assert_invalid_input;
use leto::Array3;

// ── Finite-domain contract (ADR 129): constructors reject every non-finite
// scalar and field element before allocating solver state, and the absorbing
// layer becomes fallible with overflow-free extent arithmetic. ──────────────

/// Structural invalid-input cases for scalar constructor parameters. Every
/// case must be rejected with `InvalidInput` — not `NaN`, which compares
/// false against `0.0` and slips a bare `v <= 0.0` guard.
const BAD_SCALARS: &[(&str, f64)] = &[
    ("NaN", f64::NAN),
    ("+inf", f64::INFINITY),
    ("-inf", f64::NEG_INFINITY),
    ("-1.0", -1.0),
];

/// Negative zero: `<= 0.0` rejects it, and positivity here is the physical
/// requirement, so the guard must keep rejecting it. (Tracked separately
/// from `BAD_SCALARS` because the *reason* differs: it is finite but not
/// positive.)
const NEG_ZERO: f64 = -0.0;

/// Construction validation and the `GeneralizedMaxwellModel` convenience path.
#[test]
fn construction_validates_and_accepts_model() {
    use kwavers_medium::viscoelastic::GeneralizedMaxwellModel;

    assert!(ViscoacousticMemorySolver::new_1d(0, 1e-4, 1e-8, RHO, M_INF, &[]).is_err());
    assert!(ViscoacousticMemorySolver::new(
        ViscoacousticGrid::new(8, 8, 0, 1e-4, 1e-4, 1e-4, 1e-8),
        RHO,
        M_INF,
        &[]
    )
    .is_err());
    assert!(ViscoacousticMemorySolver::new_1d(64, -1.0, 1e-8, RHO, M_INF, &[]).is_err());
    assert!(
        ViscoacousticMemorySolver::new_1d(64, 1e-4, 1e-8, RHO, M_INF, &[(-1.0, 1e-7)]).is_err()
    );

    let model =
        GeneralizedMaxwellModel::power_law(M_INF, 2.0e8, 1.0e5, 2.0e6, 6, 1.3, RHO).unwrap();
    let solver = ViscoacousticMemorySolver::from_generalized_maxwell(
        &model,
        ViscoacousticGrid::new(16, 16, 16, 1e-4, 1e-4, 1e-4, 1e-8),
    )
    .expect("model-backed solver");
    assert!(solver.unrelaxed_speed() > (M_INF / RHO).sqrt());
}

/// The finite-domain boundary tests. Every valid parameter vector stays
/// accepted (the finite-domain contract changes no accepted finite value);
/// every injection of a bad scalar into any parameter slot is rejected.
#[test]
fn finite_domain_rejects_non_finite_and_non_positive_scalars() {
    let valid = |dx: f64, dy: f64, dz: f64, dt: f64, rho: f64, m_inf: f64| {
        ViscoacousticMemorySolver::new(
            ViscoacousticGrid::new(8, 1, 1, dx, dy, dz, dt),
            rho,
            m_inf,
            &ARMS,
        )
    };
    // Baseline: the valid vector is accepted.
    valid(1.0e-4, 1.0, 1.0, 1.0e-8, RHO, M_INF)
        .unwrap_or_else(|e| panic!("valid parameters must stay accepted: {e}"));
    // Every slot × every bad value is rejected.
    for &(name, v) in BAD_SCALARS {
        for slot in ["dx", "dy", "dz", "dt", "rho", "m_inf"] {
            let r = match slot {
                "dx" => valid(v, 1.0, 1.0, 1.0e-8, RHO, M_INF),
                "dy" => valid(1.0e-4, v, 1.0, 1.0e-8, RHO, M_INF),
                "dz" => valid(1.0, 1.0, v, 1.0e-8, RHO, M_INF),
                "dt" => valid(1.0e-4, 1.0, 1.0, v, RHO, M_INF),
                "rho" => valid(1.0e-4, 1.0, 1.0, 1.0e-8, v, M_INF),
                _ => valid(1.0e-4, 1.0, 1.0, 1.0e-8, RHO, v),
            };
            assert!(
                matches!(r, Err(KwaversError::InvalidInput(_))),
                "{slot} = {name} ({v}) must be rejected as InvalidInput"
            );
        }
    }
    // Negative zero in a spacing slot: finite, but not positive.
    let r = valid(NEG_ZERO, 1.0, 1.0, 1.0e-8, RHO, M_INF);
    assert!(
        matches!(r, Err(KwaversError::InvalidInput(_))),
        "dx = -0.0 must be rejected"
    );
}

#[test]
fn finite_domain_rejects_bad_relaxation_arm_scalars() {
    for &(name, v) in BAD_SCALARS {
        for slot in ["dm", "tau"] {
            let arms: &[(f64, f64)] = match slot {
                "dm" => &[(v, 3.2e-7)],
                _ => &[(1.5e8, v)],
            };
            let r = ViscoacousticMemorySolver::new_1d(8, 1.0e-4, 1.0e-8, RHO, M_INF, arms);
            assert!(
                matches!(r, Err(KwaversError::InvalidInput(_))),
                "arm {slot} = {name} ({v}) must be rejected as InvalidInput"
            );
        }
    }
}

#[test]
fn finite_domain_rejects_non_finite_field_elements() {
    let n = 8usize;
    let shape = [n, 1, 1];
    let ones = Array3::from_elem(shape, 1.0);
    for &(name, v) in BAD_SCALARS {
        // ρ and M_∞ fields: every element must be positive finite.
        let mut bad_rho = Array3::from_elem(shape, RHO);
        bad_rho[[3, 0, 0]] = v;
        let r = ViscoacousticMemorySolver::new_heterogeneous(
            ViscoacousticGrid::new(n, 1, 1, 1.0e-4, 1.0, 1.0, 1.0e-8),
            &bad_rho,
            &ones,
            &[],
        );
        assert!(
            matches!(r, Err(KwaversError::InvalidInput(_))),
            "ρ field element {name} ({v}) must be rejected"
        );

        let mut bad_m = Array3::from_elem(shape, M_INF);
        bad_m[[3, 0, 0]] = v;
        let r = ViscoacousticMemorySolver::new_heterogeneous(
            ViscoacousticGrid::new(n, 1, 1, 1.0e-4, 1.0, 1.0, 1.0e-8),
            &ones,
            &bad_m,
            &[],
        );
        assert!(
            matches!(r, Err(KwaversError::InvalidInput(_))),
            "M_∞ field element {name} ({v}) must be rejected"
        );

        // τ arm field: must be positive finite.
        let mut bad_tau = Array3::from_elem(shape, 3.2e-7);
        bad_tau[[3, 0, 0]] = v;
        let dm = Array3::from_elem(shape, 1.5e8);
        let r = ViscoacousticMemorySolver::new_heterogeneous(
            ViscoacousticGrid::new(n, 1, 1, 1.0e-4, 1.0, 1.0, 1.0e-8),
            &ones,
            &ones,
            &[(dm, bad_tau)],
        );
        assert!(
            matches!(r, Err(KwaversError::InvalidInput(_))),
            "τ field element {name} ({v}) must be rejected"
        );
    }
    // ΔM field admits exactly zero (lossless voxel) but not negative.
    let dm_zero = Array3::from_elem(shape, 0.0);
    let r = ViscoacousticMemorySolver::new_heterogeneous(
        ViscoacousticGrid::new(n, 1, 1, 1.0e-4, 1.0, 1.0, 1.0e-8),
        &ones,
        &ones,
        &[(dm_zero.clone(), ones.clone())],
    );
    r.expect("ΔM = 0 is a lossless voxel and must stay accepted");
    let mut dm_neg = dm_zero;
    dm_neg[[2, 0, 0]] = -1.0;
    let r = ViscoacousticMemorySolver::new_heterogeneous(
        ViscoacousticGrid::new(n, 1, 1, 1.0e-4, 1.0, 1.0, 1.0e-8),
        &ones.clone(),
        &ones.clone(),
        &[(dm_neg, ones)],
    );
    assert_invalid_input(r, "finite ΔM≥0");
}

#[test]
fn absorbing_layer_rejects_non_finite_gamma_and_preserves_state() {
    let n = 256;
    let dx = 1.0e-4;
    let dt = 8.0e-9;
    let mut s = ViscoacousticMemorySolver::new_1d(n, dx, dt, RHO, M_INF, &ARMS).unwrap();
    s.enable_absorbing_layer(48, 2.0e6).unwrap();
    let baseline = s.clone();

    // A non-finite gamma is rejected before the retained decay field is
    // replaced: the solver keeps stepping the previously configured layer.
    // (A negative *finite* gamma remains the established disable path —
    // positivity is a caller concern, finiteness is the domain contract.)
    for &(name, g) in BAD_SCALARS.iter().take(3) {
        let r = s.enable_absorbing_layer(48, g);
        assert!(
            matches!(r, Err(KwaversError::InvalidInput(_))),
            "gamma_max = {name} ({g}) must be rejected"
        );
        assert_eq!(s.damping_decay_state(), baseline.damping_decay_state());
    }
    // The valid path still configures, and thickness = 0 still disables.
    s.enable_absorbing_layer(64, 1.0e6).unwrap();
    assert!(s.damping_decay_state().is_some());
    s.enable_absorbing_layer(0, 1.0e6).unwrap();
    assert!(s.damping_decay_state().is_none());
}

#[test]
fn absorbing_layer_extreme_thickness_never_panics() {
    let n = 8;
    let mut s = ViscoacousticMemorySolver::new_1d(n, 1.0e-4, 1.0e-8, RHO, M_INF, &[]).unwrap();
    // Before the overflow-free extent test, `2 * thickness` wrapped for these
    // inputs, the guard read false, and `n - thickness` underflowed. None of
    // these can be hosted by an 8-cell axis, so the layer stays cleared.
    for thickness in [usize::MAX, usize::MAX - 1, n, n + 1, 5, 4] {
        let r = s.enable_absorbing_layer(thickness, 2.0e6);
        r.expect("an 8-cell axis accepts every thickness by clearing the layer");
        assert!(
            s.damping_decay_state().is_none(),
            "an 8-cell axis cannot host thickness {thickness}"
        );
    }
    // And a genuinely hostable thickness still configures (2 < 8/2).
    s.enable_absorbing_layer(2, 2.0e6).unwrap();
    assert!(s.damping_decay_state().is_some());
}
