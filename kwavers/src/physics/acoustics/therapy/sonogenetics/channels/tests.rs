use super::*;
use approx::assert_relative_eq;
use ndarray::Array3;

/// At Delta T = T_half, P_open must equal exactly 0.5.
/// # Panics
/// - Panics with `"MscLG22S must use Boltzmann model"`.
///
#[test]
fn test_boltzmann_half_activation() {
    let params = MechanoChannel::MscLG22S.canonical_params();
    let GatingModel::Boltzmann(ref bp) = params else {
        panic!("MscLG22S must use Boltzmann model");
    };
    let tension = Array3::from_elem((2, 2, 2), bp.half_tension_n_per_m);
    let p_open = boltzmann_p_open(&tension, bp, BODY_TEMP_K).unwrap();
    for &v in &p_open {
        assert_relative_eq!(v, 0.5, max_relative = 1e-12);
    }
}

/// Well below T_half, P_open must match the analytical closed-state scale.
/// # Panics
/// - Panics with `"expected Boltzmann"`.
///
#[test]
fn test_boltzmann_deep_closed() {
    let params = MechanoChannel::MscLG22S.canonical_params();
    let GatingModel::Boltzmann(ref bp) = params else {
        panic!("expected Boltzmann");
    };
    let tension = Array3::zeros((2, 2, 2));
    let p_open = boltzmann_p_open(&tension, bp, BODY_TEMP_K).unwrap();
    for &v in &p_open {
        assert!(
            v < 0.01,
            "P_open at zero tension should be < 1%, got {v:.3e}"
        );
    }
}

/// Well above T_half, P_open must be close to 1.
/// # Panics
/// - Panics with `"expected Boltzmann"`.
///
#[test]
fn test_boltzmann_deep_open() {
    let params = MechanoChannel::MscLG22S.canonical_params();
    let GatingModel::Boltzmann(ref bp) = params else {
        panic!("expected Boltzmann");
    };
    let tension = Array3::from_elem((2, 2, 2), 10.0 * bp.half_tension_n_per_m);
    let p_open = boltzmann_p_open(&tension, bp, BODY_TEMP_K).unwrap();
    for &v in &p_open {
        assert!(
            v > 1.0 - 1e-10,
            "P_open at 10*T_half should approach 1, got {v:.6}"
        );
    }
}

/// Boltzmann opening probability is monotonically increasing with tension.
/// # Panics
/// - Panics with `"expected Boltzmann"`.
///
#[test]
fn test_boltzmann_monotone() {
    let params = MechanoChannel::Piezo1.canonical_params();
    let GatingModel::Boltzmann(ref bp) = params else {
        panic!("expected Boltzmann");
    };
    let tensions: Vec<f64> = (0..10)
        .map(|i| i as f64 * bp.half_tension_n_per_m / 4.0)
        .collect();
    let mut prev = 0.0_f64;
    for &t in &tensions {
        let arr = Array3::from_elem((1, 1, 1), t);
        let p = boltzmann_p_open(&arr, bp, BODY_TEMP_K).unwrap()[[0, 0, 0]];
        assert!(
            p > prev,
            "P_open must increase with tension: {prev} -> {p} at t={t}"
        );
        prev = p;
    }
}

/// Absolute temperature must be positive.
/// # Panics
/// - Panics with `"expected Boltzmann"`.
///
#[test]
fn test_boltzmann_zero_temperature_is_error() {
    let params = MechanoChannel::MscLG22S.canonical_params();
    let GatingModel::Boltzmann(ref bp) = params else {
        panic!("expected Boltzmann");
    };
    let tension = Array3::zeros((2, 2, 2));
    assert!(boltzmann_p_open(&tension, bp, 0.0).is_err());
    assert!(boltzmann_p_open(&tension, bp, -1.0).is_err());
}

/// At P_rad = P_half, P_open = 0.5.
/// # Panics
/// - Panics with `"hsTRPA1 must use PressureThreshold model"`.
///
#[test]
fn test_pressure_threshold_half_activation() {
    let params = MechanoChannel::HsTrpa1.canonical_params();
    let GatingModel::PressureThreshold(ref pp) = params else {
        panic!("hsTRPA1 must use PressureThreshold model");
    };
    let p_rad = Array3::from_elem((2, 2, 2), pp.half_pressure_pa);
    let p_open = pressure_threshold_p_open(&p_rad, pp).unwrap();
    for &v in &p_open {
        assert_relative_eq!(v, 0.5, max_relative = 1e-12);
    }
}

/// Zero or negative sigmoid steepness is invalid.
/// # Panics
/// - Panics if an internal precondition is violated.
///
#[test]
fn test_pressure_threshold_zero_steepness_is_error() {
    let mut pp = PressureThresholdParams {
        half_pressure_pa: 35.6,
        steepness_pa: 0.0,
        single_channel_conductance_s: 60.0e-12,
        reversal_potential_v: 0.0,
    };
    let p_rad = Array3::zeros((2, 2, 2));
    assert!(pressure_threshold_p_open(&p_rad, &pp).is_err());
    pp.steepness_pa = -1.0;
    assert!(pressure_threshold_p_open(&p_rad, &pp).is_err());
}

/// I_ion = g * n * P_open * (E_rev - V_m).
#[test]
fn test_ion_current_analytical() {
    let p_open = Array3::from_elem((2, 2, 2), 0.5_f64);
    let current = ion_current(&p_open, 3.0e-9, 1000.0, -60.0e-3, 0.0);
    let expected = 3.0e-9 * 1000.0 * 0.5 * (0.0 - (-60.0e-3));
    for &v in &current {
        assert_relative_eq!(v, expected, max_relative = 1e-12);
    }
}

/// At P_open = 0, current is zero regardless of driving force.
/// # Panics
/// - Panics if an internal precondition is violated.
///
#[test]
fn test_ion_current_zero_when_closed() {
    let p_open = Array3::zeros((2, 2, 2));
    let current = ion_current(&p_open, 3.0e-9, 1000.0, -60.0e-3, 0.0);
    for &v in &current {
        assert_eq!(v, 0.0);
    }
}

/// At V_m = E_rev, current is zero regardless of P_open.
/// # Panics
/// - Panics if an internal precondition is violated.
///
#[test]
fn test_ion_current_zero_at_reversal() {
    let p_open = Array3::from_elem((2, 2, 2), 0.8_f64);
    let e_rev = -10.0e-3_f64;
    let current = ion_current(&p_open, 35.0e-12, 5000.0, e_rev, e_rev);
    for &v in &current {
        assert_eq!(v, 0.0);
    }
}

/// All canonical parameters must be positive where positivity is required.
/// # Panics
/// - Panics if assertion fails: `{ch:?}: gating_area must be > 0`.
/// - Panics if assertion fails: `{ch:?}: half_tension must be > 0`.
/// - Panics if assertion fails: `{ch:?}: conductance must be > 0`.
/// - Panics if assertion fails: `{ch:?}: half_pressure must be > 0`.
/// - Panics if assertion fails: `{ch:?}: steepness must be > 0`.
///
#[test]
fn test_canonical_params_are_positive() {
    for ch in [
        MechanoChannel::MscLG22S,
        MechanoChannel::MscLG22N,
        MechanoChannel::MscS,
        MechanoChannel::Piezo1,
        MechanoChannel::Trpc6,
        MechanoChannel::HsTrpa1,
    ] {
        match ch.canonical_params() {
            GatingModel::Boltzmann(p) => {
                assert!(p.gating_area_m2 > 0.0, "{ch:?}: gating_area must be > 0");
                assert!(
                    p.half_tension_n_per_m > 0.0,
                    "{ch:?}: half_tension must be > 0"
                );
                assert!(
                    p.single_channel_conductance_s > 0.0,
                    "{ch:?}: conductance must be > 0"
                );
            }
            GatingModel::PressureThreshold(p) => {
                assert!(
                    p.half_pressure_pa > 0.0,
                    "{ch:?}: half_pressure must be > 0"
                );
                assert!(p.steepness_pa > 0.0, "{ch:?}: steepness must be > 0");
                assert!(
                    p.single_channel_conductance_s > 0.0,
                    "{ch:?}: conductance must be > 0"
                );
            }
        }
    }
}

/// G22N encodes lower mechanical threshold than G22S; MscS remains lower
/// conductance than MscL.
/// # Panics
/// - Panics with `"MscLG22S must use Boltzmann model"`.
/// - Panics with `"MscLG22N must use Boltzmann model"`.
/// - Panics with `"MscS must use Boltzmann model"`.
///
#[test]
fn test_bacterial_channel_variant_ordering() {
    let GatingModel::Boltzmann(g22s) = MechanoChannel::MscLG22S.canonical_params() else {
        panic!("MscLG22S must use Boltzmann model");
    };
    let GatingModel::Boltzmann(g22n) = MechanoChannel::MscLG22N.canonical_params() else {
        panic!("MscLG22N must use Boltzmann model");
    };
    let GatingModel::Boltzmann(mscs) = MechanoChannel::MscS.canonical_params() else {
        panic!("MscS must use Boltzmann model");
    };

    assert!(
        g22n.half_tension_n_per_m < g22s.half_tension_n_per_m,
        "G22N must encode lower mechanical threshold than G22S"
    );
    assert!(
        mscs.single_channel_conductance_s < g22s.single_channel_conductance_s,
        "MscS must remain lower-conductance than MscL variants"
    );
}
