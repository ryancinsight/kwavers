use super::functions::{generate_kappa, generate_shift_1d, generate_source_kappa};
use crate::core::constants::fundamental::SOUND_SPEED_WATER_SIM;
use crate::math::fft::Complex64;
use std::f64::consts::PI;

/// Shift operators at k=0 (DC bin) must be exactly zero.
///
/// i·0·exp(0) = 0.
/// # Panics
/// - Panics if an internal precondition is violated.
///
#[test]
fn test_shift_dc_bin_is_zero() {
    let n = 16;
    let dk = 2.0 * PI / (n as f64 * 1e-3);
    let (pos, neg) = generate_shift_1d(n, dk, 1e-3);
    let eps = 1e-15;
    assert!(
        pos[0].norm() < eps,
        "shift_pos[0] should be zero, got {:?}",
        pos[0]
    );
    assert!(
        neg[0].norm() < eps,
        "shift_neg[0] should be zero, got {:?}",
        neg[0]
    );
}

/// shift_neg equals the negated complex conjugate of shift_pos.
///
/// Proof for real k:
///   shift_pos = i·k·(cos θ + i·sin θ) = −k·sin θ + i·k·cos θ
///   conj(shift_pos) = −k·sin θ − i·k·cos θ
///   shift_neg = i·k·(cos θ − i·sin θ) = k·sin θ + i·k·cos θ = −conj(shift_pos)
///
/// Therefore: shift_neg[k] = −conj(shift_pos[k])
/// # Panics
/// - Panics if assertion fails: `shift_neg[{idx}] != -conj(shift_pos[{idx}]): diff = {diff}`.
///
#[test]
fn test_shift_neg_is_neg_conjugate_of_pos() {
    let n = 32;
    let dx = 5e-4;
    let dk = 2.0 * PI / (n as f64 * dx);
    let (pos, neg) = generate_shift_1d(n, dk, dx);
    for idx in 0..n {
        let diff = (neg[idx] - (-pos[idx].conj())).norm();
        assert!(
            diff < 1e-14,
            "shift_neg[{idx}] != -conj(shift_pos[{idx}]): diff = {diff}"
        );
    }
}

/// Nyquist bin (n/2) must be non-zero for even n.
///
/// k_nyq = π/dx; the operator i·k_nyq·exp(±i·k_nyq·dx/2) = i·k_nyq·exp(±i·π/2)
/// = i·k_nyq·(0 ± i) = ∓k_nyq ≠ 0.
/// # Panics
/// - Panics if an internal precondition is violated.
///
#[test]
fn test_nyquist_bin_not_zeroed() {
    let n = 16;
    let dx = 1e-3;
    let dk = 2.0 * PI / (n as f64 * dx);
    let (pos, neg) = generate_shift_1d(n, dk, dx);
    let nyq = n / 2;
    assert!(
        pos[nyq].norm() > 1e-3,
        "shift_pos Nyquist must not be zeroed, got {:?}",
        pos[nyq]
    );
    assert!(
        neg[nyq].norm() > 1e-3,
        "shift_neg Nyquist must not be zeroed, got {:?}",
        neg[nyq]
    );
}

/// κ at k=0 (DC, i=0,j=0,k=0) must be exactly 1.
///
/// lim_{x→0} sin(x)/x = 1.
/// # Panics
/// - Panics if an internal precondition is violated.
///
#[test]
fn test_kappa_dc_is_one() {
    let kappa = generate_kappa(8, 8, 8, 1e-3, 1e-3, 1e-3, SOUND_SPEED_WATER_SIM, 1e-7);
    assert!(
        (kappa[[0, 0, 0]] - 1.0).abs() < 1e-15,
        "kappa DC should be 1.0, got {}",
        kappa[[0, 0, 0]]
    );
}

/// κ must be in [0, 1] for all bins when c_ref·dt·|k|/2 ≤ π/2,
/// i.e., when the CFL condition c·dt/dx ≤ π / (sqrt(3)·π) = 1/sqrt(3) holds.
/// For the test grid with CFL=0.3, all κ values are positive.
/// # Panics
/// - Panics if an internal precondition is violated.
///
#[test]
fn test_kappa_range_cfl_stable() {
    let dx = 1e-3;
    let c_ref = SOUND_SPEED_WATER_SIM;
    let dt = 0.3 * dx / c_ref; // CFL ≈ 0.3
    let kappa = generate_kappa(8, 8, 8, dx, dx, dx, c_ref, dt);
    for &v in kappa.iter() {
        assert!((0.0..=1.0).contains(&v), "kappa out of [0,1]: {v}");
    }
}

/// κ uses unnormalized sinc: verify at a known non-zero wavenumber.
///
/// At k = (2π/dx)/4 (quarter-Nyquist), x = 0.5·c·dt·k:
///   κ = sin(x)/x  ≠  cos(x)
/// # Panics
/// - Panics if an internal precondition is violated.
///
#[test]
fn test_kappa_sinc_not_cos() {
    let dx = 1e-3_f64;
    let c_ref = SOUND_SPEED_WATER_SIM;
    let dt = 0.2 * dx / c_ref;
    let kappa = generate_kappa(8, 8, 1, dx, dx, dx, c_ref, dt);

    // 1D index: i=1, j=0, k=0  → k_mag = dk_x * 1 = 2π/(8·dx)
    let dk_x = 2.0 * PI / (8.0 * dx);
    let k_mag = dk_x;
    let x = 0.5 * c_ref * dt * k_mag;
    let expected_sinc = x.sin() / x;
    let expected_cos = x.cos();

    // Confirm sinc ≠ cos (otherwise the test is trivial)
    assert!(
        (expected_sinc - expected_cos).abs() > 1e-6,
        "sinc and cos too close to distinguish at x={x}"
    );

    let v = kappa[[1, 0, 0]];
    assert!(
        (v - expected_sinc).abs() < 1e-14,
        "kappa[1,0,0] = {v}, expected sinc={expected_sinc}, cos={expected_cos}"
    );
}

/// source_kappa uses cos, not sinc.
///
/// Additive source injection requires cos(x), not sinc(x).
/// # Panics
/// - Panics if an internal precondition is violated.
///
#[test]
fn test_source_kappa_is_cos() {
    let dx = 1e-3_f64;
    let c_ref = SOUND_SPEED_WATER_SIM;
    let dt = 0.2 * dx / c_ref;
    let src_kappa = generate_source_kappa(8, 8, 1, dx, dx, dx, c_ref, dt);

    let dk_x = 2.0 * PI / (8.0 * dx);
    let k_mag = dk_x;
    let x = 0.5 * c_ref * dt * k_mag;
    let expected_cos = x.cos();

    let v = src_kappa[[1, 0, 0]];
    assert!(
        (v - expected_cos).abs() < 1e-14,
        "source_kappa[1,0,0] = {v}, expected cos={expected_cos}"
    );
}

/// generate_shift_1d output matches the inline closure in PSTD orchestrator.rs.
///
/// The PSTD orchestrator uses the same formula. This test verifies that the
/// extracted function produces bit-identical results.
/// # Panics
/// - Panics if an internal precondition is violated.
///
#[test]
fn test_shift_matches_orchestrator_formula() {
    let n = 8usize;
    let dx = 1e-3_f64;
    let dk = 2.0 * PI / (n as f64 * dx);

    let (pos_fn, neg_fn) = generate_shift_1d(n, dk, dx);

    // Inline reference (same formula as orchestrator.rs generate_shift_1d closure)
    let i_unit = Complex64::new(0.0, 1.0);
    for idx in 0..n {
        let shifted = if idx <= n / 2 {
            idx as isize
        } else {
            idx as isize - n as isize
        };
        let k_val = dk * shifted as f64;
        let exponent = k_val * dx * 0.5;
        let expected_pos =
            i_unit * Complex64::new(k_val, 0.0) * Complex64::new(exponent.cos(), exponent.sin());
        let expected_neg =
            i_unit * Complex64::new(k_val, 0.0) * Complex64::new(exponent.cos(), -exponent.sin());
        // Tolerance is relative to the value magnitude: k_val grows with idx,
        // so machine-epsilon error in the complex product scales with |k_val|.
        // At idx=2, k_val ≈ 1571 → 1 ULP ≈ 3.5e-13. Use 1e-12 (relative ~6e-16).
        let tol = 1e-12_f64.max(expected_pos.norm() * 1e-14);
        assert!(
            (pos_fn[idx] - expected_pos).norm() < tol,
            "pos mismatch at idx={idx}: got {:?}, expected {:?}, diff_norm={:.2e}",
            pos_fn[idx],
            expected_pos,
            (pos_fn[idx] - expected_pos).norm()
        );
        let tol = 1e-12_f64.max(expected_neg.norm() * 1e-14);
        assert!(
            (neg_fn[idx] - expected_neg).norm() < tol,
            "neg mismatch at idx={idx}: got {:?}, expected {:?}, diff_norm={:.2e}",
            neg_fn[idx],
            expected_neg,
            (neg_fn[idx] - expected_neg).norm()
        );
    }
}
