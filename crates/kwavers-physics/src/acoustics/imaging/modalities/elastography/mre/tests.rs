//! Value-semantic tests for the MRE front end. Expected harmonic amplitudes and
//! phases are derived analytically from the single-bin DFT of a sampled sinusoid.

use super::{extract_first_harmonic, harmonic_snapshot, mre_displacement_field_z};
use leto::Array4;
use std::f64::consts::PI;

const KAPPA: f64 = 1.0e4; // rad/m encoding sensitivity
const N: usize = 8; // phase offsets over one period

/// Build a 1×1×1×N stack of φ[k] = κ·A·cos(2πk/N + θ).
fn single_voxel_stack(amplitude: f64, theta: f64) -> Array4<f64> {
    Array4::from_shape_fn([1, 1, 1, N], |[_i, _j, _k, phase]| {
        KAPPA * amplitude * (2.0 * PI * phase as f64 / N as f64 + theta).cos()
    })
}

#[test]
fn recovers_amplitude_and_phase_of_a_pure_harmonic() {
    let amplitude = 2.0e-6; // 2 µm
    let theta = PI / 4.0;
    let stack = single_voxel_stack(amplitude, theta);
    let h = extract_first_harmonic(&stack, KAPPA).expect("extract");
    let u = h[[0, 0, 0]];
    assert!(
        (u.norm() - amplitude).abs() < 1e-15,
        "amplitude: got {}",
        u.norm()
    );
    assert!((u.arg() - theta).abs() < 1e-12, "phase: got {}", u.arg());
}

#[test]
fn rejects_dc_phase_offset() {
    // A constant phase bias (e.g. B0 inhomogeneity) is DC (bin 0) and must not
    // leak into the first harmonic.
    let amplitude = 1.5e-6;
    let theta = -0.7;
    let mut stack = single_voxel_stack(amplitude, theta);
    stack.iter_mut().for_each(|value| *value += 3.3); // add DC bias
    let h = extract_first_harmonic(&stack, KAPPA).expect("extract");
    assert!(
        (h[[0, 0, 0]].norm() - amplitude).abs() < 1e-12,
        "DC must not affect fundamental"
    );
}

#[test]
fn snapshot_recovers_amplitude_at_matched_phase() {
    // Re{U·e^{iθ_snap}} = A·cos(arg(U)+θ_snap); at θ_snap = −arg(U) it equals A.
    let amplitude = 3.0e-6;
    let theta = 1.1;
    let h = extract_first_harmonic(&single_voxel_stack(amplitude, theta), KAPPA).expect("extract");
    let snap = harmonic_snapshot(&h, -theta);
    assert!(
        (snap[[0, 0, 0]] - amplitude).abs() < 1e-12,
        "got {}",
        snap[[0, 0, 0]]
    );
    // At quadrature (θ_snap = −θ + π/2) the in-phase projection is ~0.
    let quad = harmonic_snapshot(&h, -theta + PI / 2.0);
    assert!(quad[[0, 0, 0]].abs() < 1e-12);
}

#[test]
fn zero_phase_stack_gives_zero_displacement() {
    let stack = Array4::<f64>::zeros([2, 2, 2, N]);
    let h = extract_first_harmonic(&stack, KAPPA).expect("extract");
    assert!(h.iter().all(|u| u.norm() == 0.0));
}

#[test]
fn displacement_field_z_carries_the_harmonic_snapshot() {
    // uz at θ=0 snapshot = A·cos(θ) for φ[k]=κ·A·cos(2πk/N+θ).
    let amplitude = 4.0e-6;
    let theta = 0.0; // in-phase ⇒ snapshot at 0 equals amplitude
    let field =
        mre_displacement_field_z(&single_voxel_stack(amplitude, theta), KAPPA).expect("field");
    assert!(
        (field.uz[[0, 0, 0]] - amplitude).abs() < 1e-12,
        "got {}",
        field.uz[[0, 0, 0]]
    );
    assert_eq!(field.ux[[0, 0, 0]], 0.0);
    assert_eq!(field.uy[[0, 0, 0]], 0.0);
}

#[test]
fn rejects_invalid_inputs() {
    let stack = single_voxel_stack(1e-6, 0.0);
    assert!(extract_first_harmonic(&stack, 0.0).is_err());
    assert!(extract_first_harmonic(&stack, -1.0).is_err());
    let too_short = Array4::<f64>::zeros([1, 1, 1, 1]);
    assert!(extract_first_harmonic(&too_short, KAPPA).is_err());
}
