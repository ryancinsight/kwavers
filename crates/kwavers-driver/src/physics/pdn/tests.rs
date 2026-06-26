//! Phase 3e consolidated test surface for the [`crate::physics::pdn`] slice.
//!
//! Four tests, migrated verbatim from the prior flat `src/pdn.rs::tests` module. All free fns
//! are re-exported at the slice root via `pub use` in [`super::mod`], so `use super::*` brings
//! every public symbol into scope exactly as `use super::*` did in the flat-module test.

use super::*;

#[test]
fn decoupling_design_arithmetic() {
    // 50 mV ripple budget under a 3 A transient ⇒ Z_target ≈ 16.7 mΩ.
    assert!((target_impedance_ohm(0.05, 3.0) - 0.01667).abs() < 1e-4);
    // Hold 1 A for 1 µs within 100 mV droop ⇒ C = I·Δt/ΔV = 10 µF.
    assert!((holdup_capacitance_f(1.0, 1e-6, 0.1) - 10e-6).abs() < 1e-9);
    // 100 nF with 0.5 nH ESL self-resonates near 22 MHz.
    let srf = self_resonant_freq_hz(0.5e-9, 100e-9);
    assert!(
        (20e6..24e6).contains(&srf),
        "expected ~22 MHz SRF, got {:.1e}",
        srf
    );
    // Decoupling distance budget: 100 nF must stay effective to 10 MHz, 0.5 nH mounting ESL,
    // 0.6 nH/mm connection loop. L_budget = 1/((2π·1e7)²·1e-7) ≈ 2.53 nH; trace budget ≈ 2.03 nH;
    // distance ≈ 2.03/0.6 ≈ 3.4 mm.
    let d = max_decoupling_distance_mm(100e-9, 10e6, 0.6, 0.5e-9);
    assert!(
        (3.0..3.8).contains(&d),
        "expected ~3.4 mm budget, got {d:.2}"
    );
    // If the cap already can't meet the target at zero length, the budget is zero.
    assert_eq!(max_decoupling_distance_mm(100e-9, 30e6, 0.6, 1.0e-9), 0.0);
    // 100×80 mm FR4 plane pair: first cavity mode ~700 MHz ≫ the 2 MHz drive.
    let f = plane_resonance_hz(0.1, 0.08, 4.3, 1, 0);
    assert!(
        (650e6..780e6).contains(&f),
        "expected ~700 MHz plane mode, got {:.2e}",
        f
    );
}

#[test]
fn pdn_impedance_at_srf_equals_esr() {
    // 100 nF, 50 mΩ ESR, 0.5 nH ESL: SRF = 1/(2π√(0.5e-9·100e-9)) ≈ 22.5 MHz.
    let srf = self_resonant_freq_hz(0.5e-9, 100e-9);
    let z_at_srf = pdn_impedance_at_freq(&[(100e-9, 50e-3, 0.5e-9)], srf);
    // At SRF the reactive terms cancel; |Z| ≈ ESR = 50 mΩ.
    assert!(
        (z_at_srf - 50e-3).abs() < 5e-3,
        "at SRF |Z| must equal ESR (50 mΩ), got {z_at_srf:.4}"
    );
    // Well below SRF (DC-like): dominated by capacitive branch, |Z| > ESR.
    let z_low = pdn_impedance_at_freq(&[(100e-9, 50e-3, 0.5e-9)], srf / 100.0);
    assert!(
        z_low > z_at_srf,
        "below SRF impedance must be higher than at SRF"
    );
    // Empty bank: ∞.
    assert!(pdn_impedance_at_freq(&[], 1e6).is_infinite());
}

#[test]
fn parallel_caps_lower_impedance() {
    // Two identical 100 nF caps in parallel should roughly halve impedance at SRF.
    let one = pdn_impedance_at_freq(&[(100e-9, 50e-3, 0.5e-9)], 10e6);
    let two = pdn_impedance_at_freq(&[(100e-9, 50e-3, 0.5e-9), (100e-9, 50e-3, 0.5e-9)], 10e6);
    assert!(
        two < one,
        "two caps in parallel must lower |Z| ({two:.4} vs {one:.4})"
    );
}

#[test]
fn anti_resonance_between_bulk_and_local_cap() {
    // Bulk: 10 µH, local: 100 nF. f_ar = 1/(2π√(10e-6·100e-9)) ≈ 159 kHz.
    let f = anti_resonance_hz(10e-6, 100e-9);
    assert!(
        (f - 159.0e3).abs() < 5.0e3,
        "expected ~159 kHz anti-resonance, got {:.1} kHz",
        f / 1e3
    );
    // Degenerate inputs.
    assert!(anti_resonance_hz(0.0, 100e-9).is_infinite());
    assert!(anti_resonance_hz(10e-6, 0.0).is_infinite());
}
