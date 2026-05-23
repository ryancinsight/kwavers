//! Unit tests for microbubble models and population dynamics.
//!
//! This file is already gated by `#[cfg(test)] mod tests;` in the parent
//! `mod.rs`, so an inner `mod tests { ... }` would be redundant nesting.

use super::dynamics::BubbleDynamics;
use crate::core::constants::cavitation::SURFACE_TENSION_WATER;
use crate::core::constants::fundamental::ATMOSPHERIC_PRESSURE;
use crate::core::constants::numerical::MHZ_TO_HZ;
use crate::domain::imaging::ultrasound::ceus::{Microbubble, MicrobubblePopulation};

#[test]
fn test_microbubble_creation() {
    let bubble = Microbubble::sono_vue();

    assert!((bubble.radius_eq - 1.5e-6).abs() < 1e-9);
    assert!(bubble.shell_elasticity > 0.0);
    bubble.validate().unwrap();
}

#[test]
fn test_resonance_frequency() {
    let bubble = Microbubble::new(2.0, 1.0, 0.5); // 2 μm radius
    let freq = bubble.resonance_frequency(ATMOSPHERIC_PRESSURE, 1000.0);

    // Typical resonance frequency for 2 μm bubble should be around 2-5 MHz
    assert!(freq > MHZ_TO_HZ && freq < 10.0 * MHZ_TO_HZ);
}

#[test]
fn test_population_creation() {
    let population = MicrobubblePopulation::new(1e6, 2.5).unwrap();

    // 1e6 bubbles/mL = 1e6 * 1e6 = 1e12 bubbles/m³
    assert!((population.concentration - 1e12).abs() < 1e10);
    assert!(population.reference_bubble.radius_eq > 0.0);
}

#[test]
fn test_bubble_dynamics() {
    let dynamics = BubbleDynamics::new();
    let bubble = Microbubble::definit_y();

    let response = dynamics
        .simulate_oscillation(
            &bubble, 50_000.0,        // 50 kPa
            2.0 * MHZ_TO_HZ, // 2 MHz
            1e-6,             // 1 μs
        )
        .unwrap();

    assert!(!response.time.is_empty());
    assert!(!response.radius.is_empty());
    assert_eq!(response.time.len(), response.radius.len());

    // Bubble should oscillate
    let radius_change = response.max_radius_change();
    assert!(radius_change > 0.0);
}

/// Nonlinear scattering efficiency via Lorentzian — qualitative validation.
///
/// ## Expected Behaviour (de Jong et al. 1994)
///
/// 1. η_NL ≥ 0 for all inputs (physical efficiency)
/// 2. η_NL scales linearly with drive amplitude P_A (perturbation regime)
/// 3. η_NL peaks at resonance Ω=1 vs. far off-resonance Ω=10
///    (Lorentzian ratio ≈ (Ω²-1)² / δ² >> 1 off-resonance)
///
/// Note: η_NL is not bounded by 1 — it is the ratio of the second-harmonic
/// to the linear scattering amplitude, which can exceed 1 near resonance.
/// # Panics
/// - Panics if assertion fails: `η_NL must be non-negative, got {eff_nominal}`.
/// - Panics if assertion fails: `η_NL at resonance ({eff_res:.3}) should exceed far off-resonance ({eff_off:.3})`.
///
#[test]
fn test_nonlinear_scattering() {
    let dynamics = BubbleDynamics::new();
    let bubble = Microbubble::sono_vue();

    // 1. Non-negative at typical CEUS drive (100 kPa, 3 MHz)
    let eff_nominal = dynamics.nonlinear_scattering_efficiency(
        &bubble, 100_000.0,       // 100 kPa
        3.0 * MHZ_TO_HZ, // 3 MHz
    );
    assert!(
        eff_nominal >= 0.0,
        "η_NL must be non-negative, got {eff_nominal}"
    );

    // 2. Linear scaling with pressure amplitude (perturbation regime)
    let eff_double = dynamics.nonlinear_scattering_efficiency(
        &bubble, 200_000.0,       // 200 kPa (2× drive)
        3.0 * MHZ_TO_HZ,
    );
    let ratio = eff_double / eff_nominal.max(f64::EPSILON);
    assert!(
        (ratio - 2.0).abs() < 0.1,
        "η_NL should scale linearly with P_A; ratio={ratio:.3} (expected ≈2.0)"
    );

    // 3. Resonance gives higher efficiency than far off-resonance (Ω=10)
    // Off-resonance: f = 10 × f_res — Lorentzian ≈ 1/Ω² → much smaller
    let f_res = bubble.resonance_frequency(ATMOSPHERIC_PRESSURE, 1000.0);
    let eff_off = dynamics.nonlinear_scattering_efficiency(
        &bubble,
        100_000.0,
        f_res * 10.0, // Ω = 10 >> 1
    );
    let eff_res =
        dynamics.nonlinear_scattering_efficiency(&bubble, 100_000.0, f_res /* Ω = 1 (resonance) */);
    assert!(
        eff_res > eff_off,
        "η_NL at resonance ({eff_res:.3}) should exceed far off-resonance ({eff_off:.3})"
    );
}

#[test]
fn test_invalid_microbubble() {
    let bubble = Microbubble {
        radius_eq: -1.0, // Invalid
        shell_thickness: 0.1e-6,
        shell_elasticity: 1000.0,
        shell_viscosity: 0.5,
        polytropic_index: 1.07,
        surface_tension: SURFACE_TENSION_WATER,
    };

    assert!(bubble.validate().is_err());
}
