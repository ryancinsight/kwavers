//! Unit tests for microbubble models and population dynamics.
//!
//! This file is already gated by `#[cfg(test)] mod tests;` in the parent
//! `mod.rs`, so an inner `mod tests { ... }` would be redundant nesting.

use super::dynamics::BubbleDynamics;
use aequitas::systems::si::quantities::{
    DynamicViscosity, Frequency, Length, MassDensity, NumberDensity, Pressure, SurfaceTension, Time,
};
use kwavers_core::constants::cavitation::SURFACE_TENSION_WATER;
use kwavers_core::constants::fundamental::ATMOSPHERIC_PRESSURE;
use kwavers_core::constants::numerical::MHZ_TO_HZ;
use kwavers_imaging::ultrasound::ceus::{Microbubble, MicrobubblePopulation};

#[test]
fn test_microbubble_creation() {
    let bubble = Microbubble::sono_vue();

    assert!((bubble.radius_eq.into_base() - 1.5e-6).abs() < 1e-9);
    assert!(bubble.shell_elasticity.into_base() > 0.0);
    bubble.validate().unwrap();
}

#[test]
fn test_resonance_frequency() {
    let bubble = Microbubble::new(
        Length::from_base(2.0e-6),
        Pressure::from_base(1_000.0),
        DynamicViscosity::from_base(0.5),
    );
    let freq = bubble.resonance_frequency(
        Pressure::from_base(ATMOSPHERIC_PRESSURE),
        MassDensity::from_base(1000.0),
    );

    // Typical resonance frequency for 2 μm bubble should be around 2-5 MHz
    assert!(freq.into_base() > MHZ_TO_HZ && freq.into_base() < 10.0 * MHZ_TO_HZ);
}

#[test]
fn test_population_creation() {
    let population =
        MicrobubblePopulation::new(NumberDensity::from_base(1e12), Length::from_base(2.5e-6))
            .unwrap();

    assert!((population.concentration.into_base() - 1e12).abs() < 1e10);
    assert!(population.reference_bubble.radius_eq.into_base() > 0.0);
}

#[test]
fn test_bubble_dynamics() {
    let dynamics = BubbleDynamics::new();
    let bubble = Microbubble::definit_y();

    let response = dynamics
        .simulate_oscillation(
            &bubble,
            Pressure::from_base(50_000.0),
            Frequency::from_base(2.0 * MHZ_TO_HZ),
            Time::from_base(1e-6),
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
#[test]
fn test_nonlinear_scattering() {
    let dynamics = BubbleDynamics::new();
    let bubble = Microbubble::sono_vue();

    // 1. Non-negative at typical CEUS drive (100 kPa, 3 MHz)
    let eff_nominal = dynamics.nonlinear_scattering_efficiency(
        &bubble,
        Pressure::from_base(100_000.0),
        Frequency::from_base(3.0 * MHZ_TO_HZ),
    );
    assert!(
        eff_nominal >= 0.0,
        "η_NL must be non-negative, got {eff_nominal}"
    );

    // 2. Linear scaling with pressure amplitude (perturbation regime)
    let eff_double = dynamics.nonlinear_scattering_efficiency(
        &bubble,
        Pressure::from_base(200_000.0),
        Frequency::from_base(3.0 * MHZ_TO_HZ),
    );
    let ratio = eff_double / eff_nominal.max(f64::EPSILON);
    assert!(
        (ratio - 2.0).abs() < 0.1,
        "η_NL should scale linearly with P_A; ratio={ratio:.3} (expected ≈2.0)"
    );

    // 3. Resonance gives higher efficiency than far off-resonance (Ω=10)
    let f_res = bubble.resonance_frequency(
        Pressure::from_base(ATMOSPHERIC_PRESSURE),
        MassDensity::from_base(1000.0),
    );
    let eff_off = dynamics.nonlinear_scattering_efficiency(
        &bubble,
        Pressure::from_base(100_000.0),
        Frequency::from_base(f_res.into_base() * 10.0),
    );
    let eff_res =
        dynamics.nonlinear_scattering_efficiency(&bubble, Pressure::from_base(100_000.0), f_res);
    assert!(
        eff_res > eff_off,
        "η_NL at resonance ({eff_res:.3}) should exceed far off-resonance ({eff_off:.3})"
    );
}

#[test]
fn test_invalid_microbubble() {
    let bubble = Microbubble {
        radius_eq: Length::from_base(-1.0),
        shell_thickness: Length::from_base(0.1e-6),
        shell_elasticity: Pressure::from_base(1000.0),
        shell_viscosity: DynamicViscosity::from_base(0.5),
        polytropic_index: 1.07,
        surface_tension: SurfaceTension::from_base(SURFACE_TENSION_WATER),
    };

    assert!(bubble.validate().is_err());
}
