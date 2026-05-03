//! Tests for multi-physics interface boundary conditions.

use super::super::types::{CouplingType, PhysicsDomain};
use super::interface::MultiPhysicsInterface;

const Z_WATER: f64 = 1_479_036.0;
const Z_SOFT_TISSUE: f64 = 1_632_400.0;
const Z_BONE: f64 = 6_258_600.0;

#[test]
fn test_multiphysics_interface_photoacoustic() {
    let interface = MultiPhysicsInterface::new(
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        PhysicsDomain::Electromagnetic,
        PhysicsDomain::Acoustic,
        CouplingType::ElectromagneticAcoustic {
            optical_absorption: 100.0,
            gruneisen: 0.15,
        },
    );
    let tau = interface.transmission_coefficient(1e6);
    assert!(tau > 0.0);
    assert!((0.0..=1.0).contains(&tau));
}

/// Water–soft-tissue: nearly impedance-matched → τ ≈ 0.9978 (high transmission).
#[test]
fn test_acoustic_elastic_water_soft_tissue_transmission() {
    let interface = MultiPhysicsInterface::new(
        [0.0, 0.0, 0.0],
        [0.0, 0.0, 1.0],
        PhysicsDomain::Acoustic,
        PhysicsDomain::Elastic,
        CouplingType::AcousticElastic {
            z1_rayl: Z_WATER,
            z2_rayl: Z_SOFT_TISSUE,
        },
    );
    let tau = interface.transmission_coefficient(1e6);
    let expected = 4.0 * Z_WATER * Z_SOFT_TISSUE / (Z_WATER + Z_SOFT_TISSUE).powi(2);
    assert!(
        (tau - expected).abs() < 1e-10,
        "τ = {:.6}, expected {:.6}",
        tau,
        expected
    );
    assert!(
        tau > 0.99,
        "water/tissue should be nearly impedance-matched; got τ = {:.6}",
        tau
    );
    let r = (Z_SOFT_TISSUE - Z_WATER).powi(2) / (Z_SOFT_TISSUE + Z_WATER).powi(2);
    assert!((tau + r - 1.0).abs() < 1e-12, "τ + R = {:.15}", tau + r);
}

/// Water–cortical bone: significant impedance mismatch → τ ≈ 0.618, R ≈ 0.382.
#[test]
fn test_acoustic_elastic_water_bone_transmission() {
    let interface = MultiPhysicsInterface::new(
        [0.0, 0.0, 0.0],
        [0.0, 0.0, 1.0],
        PhysicsDomain::Acoustic,
        PhysicsDomain::Elastic,
        CouplingType::AcousticElastic {
            z1_rayl: Z_WATER,
            z2_rayl: Z_BONE,
        },
    );
    let tau = interface.transmission_coefficient(1e6);
    let expected = 4.0 * Z_WATER * Z_BONE / (Z_WATER + Z_BONE).powi(2);
    assert!(
        (tau - expected).abs() < 1e-10,
        "τ = {:.6}, expected {:.6}",
        tau,
        expected
    );
    assert!(
        tau > 0.3 && tau < 0.75,
        "water/bone τ should be in (0.3, 0.75); got τ = {:.4}",
        tau
    );
    let r = (Z_BONE - Z_WATER).powi(2) / (Z_BONE + Z_WATER).powi(2);
    assert!((tau + r - 1.0).abs() < 1e-12, "τ + R = {:.15}", tau + r);
    assert!(
        r > 0.2,
        "bone interface should have >20% reflection; R = {:.4}",
        r
    );
}

/// Photoacoustic: higher absorption → higher coupling (monotone property).
#[test]
fn test_multiphysics_photoacoustic_monotone() {
    let gruneisen = 0.12;
    let make = |mu_a: f64| {
        MultiPhysicsInterface::new(
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            PhysicsDomain::Electromagnetic,
            PhysicsDomain::Acoustic,
            CouplingType::ElectromagneticAcoustic {
                optical_absorption: mu_a,
                gruneisen,
            },
        )
        .transmission_coefficient(1e6)
    };
    let tau_lo = make(1.0);
    let tau_hi = make(10.0);
    assert!(tau_hi >= tau_lo, "coupling must be monotone in μ_a");
    assert!(
        (tau_lo - 0.12).abs() < 1e-12,
        "τ(μ_a=1) = Γ·μ_a = {:.4}",
        tau_lo
    );
    assert!(
        (tau_hi - 1.0).abs() < 1e-12,
        "τ must saturate at 1 for Γ·μ_a > 1"
    );
}

/// Acoustic-thermal: coupling is physically positive and ≤ 1.
#[test]
fn test_acoustic_thermal_coupling_bounds() {
    let interface = MultiPhysicsInterface::new(
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        PhysicsDomain::Acoustic,
        PhysicsDomain::Thermal,
        CouplingType::AcousticThermal {
            alpha_np_per_m: 2.0,
            rho_kg_per_m3: 1060.0,
            c_p_j_per_kg_k: 3500.0,
        },
    );
    let tau = interface.transmission_coefficient(1e6);
    assert!((0.0..=1.0).contains(&tau), "τ = {}", tau);
    let expected = (2.0 * 2.0 / (1060.0 * 3500.0_f64)).clamp(0.0, 1.0);
    assert!((tau - expected).abs() < 1e-15);
}

#[test]
fn test_multiphysics_electromagnetic_thermal() {
    let interface = MultiPhysicsInterface::new(
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        PhysicsDomain::Electromagnetic,
        PhysicsDomain::Thermal,
        CouplingType::ElectromagneticThermal,
    );
    let tau = interface.transmission_coefficient(1e6);
    assert!(tau > 0.9, "photothermal coupling should exceed 90%");
    assert!(tau <= 1.0);
}

#[test]
fn test_multiphysics_custom_coupling() {
    let interface = MultiPhysicsInterface::new(
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        PhysicsDomain::Custom(1),
        PhysicsDomain::Custom(2),
        CouplingType::Custom("user_defined".to_string()),
    );
    assert_eq!(interface.transmission_coefficient(1e6), 1.0);
}

/// Impedance self-matching (Z₁ = Z₂) gives τ = 1.0 exactly.
#[test]
fn test_acoustic_elastic_self_matched() {
    let interface = MultiPhysicsInterface::new(
        [0.0, 0.0, 0.0],
        [0.0, 0.0, 1.0],
        PhysicsDomain::Acoustic,
        PhysicsDomain::Elastic,
        CouplingType::AcousticElastic {
            z1_rayl: Z_WATER,
            z2_rayl: Z_WATER,
        },
    );
    let tau = interface.transmission_coefficient(1e6);
    assert!(
        (tau - 1.0).abs() < 1e-14,
        "Z₁=Z₂ must give τ=1; got {}",
        tau
    );
}
