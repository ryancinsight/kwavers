use super::solver::EpsteinPlessetStabilitySolver;
use super::types::{AmplitudeEvolution, OscillationType};
use kwavers_core::constants::cavitation::{SURFACE_TENSION_WATER, VISCOSITY_WATER};
use kwavers_core::constants::fundamental::{ATMOSPHERIC_PRESSURE, DENSITY_WATER_NOMINAL};
use kwavers_core::constants::numerical::TWO_PI;
use kwavers_core::constants::thermodynamic::HEAT_CAPACITY_RATIO_DIATOMIC;
use crate::acoustics::bubble_dynamics::bubble_state::BubbleParameters;
use approx::assert_relative_eq;

#[test]
fn test_epstein_plesset_stability_analysis() {
    let params = BubbleParameters {
        r0: 1e-3,
        p0: ATMOSPHERIC_PRESSURE,
        rho_liquid: DENSITY_WATER_NOMINAL,
        sigma: SURFACE_TENSION_WATER,
        mu_liquid: VISCOSITY_WATER,
        gamma: HEAT_CAPACITY_RATIO_DIATOMIC,
        ..Default::default()
    };

    let solver = EpsteinPlessetStabilitySolver::new(params);
    let analysis = solver.analyze_stability();

    assert!(analysis.is_stable);
    assert_eq!(analysis.oscillation_type, OscillationType::StableHarmonic);
    assert!(analysis.resonance_frequency > 3000.0 && analysis.resonance_frequency < 4000.0);
    assert!(analysis.quality_factor > 1.0);
    assert!(analysis.stability_parameter > 0.0);
}

#[test]
fn test_stability_boundary_analysis() {
    let params = BubbleParameters {
        r0: 1e-4,
        p0: ATMOSPHERIC_PRESSURE,
        rho_liquid: DENSITY_WATER_NOMINAL,
        sigma: SURFACE_TENSION_WATER,
        mu_liquid: VISCOSITY_WATER,
        gamma: HEAT_CAPACITY_RATIO_DIATOMIC,
        ..Default::default()
    };

    let solver = EpsteinPlessetStabilitySolver::new(params);
    let boundary = solver.compute_stability_boundary();

    assert!(boundary.critical_surface_tension > 0.0);
    assert!(boundary.critical_viscosity > 0.0);
    assert!(boundary.is_currently_stable());
}

#[test]
fn test_amplitude_evolution_prediction() {
    let params = BubbleParameters::default();
    let solver = EpsteinPlessetStabilitySolver::new(params);

    let evolution = solver.predict_amplitude_evolution(1e-6);

    match evolution {
        AmplitudeEvolution::Decaying {
            initial_amplitude,
            final_amplitude,
            decay_rate,
            time_constant,
        } => {
            assert_eq!(initial_amplitude, 1e-6);
            assert!(final_amplitude < initial_amplitude);
            assert!(decay_rate > 0.0);
            assert!(time_constant > 0.0);
        }
        _ => panic!("Expected decaying evolution for stable bubble"),
    }
}

#[test]
fn test_validation_against_literature() {
    let params = BubbleParameters {
        r0: 1e-3,
        p0: ATMOSPHERIC_PRESSURE,
        rho_liquid: DENSITY_WATER_NOMINAL,
        sigma: SURFACE_TENSION_WATER,
        mu_liquid: VISCOSITY_WATER,
        gamma: HEAT_CAPACITY_RATIO_DIATOMIC,
        ..Default::default()
    };

    let solver = EpsteinPlessetStabilitySolver::new(params);
    let validation = solver.validate_implementation().unwrap();

    assert!(validation.all_tests_passed);
    assert!(validation.resonance_frequency_error < 1e-10);
    assert!(validation.quality_factor_valid);
    assert!(validation.stability_parameter_valid);
}

#[test]
fn test_epstein_plesset_vs_minnaert_frequency() {
    let r0 = 1e-3;
    let p0 = ATMOSPHERIC_PRESSURE;
    let rho = DENSITY_WATER_NOMINAL;
    let gamma = HEAT_CAPACITY_RATIO_DIATOMIC;

    let params = BubbleParameters {
        r0,
        p0,
        rho_liquid: rho,
        gamma,
        ..Default::default()
    };

    let solver = EpsteinPlessetStabilitySolver::new(params);
    let analysis = solver.analyze_stability();

    let minnaert_freq = (1.0 / (TWO_PI * r0)) * ((3.0 * gamma * p0) / rho).sqrt();

    assert_relative_eq!(analysis.resonance_frequency, minnaert_freq, epsilon = 1e-10);
}

/// Validate against Minnaert's published numeric result (external reference,
/// not the code's own formula): an air bubble in water has resonance
/// `f₀ · R₀ ≈ 3.26 m·Hz` (Minnaert 1933; Leighton 1994, *The Acoustic Bubble*
/// §3.2.1, citing `f₀ ≈ 3.26 / R₀` Hz for R₀ in metres at 1 atm).
///
/// This guards against a self-consistent-but-wrong implementation: it pins the
/// dimensional constant `f₀·R₀` to a value taken from the literature, so an
/// erroneous stiffness/density/γ grouping that still round-trips through the
/// code's own `compute_resonance_frequency` would be caught here.
#[test]
fn test_minnaert_constant_matches_literature_value() {
    // Air bubble in water at 1 atm: γ = 1.4 (diatomic), the textbook case.
    let p0 = ATMOSPHERIC_PRESSURE;
    let rho = DENSITY_WATER_NOMINAL;
    let gamma = HEAT_CAPACITY_RATIO_DIATOMIC;

    // f₀·R₀ = (1/2π)·√(3γp₀/ρ) — independent of R₀; the Minnaert constant.
    let f0_times_r0 =
        |r0: f64| -> f64 {
            let params = BubbleParameters {
                r0,
                p0,
                rho_liquid: rho,
                gamma,
                ..Default::default()
            };
            EpsteinPlessetStabilitySolver::new(params)
                .analyze_stability()
                .resonance_frequency
                * r0
        };

    // Published value ≈ 3.26 m·Hz. Allow 2% (γ/p₀/ρ rounding across sources).
    let literature_constant = 3.26;
    for &r0 in &[1e-6, 1e-5, 1e-4, 1e-3] {
        let constant = f0_times_r0(r0);
        assert_relative_eq!(constant, literature_constant, max_relative = 0.02);
    }
}
