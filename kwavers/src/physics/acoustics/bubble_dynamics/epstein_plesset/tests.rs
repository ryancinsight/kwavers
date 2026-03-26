use super::solver::EpsteinPlessetStabilitySolver;
use super::types::{AmplitudeEvolution, OscillationType};
use crate::physics::acoustics::bubble_dynamics::bubble_state::BubbleParameters;
use approx::assert_relative_eq;

#[test]
fn test_epstein_plesset_stability_analysis() {
    let params = BubbleParameters {
        r0: 1e-3,
        p0: 101325.0,
        rho_liquid: 1000.0,
        sigma: 0.072,
        mu_liquid: 0.001,
        gamma: 1.4,
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
        p0: 101325.0,
        rho_liquid: 1000.0,
        sigma: 0.072,
        mu_liquid: 0.001,
        gamma: 1.4,
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
        p0: 101325.0,
        rho_liquid: 1000.0,
        sigma: 0.072,
        mu_liquid: 0.001,
        gamma: 1.4,
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
    let p0 = 101325.0;
    let rho = 1000.0;
    let gamma = 1.4;

    let params = BubbleParameters {
        r0,
        p0,
        rho_liquid: rho,
        gamma,
        ..Default::default()
    };

    let solver = EpsteinPlessetStabilitySolver::new(params);
    let analysis = solver.analyze_stability();

    let minnaert_freq =
        (1.0 / (2.0 * std::f64::consts::PI * r0)) * ((3.0 * gamma * p0) / rho).sqrt();

    assert_relative_eq!(analysis.resonance_frequency, minnaert_freq, epsilon = 1e-10);
}
