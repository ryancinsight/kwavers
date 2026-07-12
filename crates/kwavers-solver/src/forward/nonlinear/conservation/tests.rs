use super::*;
use kwavers_core::constants::fundamental::{DENSITY_WATER_NOMINAL, SOUND_SPEED_WATER_SIM};

#[test]
fn test_conservation_diagnostic_severity() {
    let tolerances = ConservationTolerances::default();

    let diag = ConservationDiagnostic::new(
        NonlinearConservationLaw::Energy,
        1000.0,
        1000.0 + 1e-9,
        100,
        0.1,
        &tolerances,
    );
    assert_eq!(diag.severity, ViolationSeverity::Acceptable);
    assert!(diag.is_acceptable());
    assert!(!diag.requires_action());

    let diag_critical = ConservationDiagnostic::new(
        NonlinearConservationLaw::Energy,
        1000.0,
        2000.0,
        100,
        0.1,
        &tolerances,
    );
    assert_eq!(diag_critical.severity, ViolationSeverity::Critical);
    assert!(diag_critical.requires_action());
}

#[test]
fn test_conservation_tracker() {
    let tolerances = ConservationTolerances {
        check_interval: 1,
        ..Default::default()
    };

    let mut tracker = ConservationTracker::new(1000.0, (0.0, 0.0, 0.0), 100.0, tolerances);

    struct MockSolver {
        energy: f64,
    }

    impl ConservationDiagnostics for MockSolver {
        fn calculate_total_energy(&self) -> f64 {
            self.energy
        }
        fn calculate_total_momentum(&self) -> (f64, f64, f64) {
            (0.0, 0.0, 0.0)
        }
        fn calculate_total_mass(&self) -> f64 {
            100.0
        }
    }

    let solver = MockSolver {
        energy: 1000.0 + 1e-9,
    };
    let diagnostics = tracker.update(&solver, 1, 0.001);

    assert_eq!((diagnostics.len()), 3);
    assert_eq!(diagnostics[0].law, NonlinearConservationLaw::Energy);
    assert!(diagnostics[0].is_acceptable());

    let solver_warning = MockSolver { energy: 1001.0 };
    let diagnostics_warning = tracker.update(&solver_warning, 2, 0.002);
    assert!(diagnostics_warning[0].severity >= ViolationSeverity::Warning);
}

#[test]
fn test_energy_density_calculation() {
    let density = DENSITY_WATER_NOMINAL;
    let sound_speed = SOUND_SPEED_WATER_SIM;
    let pressure = 1000.0;
    let velocity = (0.1, 0.0, 0.0);

    let energy_density = helpers::acoustic_energy_density(pressure, velocity, density, sound_speed);

    let kinetic = 0.5 * density * 0.1_f64.powi(2);
    let potential = pressure.powi(2) / (2.0 * density * sound_speed.powi(2));
    let expected = kinetic + potential;

    assert!((energy_density - expected).abs() < 1e-10);
}

#[test]
fn test_conservation_tolerances() {
    let default = ConservationTolerances::default();
    assert!(default.absolute_tolerance > 0.0);
    assert!(default.relative_tolerance > 0.0);

    let strict = ConservationTolerances::strict();
    assert!(strict.absolute_tolerance < default.absolute_tolerance);

    let relaxed = ConservationTolerances::relaxed();
    assert!(relaxed.absolute_tolerance > default.absolute_tolerance);
}

#[test]
fn test_field_integration() {
    use leto::Array3;
    let field = Array3::<f64>::ones((10, 10, 10));
    let dx = 0.1;
    let dy = 0.1;
    let dz = 0.1;

    let integral = helpers::integrate_field(&field, dx, dy, dz);
    let expected = 10.0 * 10.0 * 10.0 * 0.1 * 0.1 * 0.1;

    assert!((integral - expected).abs() < 1e-10);
}

#[test]
fn test_field_rms() {
    use leto::Array3;
    let field = Array3::<f64>::from_elem([5, 5, 5], 2.0);
    let rms = helpers::field_rms(&field);
    assert!((rms - 2.0).abs() < 1e-10);
}
