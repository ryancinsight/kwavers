use super::*;

#[test]
fn test_fdtd_config_validation() {
    let config = FDTDConfig::default();
    assert!(config.validate().is_ok());

    let mut bad_config = config.clone();
    bad_config.wave_speed = -1.0;
    assert!(bad_config.validate().is_err());

    let mut bad_config = config.clone();
    bad_config.dx = 0.0;
    assert!(bad_config.validate().is_err());

    let mut bad_config = config.clone();
    bad_config.dt = 1.0;
    assert!(bad_config.validate().is_err());
}

#[test]
fn test_fdtd_config_cfl() {
    let config = FDTDConfig::default();
    let cfl = config.cfl_number();
    assert!(cfl <= 1.0);
    assert!(cfl > 0.0);
}

#[test]
fn test_fdtd_solver_creation() {
    let config = FDTDConfig::default();
    let solver = FDTD1DWaveSolver::new(config).unwrap();
    assert_eq!(solver.current_step(), 0);
    assert_eq!(solver.current_field().len(), 100);
}

#[test]
fn test_fdtd_step() {
    let config = FDTDConfig::default();
    let mut solver = FDTD1DWaveSolver::new(config).unwrap();

    let initial_field = solver.current_field().clone();
    solver.step().unwrap();

    assert_eq!(solver.current_step(), 1);
    assert_ne!(solver.current_field(), &initial_field);
}

#[test]
fn test_fdtd_solve() {
    let config = FDTDConfig {
        nx: 50,
        nt: 50,
        ..Default::default()
    };
    let mut solver = FDTD1DWaveSolver::new(config).unwrap();
    let solution = solver.solve().unwrap();
    assert_eq!(solution.dim(), (50, 50));

    for &val in solution.iter() {
        assert!(val.is_finite());
    }
}

#[test]
fn test_boundary_conditions() {
    let config = FDTDConfig::default();
    let mut solver = FDTD1DWaveSolver::new(config).unwrap();

    for _ in 0..10 {
        solver.step().unwrap();
        assert_eq!(solver.current_field()[0], 0.0);
        assert_eq!(solver.current_field()[99], 0.0);
    }
}

#[test]
fn test_gaussian_initial_condition() {
    let config = FDTDConfig {
        initial_condition: InitialCondition::GaussianPulse {
            width: 0.05,
            amplitude: 2.0,
        },
        ..Default::default()
    };
    let solver = FDTD1DWaveSolver::new(config).unwrap();

    let field = solver.current_field();
    let max_val = field.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));
    assert!(max_val > 1.0);
    assert!(max_val < 2.5);
}

#[test]
fn test_sine_initial_condition() {
    let config = FDTDConfig {
        initial_condition: InitialCondition::SineWave {
            frequency: 10.0,
            amplitude: 1.0,
        },
        ..Default::default()
    };
    let solver = FDTD1DWaveSolver::new(config).unwrap();

    let field = solver.current_field();
    let max_val = field.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));
    let min_val = field.iter().fold(f64::INFINITY, |a, &b| a.min(b));
    assert!(max_val > 0.5);
    assert!(min_val < -0.5);
}
