use super::traits::GridParameters;
use super::types::{FactoryConfiguration, FactoryError};
use crate::solver::config::SolverType;

struct MockGridParams {
    nx: usize,
    ny: usize,
    nz: usize,
    dx: f64,
    dy: f64,
    dz: f64,
}

impl GridParameters for MockGridParams {
    fn nx(&self) -> usize {
        self.nx
    }
    fn ny(&self) -> usize {
        self.ny
    }
    fn nz(&self) -> usize {
        self.nz
    }
    fn dx(&self) -> f64 {
        self.dx
    }
    fn dy(&self) -> f64 {
        self.dy
    }
    fn dz(&self) -> f64 {
        self.dz
    }
}

#[test]
fn grid_parameters_total_points() {
    let grid = MockGridParams {
        nx: 64,
        ny: 64,
        nz: 64,
        dx: 1e-3,
        dy: 1e-3,
        dz: 1e-3,
    };
    assert_eq!(grid.total_points(), 64_usize.pow(3));
}

#[test]
fn factory_configuration_defaults() {
    let config = FactoryConfiguration::default();
    assert!(config.enable_auto_selection);
    assert_eq!(config.performance_target, 1.0);
}

#[test]
fn factory_error_display() {
    let err = FactoryError::SolverTypeNotSupported(SolverType::FDTD);
    assert!(err.to_string().contains("FDTD"));
}
