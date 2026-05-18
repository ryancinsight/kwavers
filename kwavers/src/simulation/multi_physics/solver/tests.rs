//! Tests for multi-physics solver.

use super::super::{
    CoupledPhysicsSolver, MultiPhysicsConfig, PhysicsDomain, SimulationCouplingStrategy,
};
use super::core::SimulationMultiPhysicsSolver;
use crate::core::error::KwaversResult;
use crate::domain::grid::Grid;
use ndarray::{Array3, ArrayView3};

struct MockSolver {
    domain: PhysicsDomain,
    grid: Grid,
    field: Array3<f64>,
}

impl MockSolver {
    fn new(domain: PhysicsDomain, grid: Grid) -> Self {
        let field = Array3::zeros((grid.nx, grid.ny, grid.nz));
        Self {
            domain,
            grid,
            field,
        }
    }
}

impl CoupledPhysicsSolver for MockSolver {
    fn domain_type(&self) -> PhysicsDomain {
        self.domain
    }

    fn grid(&self) -> &Grid {
        &self.grid
    }

    fn get_field(&self, _field_name: &str) -> KwaversResult<ArrayView3<'_, f64>> {
        Ok(self.field.view())
    }

    fn set_field(&mut self, _field_name: &str, field: ArrayView3<f64>) -> KwaversResult<()> {
        self.field.assign(&field);
        Ok(())
    }

    fn step(&mut self, _dt: f64) -> KwaversResult<()> {
        self.field.fill(1.0);
        Ok(())
    }

    fn get_coupling_source(
        &self,
        _target_domain: PhysicsDomain,
    ) -> KwaversResult<Option<Array3<f64>>> {
        Ok(Some(self.field.clone()))
    }

    fn apply_coupling_source(
        &mut self,
        _source_domain: PhysicsDomain,
        source: ArrayView3<f64>,
    ) -> KwaversResult<()> {
        self.field += &source;
        Ok(())
    }
}

#[test]
fn test_multi_physics_solver_creation() {
    let config = MultiPhysicsConfig::default();
    let solver = SimulationMultiPhysicsSolver::new(config);

    assert_eq!(solver.solvers.len(), 0);
    assert_eq!(solver.convergence_history.len(), 0);
}

#[test]
fn test_add_solver() {
    let mut solver = SimulationMultiPhysicsSolver::new(MultiPhysicsConfig::default());
    let grid = Grid::new(10, 10, 10, 0.001, 0.001, 0.001).unwrap();
    let mock_solver = Box::new(MockSolver::new(PhysicsDomain::Acoustic, grid));

    solver.add_solver(mock_solver).unwrap();
    assert_eq!(solver.solvers.len(), 1);
}

#[test]
fn test_explicit_coupling() {
    let mut solver = SimulationMultiPhysicsSolver::new(MultiPhysicsConfig {
        coupling_strategy: SimulationCouplingStrategy::Explicit,
        ..Default::default()
    });

    let grid = Grid::new(8, 8, 8, 0.001, 0.001, 0.001).unwrap();
    let acoustic_solver = Box::new(MockSolver::new(PhysicsDomain::Acoustic, grid.clone()));

    solver.add_solver(acoustic_solver).unwrap();

    let residual = solver.step_coupled(1e-6).unwrap();
    assert_eq!(residual, 1.0);
    assert_eq!(solver.convergence_history(), &[1.0]);
}

/// Monolithic coupling with a single physics domain is equivalent to uncoupled:
/// no inter-domain transfers occur, and the solver simply steps once per iteration.
/// # Panics
/// - Panics if an internal invariant assumed to hold at this call site is violated.
///
#[test]
fn test_monolithic_coupling_single_domain() {
    let mut solver = SimulationMultiPhysicsSolver::new(MultiPhysicsConfig {
        coupling_strategy: SimulationCouplingStrategy::Monolithic,
        max_iterations: 10,
        tolerance: 1e-8,
        ..Default::default()
    });

    let grid = Grid::new(4, 4, 4, 0.001, 0.001, 0.001).unwrap();
    let acoustic_solver = Box::new(MockSolver::new(PhysicsDomain::Acoustic, grid));
    solver.add_solver(acoustic_solver).unwrap();

    let residual = solver.step_coupled(1e-6).unwrap();
    assert_eq!(residual, 0.0);
    assert_eq!(solver.convergence_history(), &[1.0, 0.0]);
}

/// Monolithic coupling with two identical domains reaches the exact fixed point
/// after one corrective iteration.
/// # Panics
/// - Panics if an internal invariant assumed to hold at this call site is violated.
///
#[test]
fn test_monolithic_coupling_two_domains_reaches_fixed_point() {
    let mut solver = SimulationMultiPhysicsSolver::new(MultiPhysicsConfig {
        coupling_strategy: SimulationCouplingStrategy::Monolithic,
        max_iterations: 5,
        tolerance: 1e-8,
        relaxation_factor: 0.5,
        adaptive_timestep: false,
        min_dt: 1e-9,
        max_dt: 1e-3,
    });

    let grid = Grid::new(4, 4, 4, 0.001, 0.001, 0.001).unwrap();
    let acoustic = Box::new(MockSolver::new(PhysicsDomain::Acoustic, grid.clone()));
    let thermal = Box::new(MockSolver::new(PhysicsDomain::Thermal, grid.clone()));
    solver.add_solver(acoustic).unwrap();
    solver.add_solver(thermal).unwrap();
    solver
        .add_coupling(PhysicsDomain::Acoustic, PhysicsDomain::Thermal)
        .unwrap();
    solver
        .add_coupling(PhysicsDomain::Thermal, PhysicsDomain::Acoustic)
        .unwrap();

    let residual = solver.step_coupled(1e-6).unwrap();
    assert_eq!(residual, 0.0);
    assert_eq!(solver.convergence_history(), &[1.0, 0.0]);
}
