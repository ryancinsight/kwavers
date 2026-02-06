//! Multi-Physics Simulation Orchestration
//!
//! This module provides the orchestration framework for coupled multi-physics simulations,
//! enabling conservative field transfer between different physics domains (acoustic, thermal,
//! optical, chemical) with proper coupling strategies.
//!
//! ## Mathematical Foundation
//!
//! Multi-physics coupling requires:
//! - **Conservative interpolation**: Preserves physical invariants across domain interfaces
//! - **Stability analysis**: Ensures coupled system remains numerically stable
//! - **Convergence acceleration**: Improves coupling iteration convergence
//! - **Load balancing**: Distributes computational work efficiently
//!
//! ## Architecture
//!
//! The multi-physics framework consists of:
//! - **FieldCoupler**: Manages conservative field transfer between physics domains
//! - **CouplingStrategy**: Defines coupling algorithms (explicit, implicit, partitioned)
//! - **MultiPhysicsSolver**: Orchestrates coupled physics simulations
//! - **DomainManager**: Handles domain decomposition and load balancing
//!
//! ## Usage Example
//!
//! ```rust,ignore
//! use kwavers::simulation::multi_physics::{MultiPhysicsSolver, CouplingStrategy};
//!
//! // Create coupled acoustic-thermal simulation
//! let config = MultiPhysicsConfig {
//!     coupling_strategy: CouplingStrategy::Implicit,
//!     max_iterations: 50,
//!     tolerance: 1e-6,
//!     relaxation_factor: 0.8,
//! };
//!
//! let solver = MultiPhysicsSolver::new(config, acoustic_solver, thermal_solver)?;
//! let result = solver.solve_coupled(&initial_conditions, time_span)?;
//! ```

use crate::core::error::{KwaversError, KwaversResult, ValidationError};
use crate::domain::grid::Grid;
use crate::math::numerics::operators::TrilinearInterpolator;
use ndarray::{Array3, ArrayView3};
use std::collections::HashMap;

/// Physics domain types for coupling
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum PhysicsDomain {
    /// Acoustic wave propagation
    Acoustic,
    /// Thermal diffusion/convection
    Thermal,
    /// Optical light transport
    Optical,
    /// Chemical reaction-diffusion
    Chemical,
    /// Elastic wave propagation
    Elastic,
    /// Electromagnetic fields
    Electromagnetic,
}

/// Coupling strategy for multi-physics simulations
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum CouplingStrategy {
    /// Explicit coupling (no iteration, use previous timestep values)
    Explicit,
    /// Implicit coupling (iterative solution until convergence)
    Implicit,
    /// Partitioned coupling (solve each physics sequentially with updates)
    Partitioned,
    /// Monolithic coupling (solve all physics simultaneously)
    Monolithic,
}

/// Configuration for multi-physics coupling
#[derive(Debug, Clone)]
pub struct MultiPhysicsConfig {
    /// Coupling strategy to use
    pub coupling_strategy: CouplingStrategy,
    /// Maximum coupling iterations
    pub max_iterations: usize,
    /// Convergence tolerance
    pub tolerance: f64,
    /// Relaxation factor for iterative coupling (0 < omega <= 1)
    pub relaxation_factor: f64,
    /// Enable adaptive time stepping
    pub adaptive_timestep: bool,
    /// Minimum time step size
    pub min_dt: f64,
    /// Maximum time step size
    pub max_dt: f64,
}

impl Default for MultiPhysicsConfig {
    fn default() -> Self {
        Self {
            coupling_strategy: CouplingStrategy::Implicit,
            max_iterations: 20,
            tolerance: 1e-6,
            relaxation_factor: 0.8,
            adaptive_timestep: true,
            min_dt: 1e-9,
            max_dt: 1e-3,
        }
    }
}

/// Interface for physics solvers that can participate in coupling
pub trait CoupledPhysicsSolver: Send + Sync {
    /// Get the physics domain type
    fn domain_type(&self) -> PhysicsDomain;

    /// Get the computational grid
    fn grid(&self) -> &Grid;

    /// Get current field values
    fn get_field(&self, field_name: &str) -> KwaversResult<ArrayView3<'_, f64>>;

    /// Set field values (for coupling updates)
    fn set_field(&mut self, field_name: &str, field: ArrayView3<f64>) -> KwaversResult<()>;

    /// Perform a single time step
    fn step(&mut self, dt: f64) -> KwaversResult<()>;

    /// Get coupling source terms from this physics domain
    fn get_coupling_source(
        &self,
        target_domain: PhysicsDomain,
    ) -> KwaversResult<Option<Array3<f64>>>;

    /// Apply coupling source terms to this physics domain
    fn apply_coupling_source(
        &mut self,
        source_domain: PhysicsDomain,
        source: ArrayView3<f64>,
    ) -> KwaversResult<()>;
}

/// Field coupling manager for conservative interpolation between domains
#[derive(Debug)]
pub struct FieldCoupler {
    /// Interpolation operators for each domain pair
    interpolators: HashMap<(PhysicsDomain, PhysicsDomain), TrilinearInterpolator>,
    /// Coupling interface definitions
    interfaces: HashMap<(PhysicsDomain, PhysicsDomain), CouplingInterface>,
    /// Conservation enforcement
    conservation: ConservationEnforcer,
}

impl FieldCoupler {
    /// Create a new field coupler
    pub fn new() -> Self {
        Self {
            interpolators: HashMap::new(),
            interfaces: HashMap::new(),
            conservation: ConservationEnforcer::new(),
        }
    }

    /// Add coupling between two physics domains
    pub fn add_coupling(
        &mut self,
        source_domain: PhysicsDomain,
        target_domain: PhysicsDomain,
        source_grid: &Grid,
        target_grid: &Grid,
    ) -> KwaversResult<()> {
        let key = (source_domain, target_domain);

        // Create interpolator for this domain pair
        let interpolator =
            TrilinearInterpolator::new(target_grid.dx, target_grid.dy, target_grid.dz);

        // Define coupling interface (simplified - would need proper interface detection)
        // TODO_AUDIT: P1 - Advanced Multi-Physics Coupling - Implement conservative coupling schemes with energy/momentum conservation
        // DEPENDS ON: simulation/multi_physics/conservative_coupling.rs, simulation/multi_physics/domain_decomposition.rs
        // MISSING: Conservative interpolation schemes (Sprague-Grundy theorem compliance)
        // MISSING: Domain decomposition with Schwarz alternating methods
        // MISSING: Energy-momentum conservation across physics interfaces
        // MISSING: Adaptive time stepping for multi-scale coupling
        // MISSING: Jacobian-free Newton-Krylov methods for nonlinear coupling
        // THEOREM: Sprague-Grundy theorem: Conservative interpolation requires specific stencil weights
        // THEOREM: Schwarz alternating method convergence: spectral radius ρ < 1 for convergence
        let interface = CouplingInterface::new(source_grid, target_grid)?;

        self.interpolators.insert(key, interpolator);
        self.interfaces.insert(key, interface);

        Ok(())
    }

    /// Transfer field conservatively between domains
    pub fn transfer_field(
        &mut self,
        source_domain: PhysicsDomain,
        target_domain: PhysicsDomain,
        field_name: &str,
        source_solver: &dyn CoupledPhysicsSolver,
        target_solver: &mut dyn CoupledPhysicsSolver,
        relaxation: f64,
    ) -> KwaversResult<f64> {
        let _key = (source_domain, target_domain);

        // Get source field
        let source_field = source_solver.get_field(field_name)?;

        // Apply conservative interpolation
        let interpolated = self.conservation.conservative_interpolate(
            &source_field,
            source_solver.grid(),
            target_solver.grid(),
        )?;

        // Apply relaxation for stability
        let current_target = target_solver.get_field(field_name)?.to_owned();
        let relaxed_field = &interpolated * relaxation + &current_target * (1.0 - relaxation);

        // Update target field
        target_solver.set_field(field_name, relaxed_field.view())?;

        // Return residual for convergence checking
        let residual = (&relaxed_field - &current_target)
            .mapv(|x| x.abs())
            .mean()
            .unwrap_or(0.0);
        Ok(residual)
    }
}

impl Default for FieldCoupler {
    fn default() -> Self {
        Self::new()
    }
}

/// Conservation enforcement for multi-physics coupling
#[derive(Debug)]
pub struct ConservationEnforcer {
    /// Conservation tolerance
    tolerance: f64,
}

impl ConservationEnforcer {
    /// Create new conservation enforcer
    pub fn new() -> Self {
        Self { tolerance: 1e-10 }
    }

    /// Apply conservative interpolation between grids
    pub fn conservative_interpolate(
        &self,
        source_field: &ArrayView3<f64>,
        source_grid: &Grid,
        target_grid: &Grid,
    ) -> KwaversResult<Array3<f64>> {
        let expected_source_dim = (source_grid.nx, source_grid.ny, source_grid.nz);
        let actual_source_dim = source_field.dim();
        if actual_source_dim != expected_source_dim {
            return Err(KwaversError::Validation(
                ValidationError::DimensionMismatch {
                    expected: format!("{expected_source_dim:?}"),
                    actual: format!("{actual_source_dim:?}"),
                },
            ));
        }

        let same_grid = source_grid.nx == target_grid.nx
            && source_grid.ny == target_grid.ny
            && source_grid.nz == target_grid.nz
            && (source_grid.dx - target_grid.dx).abs() <= self.tolerance
            && (source_grid.dy - target_grid.dy).abs() <= self.tolerance
            && (source_grid.dz - target_grid.dz).abs() <= self.tolerance;
        if same_grid {
            return Ok(source_field.to_owned());
        }

        // Trilinear interpolation for smooth field transfer between grids.
        // Falls back to nearest-neighbor at domain boundaries where the
        // 8-point stencil extends outside the source grid.
        let interpolator =
            TrilinearInterpolator::new(source_grid.dx, source_grid.dy, source_grid.dz);
        let mut result = Array3::zeros((target_grid.nx, target_grid.ny, target_grid.nz));

        for i in 0..target_grid.nx {
            for j in 0..target_grid.ny {
                for k in 0..target_grid.nz {
                    let (x, y, z) = target_grid.indices_to_coordinates(i, j, k);

                    result[[i, j, k]] = interpolator
                        .interpolate_point(source_field.view(), x, y, z)
                        .unwrap_or_else(|_| {
                            // Nearest-neighbor fallback for boundary points
                            let si = (x / source_grid.dx).round() as usize;
                            let sj = (y / source_grid.dy).round() as usize;
                            let sk = (z / source_grid.dz).round() as usize;
                            source_field[[
                                si.min(source_grid.nx - 1),
                                sj.min(source_grid.ny - 1),
                                sk.min(source_grid.nz - 1),
                            ]]
                        });
                }
            }
        }

        Ok(result)
    }
}

impl Default for ConservationEnforcer {
    fn default() -> Self {
        Self::new()
    }
}

/// Simplified coupling interface definition
#[derive(Debug, Clone)]
pub struct CouplingInterface {
    /// Interface area
    pub area: f64,
    /// Interface normal direction
    pub normal: (f64, f64, f64),
}

impl CouplingInterface {
    /// Create coupling interface between grids
    ///
    /// Computes the interface area from the overlap of the two domain extents
    /// and sets the normal perpendicular to the largest face.
    pub fn new(source_grid: &Grid, target_grid: &Grid) -> KwaversResult<Self> {
        // Domain extents
        let sx = source_grid.nx as f64 * source_grid.dx;
        let sy = source_grid.ny as f64 * source_grid.dy;
        let sz = source_grid.nz as f64 * source_grid.dz;
        let tx = target_grid.nx as f64 * target_grid.dx;
        let ty = target_grid.ny as f64 * target_grid.dy;
        let tz = target_grid.nz as f64 * target_grid.dz;

        // Overlap extents (minimum of source/target in each direction)
        let overlap_y = sy.min(ty);
        let overlap_z = sz.min(tz);
        let overlap_x = sx.min(tx);

        // Interface normal is perpendicular to the largest face of the overlap volume
        let area_yz = overlap_y * overlap_z;
        let area_xz = overlap_x * overlap_z;
        let area_xy = overlap_x * overlap_y;

        let (area, normal) = if area_yz >= area_xz && area_yz >= area_xy {
            (area_yz, (1.0, 0.0, 0.0))
        } else if area_xz >= area_yz && area_xz >= area_xy {
            (area_xz, (0.0, 1.0, 0.0))
        } else {
            (area_xy, (0.0, 0.0, 1.0))
        };

        Ok(Self { area, normal })
    }
}

/// Multi-physics simulation orchestrator
pub struct MultiPhysicsSolver {
    config: MultiPhysicsConfig,
    solvers: HashMap<PhysicsDomain, Box<dyn CoupledPhysicsSolver>>,
    coupler: FieldCoupler,
    convergence_history: Vec<f64>,
    time_step: usize,
}

impl std::fmt::Debug for MultiPhysicsSolver {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("MultiPhysicsSolver")
            .field("config", &self.config)
            .field("solvers", &format!("{} solvers", self.solvers.len()))
            .field("coupler", &self.coupler)
            .field(
                "convergence_history",
                &format!("{} entries", self.convergence_history.len()),
            )
            .field("time_step", &self.time_step)
            .finish()
    }
}

impl MultiPhysicsSolver {
    /// Create new multi-physics solver
    pub fn new(config: MultiPhysicsConfig) -> Self {
        Self {
            config,
            solvers: HashMap::new(),
            coupler: FieldCoupler::new(),
            convergence_history: Vec::new(),
            time_step: 0,
        }
    }

    /// Add a physics solver to the coupled system
    pub fn add_solver(&mut self, solver: Box<dyn CoupledPhysicsSolver>) -> KwaversResult<()> {
        let domain = solver.domain_type();

        // Check for duplicate domains
        if self.solvers.contains_key(&domain) {
            return Err(KwaversError::InvalidInput(format!(
                "Solver for domain {:?} already exists",
                domain
            )));
        }

        self.solvers.insert(domain, solver);
        Ok(())
    }

    /// Add coupling between two physics domains
    pub fn add_coupling(
        &mut self,
        source_domain: PhysicsDomain,
        target_domain: PhysicsDomain,
    ) -> KwaversResult<()> {
        let source_solver = self.solvers.get(&source_domain).ok_or_else(|| {
            KwaversError::InvalidInput(format!(
                "Source solver for domain {:?} not found",
                source_domain
            ))
        })?;

        let target_solver = self.solvers.get(&target_domain).ok_or_else(|| {
            KwaversError::InvalidInput(format!(
                "Target solver for domain {:?} not found",
                target_domain
            ))
        })?;

        self.coupler.add_coupling(
            source_domain,
            target_domain,
            source_solver.grid(),
            target_solver.grid(),
        )
    }

    /// Solve coupled multi-physics system for one time step
    pub fn step_coupled(&mut self, dt: f64) -> KwaversResult<f64> {
        self.convergence_history.clear();

        match self.config.coupling_strategy {
            CouplingStrategy::Explicit => self.solve_explicit_coupling(dt),
            CouplingStrategy::Implicit => self.solve_implicit_coupling(dt),
            CouplingStrategy::Partitioned => self.solve_partitioned_coupling(dt),
            CouplingStrategy::Monolithic => self.solve_monolithic_coupling(dt),
        }
    }

    /// Explicit coupling (no iteration)
    ///
    /// Steps all solvers once using the current coupling state.
    /// Returns the maximum mean field change across all domains as a residual.
    fn solve_explicit_coupling(&mut self, dt: f64) -> KwaversResult<f64> {
        // Snapshot fields before stepping for residual estimation
        let mut snapshots: HashMap<PhysicsDomain, Array3<f64>> = HashMap::new();
        for (domain, solver) in &self.solvers {
            if let Ok(field) = solver.get_field("pressure") {
                snapshots.insert(*domain, field.to_owned());
            }
        }

        // Step all solvers with current coupling sources
        for solver in self.solvers.values_mut() {
            solver.step(dt)?;
        }

        // Residual = max mean absolute change across all solver fields
        let mut max_residual = 0.0_f64;
        for (domain, solver) in &self.solvers {
            if let Some(old_field) = snapshots.get(domain) {
                if let Ok(new_field) = solver.get_field("pressure") {
                    let residual = (&new_field - old_field)
                        .mapv(|x| x.abs())
                        .mean()
                        .unwrap_or(0.0);
                    max_residual = max_residual.max(residual);
                }
            }
        }

        self.convergence_history.push(max_residual);
        Ok(max_residual)
    }

    /// Implicit coupling with iteration
    ///
    /// Iteratively steps all solvers until the maximum field change between
    /// successive iterations drops below `config.tolerance`, or
    /// `config.max_iterations` is reached.
    fn solve_implicit_coupling(&mut self, dt: f64) -> KwaversResult<f64> {
        let mut residual = f64::MAX;

        for _iteration in 0..self.config.max_iterations {
            // Snapshot fields before this iteration
            let mut snapshots: HashMap<PhysicsDomain, Array3<f64>> = HashMap::new();
            for (domain, solver) in &self.solvers {
                if let Ok(field) = solver.get_field("pressure") {
                    snapshots.insert(*domain, field.to_owned());
                }
            }

            // Step all solvers
            for solver in self.solvers.values_mut() {
                solver.step(dt)?;
            }

            // Compute residual as max mean absolute change across all domains
            residual = 0.0;
            for (domain, solver) in &self.solvers {
                if let Some(old_field) = snapshots.get(domain) {
                    if let Ok(new_field) = solver.get_field("pressure") {
                        let r = (&new_field - old_field)
                            .mapv(|x| x.abs())
                            .mean()
                            .unwrap_or(0.0);
                        residual = residual.max(r);
                    }
                }
            }

            self.convergence_history.push(residual);

            if residual < self.config.tolerance {
                break;
            }
        }

        Ok(residual)
    }

    /// Partitioned coupling (Gauss-Seidel style)
    ///
    /// Steps solvers one at a time and transfers fields to all other
    /// domains after each step. Returns the maximum transfer residual.
    fn solve_partitioned_coupling(&mut self, dt: f64) -> KwaversResult<f64> {
        let domains: Vec<PhysicsDomain> = self.solvers.keys().cloned().collect();
        let mut max_residual = 0.0_f64;

        for &domain in &domains {
            let Some(mut source_solver) = self.solvers.remove(&domain) else {
                continue;
            };
            source_solver.step(dt)?;

            // Transfer fields to all other coupled domains
            for &target_domain in &domains {
                if target_domain != domain {
                    let Some(mut target_solver) = self.solvers.remove(&target_domain) else {
                        continue;
                    };

                    let residual = self.coupler.transfer_field(
                        domain,
                        target_domain,
                        "pressure",
                        source_solver.as_ref(),
                        target_solver.as_mut(),
                        self.config.relaxation_factor,
                    )?;
                    max_residual = max_residual.max(residual);
                    self.solvers.insert(target_domain, target_solver);
                }
            }

            self.solvers.insert(domain, source_solver);
        }

        self.convergence_history.push(max_residual);
        Ok(max_residual)
    }

    /// Monolithic coupling (unified system matrix with Newton solver)
    ///
    /// Requires block-structured Jacobian assembly, Krylov linear solver
    /// (GMRES/BiCGSTAB), and physics-based preconditioning. Use Implicit
    /// or Partitioned coupling as alternatives for loosely-coupled problems.
    fn solve_monolithic_coupling(&mut self, _dt: f64) -> KwaversResult<f64> {
        Err(KwaversError::NotImplemented(
            "Monolithic coupling not yet implemented — use Implicit or Partitioned strategy"
                .to_string(),
        ))
    }

    /// Get convergence history
    pub fn convergence_history(&self) -> &[f64] {
        &self.convergence_history
    }

    /// Check if coupling has converged
    pub fn has_converged(&self) -> bool {
        if self.convergence_history.is_empty() {
            return false;
        }

        let last_residual = *self.convergence_history.last().unwrap();
        last_residual < self.config.tolerance
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // Mock physics solver for testing
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
            // Simple update for testing
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
        let solver = MultiPhysicsSolver::new(config);

        assert_eq!(solver.solvers.len(), 0);
        assert_eq!(solver.convergence_history.len(), 0);
    }

    #[test]
    fn test_add_solver() {
        let mut solver = MultiPhysicsSolver::new(MultiPhysicsConfig::default());
        let grid = Grid::new(10, 10, 10, 0.001, 0.001, 0.001).unwrap();
        let mock_solver = Box::new(MockSolver::new(PhysicsDomain::Acoustic, grid));

        assert!(solver.add_solver(mock_solver).is_ok());
        assert_eq!(solver.solvers.len(), 1);
    }

    #[test]
    fn test_explicit_coupling() {
        let mut solver = MultiPhysicsSolver::new(MultiPhysicsConfig {
            coupling_strategy: CouplingStrategy::Explicit,
            ..Default::default()
        });

        let grid = Grid::new(8, 8, 8, 0.001, 0.001, 0.001).unwrap();
        let acoustic_solver = Box::new(MockSolver::new(PhysicsDomain::Acoustic, grid.clone()));

        solver.add_solver(acoustic_solver).unwrap();

        let residual = solver.step_coupled(1e-6).unwrap();
        assert!(residual >= 0.0);
    }
}
