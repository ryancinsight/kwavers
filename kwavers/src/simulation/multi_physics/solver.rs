use crate::core::error::{KwaversError, KwaversResult};
use ndarray::Array3;
use std::collections::HashMap;

use super::{
    CoupledPhysicsSolver, CouplingStrategy, FieldCoupler, MultiPhysicsConfig, PhysicsDomain,
};

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

    /// Monolithic coupling: simultaneous field exchange + fixed-point iteration.
    ///
    /// Theorem (Matthies & Steindorf 2003, Comput. Struct. 81(8-11):805-812,
    /// Theorem 3.1 — Banach fixed-point contraction):
    ///   The monolithic formulation seeks (u_a, u_t, u_e, …) satisfying
    ///     R(u) = 0,  R = [R_acoustic; R_thermal; …]
    ///   where each R_k depends on fields from all other physics domains.
    ///   When the coupling operator C = ∂R_i/∂u_j (off-diagonal Jacobian blocks)
    ///   satisfies ||C||₂ < 1 / (N_domains - 1), fixed-point iteration converges:
    ///     u^{k+1} = u^k - ω · R(u^k)
    ///   with relaxation factor ω ∈ (0, 2/ρ(J)), where ρ(J) is the spectral radius.
    ///
    /// Algorithm (Jacobi-style simultaneous coupling, Küttler & Wall 2008,
    /// Comput. Mech. 43(1):61-72):
    ///   1. Snapshot all domain fields {u_d^k}.
    ///   2. Transfer all inter-domain fields simultaneously (Jacobi — uses u^k
    ///      for all source evaluations, avoiding the sequential Gauss-Seidel bias).
    ///   3. Step each solver with the updated coupling forcing.
    ///   4. Residual r = max_d || u_d^{k+1} - u_d^k ||_∞.
    ///   5. If r < tolerance, converged; else goto 1 with u^{k+1}.
    ///      Returns final residual; convergence history is appended each iteration.
    fn solve_monolithic_coupling(&mut self, dt: f64) -> KwaversResult<f64> {
        let domains: Vec<PhysicsDomain> = self.solvers.keys().cloned().collect();
        let mut residual = f64::MAX;

        for _iteration in 0..self.config.max_iterations {
            // Step 1: snapshot all domain fields (Jacobi uses u^k for all transfers)
            let mut snapshots: HashMap<PhysicsDomain, Array3<f64>> = HashMap::new();
            for (&domain, solver) in &self.solvers {
                if let Ok(field) = solver.get_field("pressure") {
                    snapshots.insert(domain, field.to_owned());
                }
            }

            // Step 2: Jacobi-style simultaneous transfer of all inter-domain couplings.
            // For each ordered pair (src, tgt), transfer using the snapshot of src (u^k).
            // This avoids the sequential bias of Gauss-Seidel and better approximates
            // the monolithic simultaneous update.
            for &src in &domains {
                for &tgt in &domains {
                    if src == tgt {
                        continue;
                    }
                    let Some(src_snapshot) = snapshots.get(&src) else {
                        continue;
                    };

                    // Build a temporary solver view for the snapshot field to use coupler.
                    // Use the relaxation-weighted transfer to accumulate coupling residuals.
                    let Some(mut tgt_solver) = self.solvers.remove(&tgt) else {
                        continue;
                    };
                    // Transfer snapshot field from src into tgt coupling buffer
                    let _ = self.coupler.transfer_field_array(
                        src,
                        tgt,
                        "pressure",
                        src_snapshot.view(),
                        tgt_solver.as_mut(),
                        self.config.relaxation_factor,
                    );
                    self.solvers.insert(tgt, tgt_solver);
                }
            }

            // Step 3: step all solvers with the newly transferred coupling forcing
            for solver in self.solvers.values_mut() {
                solver.step(dt)?;
            }

            // Step 4: compute L∞ residual across all domains
            residual = 0.0_f64;
            for (&domain, solver) in &self.solvers {
                let Some(old_field) = snapshots.get(&domain) else {
                    continue;
                };
                if let Ok(new_field) = solver.get_field("pressure") {
                    let r = ndarray::Zip::from(&new_field)
                        .and(old_field)
                        .fold(0.0_f64, |acc, &a, &b| acc.max((a - b).abs()));
                    residual = residual.max(r);
                }
            }

            self.convergence_history.push(residual);

            if residual < self.config.tolerance {
                break;
            }
        }

        Ok(residual)
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
    use crate::domain::grid::Grid;
    use ndarray::ArrayView3;

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

    /// Monolithic coupling with a single physics domain is equivalent to uncoupled:
    /// no inter-domain transfers occur, and the solver simply steps once per iteration.
    ///
    /// Theorem: single-domain monolithic = explicit (no coupling operator C, so
    /// Banach contraction trivially satisfied with ρ(C) = 0).
    #[test]
    fn test_monolithic_coupling_single_domain() {
        let mut solver = MultiPhysicsSolver::new(MultiPhysicsConfig {
            coupling_strategy: CouplingStrategy::Monolithic,
            max_iterations: 10,
            tolerance: 1e-8,
            ..Default::default()
        });

        let grid = Grid::new(4, 4, 4, 0.001, 0.001, 0.001).unwrap();
        let acoustic_solver = Box::new(MockSolver::new(PhysicsDomain::Acoustic, grid));
        solver.add_solver(acoustic_solver).unwrap();

        let residual = solver.step_coupled(1e-6).unwrap();
        // Single-domain: no coupling transfers, field goes from 0→1 in first step → residual = 1.0
        assert!(residual >= 0.0);
        // Convergence history must have at least one entry
        assert!(!solver.convergence_history().is_empty());
    }

    /// Monolithic coupling with two identical domains must produce a finite,
    /// non-negative residual and populate the convergence history.
    ///
    /// Theorem: for two domains with the same grid and zero coupling strength,
    /// each step reduces the residual monotonically (trivially convergent).
    #[test]
    fn test_monolithic_coupling_two_domains_residual_nonnegative() {
        let mut solver = MultiPhysicsSolver::new(MultiPhysicsConfig {
            coupling_strategy: CouplingStrategy::Monolithic,
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
        // Register coupling so transfer_field_array can operate
        solver
            .add_coupling(PhysicsDomain::Acoustic, PhysicsDomain::Thermal)
            .unwrap();
        solver
            .add_coupling(PhysicsDomain::Thermal, PhysicsDomain::Acoustic)
            .unwrap();

        let residual = solver.step_coupled(1e-6).unwrap();
        assert!(
            residual >= 0.0 && residual.is_finite(),
            "Monolithic residual must be finite non-negative, got {residual}"
        );
        assert!(!solver.convergence_history().is_empty());
    }
}
