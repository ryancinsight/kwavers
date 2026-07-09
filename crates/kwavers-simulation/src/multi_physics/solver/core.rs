//! `SimulationMultiPhysicsSolver` — multi-physics simulation orchestrator.

use kwavers_core::error::{KwaversError, KwaversResult};
use leto::Array3;
use std::collections::HashMap;

use super::super::residual::max_abs_difference;
use super::super::{
    CoupledPhysicsSolver, MultiPhysicsConfig, MultiPhysicsFieldCoupler, SimulationCouplingStrategy,
    SimulationPhysicsDomain,
};

/// Multi-physics simulation orchestrator
pub struct SimulationMultiPhysicsSolver {
    config: MultiPhysicsConfig,
    pub(super) solvers: HashMap<SimulationPhysicsDomain, Box<dyn CoupledPhysicsSolver>>,
    coupler: MultiPhysicsFieldCoupler,
    pub(super) convergence_history: Vec<f64>,
    time_step: usize,
}

impl std::fmt::Debug for SimulationMultiPhysicsSolver {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("SimulationMultiPhysicsSolver")
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

impl SimulationMultiPhysicsSolver {
    /// Create new multi-physics solver
    /// # Errors
    /// - Returns [`Err`] if an internal constraint is violated.
    ///
    #[must_use]
    pub fn new(config: MultiPhysicsConfig) -> Self {
        Self {
            config,
            solvers: HashMap::new(),
            coupler: MultiPhysicsFieldCoupler::new(),
            convergence_history: Vec::new(),
            time_step: 0,
        }
    }

    /// Add a physics solver to the coupled system
    /// # Errors
    /// - Returns [`KwaversError::InvalidInput`] if the precondition for invalid or out-of-range input parameters is violated.
    /// - Propagates any [`KwaversError`] returned by called functions.
    ///
    pub fn add_solver(&mut self, solver: Box<dyn CoupledPhysicsSolver>) -> KwaversResult<()> {
        let domain = solver.domain_type();
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
    /// # Errors
    /// - Propagates any [`KwaversError`] returned by called functions.
    ///
    pub fn add_coupling(
        &mut self,
        source_domain: SimulationPhysicsDomain,
        target_domain: SimulationPhysicsDomain,
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
    /// # Errors
    /// - Returns [`Err`] if an internal constraint is violated.
    ///
    pub fn step_coupled(&mut self, dt: f64) -> KwaversResult<f64> {
        self.convergence_history.clear();
        match self.config.coupling_strategy {
            SimulationCouplingStrategy::Explicit => self.solve_explicit_coupling(dt),
            SimulationCouplingStrategy::Implicit => self.solve_implicit_coupling(dt),
            SimulationCouplingStrategy::Partitioned => self.solve_partitioned_coupling(dt),
            SimulationCouplingStrategy::Monolithic => self.solve_monolithic_coupling(dt),
        }
    }

    /// One Jacobi coupling sweep: snapshot every domain's pressure, transfer each
    /// snapshot to all other domains as (relaxed) coupling forcing, advance every
    /// solver one step, and return the L∞ field change (the coupling residual),
    /// recording it in `convergence_history`.
    ///
    /// This is the shared per-sweep physics for all multi-domain strategies — they
    /// differ only in how many sweeps they run, not in the sweep itself. Jacobi:
    /// every source uses the pre-sweep field, so the result is order-independent.
    /// (Previously the explicit/implicit paths stepped solvers WITHOUT any
    /// `transfer_field`, so the domains never actually coupled.)
    fn coupled_jacobi_sweep(&mut self, dt: f64) -> KwaversResult<f64> {
        let domains: Vec<SimulationPhysicsDomain> = self.solvers.keys().copied().collect();
        let mut snapshots: HashMap<SimulationPhysicsDomain, Array3<f64>> = HashMap::new();
        for (&domain, solver) in &self.solvers {
            if let Ok(field) = solver.get_field("pressure") {
                snapshots.insert(domain, field.to_owned());
            }
        }
        for &src in &domains {
            for &tgt in &domains {
                if src == tgt {
                    continue;
                }
                let Some(src_snapshot) = snapshots.get(&src) else {
                    continue;
                };
                let Some(mut tgt_solver) = self.solvers.remove(&tgt) else {
                    continue;
                };
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
        for solver in self.solvers.values_mut() {
            solver.step(dt)?;
        }
        let mut residual = 0.0_f64;
        for (&domain, solver) in &self.solvers {
            if let Some(old_field) = snapshots.get(&domain) {
                if let Ok(new_field) = solver.get_field("pressure") {
                    residual = residual.max(max_abs_difference(new_field, old_field.view())?);
                }
            }
        }
        self.convergence_history.push(residual);
        Ok(residual)
    }

    /// Iterate the Jacobi sweep to a coupling fixed point (strong coupling).
    fn iterate_coupled_to_fixed_point(&mut self, dt: f64) -> KwaversResult<f64> {
        let mut residual = f64::MAX;
        for _iteration in 0..self.config.max_iterations {
            residual = self.coupled_jacobi_sweep(dt)?;
            if residual < self.config.tolerance {
                break;
            }
        }
        Ok(residual)
    }

    fn solve_explicit_coupling(&mut self, dt: f64) -> KwaversResult<f64> {
        // Loose (staggered-explicit) coupling: exactly one exchange+step sweep.
        self.coupled_jacobi_sweep(dt)
    }

    fn solve_implicit_coupling(&mut self, dt: f64) -> KwaversResult<f64> {
        // Strong (implicit) coupling: sub-iterate the exchange+step sweep to a
        // fixed point within the time step.
        self.iterate_coupled_to_fixed_point(dt)
    }

    fn solve_partitioned_coupling(&mut self, dt: f64) -> KwaversResult<f64> {
        let domains: Vec<SimulationPhysicsDomain> = self.solvers.keys().copied().collect();
        let mut max_residual = 0.0_f64;
        for &domain in &domains {
            let Some(mut source_solver) = self.solvers.remove(&domain) else {
                continue;
            };
            source_solver.step(dt)?;
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
    /// Theorem (Matthies & Steindorf 2003, Comput. Struct. 81(8-11):805-812, Theorem 3.1):
    ///   Fixed-point iteration converges when ||C||₂ < 1/(N_domains − 1).
    ///
    /// Algorithm (Jacobi-style, Küttler & Wall 2008, Comput. Mech. 43(1):61-72):
    ///   1. Snapshot all domain fields.
    ///   2. Transfer all inter-domain fields simultaneously (Jacobi, u^k for all sources).
    ///   3. Step each solver with updated coupling forcing.
    ///   4. Compute L∞ residual. Converge or iterate.
    /// # Errors
    /// - Propagates any [`KwaversError`] returned by called functions.
    ///
    fn solve_monolithic_coupling(&mut self, dt: f64) -> KwaversResult<f64> {
        // Jacobi fixed-point iteration of the shared exchange+step sweep (steps 1–4
        // of the algorithm above). Same per-sweep physics as the implicit strategy.
        self.iterate_coupled_to_fixed_point(dt)
    }

    /// Get convergence history
    #[must_use]
    pub fn convergence_history(&self) -> &[f64] {
        &self.convergence_history
    }

    /// Check if coupling has converged
    /// # Panics
    /// - Panics if an internal invariant assumed to hold at this call site is violated.
    ///
    #[must_use]
    pub fn has_converged(&self) -> bool {
        if self.convergence_history.is_empty() {
            return false;
        }
        let last_residual = *self.convergence_history.last().unwrap();
        last_residual < self.config.tolerance
    }
}
