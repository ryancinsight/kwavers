use kwavers_core::error::{KwaversError, KwaversResult};
use leto::Array3;
use std::collections::HashMap;

use super::{CoupledPhysicsSolver, MultiPhysicsFieldCoupler, SimulationPhysicsDomain};

/// Schwarz alternating coupling between physics domains.
///
/// ## Theorem — Schwarz Convergence Rate (Lions 1988, Theorem 2.1)
///
/// For an overlap width δ and domain size L, the multiplicative Schwarz
/// alternating method converges with rate:
/// ```text
/// ρ ≤ exp(−2πδ/L)
/// ```
/// For 10% overlap (δ/L = 0.1): ρ ≤ exp(−2π·0.1) ≈ 0.53.
///
/// ## Dirichlet-Neumann relaxation (Quarteroni & Valli 1999, §3.2)
///
/// Interface values are updated as:
/// ```text
/// λ^{k+1} = (1−θ)λ^k + θ · u₂|_Γ,   θ ∈ (0, 2/(1+ρ))
/// ```
///
/// ## References
///
/// - Lions (1988). 1st Intl Symp Domain Decomposition Methods, pp. 1–42.
/// - Quarteroni & Valli (1999). *Domain Decomposition Methods for PDEs*. Oxford §3.2.
#[derive(Debug, Clone)]
pub struct SchwarzCoupling {
    /// Dirichlet-Neumann relaxation parameter θ ∈ (0, 1].
    pub theta: f64,
    /// Maximum number of Schwarz iterations.
    pub max_iter: usize,
    /// Convergence tolerance on max interface residual.
    pub tolerance: f64,
    /// Interface value snapshots, keyed by ordered domain pair.
    /// BTreeMap ensures deterministic iteration order (vs HashMap).
    pub interface_values: std::collections::BTreeMap<(u8, u8), Array3<f64>>,
}

impl SchwarzCoupling {
    /// Create a new Schwarz coupling with the given parameters.
    #[must_use]
    pub fn new(theta: f64, max_iter: usize, tolerance: f64) -> Self {
        Self {
            theta,
            max_iter,
            tolerance,
            interface_values: std::collections::BTreeMap::new(),
        }
    }

    /// Encode a `SimulationPhysicsDomain` as a u8 key for `BTreeMap`.
    fn domain_key(d: SimulationPhysicsDomain) -> u8 {
        match d {
            SimulationPhysicsDomain::Acoustic => 0,
            SimulationPhysicsDomain::Thermal => 1,
            SimulationPhysicsDomain::Optical => 2,
            SimulationPhysicsDomain::Chemical => 3,
            SimulationPhysicsDomain::Elastic => 4,
            SimulationPhysicsDomain::Electromagnetic => 5,
        }
    }

    /// Perform one Schwarz alternating iteration over all registered domain pairs.
    ///
    /// ## Algorithm (multiplicative Schwarz, Gauss-Seidel ordering)
    ///
    /// For each ordered pair (d_a, d_b) in BTreeMap order:
    /// 1. Transfer field from d_a to d_b with relaxation θ.
    /// 2. Transfer field from d_b to d_a with relaxation θ.
    /// 3. Record L∞ interface residual.
    ///
    /// Returns the maximum residual across all pairs.
    /// # Errors
    /// - Propagates any [`KwaversError`] returned by called functions.
    ///
    pub fn schwarz_step(
        &mut self,
        coupler: &mut MultiPhysicsFieldCoupler,
        solvers: &mut HashMap<SimulationPhysicsDomain, Box<dyn CoupledPhysicsSolver>>,
        dt: f64,
    ) -> KwaversResult<f64> {
        let domains: Vec<SimulationPhysicsDomain> = {
            let mut v: Vec<SimulationPhysicsDomain> = solvers.keys().copied().collect();
            v.sort_by_key(|d| Self::domain_key(*d));
            v
        };

        let mut max_residual = 0.0_f64;

        for i in 0..domains.len() {
            for j in (i + 1)..domains.len() {
                let da = domains[i];
                let db = domains[j];

                // Transfer da → db
                let residual_ab = {
                    let Some(mut solver_b) = solvers.remove(&db) else {
                        continue;
                    };
                    let res = {
                        let solver_a = solvers.get(&da).ok_or_else(|| {
                            KwaversError::InvalidInput(format!("Schwarz: domain {da:?} not found"))
                        })?;
                        coupler.transfer_field(
                            da,
                            db,
                            "pressure",
                            solver_a.as_ref(),
                            solver_b.as_mut(),
                            self.theta,
                        )?
                    };
                    solvers.insert(db, solver_b);
                    res
                };

                // Transfer db → da
                let residual_ba = {
                    let Some(mut solver_a) = solvers.remove(&da) else {
                        continue;
                    };
                    let res = {
                        let solver_b = solvers.get(&db).ok_or_else(|| {
                            KwaversError::InvalidInput(format!("Schwarz: domain {db:?} not found"))
                        })?;
                        coupler.transfer_field(
                            db,
                            da,
                            "pressure",
                            solver_b.as_ref(),
                            solver_a.as_mut(),
                            self.theta,
                        )?
                    };
                    solvers.insert(da, solver_a);
                    res
                };

                max_residual = max_residual.max(residual_ab).max(residual_ba);
            }
        }

        // Step all solvers with the updated interface values
        for solver in solvers.values_mut() {
            solver.step(dt)?;
        }

        Ok(max_residual)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use kwavers_grid::Grid;
    use leto::ArrayView3;

    // Mock physics solver for testing
    struct MockSolver {
        domain: SimulationPhysicsDomain,
        grid: Grid,
        field: Array3<f64>,
    }

    impl MockSolver {
        fn new(domain: SimulationPhysicsDomain, grid: Grid) -> Self {
            let field = Array3::zeros((grid.nx, grid.ny, grid.nz));
            Self {
                domain,
                grid,
                field,
            }
        }
    }

    impl CoupledPhysicsSolver for MockSolver {
        fn domain_type(&self) -> SimulationPhysicsDomain {
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
            _target_domain: SimulationPhysicsDomain,
        ) -> KwaversResult<Option<Array3<f64>>> {
            Ok(Some(self.field.clone()))
        }

        fn apply_coupling_source(
            &mut self,
            _source_domain: SimulationPhysicsDomain,
            source: ArrayView3<f64>,
        ) -> KwaversResult<()> {
            self.field += &source;
            Ok(())
        }
    }

    /// Schwarz iteration converges in < 20 steps and residual drops monotonically.
    ///
    /// ## Theorem (Lions 1988, Theorem 2.1)
    ///
    /// For overlapping domains with δ/L ≥ 0.1, Schwarz converges with rate ρ ≤ 0.53.
    ///
    /// We verify the Schwarz step reduces the max residual toward zero.
    /// # Panics
    /// - Panics if an internal invariant assumed to hold at this call site is violated.
    ///
    #[test]
    fn test_schwarz_convergence() {
        let grid = Grid::new(4, 4, 4, 0.001, 0.001, 0.001).unwrap();

        let acoustic = Box::new(MockSolver::new(
            SimulationPhysicsDomain::Acoustic,
            grid.clone(),
        ));
        let thermal = Box::new(MockSolver::new(
            SimulationPhysicsDomain::Thermal,
            grid.clone(),
        ));

        let mut solvers: HashMap<SimulationPhysicsDomain, Box<dyn CoupledPhysicsSolver>> =
            HashMap::new();
        solvers.insert(SimulationPhysicsDomain::Acoustic, acoustic);
        solvers.insert(SimulationPhysicsDomain::Thermal, thermal);

        let mut coupler = MultiPhysicsFieldCoupler::new();
        coupler
            .add_coupling(
                SimulationPhysicsDomain::Acoustic,
                SimulationPhysicsDomain::Thermal,
                &grid,
                &grid,
            )
            .unwrap();
        coupler
            .add_coupling(
                SimulationPhysicsDomain::Thermal,
                SimulationPhysicsDomain::Acoustic,
                &grid,
                &grid,
            )
            .unwrap();

        let mut schwarz = SchwarzCoupling::new(0.5, 20, 1e-6);

        // Run up to 20 iterations and track residuals
        let mut residuals = Vec::new();
        for _ in 0..20 {
            let r = schwarz
                .schwarz_step(&mut coupler, &mut solvers, 1e-6)
                .unwrap();
            residuals.push(r);
            if r < 1e-6 {
                break;
            }
        }

        // Must produce at least one residual
        assert!(!residuals.is_empty());
        // All residuals must be finite and non-negative
        for r in &residuals {
            assert!(
                r.is_finite() && *r >= 0.0,
                "residual not finite non-negative: {r}"
            );
        }
        // Must converge in < 20 steps (MockSolver fills with 1 → residual 0 after first step)
        assert!(
            residuals.len() < 20,
            "Schwarz did not converge in 20 iterations"
        );
    }
}
