//! `UniversalPINNSolver` accessors and high-level orchestration methods.
//!
//! SRP: changes when the public query/orchestration API changes.

use super::solver::UniversalPINNSolver;
use super::types::{
    MultiDomainTrainingResult, UniversalSolverGeometry2D, UniversalSolverStats,
    UniversalTrainingConfig,
};
use crate::inverse::pinn::ml::physics::{PinnDomainPhysicsParameters, SimulationPhysicsDomain};
use kwavers_core::error::{KwaversError, KwaversResult};
use std::collections::HashMap;
use std::time::Instant;

impl<B: coeus_ops::BackendOps<f32> + coeus_ops::CpuBackend + Default + 'static>
    UniversalPINNSolver<B>
where
    B::DeviceBuffer<f32>:
        coeus_core::CpuAddressableStorage<f32> + coeus_core::CpuAddressableStorageMut<f32>,
{
    /// Register a physics domain
    /// # Errors
    /// - Returns [`Err`] if an internal constraint is violated.
    ///
    pub fn register_physics_domain<D>(&mut self, domain: D) -> KwaversResult<()>
    where
        D: SimulationPhysicsDomain<B> + Send + Sync + 'static,
    {
        self.physics_registry.register_domain(domain)
    }

    /// Configure training for a specific physics domain
    /// # Errors
    /// - Returns [`crate::KwaversError::System`] if the precondition for a System-class constraint is violated.
    ///
    pub fn configure_domain(
        &mut self,
        domain_name: &str,
        config: UniversalTrainingConfig,
    ) -> KwaversResult<()> {
        if !self.physics_registry.has_domain(domain_name) {
            return Err(KwaversError::System(
                kwavers_core::error::SystemError::InvalidConfiguration {
                    parameter: "domain_name".to_string(),
                    reason: format!("Physics domain '{}' not registered", domain_name),
                },
            ));
        }
        self.configs.insert(domain_name.to_string(), config);
        Ok(())
    }

    /// Get available physics domains
    /// # Errors
    /// - Returns [`Err`] if an internal constraint is violated.
    ///
    pub fn available_domains(&self) -> Vec<String> {
        self.physics_registry.list_domains()
    }

    /// List all registered physics domains
    /// # Errors
    /// - Returns [`Err`] if an internal constraint is violated.
    ///
    pub fn list_registered_domains(&self) -> Vec<String> {
        self.available_domains()
    }

    /// Train a specific physics domain with default geometry
    /// # Errors
    /// - Propagates any [`crate::KwaversError`] returned by called functions.
    ///
    pub fn train_domain(
        &mut self,
        domain_name: &str,
        config: &UniversalTrainingConfig,
        physics_params: &PinnDomainPhysicsParameters,
    ) -> KwaversResult<UniversalSolverStats> {
        let geometry = UniversalSolverGeometry2D::rectangle(0.0, 1.0, 0.0, 1.0);
        let solution =
            self.solve_physics_domain(domain_name, &geometry, physics_params, Some(config))?;
        Ok(solution.stats)
    }

    /// Train all registered physics domains
    /// # Errors
    /// - Propagates any [`crate::KwaversError`] returned by called functions.
    ///
    pub fn train_all_domains(
        &mut self,
        config: &UniversalTrainingConfig,
        physics_params: &PinnDomainPhysicsParameters,
    ) -> KwaversResult<MultiDomainTrainingResult> {
        let start_time = Instant::now();
        let domains = self.list_registered_domains();
        let mut domain_stats = HashMap::new();
        let mut total_loss = 0.0;

        for domain_name in domains {
            let stats = self.train_domain(&domain_name, config, physics_params)?;
            total_loss += stats.final_loss;
            domain_stats.insert(domain_name, stats);
        }

        Ok(MultiDomainTrainingResult {
            total_loss,
            training_time: start_time.elapsed(),
            domain_stats,
        })
    }

    /// Get training statistics for a domain
    pub fn get_domain_stats(&self, domain_name: &str) -> Option<&UniversalSolverStats> {
        self.stats.get(domain_name)
    }

    /// Check if a domain is registered
    pub fn has_domain(&self, domain_name: &str) -> bool {
        self.physics_registry.has_domain(domain_name)
    }

    /// Get domain information
    pub fn get_domain_info(
        &self,
        domain_name: &str,
    ) -> Option<&(dyn SimulationPhysicsDomain<B> + Send + Sync)> {
        self.physics_registry.get_domain(domain_name)
    }
}
