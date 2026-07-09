//! `UniversalPINNSolver` struct definition.
//!
//! SRP: changes when the solver's fields change.

use super::types::{UniversalSolverStats, UniversalTrainingConfig};
use crate::inverse::pinn::ml::physics::PhysicsDomainRegistry;
use std::collections::HashMap;

/// Universal PINN solver for any physics domain
pub struct UniversalPINNSolver<B: coeus_ops::BackendOps<f32> + coeus_ops::CpuBackend + Default> {
    pub(super) physics_registry: PhysicsDomainRegistry<B>,
    pub(super) models: HashMap<String, crate::inverse::pinn::ml::PinnWave2D<B>>,
    pub(super) configs: HashMap<String, UniversalTrainingConfig>,
    pub(super) stats: HashMap<String, UniversalSolverStats>,
}

// Manual `Debug` impl: `PinnWave2D<B>` requires the `CpuAddressableStorage`
// bound to implement `Debug`, which this struct's own bound does not carry.
impl<B: coeus_ops::BackendOps<f32> + coeus_ops::CpuBackend + Default> std::fmt::Debug
    for UniversalPINNSolver<B>
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("UniversalPINNSolver")
            .field("num_models", &(self.models.shape()[0] * self.models.shape()[1] * self.models.shape()[2]))
            .field("configs", &self.configs)
            .field("stats", &self.stats)
            .finish_non_exhaustive()
    }
}
