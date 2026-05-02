//! `UniversalPINNSolver` struct definition.
//!
//! SRP: changes when the solver's fields change.

use super::types::{UniversalSolverStats, UniversalTrainingConfig};
use crate::solver::inverse::pinn::ml::physics::PhysicsDomainRegistry;
use burn::tensor::backend::AutodiffBackend;
use std::collections::HashMap;

/// Universal PINN solver for any physics domain
#[derive(Debug)]
pub struct UniversalPINNSolver<B: AutodiffBackend> {
    pub(super) physics_registry: PhysicsDomainRegistry<B>,
    pub(super) models: HashMap<String, crate::solver::inverse::pinn::ml::BurnPINN2DWave<B>>,
    pub(super) configs: HashMap<String, UniversalTrainingConfig>,
    pub(super) stats: HashMap<String, UniversalSolverStats>,
}
