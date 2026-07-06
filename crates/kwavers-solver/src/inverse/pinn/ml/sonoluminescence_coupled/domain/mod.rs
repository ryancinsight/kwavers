//! Sonoluminescence coupled physics domain.
//!
//! Partitioned by responsibility:
//! - `construction` — struct construction, coupling interface factory, state update, `Clone`.
//! - `residuals` — light-source computation and Maxwell residual with sonoluminescence sources.
//! - `physics_domain_impl` — `SimulationPhysicsDomain<B>` trait implementation.

mod construction;
mod physics_domain_impl;
mod residuals;

use crate::inverse::pinn::ml::physics::PinnCouplingInterface;
use kwavers_physics::optics::sonoluminescence::SonoluminescenceEmission;
use ndarray::Array3;

use super::config::{SonoluminescenceCouplingConfig, SonoluminescenceCouplingType};

/// Sonoluminescence coupled physics domain implementation.
pub struct SonoluminescenceCoupledDomain<
    B: coeus_ops::BackendOps<f32> + coeus_ops::CpuBackend + Default,
> {
    /// Coupling configuration.
    pub config: SonoluminescenceCouplingConfig,
    /// Coupling type.
    pub coupling_type: SonoluminescenceCouplingType,
    /// Sonoluminescence emission calculator.
    pub emission_calculator: SonoluminescenceEmission,
    /// Current bubble state fields.
    pub bubble_states: Array3<f64>,
    /// Current temperature field from bubbles.
    pub temperature_field: Array3<f64>,
    /// Coupling interfaces.
    pub coupling_interfaces: Vec<PinnCouplingInterface>,
    /// Backend marker.
    pub(super) _backend: std::marker::PhantomData<B>,
}

// Manual `Debug` impl: `#[derive(Debug)]` on a generic struct adds a spurious
// `B: Debug` bound even though `B` appears only in `PhantomData<B>`.
impl<B: coeus_ops::BackendOps<f32> + coeus_ops::CpuBackend + Default> std::fmt::Debug
    for SonoluminescenceCoupledDomain<B>
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("SonoluminescenceCoupledDomain")
            .field("config", &self.config)
            .field("coupling_type", &self.coupling_type)
            .field("emission_calculator", &self.emission_calculator)
            .field("bubble_states", &self.bubble_states)
            .field("temperature_field", &self.temperature_field)
            .field("coupling_interfaces", &self.coupling_interfaces)
            .finish_non_exhaustive()
    }
}
