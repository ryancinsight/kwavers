//! Sonoluminescence coupled physics domain.
//!
//! Partitioned by responsibility:
//! - `construction` — struct construction, coupling interface factory, state update, `Clone`.
//! - `residuals` — light-source computation and Maxwell residual with sonoluminescence sources.
//! - `physics_domain_impl` — `SimulationPhysicsDomain<B>` trait implementation.

mod construction;
mod physics_domain_impl;
mod residuals;

use kwavers_physics::optics::sonoluminescence::SonoluminescenceEmission;
use crate::inverse::pinn::ml::physics::PinnCouplingInterface;
use burn::tensor::backend::AutodiffBackend;
use ndarray::Array3;

use super::config::{SonoluminescenceCouplingConfig, SonoluminescenceCouplingType};

/// Sonoluminescence coupled physics domain implementation.
#[derive(Debug)]
pub struct SonoluminescenceCoupledDomain<B: AutodiffBackend> {
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
