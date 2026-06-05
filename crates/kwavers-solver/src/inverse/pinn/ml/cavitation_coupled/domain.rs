use super::config::{CavitationCouplingConfig, CavitationCouplingType};
use crate::inverse::pinn::ml::physics::PinnCouplingInterface;
use burn::tensor::backend::AutodiffBackend;
use kwavers_physics::bubble_dynamics::{BubbleState, KellerMiksisModel};

/// Cavitation coupled physics domain.
#[derive(Debug)]
pub struct CavitationCoupledDomain<B: AutodiffBackend> {
    pub config: CavitationCouplingConfig,
    pub coupling_type: CavitationCouplingType,
    pub bubble_model: KellerMiksisModel,
    pub bubble_states: Vec<BubbleState>,
    /// Physics-driven nucleation sites (x, y, z) metres.
    pub bubble_locations: Vec<(f64, f64, f64)>,
    pub coupling_interfaces: Vec<PinnCouplingInterface>,
    pub domain_dims: Vec<f64>,
    /// `pub(super)` so sibling modules can write `Self { ..., _backend: PhantomData }`.
    pub(super) _backend: std::marker::PhantomData<B>,
}
