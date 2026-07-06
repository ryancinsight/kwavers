use super::config::{CavitationCouplingConfig, CavitationCouplingType};
use crate::inverse::pinn::ml::physics::PinnCouplingInterface;
use kwavers_physics::bubble_dynamics::{BubbleState, KellerMiksisModel};

/// Cavitation coupled physics domain.
pub struct CavitationCoupledDomain<B: coeus_ops::BackendOps<f32> + coeus_ops::CpuBackend + Default> {
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

// Manual `Debug` impl: `#[derive(Debug)]` on a generic struct adds a spurious
// `B: Debug` bound even though `B` appears only in `PhantomData<B>`, which
// would incorrectly propagate a `Debug` requirement onto every backend.
impl<B: coeus_ops::BackendOps<f32> + coeus_ops::CpuBackend + Default> std::fmt::Debug
    for CavitationCoupledDomain<B>
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CavitationCoupledDomain")
            .field("config", &self.config)
            .field("coupling_type", &self.coupling_type)
            .field("bubble_model", &self.bubble_model)
            .field("bubble_states_len", &self.bubble_states.len())
            .field("bubble_locations", &self.bubble_locations)
            .field("coupling_interfaces", &self.coupling_interfaces)
            .field("domain_dims", &self.domain_dims)
            .finish_non_exhaustive()
    }
}
