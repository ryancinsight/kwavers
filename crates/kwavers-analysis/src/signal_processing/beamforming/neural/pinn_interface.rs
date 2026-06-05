//! PINN-Beamforming Interface
//!
//! Explicit re-exports from the canonical [`kwavers_solver::interface::pinn_beamforming`] module
//! for use within the analysis::signal_processing::beamforming::neural subtree.

pub use kwavers_solver::interface::pinn_beamforming::{
    BeamformingTrainingMetrics, DeviceConfig, DistributedConfig, DistributedPinnProvider,
    GpuMetrics, InferenceConfig, InterfacePinnBeamformingResult, LoadBalancingStrategy,
    ModelArchitecture, ModelInfo, PinnBeamformingActivationFunction, PinnBeamformingConfig,
    PinnBeamformingDecompositionStrategy, PinnBeamformingProvider,
    PinnBeamformingUncertaintyConfig, PinnModelConfig, PinnProviderRegistry, ProcessingMetadata,
};
