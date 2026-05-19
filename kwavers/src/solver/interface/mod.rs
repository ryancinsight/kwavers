//! Unified Solver Interface
//!
//! This module defines the common interfaces, traits, and configurations
//! for all solver implementations in Kwavers.

// pub mod config; // Consolidated into crate::solver::config
pub mod factory;
pub mod pinn_beamforming;
pub mod progress;
pub mod solver;

pub use self::factory::{
    ApolloFourierBackend, FactoryConfiguration, FactoryError, FourierBackend, FactoryGridParameters,
    FactoryMediumParameters, MeshProvider, RegistrationEngine, FactorySourceParameters,
};
pub use self::pinn_beamforming::{
    BeamformingTrainingMetrics, DeviceConfig, DistributedConfig, DistributedPinnProvider,
    GpuMetrics, InferenceConfig, LoadBalancingStrategy, ModelArchitecture, ModelInfo,
    PinnBeamformingActivationFunction, PinnBeamformingConfig, PinnBeamformingDecompositionStrategy,
    PinnBeamformingProvider, InterfacePinnBeamformingResult, PinnBeamformingUncertaintyConfig,
    PinnModelConfig, PinnProviderRegistry, ProcessingMetadata,
};
pub use self::progress::{
    ConsoleProgressReporter, FieldsSummary, ProgressData, ProgressReporter, ProgressUpdate,
};
pub use self::solver::{Solver, SolverStatistics};
pub use crate::solver::feature::{FeatureManager, SolverFeature};
