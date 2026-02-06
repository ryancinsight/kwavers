//! Unified Solver Interface
//!
//! This module defines the common interfaces, traits, and configurations
//! for all solver implementations in Kwavers.

// pub mod config; // Consolidated into crate::solver::config
pub mod pinn_beamforming;
pub mod progress;
pub mod solver;

pub use self::pinn_beamforming::{
    ActivationFunction, DecompositionStrategy, DeviceConfig, DistributedConfig,
    DistributedPinnProvider, GpuMetrics, InferenceConfig, LoadBalancingStrategy, ModelArchitecture,
    ModelInfo, PinnBeamformingConfig, PinnBeamformingProvider, PinnBeamformingResult,
    PinnModelConfig, PinnProviderRegistry, ProcessingMetadata, TrainingMetrics, UncertaintyConfig,
};
pub use self::progress::{
    ConsoleProgressReporter, FieldsSummary, ProgressData, ProgressReporter, ProgressUpdate,
};
pub use self::solver::{Solver, SolverStatistics};
pub use crate::solver::config::SolverConfiguration as SolverConfig;
pub use crate::solver::feature::{FeatureManager, SolverFeature};
