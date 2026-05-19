//! # Kwavers: Acoustic & Optical Simulation Library
//!
//! A comprehensive simulation library for:
//! - Linear and nonlinear acoustic wave propagation (FDTD, PSTD, k-space)
//! - Multi-physics simulations (acoustic, thermal, optical, elastic)
//! - Physics-Informed Neural Networks (PINNs) via Burn
//! - Medical ultrasound imaging modalities (B-mode, SWE, CEUS, photoacoustic)
//! - Clinical therapy planning and safety monitoring
//!
//! ## Module Hierarchy
//!
//! - [`core`]: Error types, logging, time management, arena allocator
//! - [`math`]: FFT, geometry, linear algebra, SIMD, numerics
//! - [`domain`]: Grid, medium, source, sensor, boundary, field, signal
//! - [`physics`]: Acoustics, optics, thermal, chemistry, electromagnetic
//! - [`solver`]: Forward (FDTD, PSTD, elastic), inverse (PINN, elastography), analytical
//! - [`simulation`]: High-level simulation orchestration, backends, modalities
//! - [`analysis`]: Signal processing, beamforming, validation, performance, ML
//! - [`clinical`]: Imaging workflows, therapy planning, safety, regulatory
//! - [`infrastructure`]: I/O (DICOM, NIfTI), API, cloud, device abstraction
//! - `gpu` (feature `"gpu"`): WGPU compute shaders, GPU-accelerated kernels

// Strict warning configuration for code quality
#![warn(
    unused_imports,
    unused_mut,
    unreachable_code,
    unreachable_patterns,
    unused_must_use,
    unused_unsafe,
    path_statements,
    unused_attributes,
    unused_macros
)]
#![warn(missing_debug_implementations)]
#![warn(trivial_casts, trivial_numeric_casts)]
#![warn(unsafe_code)]
#![allow(
    clippy::type_complexity,
    clippy::assertions_on_constants,
    clippy::field_reassign_with_default
)]
#![allow(unexpected_cfgs)]

use std::collections::HashMap;

// ============================================================================
// Module declarations (layered architecture, bottom-up)
// ============================================================================

/// Architecture validation and layer enforcement
pub mod architecture;

/// Core infrastructure: errors, logging, time, arena allocator
pub mod core;

/// Pure mathematical primitives: FFT, geometry, linear algebra, SIMD
pub mod math;

/// Domain model: grid, medium, source, sensor, boundary, field, signal
pub mod domain;

/// Physics models: acoustics, optics, thermal, chemistry, electromagnetic
pub mod physics;

/// Numerical solvers: forward (FDTD/PSTD/elastic), inverse (PINN), analytical
pub mod solver;

/// High-level simulation orchestration, backends, and modality workflows
pub mod simulation;

/// Analysis tools: signal processing, beamforming, validation, ML, performance
pub mod analysis;

/// GPU profiling and allocation tracking
pub mod profiling;

/// Clinical workflows: imaging, therapy planning, safety, regulatory
pub mod clinical;

/// Infrastructure: I/O, API, cloud, device abstraction, runtime
pub mod infrastructure;

/// GPU compute acceleration (WGPU)
#[cfg(feature = "gpu")]
pub mod gpu;

// ============================================================================
// Public API re-exports
// ============================================================================

// --- Core types (most commonly used across the crate) ---
pub use crate::core::error::{KwaversError, KwaversResult};
pub use crate::domain::grid::Grid;
pub use crate::domain::medium::traits::Medium;

// --- Convenience re-export modules (used by tests, examples, benches) ---
pub mod grid {
    pub use crate::domain::grid::Grid;
}
pub mod medium {
    pub use crate::domain::medium::{
        AcousticProperties, ArrayAccess, CoreMedium, HomogeneousMedium, Medium,
    };
    pub mod heterogeneous {
        pub use crate::domain::medium::heterogeneous::HeterogeneousMedium;
    }
    pub mod homogeneous {
        pub use crate::domain::medium::HomogeneousMedium;
    }
    pub mod viscous {
        pub use crate::domain::medium::ViscousProperties;
    }
}
pub mod source {
    pub use crate::domain::source::{types::Source, GridSource, PointSource, TimeVaryingSource};
}
pub mod sensor {
    pub use crate::domain::sensor::GridSensorSet;
    pub mod beamforming {
        pub use crate::domain::sensor::beamforming::{
            BeamformerWindowType, SensorBeamformer, SensorProcessingParams,
        };
    }
}
pub mod boundary {
    pub use crate::domain::boundary::{DomainPmlConfig, DomainPMLBoundary};
}
pub mod error {
    pub use crate::core::error::{GridError, KwaversError, KwaversResult};
}
pub mod time {
    pub use crate::core::time::Time;
}

/// Testing utilities for property-based tests
pub mod testing {
    pub use crate::analysis::testing::{acoustic_properties, grid_properties, medium_properties};
}

// --- Signal types ---
pub use analysis::ml;
pub use domain::signal::{Signal, SineWave};

// --- Plugin system ---
pub use domain::field::indices as field_indices;
pub use domain::field::mapping::{
    FieldAccessor as UnifiedFieldAccessor, FieldAccessorMut, UnifiedFieldType,
};
pub use domain::plugin::{Plugin, PluginContext, PluginMetadata};
pub use physics::acoustics::state::{FieldView, FieldViewMut, PhysicsState};
pub use solver::plugin::PluginManager;

/// Unified plugin API.
///
/// Single import surface for the plugin system. The internal four-layer split
/// (`domain::plugin` contract, `solver::plugin` orchestration, state-based
/// accessors in `physics::acoustics::state::access`, `physics::factory`
/// capability catalog) is preserved for DIP but hidden behind this facade.
/// Consumers write `use kwavers::plugin::*;` and receive every type needed to
/// declare capabilities, build a populated `PluginManager`, and author or
/// register plugins.
///
/// ## Typical use
///
/// ```ignore
/// use kwavers::plugin::*;
///
/// let mut config = PhysicsConfig::new();
/// config.models.clear();
/// config.models.push(PhysicsModelConfig {
///     model_type: PhysicsModelType::LinearAcoustics {
///         solver_type: AcousticSolver::PSTD { spectral_accuracy: true },
///         boundary_conditions: crate::physics::factory::PhysicsBoundaryCondition::Absorbing { pml_layers: 10 },
///     },
///     enabled: true,
///     parameters: Default::default(),
/// });
/// let manager = PhysicsCatalog::build(&config, &grid, &medium, dt)?;
/// ```
pub mod plugin {
    pub use crate::domain::plugin::{
        DirectPluginFieldAccess, Plugin, PluginContext, PluginFields, PluginMetadata,
        PluginPriority, PluginState,
    };
    pub use crate::physics::acoustics::state::{PluginFieldAccess, PluginFieldAccessMut};
    pub use crate::physics::factory::{
        AcousticSolver, BubbleModel, NonlinearEquation, PhysicsBoundaryCondition, PhysicsCatalog,
        PhysicsConfig, PhysicsModelConfig, PhysicsModelType,
    };
    pub use crate::solver::plugin::{
        ExecutionStrategy, ParallelStrategy, PluginManager, SequentialStrategy,
    };
}

// --- Solver re-exports ---
pub use solver::fdtd::{FdtdConfig, FdtdPlugin, FdtdSolver};
pub use solver::forward::nonlinear::kuznetsov::{KuznetsovConfig, KuznetsovWave};
pub use solver::pstd::dg::shock_capturing::{ArtificialViscosity, ShockDetector, WENOLimiter};
pub use solver::pstd::dg::{HybridSpectralDGConfig, HybridSpectralDGSolver};
pub use solver::pstd::{PSTDConfig, PSTDPlugin, PSTDSolver};

// --- Physics model re-exports ---
pub use domain::medium::AnisotropicStiffnessTensor;
pub use physics::acoustics::mechanics::acoustic_wave::nonlinear::NonlinearWave;
pub use physics::acoustics::mechanics::elastic_wave::{
    mode_conversion::{MaterialSymmetry, ModeConversionConfig, ViscoelasticConfig},
    ElasticWave,
};
pub use physics::chemistry::ChemicalModel;
pub use physics::mechanics::{CavitationModel, StreamingModel};
pub use physics::traits::{AcousticWaveModel, CavitationModelBehavior, ChemicalModelTrait};

// --- Simulation factory ---
pub use simulation::factory::PhysicsFactory;

// --- I/O functions ---
pub use infrastructure::io::{
    generate_summary, save_data_csv, save_light_data, save_pressure_data,
};

// --- Source types ---
pub use domain::source::{HanningApodization, LinearArray};

// --- Configuration types ---
pub use domain::sensor::recorder::RecorderConfig;

// --- Performance optimization ---
pub use analysis::performance::{
    CacheProfile, HardwareOptimizationConfig, MemoryProfile, PerfOptSimdLevel,
    PerformanceOptimizer, PerformanceProfiler, ProfileReport, RooflineAnalysis, TimingScope,
};

// --- Feature-gated API ---
#[cfg(feature = "api")]
pub use crate::infrastructure::api;

#[cfg(all(
    feature = "gpu",
    any(
        feature = "advanced-visualization",
        feature = "web-visualization",
        feature = "vr-support"
    )
))]
pub use analysis::visualization;
// ============================================================================
// Library utilities
// ============================================================================

/// Initialize logging for the kwavers library.
/// # Errors
/// - Returns [`Err`] if an internal constraint is violated.
///
pub fn init_logging() -> crate::core::error::KwaversResult<()> {
    env_logger::init();
    Ok(())
}

/// Get library version and build information.
#[must_use]
pub fn get_version_info() -> HashMap<String, String> {
    let mut info = HashMap::new();
    info.insert("version".to_owned(), env!("CARGO_PKG_VERSION").to_owned());
    info.insert("name".to_owned(), env!("CARGO_PKG_NAME").to_owned());
    info.insert(
        "description".to_owned(),
        env!("CARGO_PKG_DESCRIPTION").to_owned(),
    );
    info.insert("authors".to_owned(), env!("CARGO_PKG_AUTHORS").to_owned());
    info.insert(
        "repository".to_owned(),
        env!("CARGO_PKG_REPOSITORY").to_owned(),
    );
    info.insert("license".to_owned(), env!("CARGO_PKG_LICENSE").to_owned());
    info
}

#[cfg(test)]
mod tests;
