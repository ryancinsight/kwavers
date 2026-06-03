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
//! - [`infrastructure`]: result/data I/O (CSV, pressure-field export)
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

/// Core infrastructure: errors, logging, time, arena allocator.
///
/// Extracted to the `kwavers-core` workspace crate (ADR 009); re-exported here
/// under the original `core` name so `crate::core::…` / `kwavers::core::…` paths
/// resolve unchanged. This `kwavers` crate is the facade over the layered
/// `kwavers-*` crates.
pub use kwavers_core as core;

/// Pure mathematical primitives: FFT, geometry, linear algebra, SIMD.
///
/// Extracted to the `kwavers-math` workspace crate (ADR 009); re-exported here
/// under the original `math` name so `crate::math::…` paths resolve unchanged.
pub use kwavers_math as math;

/// Domain model: grid, medium, source, sensor, boundary, field, signal.
///
/// Extracted to the `kwavers-domain` workspace crate (ADR 009); re-exported here
/// under the original `domain` name so `crate::domain::…` paths resolve unchanged.
pub use kwavers_domain as domain;

/// Physics models: acoustics, optics, thermal, chemistry, electromagnetic.
///
/// Extracted to the `kwavers-physics` workspace crate (ADR 009); re-exported here
/// under the original `physics` name so `crate::physics::…` paths resolve unchanged.
pub use kwavers_physics as physics;

/// Numerical solvers: forward (FDTD/PSTD/elastic), inverse (FWI/PINN), analytical.
///
/// Extracted to the `kwavers-solver` workspace crate (ADR 009); re-exported here
/// under the original `solver` name so `crate::solver::…` paths resolve unchanged.
pub use kwavers_solver as solver;

/// High-level simulation orchestration, backends, and modality workflows.
///
/// Extracted to the `kwavers-simulation` workspace crate (ADR 009); re-exported here
/// under the original `simulation` name so `crate::simulation::…` paths resolve unchanged.
pub use kwavers_simulation as simulation;

/// Analysis tools: signal processing, beamforming, validation, ML, performance.
///
/// Extracted to the `kwavers-analysis` workspace crate (ADR 009); re-exported here
/// under the original `analysis` name so `crate::analysis::…` paths resolve unchanged.
pub use kwavers_analysis as analysis;

/// GPU profiling and allocation tracking
pub mod profiling;

/// Clinical application layer (ADR 009): split into `kwavers-diagnostics`
/// (diagnostic imaging workflows) and `kwavers-therapy` (therapy planning,
/// theranostic guidance, safety, regulatory, patient management). Re-exported
/// here under the original `clinical::{imaging,therapy,safety,regulatory,
/// patient_management}` paths so `crate::clinical::…` resolves unchanged.
pub mod clinical {
    pub use kwavers_diagnostics as imaging;
    pub use kwavers_therapy::{patient_management, regulatory, safety, therapy};

    pub use imaging::{
        ClinicalApplication, ClinicalExaminationResult, ClinicalProtocol, ClinicalWorkflowConfig,
        ClinicalWorkflowOrchestrator, DiagnosticRecommendation, DiagnosticUrgency,
        QualityPreference, WorkflowPriority, WorkflowState, WorkflowTimingMetrics,
    };
    pub use patient_management::{
        ClinicalEncounter, ClinicalNote, ConsentRecord, ConsentType, EncounterId, EncounterType,
        MedicalHistoryEntry, MedicationRecord, PatientDemographics, PatientId,
        PatientManagementSystem, PatientMedicalProfile, PatientTreatmentPlan, TreatmentStatus,
        VitalSigns,
    };
    pub use regulatory::{
        ClinicalEvidence, DeviceClass, DeviceDescription, PerformanceTest, PredicateDevice,
        RiskRecord, SubmissionDocument,
    };
    pub use safety::{
        mechanical_index::{
            MechanicalIndexCalculator, MechanicalIndexResult, MechanicalIndexSafetyStatus,
            MechanicalIndexTissueType,
        },
        AuditEntry, AuditSafetyEventType, ClinicalSafetyLevel, ClinicalSafetyLimits,
        ClinicalSafetyMonitor, ComplianceResult, ComplianceValidator, DoseController, Interlock,
        InterlockSystem, SafetyAuditLogger, SafetyComplianceReport, SafetyViolation,
        SystemConfiguration, TreatmentRecord,
    };
    pub use therapy::ClinicalTherapyParameters;
}

/// Infrastructure: result/data I/O (CSV, pressure-field export)
pub mod infrastructure;

/// GPU compute acceleration (WGPU) — consolidated in the `kwavers-gpu` leaf
/// crate; re-exported so existing `kwavers::gpu::*` paths keep working.
#[cfg(feature = "gpu")]
pub use kwavers_gpu::gpu;

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
    pub use crate::domain::boundary::{DomainPMLBoundary, DomainPmlConfig};
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
        AcousticSolver, BubbleModel, NonlinearEquation, PhysicsBoundaryCondition, PhysicsConfig,
        PhysicsModelConfig, PhysicsModelType,
    };
    pub use crate::solver::plugin::{
        ExecutionStrategy, ParallelStrategy, PhysicsCatalog, PluginManager, SequentialStrategy,
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
pub use physics::acoustics::mechanics::cavitation::CavitationModel;
pub use physics::acoustics::mechanics::elastic_wave::{
    mode_conversion::{MaterialSymmetry, ModeConversionConfig, ViscoelasticConfig},
    ElasticWave,
};
pub use physics::acoustics::mechanics::streaming::StreamingModel;
pub use physics::chemistry::ChemicalModel;
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
