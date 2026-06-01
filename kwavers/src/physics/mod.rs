//! Physics Module for Multi-Physics Simulation
//!
//! This module contains comprehensive physics implementations for acoustic wave simulation,
//! multi-physics coupling, and material property modeling.
//!
//! ## Module Organization (9-Layer Deep Vertical Hierarchy)
//!
//! - **foundations**: Wave equation specifications and coupling traits (SSOT for physics specs)
//! - **acoustics**: Acoustic propagation, bubble dynamics, medical imaging, therapy
//! - **thermal**: Heat transfer and thermal diffusion
//! - **chemistry**: Chemical kinetics and sonochemistry
//! - **electromagnetic**: Electromagnetic wave equations and photoacoustic coupling
//! - **optics**: Light propagation and sonoluminescence
//!
//! ## Design Philosophy
//!
//! This module maintains strict architectural separation using a deep vertical hierarchy
//! with Single Source of Truth (SSOT) principles. The domain layer provides definitive
//! data models; physics implements the physics specifications from foundations.
//!
//! ## Namespace Management
//!
//! The wildcard re-export below maintains backward compatibility while new code should
//! use explicit imports: `use crate::physics::acoustics::Type;`
//!
//! See ARCHITECTURE_AUDIT_REPORT.md for detailed namespace management strategy.

pub mod acoustics;
pub mod analytical; // Cross-domain analytical physics kernels
pub mod chemistry;
pub mod electromagnetic; // Electromagnetic wave implementations
pub mod factory; // Capability-driven plugin catalog (PhysicsConfig → PluginManager)
pub mod field_surrogate; // Cached focal-pressure kernels for fast planner queries
pub mod foundations; // Physics specifications and wave equation traits
pub mod optics; // Optical physics (elevated from electromagnetic)
pub mod photoacoustics; // Photoacoustic physics (thermoelastic coupling)
pub mod thermal;

// ============================================================================
// CORE ACOUSTIC PHYSICS RE-EXPORTS
// ============================================================================
// Explicitly re-export only the most commonly used acoustic types to maintain
// a clean public API. Users needing specialized types should import directly
// from the acoustics submodules.

/// Core acoustic wave propagation, cavitation, traits, and conservation validation.
///
/// Consolidated single source of truth (SSOT) for acoustics re-exports.
/// Types are grouped logically; bubble dynamics items use the nested path
/// `acoustics::bubble_dynamics` for namespace clarity.
pub use acoustics::{
    // ── Conservation validation ───────────────────────────────────────
    validate_conservation,
    AcousticConservationMetrics,
    AcousticStateRefs,
    // ── Wave propagation ───────────────────────────────────────────────
    AcousticWaveModel,
    // ── Cavitation ────────────────────────────────────────────────────
    CavitationModelBehavior,
    // ── Physics traits ────────────────────────────────────────────────
    ChemicalModelTrait,
    ConservationParams,
    HasPhysicsState,
    HeterogeneityModelTrait,
    PhysicsState,
    PreviousFields,
    StreamingModelTrait,
    ThermalModelTrait,
    VelocityFieldRefs,
};

/// Bubble dynamics types (namespace:`acoustics::bubble_dynamics`).
pub use acoustics::bubble_dynamics::{
    BubbleParameters,      // Bubble physical parameters
    BubbleState,           // Bubble state representation
    KellerMiksisModel,     // Keller-Miksis equation solver
    RayleighPlessetSolver, // Rayleigh-Plesset equation solver
};

/// Backward-compatible traits re-export module (deprecated path — prefer `crate::physics::`).
pub mod traits {
    pub use crate::physics::acoustics::{
        AcousticWaveModel, CavitationModelBehavior, ChemicalModelTrait, HeterogeneityModelTrait,
        StreamingModelTrait, ThermalModelTrait,
    };
}

// Re-export core physics specifications from foundations
pub use foundations::{
    AcousticWaveEquation, AutodiffElasticWaveEquation, AutodiffWaveEquation, Domain,
    ElasticWaveEquation, SourceTerm, SpatialDimension, TimeIntegration, WaveEquation,
};

// Re-export coupling traits
pub use foundations::{
    AcousticElasticCoupling, AcousticThermalCoupling, CouplingInterfaceCondition, CouplingStrength,
    ElectromagneticAcousticCoupling, ElectromagneticThermalCoupling, MultiPhysicsCoupling,
};

// Backward-compatible re-exports for moved modules
// These ensure existing code continues to work during refactoring

/// Re-export wave_propagation from its new location
pub use acoustics::analytical::propagation as wave_propagation;

/// Re-export bubble_dynamics from its new location
pub use acoustics::bubble_dynamics;

/// Re-export cavitation_control from its new location
pub use acoustics::bubble_dynamics::cavitation_control;

/// Re-export skull modeling from acoustics
pub use acoustics::skull;

/// Re-export transcranial aberration correction from acoustics
pub use acoustics::transcranial;

/// Re-export phase_modulation from its new location
pub use acoustics::analytical::patterns as phase_modulation;

/// Re-export sonoluminescence_detector from its new location in sensor module
pub use crate::domain::sensor::sonoluminescence as sonoluminescence_detector;

/// Re-export material properties from domain layer (SSOT for material specifications)
/// This was previously in physics::materials but has been moved to domain::medium::properties
/// as material property definitions belong in the domain layer, not physics.
pub use crate::domain::medium::properties::{fluids, implants, tissue, AcousticMaterialProperties};
