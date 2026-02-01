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
pub mod chemistry;
pub mod electromagnetic; // Electromagnetic wave implementations
pub mod foundations; // Physics specifications and wave equation traits
pub mod optics; // Optical physics (elevated from electromagnetic)
pub mod thermal;
/// Physical constants re-exported for convenience
///
/// SSOT: All constants are defined in `crate::core::constants`
/// This module re-exports them for backward compatibility.
pub mod constants {
    // Re-export all core constants
    pub use crate::core::constants::*;

    // Explicit re-exports for commonly used constants
    pub use crate::core::constants::acoustic_parameters::REFERENCE_FREQUENCY_FOR_ABSORPTION_HZ;
    pub use crate::core::constants::chemistry::BASE_PHOTOCHEMICAL_RATE;
    pub use crate::core::constants::fundamental::{
        DENSITY_TISSUE, DENSITY_WATER, SOUND_SPEED_TISSUE, SOUND_SPEED_WATER,
    };
    pub use crate::core::constants::numerical::{
        B_OVER_A_DIVISOR, NONLINEARITY_COEFFICIENT_OFFSET,
    };
    pub use crate::core::constants::thermodynamic::{
        kelvin_to_celsius, WATER_LATENT_HEAT_VAPORIZATION,
    };
    pub use crate::core::constants::{
        BULK_MODULUS_WATER, C_WATER, SURFACE_TENSION_WATER, VAPOR_PRESSURE_WATER, VISCOSITY_WATER,
    };
}

// ============================================================================
// CORE ACOUSTIC PHYSICS RE-EXPORTS
// ============================================================================
// Explicitly re-export only the most commonly used acoustic types to maintain
// a clean public API. Users needing specialized types should import directly
// from the acoustics submodules.

/// Core acoustic wave propagation models
pub use acoustics::{
    AcousticWaveModel, // Primary wave propagation interface
    HasPhysicsState,   // State access trait
    PhysicsState,      // Physics state container
};

/// Cavitation and bubble dynamics
pub use acoustics::{
    CavitationModelBehavior, // Bubble cavitation interface
};

/// Bubble dynamics types (from acoustics::bubble_dynamics)
pub use acoustics::bubble_dynamics::{
    BubbleParameters,      // Bubble physical parameters
    BubbleState,           // Bubble state representation
    KellerMiksisModel,     // Keller-Miksis equation solver
    RayleighPlessetSolver, // Rayleigh-Plesset equation solver
};

/// Core acoustic traits
pub use acoustics::{
    ChemicalModelTrait,      // Chemical kinetics interface
    HeterogeneityModelTrait, // Heterogeneous media modeling
    StreamingModelTrait,     // Acoustic streaming interface
    ThermalModelTrait,       // Thermal effects interface
};

/// Backward-compatible traits re-export module
pub mod traits {
    pub use crate::physics::acoustics::{
        AcousticWaveModel, CavitationModelBehavior, ChemicalModelTrait, HeterogeneityModelTrait,
        StreamingModelTrait, ThermalModelTrait,
    };
}

/// Conservation validation
pub use acoustics::{
    validate_conservation, // Conservation validation
    ConservationMetrics,   // Conservation metrics
};

pub mod plugin;

/// Acoustic mechanics re-exported for backward compatibility
///
/// Users should prefer explicit imports: `use crate::physics::acoustics::mechanics::Type;`
pub mod mechanics {
    /// Absorption models and modes
    pub use crate::physics::acoustics::mechanics::absorption::AbsorptionMode;
    // Note: AbsorptionModel is implemented as a trait, not a struct

    /// Acoustic wave propagation
    // Note: acoustic_wave module exists but types may be in submodules
    pub use crate::physics::acoustics::mechanics::acoustic_wave::SpatialOrder;

    /// Elastic wave propagation
    pub use crate::physics::acoustics::mechanics::elastic_wave::ElasticWave;
    // Note: ElasticWaveConfig doesn't exist - configuration is via ElasticBodyForceConfig

    /// Cavitation models
    pub use crate::physics::acoustics::mechanics::cavitation::CavitationModel;

    /// Acoustic streaming models
    pub use crate::physics::acoustics::mechanics::streaming::StreamingModel;

    /// Poroelastic tissue modeling
    pub use crate::physics::acoustics::mechanics::poroelastic;

    /// Radiation force calculations (re-exported from domain::therapy::microbubble)
    pub use crate::domain::therapy::microbubble::forces::RadiationForce;
}

// Re-export core physics specifications from foundations
pub use foundations::{
    AcousticWaveEquation, AutodiffElasticWaveEquation, AutodiffWaveEquation, BoundaryCondition,
    Domain, ElasticWaveEquation, SourceTerm, SpatialDimension, TimeIntegration, WaveEquation,
};

// Re-export coupling traits
pub use foundations::{
    AcousticElasticCoupling, AcousticThermalCoupling, CouplingStrength,
    ElectromagneticAcousticCoupling, ElectromagneticThermalCoupling, InterfaceCondition,
    MultiPhysicsCoupling,
};

/// Medical imaging physics re-exported for backward compatibility
///
/// Users should prefer explicit imports: `use crate::physics::acoustics::imaging::Type;`
pub mod imaging {
    /// Imaging modality interfaces
    pub use crate::physics::acoustics::imaging::modalities::{ceus, elastography, ultrasound};

    /// Multi-modal fusion capabilities
    pub mod fusion {
        /// Multi-modal image fusion
        pub use crate::physics::acoustics::imaging::fusion::{
            FusedImageResult, FusionConfig, FusionMethod, MultiModalFusion,
        };
    }

    /// Image registration capabilities
    pub mod registration {
        /// Image registration algorithms
        pub use crate::physics::acoustics::imaging::registration::ImageRegistration;
        // Note: RegistrationMethod and TransformParameters don't exist as public types
    }
}

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
pub use crate::domain::medium::properties::{fluids, implants, tissue, MaterialProperties};

/// Optics compatibility layer for backward compatibility
///
/// Optics is now a top-level physics module. This provides re-exports for legacy code.
pub mod optics_compat {
    /// Core optical properties
    pub use crate::physics::optics::OpticalProperties;

    /// Sonoluminescence detection
    pub mod sonoluminescence {
        pub use crate::domain::sensor::sonoluminescence::SonoluminescenceDetector;
        /// Sonoluminescence emission and spectrum
        pub use crate::physics::optics::sonoluminescence::EmissionSpectrum;
    }

    /// Optical diffusion
    #[cfg(feature = "advanced-visualization")]
    pub mod diffusion {
        pub use crate::physics::optics::diffusion::OpticalProperties;
    }
}
