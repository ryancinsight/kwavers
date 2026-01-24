//! Physics module for multi-physics simulation
//!
//! This module contains the core physics implementations for acoustic wave simulation,
//! including material properties, wave equations, and numerical constants.

pub mod acoustics;
pub mod chemistry;
pub mod electromagnetic; // Electromagnetic wave implementations
pub mod foundations; // Physics specifications and wave equation traits
pub mod optics; // Optical physics (elevated from electromagnetic)
pub mod thermal;
pub mod constants {
    pub use crate::core::constants::*;
}

pub use acoustics::*;

pub mod plugin;

// Re-export mechanics from acoustics for backward compatibility
pub mod mechanics {
    pub use crate::physics::acoustics::mechanics::*;
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

pub mod imaging {
    pub use crate::physics::acoustics::imaging::modalities::{ceus, elastography, ultrasound};
    pub use crate::physics::acoustics::imaging::*;

    // Re-export fusion and registration from acoustics imaging
    pub mod fusion {
        pub use crate::physics::acoustics::imaging::fusion::*;
    }
    pub mod registration {
        pub use crate::physics::acoustics::imaging::registration::*;
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

/// Re-export phase_modulation from its new location
pub use acoustics::analytical::patterns as phase_modulation;

/// Re-export sonoluminescence_detector from its new location in sensor module
pub use crate::domain::sensor::sonoluminescence as sonoluminescence_detector;

// Note: optics is now a top-level physics module (line 9)
// Provide backward-compatible re-exports for specific optics functionality
pub mod optics_compat {
    pub use crate::physics::optics::*;

    // Re-export sonoluminescence from domain
    pub mod sonoluminescence {
        pub use crate::domain::sensor::sonoluminescence::*;
    }

    // Re-export diffusion solver if available
    #[cfg(feature = "advanced-visualization")]
    pub mod diffusion {
        pub mod solver {
            pub use crate::physics::optics::diffusion::*;
        }
    }
}
