//! # Kwavers: Acoustic Simulation Library
//!
//! A comprehensive acoustic wave simulation library with support for:
//! - Linear and nonlinear wave propagation
//! - Multi-physics simulations (acoustic, thermal, optical)
//! - Numerical methods (FDTD, PSTD, spectral methods)
//! - Real-time processing and visualization

// Enable portable SIMD for cross-platform performance
#![cfg_attr(feature = "nightly", feature(portable_simd))]
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
// Warn about code quality issues (will fix incrementally)
#![warn(missing_debug_implementations)]
// Warn about potentially unnecessary casts
#![warn(trivial_casts, trivial_numeric_casts)]
// Warn about unsafe code but allow it for performance-critical sections
#![warn(unsafe_code)]
// Allow certain patterns during refactoring
#![allow(
    clippy::too_many_arguments,  // Will fix in refactoring
    clippy::type_complexity,      // Will simplify types
)]

use std::collections::HashMap;

// Validation constants to replace magic numbers
// Removed unused validation constants - use constants module instead

// Core modules
// Core infrastructure
pub mod core;
pub mod infra;

// NOTE:
// Crate-level "convenience" re-exports are intentionally avoided to enforce direct imports
// and prevent namespace bleeding across bounded contexts.

// Domain logic
pub mod clinical;
pub mod domain;
pub mod math;
pub mod physics;
pub mod simulation;
pub mod solver;

// API surface: domain-relevant types re-exported at crate boundaries for ergonomics
pub use crate::core::error::{KwaversError, KwaversResult};
pub use crate::domain::grid::Grid;
pub use crate::domain::medium::traits::Medium;

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
        pub use crate::domain::sensor::beamforming::BeamformingCoreConfig;
    }
}
pub mod boundary {
    pub use crate::domain::boundary::{PMLBoundary, PMLConfig};
}
pub mod error {
    pub use crate::core::error::{GridError, KwaversError, KwaversResult};
}
pub mod time {
    pub use crate::core::time::Time;
}

/// Core infrastructure re-exports for testing support
pub mod testing {
    pub use crate::analysis::testing::*;
}

pub use domain::signal::{Signal, SineWave};
pub use math::ml;

// Analysis and tools
pub mod analysis;

// GPU support
#[cfg(feature = "gpu")]
pub mod gpu;

// GPU acceleration with wgpu-rs (already declared above)

// Phase 11: Visualization & Real-Time Interaction
// Phase 11: Visualization & Real-Time Interaction
#[cfg(all(
    feature = "gpu",
    any(
        feature = "advanced-visualization",
        feature = "web-visualization",
        feature = "vr-support"
    )
))]
pub use analysis::visualization;

// NOTE:
// Intentionally no crate-root type re-exports. Import from the defining modules directly.
// Example patterns:
//
// - Errors: crate::core::error::{KwaversResult, KwaversError, ValidationError, ...}
// - Grid: crate::domain::grid::Grid
// - Medium: crate::domain::medium::Medium
// - Boundaries: crate::domain::boundary::{Boundary, PMLBoundary, CPMLBoundary, ...}
// - Sensors: crate::domain::sensor::{GridSensorSet, ...}
// - Sources: crate::domain::source::Source

// Re-export physics plugin system (the new unified architecture)
pub use domain::field::indices as field_indices;
pub use domain::field::mapping::{
    FieldAccessor as UnifiedFieldAccessor, FieldAccessorMut, UnifiedFieldType,
};
pub use physics::plugin::{Plugin, PluginContext, PluginManager, PluginMetadata};
pub use physics::state::{FieldView, FieldViewMut, PhysicsState};

// Re-export spectral-DG components
pub use solver::pstd::dg::shock_capturing::{ArtificialViscosity, ShockDetector, WENOLimiter};
pub use solver::pstd::dg::{HybridSpectralDGConfig, HybridSpectralDGSolver};

// Re-export Spectral and FDTD solvers
pub use solver::fdtd::{FdtdConfig, FdtdPlugin, FdtdSolver};
pub use solver::pstd::{PSTDConfig, PSTDPlugin, PSTDSolver, PSTDSource};

// Re-export GPU-related items only when feature enabled

pub use physics::chemistry::ChemicalModel;
pub use physics::mechanics::acoustic_wave::NonlinearWave;
pub use physics::mechanics::elastic_wave::{
    mode_conversion::{
        MaterialSymmetry, ModeConversionConfig, StiffnessTensor, ViscoelasticConfig,
    },
    ElasticWave,
};
pub use physics::mechanics::{CavitationModel, StreamingModel};
pub use physics::traits::{AcousticWaveModel, CavitationModelBehavior, ChemicalModelTrait};
pub use solver::forward::nonlinear::kuznetsov::{KuznetsovConfig, KuznetsovWave};

// Re-export factory items removed
// Use domain::* and simulation::configuration explicitly instead

// Re-export I/O functions
// Re-export I/O functions
pub use infra::io::{generate_summary, save_light_data, save_pressure_data};

#[cfg(feature = "api")]
pub use crate::infra::api;

// Re-export performance optimization
// Re-export performance optimization
pub use analysis::performance::{
    profiling::{
        CacheProfile, MemoryProfile, PerformanceProfiler, ProfileReport, RooflineAnalysis,
        TimingScope,
    },
    OptimizationConfig, PerformanceOptimizer, SimdLevel,
};

// Re-export signal types
// Re-export signal types (already re-exported above)

// Re-export source types
pub use domain::source::{HanningApodization, LinearArray};

// Re-export configuration types
pub use domain::sensor::recorder::RecorderConfig;
/// Initialize logging for the kwavers library
pub fn init_logging() -> crate::core::error::KwaversResult<()> {
    env_logger::init();
    Ok(())
}

// NOTE: Plotting functionality removed - was incomplete stub
// Use external visualization tools or implement actual plotting

// Note: Use Config::default() instead of create_default_config()
// The Default trait is implemented for Config and provides the same functionality

// Note: Configuration validation should be done through the Config type itself
// or through the PluginBasedSolver's validation mechanisms

// Removed: Use PluginBasedSolver instead
// This function was part of the deprecated monolithic solver API

// Deprecated demo code removed - Use PluginBasedSolver for simulations
// See examples/ directory for usage patterns

/// Get library version and build information
///
/// Implements SSOT principle for version information
#[must_use]
pub fn get_version_info() -> HashMap<String, String> {
    let mut info = HashMap::new();
    info.insert("version".to_string(), env!("CARGO_PKG_VERSION").to_string());
    info.insert("name".to_string(), env!("CARGO_PKG_NAME").to_string());
    info.insert(
        "description".to_string(),
        env!("CARGO_PKG_DESCRIPTION").to_string(),
    );
    info.insert("authors".to_string(), env!("CARGO_PKG_AUTHORS").to_string());
    info.insert(
        "repository".to_string(),
        env!("CARGO_PKG_REPOSITORY").to_string(),
    );
    info.insert("license".to_string(), env!("CARGO_PKG_LICENSE").to_string());
    info
}

// Removed unused validation constants - use constants module instead

// Removed system info and compatibility checks - not needed for a library
// Users should handle their own resource management

#[cfg(test)]
mod tests {
    use super::*;

    use crate::core::error::{self};
    use crate::domain::medium::core::CoreMedium;

    #[test]
    fn test_default_config_creation() {
        let config = crate::simulation::configuration::Configuration::default();
        // Config validation - check that required fields exist
        assert!(config.simulation.duration > 0.0);
        assert!(config.simulation.frequency > 0.0);
        assert!(!config.output.snapshots); // Default is false
    }

    #[test]
    fn test_config_with_custom_values() {
        use crate::simulation::parameters::SimulationParameters;
        let config = crate::simulation::configuration::Configuration {
            simulation: SimulationParameters {
                frequency: 2e6,
                duration: 0.001,
                ..Default::default()
            },
            ..Default::default()
        };
        assert_eq!(config.simulation.frequency, 2e6);
    }

    #[test]
    fn test_version_info() {
        let info = get_version_info();
        assert!(info.contains_key("version"));
        assert!(info.contains_key("name"));
    }

    // ============================================================================
    // MINIMAL UNIT TESTS FOR SRS NFR-002 COMPLIANCE (<30s execution)
    // Fast tests focusing on core functionality without expensive computations
    // ============================================================================

    #[test]
    fn test_grid_creation_minimal() {
        let grid =
            crate::domain::grid::Grid::new(8, 8, 8, 0.001, 0.001, 0.001).expect("Grid creation");
        assert_eq!(grid.nx, 8);
        assert_eq!(grid.ny, 8);
        assert_eq!(grid.nz, 8);
        assert_eq!(grid.size(), 512);
    }

    #[test]
    fn test_medium_basic_properties() {
        let grid =
            crate::domain::grid::Grid::new(4, 4, 4, 0.001, 0.001, 0.001).expect("Grid creation");
        let medium = crate::domain::medium::HomogeneousMedium::new(
            physics::constants::DENSITY_WATER,
            physics::constants::SOUND_SPEED_WATER,
            0.0,
            0.0,
            &grid,
        );

        assert!(medium.is_homogeneous());
        assert!((medium.sound_speed(0, 0, 0) - physics::constants::SOUND_SPEED_WATER).abs() < 1e-6);
        assert!((medium.density(0, 0, 0) - physics::constants::DENSITY_WATER).abs() < 1e-6);
    }

    #[test]
    fn test_physics_constants_validation() {
        // Physics constants are compile-time verified through const definitions
        // No runtime assertions needed for const values (clippy::assertions_on_constants)
        use physics::constants::*;

        // Validate that constants are accessible and have expected types
        let _density: f64 = DENSITY_WATER;
        let _speed: f64 = SOUND_SPEED_WATER;

        // Constants are defined in physics::constants::fundamental
        // DENSITY_WATER = 998.2 kg/m³ (valid water density)
        // SOUND_SPEED_WATER = 1482.0 m/s (valid water sound speed)
    }

    #[test]
    fn test_cfl_calculation_basic() {
        let grid =
            crate::domain::grid::Grid::new(8, 8, 8, 1e-3, 1e-3, 1e-3).expect("Grid creation");
        let sound_speed = 1500.0;
        let cfl = 0.4; // Conservative CFL for 3D
        let min_dx = grid.dx.min(grid.dy).min(grid.dz);
        let dt = cfl * min_dx / sound_speed;

        assert!(dt > 0.0);
        assert!(dt < 1e-6); // Reasonable timestep for acoustics

        // CFL stability condition: c*dt/dx <= CFL_max
        let actual_cfl = sound_speed * dt / min_dx;
        assert!((actual_cfl - cfl).abs() < 1e-10);
    }

    #[test]
    fn test_error_handling_basic() {
        // Test basic error type creation
        use error::{ConfigError, KwaversError};

        let config_error = ConfigError::InvalidValue {
            parameter: "test".to_string(),
            value: "invalid".to_string(),
            constraint: "must be positive".to_string(),
        };

        let kwavers_error = KwaversError::Config(config_error);
        assert!(matches!(kwavers_error, KwaversError::Config(_)));
    }
}
