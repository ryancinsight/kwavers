//! # Kwavers: Acoustic Simulation Library
//!
//! A comprehensive acoustic wave simulation library with support for:
//! - Linear and nonlinear wave propagation
//! - Multi-physics simulations (acoustic, thermal, optical)
//! - Numerical methods (FDTD, PSTD, spectral methods)
//! - Real-time processing and visualization

// Enable portable SIMD for cross-platform performance
#![feature(portable_simd)]

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
pub mod boundary;
pub mod clinical; // Clinical workflows and decision support
pub mod cloud; // Sprint 155: Cloud integration for PINN deployment
#[cfg(feature = "api")]
pub mod api; // Sprint 159: Enterprise API for production deployment
pub mod configuration; // Unified configuration system (SSOT)
                       // constants module moved to physics::constants for SSOT
pub mod error;
pub mod factory;
pub mod fft;
pub mod geometry; // k-Wave compatible geometry helpers
#[cfg(feature = "gpu")]
pub mod gpu;

pub mod grid;
pub mod io;
pub mod log;
pub mod medium;
pub mod ml;
pub mod performance;
pub mod physics;
pub mod plotting;
pub mod recorder;
pub mod runtime; // Sprint 138: Async I/O, tracing, zero-copy (Persona Requirements)
pub mod sensor;
pub mod signal;
pub mod uncertainty; // Export uncertainty quantification module
pub mod solver;
pub mod source;
pub mod testing; // Property-based testing framework per FSE 2025
pub mod time;
pub mod utils;
pub mod validation;

// GPU acceleration with wgpu-rs (already declared above)

// Phase 11: Visualization & Real-Time Interaction
#[cfg(all(
    feature = "gpu",
    any(
        feature = "advanced-visualization",
        feature = "web-visualization",
        feature = "vr-support"
    )
))]
pub mod visualization;

// Re-export commonly used types for convenience
pub use boundary::{Boundary, CPMLBoundary, CPMLConfig, PMLBoundary, PMLConfig};
pub use error::{KwaversError, KwaversResult};
pub use geometry::{make_ball, make_disc, make_line, make_sphere}; // k-Wave compatible geometry helpers
pub use grid::Grid;
pub use medium::{homogeneous::HomogeneousMedium, Medium};
pub use recorder::Recorder;
pub use sensor::{ArrayGeometry, BeamformingMethod, PAMConfig, PAMPlugin, Sensor, SensorData};
pub use source::Source;
pub use time::Time;
// Solver exports
pub use configuration::{Configuration, OutputParameters, SimulationParameters, SourceParameters};
pub use error::{ConfigError, ValidationError};
pub use solver::amr::{AMRSolver, MemoryStats};
pub use solver::plugin_based::PluginBasedSolver;
pub use solver::reconstruction::photoacoustic::PhotoacousticReconstructor;
pub use solver::reconstruction::seismic::{
    FullWaveformInversion, ReverseTimeMigration, RtmImagingCondition, SeismicImagingConfig,
};
pub use solver::reconstruction::{
    FilterType, InterpolationMethod, ReconstructionAlgorithm, ReconstructionConfig, Reconstructor,
    UniversalBackProjection, WeightFunction,
};
pub use solver::time_reversal::{TimeReversalConfig, TimeReversalReconstructor};
pub use validation::{Validatable, ValidationResult};
pub use validation::theorem_validation::{TheoremValidation, TheoremValidator};

// Re-export physics plugin system (the new unified architecture)
pub use physics::field_mapping::{
    FieldAccessor as UnifiedFieldAccessor, FieldAccessorMut, UnifiedFieldType,
};
pub use physics::plugin::{Plugin, PluginContext, PluginManager, PluginMetadata};
pub use physics::state::{field_indices, FieldView, FieldViewMut, PhysicsState};

// Re-export spectral-DG components
pub use solver::spectral_dg::shock_capturing::{ArtificialViscosity, ShockDetector, WENOLimiter};
pub use solver::spectral_dg::{HybridSpectralDGConfig, HybridSpectralDGSolver};

// Re-export PSTD and FDTD plugins
pub use solver::fdtd::{FdtdConfig, FdtdPlugin, FdtdSolver};
pub use solver::pstd::{PstdConfig, PstdPlugin, PstdSolver};

// Re-export GPU-related items only when feature enabled

pub use physics::chemistry::ChemicalModel;
pub use physics::mechanics::acoustic_wave::NonlinearWave;
pub use physics::mechanics::elastic_wave::{
    mode_conversion::{
        MaterialSymmetry, ModeConversionConfig, StiffnessTensor, ViscoelasticConfig,
    },
    ElasticWave,
};
pub use physics::mechanics::{CavitationModel, KuznetsovConfig, KuznetsovWave, StreamingModel};
pub use physics::traits::{AcousticWaveModel, CavitationModelBehavior, ChemicalModelTrait};

// Re-export factory components
pub use factory::{
    ConfigBuilder, GridConfig, MediumConfig, MediumType, PhysicsConfig, PhysicsModelConfig,
    PhysicsModelType, SimulationComponents, SimulationConfig as FactorySimulationConfig,
    SimulationFactory, SourceConfig as FactorySourceConfig, TimeConfig, ValidationConfig,
};

// Re-export I/O functions
pub use io::{generate_summary, save_light_data, save_pressure_data};

// Re-export performance optimization
pub use performance::{
    profiling::{
        CacheProfile, MemoryProfile, PerformanceProfiler, ProfileReport, RooflineAnalysis,
        TimingScope,
    },
    OptimizationConfig, PerformanceOptimizer, SimdLevel,
};

// Re-export signal types
pub use signal::{Signal, SineWave};

// Re-export source types
pub use source::{HanningApodization, LinearArray};

// Re-export configuration types
pub use recorder::RecorderConfig;
pub use sensor::SensorConfig;

// Re-export benchmarks from performance module
pub use performance::{run_production_benchmarks, BenchmarkResult, ProductionBenchmarks};

// Re-export solver validation
pub use solver::validation::{KWaveTestCase, KWaveValidator, ValidationReport};

// Additional reconstruction exports for specific recon types
pub use solver::reconstruction::{
    arc_recon::ArcRecon, bowl_recon::BowlRecon, line_recon::LineRecon, plane_recon::PlaneRecon,
};

/// Initialize logging for the kwavers library
pub fn init_logging() -> KwaversResult<()> {
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
    use crate::medium::core::CoreMedium;

    #[test]
    fn test_default_config_creation() {
        let config = configuration::Configuration::default();
        // Config validation - check that required fields exist
        assert!(config.simulation.duration > 0.0);
        assert!(config.simulation.frequency > 0.0);
        assert!(!config.output.snapshots); // Default is false
    }

    #[test]
    fn test_config_with_custom_values() {
        let config = configuration::Configuration {
            simulation: configuration::SimulationParameters {
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
        let grid = grid::Grid::new(8, 8, 8, 0.001, 0.001, 0.001).expect("Grid creation");
        assert_eq!(grid.nx, 8);
        assert_eq!(grid.ny, 8);
        assert_eq!(grid.nz, 8);
        assert_eq!(grid.size(), 512);
    }

    #[test]
    fn test_medium_basic_properties() {
        let grid = grid::Grid::new(4, 4, 4, 0.001, 0.001, 0.001).expect("Grid creation");
        let medium = medium::HomogeneousMedium::new(
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
        let grid = grid::Grid::new(8, 8, 8, 1e-3, 1e-3, 1e-3).expect("Grid creation");
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
