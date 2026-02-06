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
//! - [`gpu`]: WGPU compute shaders, GPU-accelerated kernels (feature-gated)

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
    clippy::too_many_arguments,
    clippy::type_complexity,
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
            SensorBeamformer, SensorProcessingParams, WindowType,
        };
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
pub use solver::plugin::{PluginExecutor, PluginManager};

// --- Solver re-exports ---
pub use solver::fdtd::{FdtdConfig, FdtdPlugin, FdtdSolver};
pub use solver::pstd::{PSTDConfig, PSTDPlugin, PSTDSolver};
pub use solver::pstd::dg::shock_capturing::{ArtificialViscosity, ShockDetector, WENOLimiter};
pub use solver::pstd::dg::{HybridSpectralDGConfig, HybridSpectralDGSolver};
pub use solver::forward::nonlinear::kuznetsov::{KuznetsovConfig, KuznetsovWave};

// --- Physics model re-exports ---
pub use physics::acoustics::mechanics::acoustic_wave::nonlinear::NonlinearWave;
pub use physics::acoustics::mechanics::elastic_wave::{
    mode_conversion::{
        MaterialSymmetry, ModeConversionConfig, StiffnessTensor, ViscoelasticConfig,
    },
    ElasticWave,
};
pub use physics::chemistry::ChemicalModel;
pub use physics::mechanics::{CavitationModel, StreamingModel};
pub use physics::traits::{AcousticWaveModel, CavitationModelBehavior, ChemicalModelTrait};

// --- Simulation factory ---
pub use simulation::factory::PhysicsFactory;

// --- I/O functions ---
pub use infrastructure::io::{generate_summary, save_data_csv, save_light_data, save_pressure_data};

// --- Source types ---
pub use domain::source::{HanningApodization, LinearArray};

// --- Configuration types ---
pub use domain::sensor::recorder::RecorderConfig;

// --- Performance optimization ---
pub use analysis::performance::{
    CacheProfile, MemoryProfile, OptimizationConfig, PerformanceOptimizer, PerformanceProfiler,
    ProfileReport, RooflineAnalysis, SimdLevel, TimingScope,
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
pub fn init_logging() -> crate::core::error::KwaversResult<()> {
    env_logger::init();
    Ok(())
}

/// Get library version and build information.
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
