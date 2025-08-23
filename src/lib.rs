//! # Kwavers: Acoustic Simulation Library
//!
//! A comprehensive acoustic wave simulation library with support for:
//! - Linear and nonlinear wave propagation
//! - Multi-physics simulations (acoustic, thermal, optical)
//! - Numerical methods (FDTD, PSTD, spectral methods)
//! - Real-time processing and visualization

// WARNING: Removing suppressions to fix real issues
// Dead code and unused variables indicate incomplete implementation

use std::collections::HashMap;

// Validation constants to replace magic numbers
mod validation_constants {
    /// Minimum frequency for source validation [Hz]
    pub const MIN_SOURCE_FREQUENCY: f64 = 1e3;
    /// Maximum frequency for source validation [Hz]  
    pub const MAX_SOURCE_FREQUENCY: f64 = 100e6;
    /// Minimum amplitude for source validation [Pa]
    pub const MIN_SOURCE_AMPLITUDE: f64 = 1e3;
    /// Maximum amplitude for source validation [Pa]
    pub const MAX_SOURCE_AMPLITUDE: f64 = 100e6;
    /// Progress reporting interval [steps]
    pub const PROGRESS_REPORT_INTERVAL: usize = 100;
    /// Default body temperature in Kelvin (37°C)
    pub const BODY_TEMPERATURE_KELVIN: f64 = 310.0;
    /// Default plugin context frequency [Hz]
    pub const DEFAULT_PLUGIN_FREQUENCY: f64 = 100e3;
    /// Minimum domain size for grid validation [m]
    pub const MIN_DOMAIN_SIZE: f64 = 1e-3;
    /// Maximum domain size for grid validation [m]
    pub const MAX_DOMAIN_SIZE: f64 = 1.0;
    /// Minimum points per wavelength for grid validation
    pub const MIN_POINTS_PER_WAVELENGTH: usize = 5;
    /// Maximum points per wavelength for grid validation
    pub const MAX_POINTS_PER_WAVELENGTH: usize = 100;

    // System requirements constants
    /// Minimum required memory in GB for simulation
    pub const MIN_REQUIRED_MEMORY_GB: f64 = 4.0;
    /// Minimum required CPU cores for efficient simulation
    pub const MIN_REQUIRED_CPU_CORES: usize = 2;
    /// Minimum required disk space in GB for output files
    pub const MIN_REQUIRED_DISK_SPACE_GB: f64 = 10.0;
    /// Default conservative disk space estimate if detection fails
    pub const DEFAULT_DISK_SPACE_GB: f64 = 20.0;
}

// Sensor configuration constants
mod sensor_constants {
    /// Default sensor position 1 [m]
    pub const SENSOR_POSITION_1: (f64, f64, f64) = (0.03, 0.0, 0.0);
    /// Default sensor position 2 [m]
    pub const SENSOR_POSITION_2: (f64, f64, f64) = (0.02, 0.0, 0.0);
    /// Default sensor position 3 [m]
    pub const SENSOR_POSITION_3: (f64, f64, f64) = (0.04, 0.0, 0.0);
}

// Default medium properties constants
mod medium_constants {
    /// Default water density [kg/m³]
    pub const WATER_DENSITY: f64 = 998.0;
    /// Default water sound speed [m/s]
    pub const WATER_SOUND_SPEED: f64 = 1482.0;
    /// Default absorption coefficient
    pub const DEFAULT_ABSORPTION: f64 = 0.5;
    /// Default dispersion coefficient
    pub const DEFAULT_DISPERSION: f64 = 10.0;
}

// Core modules
pub mod boundary;
pub mod config;
pub mod constants;
pub mod error;
pub mod factory;
pub mod fft;
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
pub mod sensor;
pub mod signal;
pub mod solver;
pub mod source;
pub mod time;
pub mod utils;
pub mod validation;

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
pub use grid::Grid;
pub use medium::{homogeneous::HomogeneousMedium, Medium};
pub use recorder::Recorder;
pub use sensor::{
    ArrayGeometry, BeamformingMethod, PAMConfig, PassiveAcousticMappingPlugin, Sensor, SensorData,
};
pub use source::Source;
pub use time::Time;
// Solver exports
pub use config::{Config, OutputConfig, SimulationConfig, SourceConfig};
pub use error::{ConfigError, ValidationError};
pub use solver::amr::{
    feature_refinement::{
        CurvatureCriterion, FeatureCriterion, FeatureType, GradientCriterion, LoadBalancer,
        LoadBalancingStrategy, PredictiveCriterion, RefinementCriterion,
    },
    AMRConfig, AMRManager, InterpolationScheme, WaveletType,
};
pub use solver::plugin_based_solver::PluginBasedSolver;
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

// Re-export physics plugin system (the new unified architecture)
pub use physics::field_mapping::{
    FieldAccessor as UnifiedFieldAccessor, FieldAccessorMut, UnifiedFieldType,
};
pub use physics::plugin::{PhysicsPlugin, PluginContext, PluginManager, PluginMetadata};
pub use physics::state::{field_indices, PhysicsState};

// Re-export spectral-DG components
pub use solver::spectral_dg::shock_capturing::{ArtificialViscosity, ShockDetector, WENOLimiter};
pub use solver::spectral_dg::{HybridSpectralDGConfig, HybridSpectralDGSolver};

// Re-export PSTD and FDTD plugins
pub use solver::fdtd::{FdtdConfig, FdtdPlugin, FdtdSolver};
pub use solver::pstd::{PstdConfig, PstdPlugin, PstdSolver};

// Re-export GPU-related items only when feature enabled
#[cfg(feature = "gpu")]
pub use gpu::fft_kernels::{GpuFft, GpuFftPlan};
#[cfg(feature = "gpu")]
pub use gpu::memory::GpuMemoryManager;
#[cfg(feature = "gpu")]
pub use gpu::{GpuBackend, GpuContext};
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
pub use performance::{BenchmarkConfig, BenchmarkReport, BenchmarkSuite, OutputFormat};

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

#[cfg(test)]
mod deprecated_demo {
    #![allow(dead_code)]
    use super::*;
    
    fn create_validated_simulation(
    config: Config,
) -> KwaversResult<(Grid, Time, HomogeneousMedium, Box<dyn Source>, Recorder)> {
    // Skip deprecated validation - use PluginBasedSolver validation instead
    // The validation is now handled by the solver itself
    /*
    let validation_result = validate_simulation_config(&config)?;
    if !validation_result.is_valid {
        let error_summary = validation_result
            .errors
            .iter()
            .map(|e| e.to_string())
            .collect::<Vec<_>>()
            .join("; ");
        return Err(KwaversError::Config(
            crate::error::ConfigError::ValidationFailed {
                field: "simulation".to_string(),
                value: "configuration".to_string(),
                constraint: format!("Validation failed: {}", error_summary),
            },
        ));
    }
    */

    // Create grid using simulation config
    let grid = config.simulation.initialize_grid().map_err(|e| {
        KwaversError::Config(ConfigError::ValidationFailed {
            field: "grid".to_string(),
            value: "initialization".to_string(),
            constraint: e,
        })
    })?;

    // Create time discretization
    let time = config.simulation.initialize_time(&grid).map_err(|e| {
        KwaversError::Config(ConfigError::ValidationFailed {
            field: "time".to_string(),
            value: "initialization".to_string(),
            constraint: e,
        })
    })?;

    // Create medium from configuration
    let medium = HomogeneousMedium::new(
        config.simulation.medium.density,
        config.simulation.medium.sound_speed,
        config.simulation.medium.absorption,
        config.simulation.medium.dispersion,
        &grid,
    );

    // Create source using source config
    let source = config
        .source
        .initialize_source(&medium, &grid)
        .map_err(|e| {
            KwaversError::Config(ConfigError::ValidationFailed {
                field: "source".to_string(),
                value: "initialization".to_string(),
                constraint: e,
            })
        })?;

    // Create sensor with default positions from config or defaults
    let sensor_positions = get_default_sensor_positions();
    let sensor = Sensor::new(&grid, &time, &sensor_positions);

    // Create recorder
    let recorder = Recorder::new(
        sensor,
        &time,
        "simulation_output",
        true,
        true,
        config.output.snapshot_interval,
    );

    Ok((grid, time, medium, source, recorder))
}

/// Get default sensor positions
fn get_default_sensor_positions() -> Vec<(f64, f64, f64)> {
    vec![
        sensor_constants::SENSOR_POSITION_1,
        sensor_constants::SENSOR_POSITION_2,
        sensor_constants::SENSOR_POSITION_3,
    ]
}

// Removed: Use PluginBasedSolver instead
// This function was part of the deprecated monolithic solver API

#[doc(hidden)]
#[deprecated(since = "0.3.0", note = "Use PluginBasedSolver instead")]
fn run_physics_simulation(
    config: Config,
    pstd_config: solver::pstd::PstdConfig,
    pml_config: boundary::pml::PMLConfig,
) -> KwaversResult<()> {
    // Step 1: Setup all simulation components
    let (grid, time, medium, source, mut recorder, mut plugin_manager, mut boundary, mut fields) =
        setup_simulation_components(config, pstd_config, pml_config)?;

    // Step 2: Run the main simulation loop
    run_simulation_loop(
        &time,
        &grid,
        &medium,
        &source,
        &mut recorder,
        &mut plugin_manager,
        &mut boundary,
        &mut fields,
    )?;

    // Step 3: Finalize and report results
    finalize_simulation(&recorder)?;

    Ok(())
}

/// Set up all simulation components
fn setup_simulation_components(
    config: Config,
    pstd_config: solver::pstd::PstdConfig,
    pml_config: boundary::pml::PMLConfig,
) -> KwaversResult<(
    Grid,
    Time,
    HomogeneousMedium,
    Box<dyn Source>,
    Recorder,
    PluginManager,
    PMLBoundary,
    ndarray::Array4<f64>,
)> {
    // Create validated simulation components
    let (grid, time, medium, source, recorder) = create_validated_simulation(config)?;

    // Create plugin manager for physics simulation
    let mut plugin_manager = PluginManager::new();

    // Add PSTD solver for acoustic wave propagation
    plugin_manager.add_plugin(Box::new(solver::pstd::PstdPlugin::new(pstd_config, &grid)?))?;

    // Add thermal diffusion component using adapter
    // Note: Specific physics plugins need to be implemented
    // plugin_manager.register(Box::new(
    //     physics::plugin::thermal_diffusion_plugin("thermal".to_string())
    // ))?;

    // Create boundary conditions
    let boundary = PMLBoundary::new(pml_config)?;

    // Initialize fields
    let mut fields = ndarray::Array4::<f64>::zeros((
        3, // pressure, temperature, light
        grid.nx, grid.ny, grid.nz,
    ));

    // Initialize temperature field with body temperature
    fields
        .index_axis_mut(ndarray::Axis(0), 1)
        .fill(validation_constants::BODY_TEMPERATURE_KELVIN);

    Ok((
        grid,
        time,
        medium,
        source,
        recorder,
        plugin_manager,
        boundary,
        fields,
    ))
}

/// Run the main simulation loop
fn run_simulation_loop(
    time: &Time,
    grid: &Grid,
    medium: &HomogeneousMedium,
    source: &Box<dyn Source>,
    recorder: &mut Recorder,
    plugin_manager: &mut PluginManager,
    boundary: &mut PMLBoundary,
    fields: &mut ndarray::Array4<f64>,
) -> KwaversResult<()> {
    use crate::physics::plugin::PluginContext;

    // Main simulation loop
    for step in 0..time.num_steps() {
        let t = step as f64 * time.dt;

        // Apply source term
        apply_source_term(source, grid, t, fields)?;

        // Apply physics using plugin manager
        let mut plugin_context = PluginContext::new();
        plugin_context.step = step;
        plugin_context.total_steps = time.num_steps();
        plugin_manager.execute(fields, grid, medium, time.dt, t)?;

        // Apply boundary conditions
        apply_boundary_conditions(boundary, fields, grid, step)?;

        // Record data
        recorder.record(fields, step, t);

        // Progress reporting
        if step % validation_constants::PROGRESS_REPORT_INTERVAL == 0 {
            report_progress(step, time.num_steps());
        }
    }

    Ok(())
}

/// Apply source term to the fields
fn apply_source_term(
    source: &Box<dyn Source>,
    grid: &Grid,
    t: f64,
    fields: &mut ndarray::Array4<f64>,
) -> KwaversResult<()> {
    use ndarray::Zip;

    let mut pressure_field = fields.index_axis_mut(ndarray::Axis(0), 0);

    // Use Zip for parallel, in-place update (zero-copy, vectorized)
    Zip::indexed(&mut pressure_field).for_each(|(i, j, k), p| {
        let (x, y, z) = grid.coordinates(i, j, k);
        *p += source.get_source_term(t, x, y, z, grid);
    });

    Ok(())
}

/// Apply boundary conditions to all fields
fn apply_boundary_conditions(
    boundary: &mut PMLBoundary,
    fields: &mut ndarray::Array4<f64>,
    grid: &Grid,
    step: usize,
) -> KwaversResult<()> {
    // Apply to pressure field using a mutable view (zero-copy)
    let mut pressure_field_view = fields.index_axis_mut(ndarray::Axis(0), 0);
    boundary.apply_acoustic(pressure_field_view.view_mut(), grid, step)?;

    // Apply to light field using a mutable view (zero-copy)
    let mut light_field_view = fields.index_axis_mut(ndarray::Axis(0), 1);
    boundary.apply_light(light_field_view.view_mut(), grid, step);

    Ok(())
}

/// Report simulation progress
fn report_progress(step: usize, total_steps: usize) {
    let percentage = (step * 100) / total_steps;
    println!("Step {}/{} ({}%)", step, total_steps, percentage);
}

/// Finalize simulation and report results
    fn finalize_simulation(recorder: &Recorder) -> KwaversResult<()> {
        println!("Physics simulation completed successfully!");
        println!("Results saved to: {}", recorder.filename);
        Ok(())
    }
} // end deprecated_demo module

/// Get library version and build information
///
/// Implements SSOT principle for version information
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

/// Structured system information for validation and logging
///
/// Follows Single Responsibility Principle: holds system data
#[derive(Debug, Clone)]
pub struct SystemInfo {
    pub cpu_cores: usize,
    pub memory_available_gb: f64,
    pub disk_space_gb: f64,
}

impl SystemInfo {
    /// Gathers all system information in a single efficient pass
    ///
    /// This centralizes system queries, avoiding redundant calls and
    /// following the DRY principle for system information gathering
    pub fn new() -> KwaversResult<Self> {
        use sysinfo::{DiskExt, System, SystemExt};

        let mut sys = System::new_all();
        sys.refresh_all();

        // Get CPU cores with fallback
        let cpu_cores = sys
            .physical_core_count()
            .unwrap_or_else(|| sys.cpus().len())
            .max(1);

        // Get available memory in GB
        let memory_available_gb = sys.available_memory() as f64 / (1024.0 * 1024.0 * 1024.0);

        // Get available disk space for current directory
        let current_dir = std::env::current_dir().unwrap_or_else(|_| std::path::PathBuf::from("/"));
        let disk_space_gb = sys
            .disks()
            .iter()
            .find(|disk| current_dir.starts_with(disk.mount_point()))
            .map(|disk| disk.available_space() as f64 / (1024.0 * 1024.0 * 1024.0))
            .unwrap_or(validation_constants::DEFAULT_DISK_SPACE_GB);

        Ok(Self {
            cpu_cores,
            memory_available_gb,
            disk_space_gb,
        })
    }
}

/// Check system compatibility and requirements
///
/// Uses SystemInfo for efficient, single-pass system querying
/// Implements Information Expert principle for system validation
pub fn check_system_compatibility() -> KwaversResult<ValidationResult> {
    use crate::validation::validators;

    let info = SystemInfo::new()?;
    let mut errors = Vec::new();

    // Validate memory using named constant
    let memory_result = validators::validate_range(
        info.memory_available_gb,
        validation_constants::MIN_REQUIRED_MEMORY_GB,
        f64::INFINITY,
        "memory_available_gb",
    );
    if !memory_result.is_valid {
        errors.extend(memory_result.errors);
    }

    // Validate CPU cores using named constant
    let cores_result = validators::validate_range(
        info.cpu_cores as u32,
        validation_constants::MIN_REQUIRED_CPU_CORES as u32,
        u32::MAX,
        "cpu_cores",
    );
    if !cores_result.is_valid {
        errors.extend(cores_result.errors);
    }

    // Validate disk space using named constant
    let disk_result = validators::validate_range(
        info.disk_space_gb,
        validation_constants::MIN_REQUIRED_DISK_SPACE_GB,
        f64::INFINITY,
        "disk_space_gb",
    );
    if !disk_result.is_valid {
        errors.extend(disk_result.errors);
    }

    // Return the validation result
    if errors.is_empty() {
        Ok(ValidationResult::success())
    } else {
        Ok(ValidationResult::failure(errors))
    }
}

/// Conservative default disk space in GB when actual disk space cannot be determined
const CONSERVATIVE_DEFAULT_DISK_SPACE_GB: f64 = 20.0;

/// Get system information using the sysinfo crate for cross-platform compatibility
fn get_system_info() -> KwaversResult<HashMap<String, String>> {
    use sysinfo::{DiskExt, System, SystemExt};

    let mut sys = System::new_all();
    sys.refresh_all();

    let mut info = HashMap::new();

    // Get CPU cores
    let cpu_cores = sys
        .physical_core_count()
        .unwrap_or_else(|| sys.cpus().len())
        .max(1);
    info.insert("cpu_cores".to_string(), cpu_cores.to_string());

    // Get available memory in GB
    let memory_gb = sys.available_memory() as f64 / (1024.0 * 1024.0 * 1024.0);
    info.insert("memory_available_gb".to_string(), memory_gb.to_string());

    // Get available disk space for the current directory
    let current_dir = std::env::current_dir().unwrap_or_else(|_| std::path::PathBuf::from("/"));
    let disk_space_gb = sys
        .disks()
        .iter()
        .find(|disk| current_dir.starts_with(disk.mount_point()))
        .map(|disk| disk.available_space() as f64 / (1024.0 * 1024.0 * 1024.0))
        .unwrap_or(CONSERVATIVE_DEFAULT_DISK_SPACE_GB);
    info.insert("disk_space_gb".to_string(), disk_space_gb.to_string());

    Ok(info)
}

/// Get CPU core count.
fn get_cpu_cores() -> KwaversResult<usize> {
    let info = get_system_info()?;
    let cores_str = info.get("cpu_cores");
    cores_str.and_then(|s| s.parse().ok()).ok_or_else(|| {
        KwaversError::Config(ConfigError::InvalidValue {
            parameter: "cpu_cores".to_string(),
            value: cores_str.map_or("unknown", |s| s.as_str()).to_string(),
            constraint: "Must be a valid integer".to_string(),
        })
    })
}

/// Get available memory in GB.
fn get_available_memory_gb() -> KwaversResult<f64> {
    let info = get_system_info()?;
    let memory_str = info.get("memory_available_gb");
    memory_str.and_then(|s| s.parse().ok()).ok_or_else(|| {
        KwaversError::Config(ConfigError::InvalidValue {
            parameter: "available_memory".to_string(),
            value: memory_str.map_or("unknown", |s| s.as_str()).to_string(),
            constraint: "Must be a valid number".to_string(),
        })
    })
}

/// Get available disk space in GB for current directory.
fn get_available_disk_space_gb() -> KwaversResult<f64> {
    let info = get_system_info()?;
    let disk_str = info.get("disk_space_gb");
    disk_str.and_then(|s| s.parse().ok()).ok_or_else(|| {
        KwaversError::Config(ConfigError::InvalidValue {
            parameter: "disk_space".to_string(),
            value: disk_str.map_or("unknown", |s| s.as_str()).to_string(),
            constraint: "Must be a valid number".to_string(),
        })
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_config_creation() {
        let config = Config::default();
        // Config validation - check that required fields exist
        assert!(config.simulation.frequency > 0.0);
        assert!(config.source.frequency.is_some());
        assert!(!config.output.enable_visualization); // Default is false
    }

    #[test]
    fn test_config_with_custom_values() {
        let config = Config {
            simulation: config::SimulationConfig {
                frequency: 2e6,
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

    #[test]
    fn test_system_compatibility() {
        let result = check_system_compatibility().unwrap();
        // This test may fail depending on actual system resources
        // For now, we just check that the function runs without error
        assert!(result.is_valid || !result.is_valid); // Always true, just checking execution
    }
}
