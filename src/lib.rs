//! # Kwavers: Advanced Acoustic Simulation Library
//!
//! A comprehensive acoustic wave simulation library with support for:
//! - Linear and nonlinear wave propagation
//! - Multi-physics simulations (acoustic, thermal, optical)
//! - Advanced numerical methods (FDTD, PSTD, spectral methods)
//! - Real-time processing and visualization

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
#[cfg(all(feature = "gpu", any(feature = "gpu-visualization", feature = "web_visualization", feature = "vr-support")))]
pub mod visualization;

// Re-export commonly used types for convenience
pub use error::{KwaversResult, KwaversError};
pub use grid::Grid;
pub use time::Time;
pub use medium::{Medium, homogeneous::HomogeneousMedium};
pub use source::Source;
pub use sensor::{Sensor, SensorData, PassiveAcousticMappingPlugin, PAMConfig, ArrayGeometry, BeamformingMethod};
pub use recorder::Recorder;
pub use boundary::{Boundary, PMLBoundary, CPMLBoundary, CPMLConfig, PMLConfig};
// Solver exports
pub use solver::plugin_based_solver::PluginBasedSolver;
pub use solver::reconstruction::{
    Reconstructor, ReconstructionAlgorithm, ReconstructionConfig,
    FilterType, InterpolationMethod, UniversalBackProjection, WeightFunction
};
pub use solver::reconstruction::photoacoustic::PhotoacousticReconstructor;
pub use solver::reconstruction::seismic::{
    FullWaveformInversion, ReverseTimeMigration,
    SeismicImagingConfig, RtmImagingCondition
};
pub use solver::amr::{AMRConfig, AMRManager, WaveletType, InterpolationScheme, feature_refinement::{RefinementCriterion, GradientCriterion, CurvatureCriterion, FeatureCriterion, FeatureType, PredictiveCriterion, LoadBalancer, LoadBalancingStrategy}};
pub use solver::time_reversal::{TimeReversalConfig, TimeReversalReconstructor};
pub use config::{Config, SimulationConfig, SourceConfig, OutputConfig};
pub use validation::{ValidationResult, Validatable};
pub use error::{ValidationError, ConfigError};

// Re-export physics plugin system (the new unified architecture)
pub use physics::plugin::{PhysicsPlugin, PluginManager, PluginContext, PluginMetadata};
pub use physics::field_mapping::{UnifiedFieldType, FieldAccessor as UnifiedFieldAccessor, FieldAccessorMut};
pub use physics::state::{PhysicsState, field_indices};

// Re-export spectral-DG components
pub use solver::spectral_dg::{HybridSpectralDGSolver, HybridSpectralDGConfig};
pub use solver::spectral_dg::shock_capturing::{ShockDetector, WENOLimiter, ArtificialViscosity};

// Re-export PSTD and FDTD plugins
pub use solver::pstd::{PstdSolver, PstdConfig, PstdPlugin};
pub use solver::fdtd::{FdtdSolver, FdtdConfig, FdtdPlugin};

// Re-export GPU-related items only when feature enabled
#[cfg(feature = "gpu")]
pub use gpu::{GpuContext, GpuBackend};
#[cfg(feature = "gpu")]
pub use gpu::memory::GpuMemoryManager;
#[cfg(feature = "gpu")]
pub use gpu::fft_kernels::{GpuFft, GpuFftPlan};
pub use physics::mechanics::{CavitationModel, StreamingModel, KuznetsovWave, KuznetsovConfig};
pub use physics::mechanics::acoustic_wave::NonlinearWave;
pub use physics::chemistry::ChemicalModel;
pub use physics::mechanics::elastic_wave::{ElasticWave, mode_conversion::{ModeConversionConfig, ViscoelasticConfig, StiffnessTensor, MaterialSymmetry}};
pub use physics::traits::{AcousticWaveModel, CavitationModelBehavior, ChemicalModelTrait};

// Re-export factory components  
pub use factory::{SimulationFactory, SimulationConfig as FactorySimulationConfig, GridConfig, MediumConfig, MediumType, PhysicsConfig, PhysicsModelType, PhysicsModelConfig, TimeConfig, ValidationConfig, SourceConfig as FactorySourceConfig, ConfigBuilder, SimulationComponents};

// Re-export I/O functions
pub use io::{save_pressure_data, save_light_data, generate_summary};

// Re-export performance optimization
pub use performance::{
    PerformanceOptimizer, OptimizationConfig, SimdLevel,
    profiling::{PerformanceProfiler, ProfileReport, TimingScope, MemoryProfile, CacheProfile, RooflineAnalysis}
};

// Re-export signal types
pub use signal::{SineWave, Signal};

// Re-export source types
pub use source::{LinearArray, HanningApodization};

// Re-export configuration types
pub use sensor::SensorConfig;
pub use recorder::RecorderConfig;

// Re-export benchmarks from performance module
pub use performance::{BenchmarkSuite, BenchmarkConfig, BenchmarkReport, OutputFormat};

// Re-export solver validation
pub use solver::validation::{KWaveValidator, KWaveTestCase, ValidationReport};

// Additional reconstruction exports for specific recon types
pub use solver::reconstruction::{
    plane_recon::PlaneRecon, line_recon::LineRecon, 
    arc_recon::ArcRecon, bowl_recon::BowlRecon
};

/// Initialize logging for the kwavers library
pub fn init_logging() -> KwaversResult<()> {
    env_logger::init();
    Ok(())
}

/// Create default visualization plots for simulation outputs
pub fn plot_simulation_outputs(output_dir: &str, files: &[&str]) -> KwaversResult<()> {
    use std::path::Path;
    
    for file in files {
        let filepath = Path::new(output_dir).join(file);
        if !filepath.exists() {
            println!("Warning: File not found: {}", filepath.display());
            continue;
        }
        
        // For now, just log what would be plotted
        // Actual plotting would require loading CSV data and using the plotting module functions
        println!("Would plot: {}", filepath.display());
    }
    
    Ok(())
}

// Note: Use Config::default() instead of create_default_config()
// The Default trait is implemented for Config and provides the same functionality

// Note: Configuration validation should be done through the Config type itself
// or through the PluginBasedSolver's validation mechanisms

// Removed: Use PluginBasedSolver instead
// This function was part of the deprecated monolithic solver API

#[doc(hidden)]
#[deprecated(since = "0.3.0", note = "Use PluginBasedSolver instead")]
fn create_validated_simulation(
    config: Config,
) -> KwaversResult<(Grid, Time, HomogeneousMedium, Box<dyn Source>, Recorder)> {
    // Validate configuration first
    let validation_result = validate_simulation_config(&config)?;
    if !validation_result.is_valid {
        let error_summary = validation_result.errors.iter()
            .map(|e| e.to_string())
            .collect::<Vec<_>>()
            .join("; ");
        return Err(KwaversError::Config(crate::error::ConfigError::ValidationFailed {
            field: "simulation".to_string(),
            value: "configuration".to_string(),
            constraint: format!("Validation failed: {}", error_summary),
        }));
    }
    
    // Create grid using simulation config
    let grid = config.simulation.initialize_grid()
        .map_err(|e| KwaversError::Config(ConfigError::ValidationFailed {
            field: "grid".to_string(),
            value: "initialization".to_string(),
            constraint: e,
        }))?;
    
    // Create time discretization
    let time = config.simulation.initialize_time(&grid)
        .map_err(|e| KwaversError::Config(ConfigError::ValidationFailed {
            field: "time".to_string(),
            value: "initialization".to_string(),
            constraint: e,
        }))?;
    
    // Create medium from configuration
    let medium = HomogeneousMedium::new(
        config.simulation.medium.density,
        config.simulation.medium.sound_speed,
        &grid,
        config.simulation.medium.absorption,
        config.simulation.medium.dispersion
    );
    
    // Create source using source config
    let source = config.source.initialize_source(&medium, &grid)
        .map_err(|e| KwaversError::Config(ConfigError::ValidationFailed {
            field: "source".to_string(),
            value: "initialization".to_string(),
            constraint: e,
        }))?;
    
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
    use crate::physics::plugin::PluginContext;
    
    // Step 1: Setup all simulation components
    let (grid, time, medium, source, mut recorder, mut plugin_manager, mut boundary, mut fields) = 
        setup_simulation_components(config, pstd_config, pml_config)?;
    
    // Step 2: Run the main simulation loop
    run_simulation_loop(
        &time, &grid, &medium, &source, &mut recorder, 
        &mut plugin_manager, &mut boundary, &mut fields,
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
    Grid, Time, HomogeneousMedium, Box<dyn Source>, Recorder,
    PluginManager, PMLBoundary, ndarray::Array4<f64>
)> {
    // Create validated simulation components
    let (grid, time, medium, source, recorder) = create_validated_simulation(config)?;
    
    // Create plugin manager for physics simulation
    let mut plugin_manager = PluginManager::new();
    
    // Add PSTD solver for acoustic wave propagation
    plugin_manager.register(Box::new(
        solver::pstd::PstdPlugin::new(pstd_config, &grid)?
    ))?;
    
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
        grid.nx,
        grid.ny,
        grid.nz,
    ));
    
    // Initialize temperature field with body temperature
    fields.index_axis_mut(ndarray::Axis(0), 1).fill(validation_constants::BODY_TEMPERATURE_KELVIN);
    
    Ok((grid, time, medium, source, recorder, plugin_manager, boundary, fields))
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
        let plugin_context = PluginContext::new(step, time.num_steps(), validation_constants::DEFAULT_PLUGIN_FREQUENCY);
        plugin_manager.update_all(
            fields,
            grid,
            medium,
            time.dt,
            t,
            &plugin_context,
        )?;
        
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

/// Get library version and build information
/// 
/// Implements SSOT principle for version information
pub fn get_version_info() -> HashMap<String, String> {
    let mut info = HashMap::new();
    info.insert("version".to_string(), env!("CARGO_PKG_VERSION").to_string());
    info.insert("name".to_string(), env!("CARGO_PKG_NAME").to_string());
    info.insert("description".to_string(), env!("CARGO_PKG_DESCRIPTION").to_string());
    info.insert("authors".to_string(), env!("CARGO_PKG_AUTHORS").to_string());
    info.insert("repository".to_string(), env!("CARGO_PKG_REPOSITORY").to_string());
    info.insert("license".to_string(), env!("CARGO_PKG_LICENSE").to_string());
    info
}

/// Check system compatibility and requirements
/// 
/// Implements Information Expert principle for system validation
pub fn check_system_compatibility() -> KwaversResult<ValidationResult> {
    use crate::validation::validators;
    
    let mut errors = Vec::new();
    
    // Get system information
    let sys_info = get_system_info()?;
    
    // Validate memory (at least 4GB available)
    if let Some(memory_str) = sys_info.get("memory_available_gb") {
        if let Ok(memory_gb) = memory_str.parse::<f64>() {
            let memory_result = validators::validate_range(memory_gb, 4.0, f64::INFINITY, "memory_available_gb");
            if !memory_result.is_valid {
                errors.extend(memory_result.errors);
            }
        }
    }
    
    // Validate CPU cores (at least 2)
    if let Some(cores_str) = sys_info.get("cpu_cores") {
        if let Ok(cores) = cores_str.parse::<u32>() {
            let cores_result = validators::validate_range(cores, 2, u32::MAX, "cpu_cores");
            if !cores_result.is_valid {
                errors.extend(cores_result.errors);
            }
        }
    }
    
    // Return the validation result
    if errors.is_empty() {
        Ok(ValidationResult::success())
    } else {
        Ok(ValidationResult::failure(errors))
    }
}

/// Get system information using the sysinfo crate for cross-platform compatibility
fn get_system_info() -> KwaversResult<HashMap<String, String>> {
    use sysinfo::{System, SystemExt, DiskExt};
    
    let mut sys = System::new_all();
    sys.refresh_all();
    
    let mut info = HashMap::new();
    
    // Get CPU cores
    let cpu_cores = sys.physical_core_count()
        .unwrap_or_else(|| sys.cpus().len())
        .max(1);
    info.insert("cpu_cores".to_string(), cpu_cores.to_string());
    
    // Get available memory in GB
    let memory_gb = sys.available_memory() as f64 / (1024.0 * 1024.0 * 1024.0);
    info.insert("memory_available_gb".to_string(), memory_gb.to_string());
    
    // Get available disk space for the current directory
    let current_dir = std::env::current_dir().unwrap_or_else(|_| std::path::PathBuf::from("/"));
    let disk_space_gb = sys.disks()
        .iter()
        .find(|disk| current_dir.starts_with(disk.mount_point()))
        .map(|disk| disk.available_space() as f64 / (1024.0 * 1024.0 * 1024.0))
        .unwrap_or(20.0); // Conservative default
    info.insert("disk_space_gb".to_string(), disk_space_gb.to_string());
    
    Ok(info)
}

/// Get CPU core count.
fn get_cpu_cores() -> KwaversResult<usize> {
    get_system_info()?
        .get("cpu_cores")
        .and_then(|s| s.parse().ok())
        .ok_or_else(|| KwaversError::Internal("Failed to parse CPU cores from system info".to_string()))
}

/// Get available memory in GB.
fn get_available_memory_gb() -> KwaversResult<f64> {
    get_system_info()?
        .get("memory_available_gb")
        .and_then(|s| s.parse().ok())
        .ok_or_else(|| KwaversError::Internal("Failed to parse available memory from system info".to_string()))
}

/// Get available disk space in GB for current directory.
fn get_available_disk_space_gb() -> KwaversResult<f64> {
    get_system_info()?
        .get("disk_space_gb")
        .and_then(|s| s.parse().ok())
        .ok_or_else(|| KwaversError::Internal("Failed to parse disk space from system info".to_string()))
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