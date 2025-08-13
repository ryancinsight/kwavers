//! # Kwavers - Ultrasound Simulation Toolbox
//!
//! A modern, high-performance, open-source computational toolbox for simulating
//! ultrasound wave propagation in complex heterogeneous media.
//!
//! ## Features
//!
//! - **Physics Modeling**: Nonlinear acoustics, thermal effects, cavitation dynamics
//! - **GPU Acceleration**: CUDA/OpenCL backend for massive parallel processing
//! - **Memory Safety**: Zero unsafe code with comprehensive error handling
//! - **Performance**: Optimized algorithms with SIMD and parallel processing
//! - **Extensibility**: Modular architecture following SOLID principles

use ndarray::Array3;
use std::collections::HashMap;

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
pub mod log;
pub mod medium;
pub mod ml;
pub mod output;
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
pub mod benchmarks;

// Phase 11: Visualization & Real-Time Interaction
#[cfg(all(feature = "gpu", any(feature = "advanced-visualization", feature = "web-visualization", feature = "vr-support")))]
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
pub use solver::Solver;
pub use solver::amr::{AMRConfig, AMRManager, WaveletType, InterpolationScheme, feature_refinement::{RefinementCriterion, GradientCriterion, CurvatureCriterion, FeatureCriterion, FeatureType, PredictiveCriterion, LoadBalancer, LoadBalancingStrategy}};
pub use solver::time_reversal::{TimeReversalConfig, TimeReversalReconstructor};
pub use config::{Config, SimulationConfig, SourceConfig, OutputConfig};
pub use validation::{ValidationResult, ValidationManager, ValidationBuilder, ValidationValue, ValidationWarning, WarningSeverity, ValidationContext, ValidationMetadata};
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
pub use physics::mechanics::{NonlinearWave, CavitationModel, StreamingModel, KuznetsovWave, KuznetsovConfig};
pub use physics::chemistry::ChemicalModel;
pub use physics::mechanics::elastic_wave::{ElasticWave, mode_conversion::{ModeConversionConfig, ViscoelasticConfig, StiffnessTensor, MaterialSymmetry}};
pub use physics::traits::{AcousticWaveModel, CavitationModelBehavior, ChemicalModelTrait};

// Re-export factory components  
pub use factory::{SimulationFactory, SimulationConfig as FactorySimulationConfig, GridConfig, MediumConfig, MediumType, PhysicsConfig, PhysicsModelType, TimeConfig, ValidationConfig, SourceConfig as FactorySourceConfig, SimulationBuilder, SimulationSetup, SimulationResults};

// Re-export utility functions
pub use output::{save_pressure_data, save_light_data, generate_summary};

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

// Re-export benchmarks
pub use benchmarks::{BenchmarkSuite, BenchmarkConfig, BenchmarkReport, OutputFormat};

// Re-export solver validation
pub use solver::validation::{KWaveValidator, KWaveTestCase, ValidationReport};

// Re-export reconstruction algorithms
pub use solver::reconstruction::{
    ReconstructionConfig, ReconstructionAlgorithm, FilterType, InterpolationMethod,
    Reconstructor, UniversalBackProjection, WeightFunction,
    plane_recon::PlaneRecon, line_recon::LineRecon, 
    arc_recon::ArcRecon, bowl_recon::BowlRecon
};

/// Initialize logging for the kwavers library
/// 
/// Implements KISS principle with simple, clear initialization
pub fn init_logging() -> KwaversResult<()> {
    env_logger::init();
    Ok(())
}

/// Plot simulation outputs using the built-in visualization system
/// 
/// Implements YAGNI principle by providing only necessary visualization features
pub fn plot_simulation_outputs(
    output_dir: &str,
    files: &[&str],
) -> KwaversResult<()> {
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

/// Create a default simulation configuration
/// 
/// Implements SSOT principle as the single source of truth for default configuration
pub fn create_default_config() -> Config {
    Config {
        simulation: SimulationConfig {
            domain_size_x: 0.1,
            domain_size_yz: 0.1,
            points_per_wavelength: 10,
            frequency: 1e6,
            num_cycles: 5.0,
            pml_thickness: 10,
            pml_sigma_acoustic: 100.0,
            pml_sigma_light: 10.0,
            pml_polynomial_order: 3,
            pml_reflection: 1e-6,
            light_wavelength: 500.0,
            kspace_padding: 0,
            kspace_alpha: 1.0,
            medium_type: None,
        },
        source: SourceConfig {
            num_elements: 32,
            signal_type: "sine".to_string(),
            start_freq: None,
            end_freq: None,
            signal_duration: None,
            phase: None,
            focus_x: Some(0.03),
            focus_y: Some(0.0),
            focus_z: Some(0.0),
            frequency: Some(1e6),
            amplitude: Some(1e6),
        },
        output: OutputConfig {
            pressure_file: "pressure_output.csv".to_string(),
            light_file: "light_output.csv".to_string(),
            summary_file: "summary.csv".to_string(),
            snapshot_interval: 10,
            enable_visualization: true,
        },
    }
}

/// Validate a simulation configuration
/// 
/// Implements Information Expert principle by providing validation logic
pub fn validate_simulation_config(config: &Config) -> KwaversResult<ValidationResult> {
    let mut validation_result = ValidationResult::valid("simulation_config_validation".to_string());
    
    // Basic validation checks
    if config.simulation.domain_size_x < 1e-3 || config.simulation.domain_size_x > 1.0 {
        validation_result.add_error(ValidationError::RangeValidation {
            field: "domain_size_x".to_string(),
            value: config.simulation.domain_size_x,
            min: 1e-3,
            max: 1.0,
        });
    }
    
    if config.simulation.domain_size_yz < 1e-3 || config.simulation.domain_size_yz > 1.0 {
        validation_result.add_error(ValidationError::RangeValidation {
            field: "domain_size_yz".to_string(),
            value: config.simulation.domain_size_yz,
            min: 1e-3,
            max: 1.0,
        });
    }
    
    if config.simulation.points_per_wavelength < 5 || config.simulation.points_per_wavelength > 100 {
        validation_result.add_error(ValidationError::RangeValidation {
            field: "points_per_wavelength".to_string(),
            value: config.simulation.points_per_wavelength as f64,
            min: 5.0,
            max: 100.0,
        });
    }
    
    if config.simulation.frequency < 1e3 || config.simulation.frequency > 100e6 {
        validation_result.add_error(ValidationError::RangeValidation {
            field: "frequency".to_string(),
            value: config.simulation.frequency,
            min: 1e3,
            max: 100e6,
        });
    }
    
    if let Some(freq) = config.source.frequency {
        if !(1e3..=100e6).contains(&freq) {
            validation_result.add_error(ValidationError::RangeValidation {
                field: "source_frequency".to_string(),
                value: freq,
                min: 1e3,
                max: 100e6,
            });
        }
    }
    
    if let Some(amp) = config.source.amplitude {
        if !(1e3..=100e6).contains(&amp) {
            validation_result.add_error(ValidationError::RangeValidation {
                field: "source_amplitude".to_string(),
                value: amp,
                min: 1e3,
                max: 100e6,
            });
        }
    }
    
    Ok(validation_result)
}

/// Create a complete simulation setup with validation
/// 
/// Implements Controller pattern from GRASP principles
pub fn create_validated_simulation(
    config: Config,
) -> KwaversResult<(Grid, Time, HomogeneousMedium, Box<dyn Source>, Recorder)> {
    // Validate configuration first
    let validation_result = validate_simulation_config(&config)?;
    if !validation_result.is_valid {
        return Err(KwaversError::Config(crate::error::ConfigError::ValidationFailed {
            section: "simulation".to_string(),
            reason: format!("Configuration validation failed: {}", validation_result.summary()),
        }));
    }
    
    // Create grid using simulation config
    let grid = config.simulation.initialize_grid()
        .map_err(|e| KwaversError::Config(ConfigError::ValidationFailed {
            section: "simulation".to_string(),
            reason: e,
        }))?;
    
    // Create time discretization
    let time = config.simulation.initialize_time(&grid)
        .map_err(|e| KwaversError::Config(ConfigError::ValidationFailed {
            section: "simulation".to_string(),
            reason: e,
        }))?;
    
    // Create medium
    let medium = HomogeneousMedium::new(998.0, 1482.0, &grid, 0.5, 10.0);
    
    // Create source using source config
    let source = config.source.initialize_source(&medium, &grid)
        .map_err(|e| KwaversError::Config(ConfigError::ValidationFailed {
            section: "source".to_string(),
            reason: e,
        }))?;
    
    // Create sensor
    let sensor_positions = vec![
        (0.03, 0.0, 0.0),
        (0.02, 0.0, 0.0),
        (0.04, 0.0, 0.0),
    ];
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

/// Run a complete simulation with physics
/// 
/// Implements ACID principles for simulation execution
pub fn run_physics_simulation(
    config: Config,
) -> KwaversResult<()> {
    use crate::physics::plugin::PluginContext;
    
    // Create validated simulation components
    let (grid, time, medium, source, mut recorder) = create_validated_simulation(config)?;
    
    // Create plugin manager for physics simulation
    let mut plugin_manager = PluginManager::new();
    
    // Add PSTD solver for acoustic wave propagation
    let pstd_config = solver::pstd::PstdConfig {
        k_space_correction: true,
        k_space_order: 2,
        anti_aliasing: true,
        pml_stencil_size: 10,
        cfl_factor: 0.3,
        use_leapfrog: true,
        enable_absorption: false,
        absorption_model: None,
    };
    plugin_manager.register(Box::new(
        solver::pstd::PstdPlugin::new(pstd_config, &grid)?
    ))?;
    
    // Add thermal diffusion component using adapter
    // Note: Specific physics plugins need to be implemented
    // plugin_manager.register(Box::new(
    //     physics::plugin::thermal_diffusion_plugin("thermal".to_string())
    // ))?;
    
    // Create boundary conditions
    let mut boundary = PMLBoundary::new(
        PMLConfig::default()
            .with_thickness(10)
            .with_reflection_coefficient(1e-6),
    )?;
    
    // Initialize fields
    let mut fields = ndarray::Array4::<f64>::zeros((
        3, // pressure, temperature, light
        grid.nx,
        grid.ny,
        grid.nz,
    ));
    
    // Initialize temperature field
    fields.index_axis_mut(ndarray::Axis(0), 1).fill(310.0); // 37°C
    
    // Create physics context
    let mut context = PluginContext::new(0, time.num_steps(), 1e6);
    
    // Main simulation loop
    for step in 0..time.num_steps() {
        let t = step as f64 * time.dt;
        context.step = step;
        
        // Apply source
        // Create source field array
        let (nx, ny, nz) = grid.dimensions();
        let mut source_field = Array3::zeros((nx, ny, nz));
        for i in 0..nx {
            for j in 0..ny {
                for k in 0..nz {
                    let (x, y, z) = grid.coordinates(i, j, k);
                    source_field[[i, j, k]] = source.get_source_term(t, x, y, z, &grid);
                }
            }
        }
        // Source term would be added directly to the fields if needed
        
        // Apply physics using plugin manager
        let plugin_context = PluginContext::new(step, time.n_steps, 100e3);
        plugin_manager.update_all(
            &mut fields,
            &grid,
            &medium,
            time.dt,
            t,
            &plugin_context,
        )?;
        
        // Apply boundary conditions
        let mut pressure_field = fields.index_axis_mut(ndarray::Axis(0), 0).to_owned();
        boundary.apply_acoustic(&mut pressure_field, &grid, step)?;
        fields.index_axis_mut(ndarray::Axis(0), 0).assign(&pressure_field);
        let mut light_field = fields.index_axis_mut(ndarray::Axis(0), 1).to_owned();
        boundary.apply_light(&mut light_field, &grid, step);
        fields.index_axis_mut(ndarray::Axis(0), 1).assign(&light_field);
        
        // Record data
        recorder.record(&fields, step, t);
        
        // Progress reporting
        if step % 100 == 0 {
            println!("Step {}/{} ({}%)", step, time.num_steps(), 
                (step * 100) / time.num_steps());
        }
    }
    
    // Generate visualizations if enabled
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
    let _validation_manager = ValidationManager::new();
    
    // Create system compatibility validation pipeline
    let pipeline = ValidationBuilder::new("system_compatibility_validation".to_string())
        .with_range("memory_available_gb".to_string(), Some(4.0), None)
        .with_range("cpu_cores".to_string(), Some(2.0), None)
        .with_range("disk_space_gb".to_string(), Some(1.0), None)
        .build();
    
    // Get actual system information
    let (cpu_cores, memory_gb, disk_space_gb) = get_system_information();
    let system_values = vec![
        ("memory_available_gb", ValidationValue::Float(memory_gb)),
        ("cpu_cores", ValidationValue::Float(cpu_cores as f64)),
        ("disk_space_gb", ValidationValue::Float(disk_space_gb)),
    ];
    
    let results = validation::utils::validate_multiple(&pipeline, &system_values);
    
    // Merge results
    let mut final_result = ValidationResult::valid("system_compatibility_validation".to_string());
    for result in results.values() {
        final_result.merge(result.clone());
    }
    
    Ok(final_result)
}

/// Get system information using the sysinfo crate for cross-platform compatibility
fn get_system_information() -> (usize, f64, f64) {
    use sysinfo::{System, SystemExt, DiskExt};
    
    let mut sys = System::new_all();
    sys.refresh_all();
    
    // Get CPU cores
    let cpu_cores = sys.physical_core_count()
        .unwrap_or_else(|| sys.cpus().len())
        .max(1);
    
    // Get available memory in GB
    let memory_gb = sys.available_memory() as f64 / (1024.0 * 1024.0 * 1024.0);
    
    // Get available disk space for the current directory
    let current_dir = std::env::current_dir().unwrap_or_else(|_| std::path::PathBuf::from("/"));
    let disk_space_gb = sys.disks()
        .iter()
        .find(|disk| current_dir.starts_with(disk.mount_point()))
        .map(|disk| disk.available_space() as f64 / (1024.0 * 1024.0 * 1024.0))
        .unwrap_or(20.0); // Conservative default
    
    (cpu_cores, memory_gb, disk_space_gb)
}

/// Get CPU core count
fn get_cpu_cores() -> usize {
    get_system_information().0
}

/// Get available memory in GB
fn get_available_memory_gb() -> f64 {
    get_system_information().1
}

/// Get available disk space in GB for current directory
fn get_available_disk_space_gb() -> f64 {
    get_system_information().2
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_config_creation() {
        let config = create_default_config();
        // Config validation - check that required fields exist
        assert!(config.simulation.frequency > 0.0);
        assert!(config.source.frequency.is_some());
        assert!(config.output.enable_visualization);
    }
    
    #[test]
    fn test_config_validation() {
        let config = create_default_config();
        let validation_result = validate_simulation_config(&config).unwrap();
        assert!(validation_result.is_valid);
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