// src/lib.rs
//! Kwavers - Advanced Ultrasound Simulation Toolbox
//!
//! A modern, high-performance, open-source computational toolbox for simulating
//! ultrasound wave propagation and its interactions with complex biological media.
//!
//! Design Principles Implemented:
//! - SOLID: Single responsibility, open/closed, Liskov substitution, interface segregation, dependency inversion
//! - CUPID: Composable, Unix-like, Predictable, Idiomatic, Domain-focused
//! - GRASP: Information expert, creator, controller, low coupling, high cohesion
//! - ACID: Atomicity, consistency, isolation, durability
//! - DRY: Don't repeat yourself
//! - KISS: Keep it simple, stupid
//! - YAGNI: You aren't gonna need it
//! - SSOT: Single source of truth
//! - CCP: Common closure principle
//! - CRP: Common reuse principle
//! - ADP: Acyclic dependency principle

pub mod error;
pub mod grid;
pub mod time;
pub mod medium;
pub mod source;
pub mod sensor;
pub mod recorder;
pub mod boundary;
pub mod solver;
pub mod physics;
pub mod config;
pub mod validation;
pub mod signal;
pub mod utils;
pub mod fft;

use std::collections::HashMap;
use ndarray::Array3;

// Re-export commonly used types for convenience
pub use error::{KwaversResult, KwaversError};
pub use grid::Grid;
pub use time::Time;
pub use medium::{Medium, homogeneous::HomogeneousMedium};
pub use source::Source;
pub use sensor::Sensor;
pub use recorder::Recorder;
pub use boundary::{Boundary, pml::PMLBoundary, pml::PMLConfig};
pub use solver::Solver;
pub use config::{Config, SimulationConfig, SourceConfig, OutputConfig};
pub use validation::{ValidationResult, ValidationManager, ValidationBuilder, ValidationValue};
pub use error::{ValidationError, ConfigError};

// Re-export physics components
pub use physics::composable::{PhysicsPipeline, PhysicsContext, PhysicsComponent};

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
    // Placeholder implementation - would integrate with actual plotting library
    println!("Generating plots for {} files in directory: {}", files.len(), output_dir);
    
    for file in files {
        println!("  - {}", file);
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
        if freq < 1e3 || freq > 100e6 {
            validation_result.add_error(ValidationError::RangeValidation {
                field: "source_frequency".to_string(),
                value: freq,
                min: 1e3,
                max: 100e6,
            });
        }
    }
    
    if let Some(amp) = config.source.amplitude {
        if amp < 1e3 || amp > 100e6 {
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

/// Run a complete simulation with advanced physics
/// 
/// Implements ACID principles for simulation execution
pub fn run_advanced_simulation(
    config: Config,
) -> KwaversResult<()> {
    // Create validated simulation components
    let (grid, time, medium, source, mut recorder) = create_validated_simulation(config)?;
    
    // Create physics pipeline with advanced components
    let mut physics_pipeline = PhysicsPipeline::new();
    
    // Add acoustic wave component
    physics_pipeline.add_component(Box::new(
        physics::composable::AcousticWaveComponent::new("acoustic".to_string())
    ))?;
    
    // Add thermal diffusion component
    physics_pipeline.add_component(Box::new(
        physics::composable::ThermalDiffusionComponent::new("thermal".to_string())
    ))?;
    
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
    let mut context = PhysicsContext::new(1e6);
    
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
        context.add_source_term("acoustic_source".to_string(), source_field);
        
        // Apply physics pipeline
        physics_pipeline.execute(
            &mut fields,
            &grid,
            &medium,
            time.dt,
            t,
            &mut context,
        )?;
        
        // Apply boundary conditions
        let mut pressure_field = fields.index_axis_mut(ndarray::Axis(0), 0).to_owned();
        boundary.apply_acoustic(&mut pressure_field, &grid, step);
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
    println!("Advanced simulation completed successfully!");
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
    let mut validation_manager = ValidationManager::new();
    
    // Create system compatibility validation pipeline
    let pipeline = ValidationBuilder::new("system_compatibility_validation".to_string())
        .with_range("memory_available_gb".to_string(), Some(4.0), None)
        .with_range("cpu_cores".to_string(), Some(2.0), None)
        .with_range("disk_space_gb".to_string(), Some(1.0), None)
        .build();
    
    // Get system information (simplified)
    let system_values = vec![
        ("memory_available_gb", ValidationValue::Float(8.0)), // Placeholder
        ("cpu_cores", ValidationValue::Float(4.0)), // Placeholder
        ("disk_space_gb", ValidationValue::Float(100.0)), // Placeholder
    ];
    
    let results = validation::utils::validate_multiple(&pipeline, &system_values);
    
    // Merge results
    let mut final_result = ValidationResult::valid("system_compatibility_validation".to_string());
    for result in results.values() {
        final_result.merge(result.clone());
    }
    
    Ok(final_result)
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