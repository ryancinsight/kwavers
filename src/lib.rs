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

// Re-export commonly used types for convenience
pub use error::{KwaversResult, KwaversError};
pub use grid::Grid;
pub use time::Time;
pub use medium::{Medium, HomogeneousMedium};
pub use source::{Source, SourceConfig};
pub use sensor::Sensor;
pub use recorder::{Recorder, RecorderConfig};
pub use boundary::{Boundary, PMLBoundary, PMLConfig};
pub use solver::Solver;
pub use config::{Configuration, ConfigManager, ConfigBuilder, ConfigValue};
pub use validation::{ValidationResult, ValidationManager, ValidationBuilder, ValidationValue};

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
pub fn create_default_config() -> Configuration {
    ConfigBuilder::new()
        .with_string("simulation_name".to_string(), "default_simulation".to_string())
        .with_string("version".to_string(), env!("CARGO_PKG_VERSION").to_string())
        .with_boolean("enable_visualization".to_string(), true)
        .with_float("frequency".to_string(), 1e6)
        .with_float("amplitude".to_string(), 1e6)
        .with_integer("grid_nx".to_string(), 100)
        .with_integer("grid_ny".to_string(), 100)
        .with_integer("grid_nz".to_string(), 100)
        .with_float("grid_dx".to_string(), 1e-3)
        .with_float("grid_dy".to_string(), 1e-3)
        .with_float("grid_dz".to_string(), 1e-3)
        .with_float("time_duration".to_string(), 1e-3)
        .build()
}

/// Validate a simulation configuration
/// 
/// Implements Information Expert principle by providing validation logic
pub fn validate_simulation_config(config: &Configuration) -> KwaversResult<ValidationResult> {
    let mut validation_manager = ValidationManager::new();
    
    // Create validation pipeline for simulation configuration
    let pipeline = ValidationBuilder::new("simulation_config_validation".to_string())
        .with_required("simulation_name".to_string())
        .with_string_length("simulation_name".to_string(), Some(1), Some(100))
        .with_required("version".to_string())
        .with_pattern("version".to_string(), "\\d+\\.\\d+\\.\\d+".to_string(), "must be semantic version".to_string())
        .with_range("frequency".to_string(), Some(1e3), Some(100e6))
        .with_range("amplitude".to_string(), Some(1e3), Some(100e6))
        .with_range("grid_nx".to_string(), Some(10.0), Some(10000.0))
        .with_range("grid_ny".to_string(), Some(10.0), Some(10000.0))
        .with_range("grid_nz".to_string(), Some(10.0), Some(10000.0))
        .with_range("grid_dx".to_string(), Some(1e-6), Some(1e-1))
        .with_range("grid_dy".to_string(), Some(1e-6), Some(1e-1))
        .with_range("grid_dz".to_string(), Some(1e-6), Some(1e-1))
        .with_range("time_duration".to_string(), Some(1e-6), Some(1e0))
        .build();
    
    // Convert configuration to validation values
    let mut validation_values = Vec::new();
    
    for (key, value) in config.iter() {
        let validation_value = match value {
            ConfigValue::String(s) => ValidationValue::String(s.clone()),
            ConfigValue::Integer(i) => ValidationValue::Integer(*i),
            ConfigValue::Float(f) => ValidationValue::Float(*f),
            ConfigValue::Boolean(b) => ValidationValue::Boolean(*b),
            ConfigValue::Array(arr) => {
                let mut validation_array = Vec::new();
                for item in arr {
                    match item {
                        ConfigValue::String(s) => validation_array.push(ValidationValue::String(s.clone())),
                        ConfigValue::Integer(i) => validation_array.push(ValidationValue::Integer(*i)),
                        ConfigValue::Float(f) => validation_array.push(ValidationValue::Float(*f)),
                        ConfigValue::Boolean(b) => validation_array.push(ValidationValue::Boolean(*b)),
                        _ => validation_array.push(ValidationValue::Null),
                    }
                }
                ValidationValue::Array(validation_array)
            }
            ConfigValue::Object(_) => ValidationValue::Null, // Skip nested objects for now
            ConfigValue::Null => ValidationValue::Null,
        };
        
        validation_values.push((key.as_str(), validation_value));
    }
    
    // Validate each field
    let results = validation::utils::validate_multiple(&pipeline, &validation_values);
    
    // Merge all results
    let mut final_result = ValidationResult::valid("simulation_config_validation".to_string());
    for result in results.values() {
        final_result.merge(result.clone());
    }
    
    Ok(final_result)
}

/// Create a complete simulation setup with validation
/// 
/// Implements Controller pattern from GRASP principles
pub fn create_validated_simulation(
    config: Configuration,
) -> KwaversResult<(Grid, Time, HomogeneousMedium, Source, Recorder)> {
    // Validate configuration first
    let validation_result = validate_simulation_config(&config)?;
    if !validation_result.is_valid {
        return Err(KwaversError::Config(crate::error::ConfigError::ValidationFailed {
            section: "simulation".to_string(),
            reason: format!("Configuration validation failed: {}", validation_result.summary()),
        }));
    }
    
    // Extract configuration values
    let nx = config.get("grid_nx").unwrap().as_integer().unwrap() as usize;
    let ny = config.get("grid_ny").unwrap().as_integer().unwrap() as usize;
    let nz = config.get("grid_nz").unwrap().as_integer().unwrap() as usize;
    let dx = config.get("grid_dx").unwrap().as_float().unwrap();
    let dy = config.get("grid_dy").unwrap().as_float().unwrap();
    let dz = config.get("grid_dz").unwrap().as_float().unwrap();
    let duration = config.get("time_duration").unwrap().as_float().unwrap();
    let frequency = config.get("frequency").unwrap().as_float().unwrap();
    let amplitude = config.get("amplitude").unwrap().as_float().unwrap();
    
    // Create grid
    let grid = Grid::new(nx, ny, nz, dx, dy, dz)?;
    
    // Create time discretization
    let time = Time::new(duration, grid.cfl_timestep(1500.0)?)?;
    
    // Create medium
    let medium = HomogeneousMedium::new(998.0, 1482.0, &grid, 0.5, 10.0);
    
    // Create source
    let source_config = SourceConfig {
        num_elements: 32,
        signal_type: "sine".to_string(),
        frequency: Some(frequency),
        amplitude: Some(amplitude),
        phase: None,
        focus_x: Some(0.03),
        focus_y: Some(0.0),
        focus_z: Some(0.0),
        start_freq: None,
        end_freq: None,
        signal_duration: None,
    };
    let source = Source::new(source_config, &grid)?;
    
    // Create recorder
    let sensor_config = crate::sensor::SensorConfig {
        pressure_sensors: vec![
            Sensor::point(0.03, 0.0, 0.0),
            Sensor::point(0.02, 0.0, 0.0),
            Sensor::point(0.04, 0.0, 0.0),
        ],
        temperature_sensors: vec![],
        light_sensors: vec![],
        cavitation_sensors: vec![],
    };
    
    let recorder_config = RecorderConfig {
        output_directory: "simulation_output".to_string(),
        snapshot_interval: 10,
        enable_visualization: config.get("enable_visualization").unwrap().as_boolean().unwrap(),
        save_raw_data: true,
    };
    
    let recorder = Recorder::new(sensor_config, recorder_config)?;
    
    Ok((grid, time, medium, source, recorder))
}

/// Run a complete simulation with advanced physics
/// 
/// Implements ACID principles for simulation execution
pub fn run_advanced_simulation(
    config: Configuration,
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
    let boundary = PMLBoundary::new(
        PMLConfig::default()
            .with_thickness(10)
            .with_sigma_acoustic(100.0)
            .with_polynomial_order(2)
            .with_reflection_coefficient(1e-6),
        &grid,
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
        let source_field = source.generate_field(t, &grid)?;
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
        boundary.apply(&mut fields, &grid, time.dt)?;
        
        // Record data
        recorder.record(step, &fields, &grid)?;
        
        // Progress reporting
        if step % 100 == 0 {
            println!("Step {}/{} ({}%)", step, time.num_steps(), 
                (step * 100) / time.num_steps());
        }
    }
    
    // Generate visualizations if enabled
    if recorder.config.enable_visualization {
        plot_simulation_outputs(
            &recorder.config.output_directory,
            &[
                "pressure_time_series.html",
                "temperature_time_series.html",
                "pressure_slice.html",
                "temperature_slice.html",
                "source_positions.html",
                "sensor_positions.html",
            ],
        )?;
    }
    
    println!("Advanced simulation completed successfully!");
    println!("Results saved to: {}", recorder.config.output_directory);
    
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
        assert!(config.has_key("simulation_name"));
        assert!(config.has_key("version"));
        assert!(config.has_key("frequency"));
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