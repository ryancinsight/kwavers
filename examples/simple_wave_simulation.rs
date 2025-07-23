// examples/simple_wave_simulation.rs
//! Simple Wave Simulation Example - The RIGHT Way with Source Configuration
//! 
//! This example demonstrates how users should actually use the kwavers library:
//! 1. Create a configuration using the factory pattern WITH source configuration
//! 2. Build a simulation setup
//! 3. Run the simulation using the built-in run_with_source() method
//! 4. Analyze results
//!
//! This is much simpler than implementing your own initial conditions!

use kwavers::{
    KwaversResult, SimulationFactory, SimulationResults,
    FactorySimulationConfig, GridConfig, MediumConfig, MediumType, 
    PhysicsConfig, TimeConfig, ValidationConfig, FactorySourceConfig,
    init_logging,
};
use kwavers::factory::{PhysicsModelConfig, PhysicsModelType};
use std::collections::HashMap;

fn main() -> KwaversResult<()> {
    // Initialize logging
    let _ = init_logging();
    
    println!("=== Simple Wave Simulation - The RIGHT Way with Source Config ===");
    println!("This example shows how to use kwavers properly with source configuration.\n");
    
    // Step 1: Create configuration using the factory pattern WITH source config
    let config = create_simulation_config();
    
    // Step 2: Build simulation setup using the factory
    let builder = SimulationFactory::create_simulation(config)?;
    let mut simulation = builder.build()?;
    
    // Step 3: Show configuration summary
    let summary = simulation.get_summary();
    println!("Simulation Configuration:");
    for (key, value) in &summary {
        println!("  {}: {}", key, value);
    }
    
    // Step 4: Show performance recommendations (optional)
    let recommendations = simulation.get_performance_recommendations();
    if !recommendations.is_empty() {
        println!("\nPerformance recommendations:");
        for (category, recommendation) in recommendations {
            println!("  - {}: {}", category, recommendation);
        }
        println!();
    }
    
    // Step 5: Run simulation with built-in source configuration
    println!("Running simulation with built-in source configuration...");
    let results = simulation.run_with_source()?;
    
    // Step 6: Analyze results
    analyze_results(&results);
    
    println!("\n✅ Simple wave simulation completed successfully!");
    println!("This demonstrates the proper way to use kwavers with source configuration.");
    
    Ok(())
}

/// Create simulation configuration with proper source configuration
fn create_simulation_config() -> FactorySimulationConfig {
    FactorySimulationConfig {
        grid: GridConfig {
            nx: 24,
            ny: 24, 
            nz: 24,
            dx: 2e-4, // 200 μm spacing
            dy: 2e-4,
            dz: 2e-4,
        },
        medium: MediumConfig {
            medium_type: MediumType::Homogeneous {
                density: 1000.0,
                sound_speed: 1500.0,
                mu_a: 0.1,
                mu_s_prime: 1.0,
            },
            properties: [
                ("density".to_string(), 1000.0),
                ("sound_speed".to_string(), 1500.0),
                ("mu_a".to_string(), 0.1),
                ("mu_s_prime".to_string(), 1.0),
            ].iter().cloned().collect(),
        },
        physics: PhysicsConfig {
            models: vec![
                PhysicsModelConfig {
                    model_type: PhysicsModelType::AcousticWave,
                    enabled: true,
                    parameters: HashMap::new(),
                },
            ],
            frequency: 1e6, // 1 MHz
            parameters: HashMap::new(),
        },
        time: TimeConfig {
            dt: 5e-8,  // 50 ns time step
            num_steps: 200,
            cfl_factor: 0.3,
        },
        source: FactorySourceConfig {
            source_type: "gaussian".to_string(),
            position: (2.4e-3, 2.4e-3, 2.4e-3), // Center of domain (12 * 200μm)
            amplitude: 1e6, // 1 MPa
            frequency: 1e6, // 1 MHz
            radius: Some(1e-3), // 1 mm radius
            focus: None,
            num_elements: None,
            signal_type: "gaussian_pulse".to_string(),
            phase: 0.0,
            duration: Some(1e-6), // 1 μs pulse
        },
        validation: ValidationConfig {
            enable_validation: true,
            strict_mode: false,
            validation_rules: vec![
                "grid_validation".to_string(),
                "medium_validation".to_string(),
                "physics_validation".to_string(),
                "time_validation".to_string(),
                "source_validation".to_string(),
            ],
        },
    }
}

/// Analyze simulation results
fn analyze_results(results: &SimulationResults) {
    println!("\n📊 Simulation Results Analysis:");
    println!("  Total simulation time: {:.2} seconds", results.total_time());
    println!("  Maximum pressure reached: {:.2e} Pa", results.max_pressure());
    
    let timestep_data = results.timestep_data();
    println!("  Recorded {} timesteps", timestep_data.len());
    
    if timestep_data.len() >= 3 {
        let initial_pressure = timestep_data[0].max_pressure;
        let mid_pressure = timestep_data[timestep_data.len() / 2].max_pressure;
        let final_pressure = timestep_data.last().unwrap().max_pressure;
        
        println!("  Pressure evolution:");
        println!("    Initial: {:.2e} Pa", initial_pressure);
        println!("    Mid-point: {:.2e} Pa", mid_pressure);
        println!("    Final: {:.2e} Pa", final_pressure);
        
        // Wave propagation analysis
        let simulation_time = timestep_data.last().unwrap().time;
        let domain_size = 0.0048; // 24 * 200μm = 4.8mm
        let sound_speed = 1500.0; // m/s
        let expected_travel_time = domain_size / sound_speed;
        
        println!("  Wave propagation analysis:");
        println!("    Simulation time: {:.3} μs", simulation_time * 1e6);
        println!("    Expected travel time across domain: {:.3} μs", expected_travel_time * 1e6);
        
        if simulation_time >= expected_travel_time * 0.5 {
            println!("    ✅ Sufficient time for wave propagation");
        } else {
            println!("    ℹ️  Limited wave propagation time");
        }
        
        // Source effectiveness analysis
        if initial_pressure > 1e5 {
            println!("    ✅ Source successfully initialized with {:.2e} Pa", initial_pressure);
        } else {
            println!("    ⚠️  Low initial pressure: {:.2e} Pa", initial_pressure);
        }
        
        // Energy conservation check
        if final_pressure > 0.1 * initial_pressure && final_pressure < 10.0 * initial_pressure {
            println!("    ✅ Pressure levels remain within reasonable bounds");
        } else if final_pressure > 100.0 * initial_pressure {
            println!("    ⚠️  Significant pressure amplification detected");
        } else {
            println!("    ⚠️  Rapid pressure dissipation detected");
        }
    }
    
    println!("\n🎯 Key Benefits of Source Configuration:");
    println!("  ✓ No manual initial condition implementation needed");
    println!("  ✓ Consistent source positioning using world coordinates");
    println!("  ✓ Built-in validation of source parameters");
    println!("  ✓ Multiple source types available (gaussian, focused, point)");
    println!("  ✓ Automatic coordinate conversion from world to grid units");
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_simulation_config_creation() {
        let config = create_simulation_config();
        assert_eq!(config.grid.nx, 24);
        assert_eq!(config.source.source_type, "gaussian");
        assert_eq!(config.source.amplitude, 1e6);
        assert!(config.validation.enable_validation);
    }
    
    #[test]
    fn test_source_config_parameters() {
        let config = create_simulation_config();
        
        // Test source positioning
        assert_eq!(config.source.position, (2.4e-3, 2.4e-3, 2.4e-3));
        
        // Test source properties
        assert_eq!(config.source.frequency, 1e6);
        assert_eq!(config.source.radius, Some(1e-3));
        assert_eq!(config.source.signal_type, "gaussian_pulse");
        
        // Test validation rules include source
        assert!(config.validation.validation_rules.contains(&"source_validation".to_string()));
    }
    
    #[test]
    fn test_grid_and_source_consistency() {
        let config = create_simulation_config();
        
        // Domain size calculation
        let domain_x = config.grid.nx as f64 * config.grid.dx;
        let domain_y = config.grid.ny as f64 * config.grid.dy;
        let domain_z = config.grid.nz as f64 * config.grid.dz;
        
        // Source should be within domain
        assert!(config.source.position.0 < domain_x);
        assert!(config.source.position.1 < domain_y);
        assert!(config.source.position.2 < domain_z);
        
        // Source should be reasonably centered
        let center_x = domain_x / 2.0;
        let center_y = domain_y / 2.0;
        let center_z = domain_z / 2.0;
        
        assert!((config.source.position.0 - center_x).abs() < 1e-3);
        assert!((config.source.position.1 - center_y).abs() < 1e-3);
        assert!((config.source.position.2 - center_z).abs() < 1e-3);
    }
}