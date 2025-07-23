// examples/simple_wave_simulation.rs
//! Simple Wave Simulation Example - The RIGHT Way
//! 
//! This example demonstrates how users should actually use the kwavers library:
//! 1. Create a configuration using the factory pattern
//! 2. Build a simulation setup
//! 3. Run the simulation using the built-in run() method
//! 4. Analyze results
//!
//! This is much simpler than implementing your own simulation loop!

use kwavers::{
    KwaversResult, SimulationFactory, SimulationResults,
    FactorySimulationConfig, GridConfig, MediumConfig, MediumType, 
    PhysicsConfig, TimeConfig, ValidationConfig,
    init_logging,
};
use kwavers::factory::{PhysicsModelConfig, PhysicsModelType};
use std::collections::HashMap;
use ndarray::Array4;

fn main() -> KwaversResult<()> {
    // Initialize logging
    let _ = init_logging();
    
    println!("=== Simple Wave Simulation - The RIGHT Way ===");
    println!("This example shows how to use kwavers properly with the factory pattern.\n");
    
    // Step 1: Create configuration using the factory pattern
    let config = create_simulation_config();
    
    // Step 2: Build simulation setup using the factory
    let builder = SimulationFactory::create_simulation(config)?;
    let mut simulation = builder.build()?;
    
    // Step 3: Show performance recommendations (optional)
    let recommendations = simulation.get_performance_recommendations();
    if !recommendations.is_empty() {
        println!("Performance recommendations:");
        for (category, recommendation) in recommendations {
            println!("  - {}: {}", category, recommendation);
        }
        println!();
    }
    
    // Step 4: Run simulation with custom initial conditions
    let results = simulation.run_with_initial_conditions(|fields, grid| {
        set_gaussian_pressure_ball(fields, grid, (12, 12, 12), 2.0, 1e6)
    })?;
    
    // Step 5: Analyze results
    analyze_results(&results);
    
    println!("\n✅ Simulation completed successfully!");
    println!("This is how kwavers should be used - simple configuration, then run!");
    
    Ok(())
}

/// Create a simulation configuration - this is what users should focus on
fn create_simulation_config() -> FactorySimulationConfig {
    FactorySimulationConfig {
        grid: GridConfig {
            nx: 24,
            ny: 24,
            nz: 24,
            dx: 2e-4, // 200 μm
            dy: 2e-4,
            dz: 2e-4,
        },
        medium: MediumConfig {
            medium_type: MediumType::Homogeneous { 
                density: 1000.0,      // Water-like
                sound_speed: 1500.0,  // m/s
                mu_a: 0.1, 
                mu_s_prime: 1.0 
            },
            properties: [
                ("density".to_string(), 1000.0),
                ("sound_speed".to_string(), 1500.0),
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
            dt: 4e-8,      // 40 ns for stability
            num_steps: 200, // Shorter simulation for demo
            cfl_factor: 0.3,
        },
        validation: ValidationConfig {
            enable_validation: true,
            strict_mode: false,
            validation_rules: vec![
                "grid_validation".to_string(),
                "medium_validation".to_string(),
                "physics_validation".to_string(),
                "time_validation".to_string(),
            ],
        },
    }
}

/// Set initial conditions - Gaussian pressure ball
/// This is much simpler than implementing the whole simulation loop!
fn set_gaussian_pressure_ball(
    fields: &mut Array4<f64>, 
    grid: &kwavers::Grid, 
    center: (usize, usize, usize),
    radius: f64,
    amplitude: f64
) -> KwaversResult<()> {
    let (cx, cy, cz) = center;
    
    println!("Setting initial conditions:");
    println!("  Gaussian pressure ball at ({}, {}, {})", cx, cy, cz);
    println!("  Radius: {:.1} grid points", radius);
    println!("  Amplitude: {:.2e} Pa", amplitude);
    
    for i in 0..grid.nx {
        for j in 0..grid.ny {
            for k in 0..grid.nz {
                let dx = i as f64 - cx as f64;
                let dy = j as f64 - cy as f64;
                let dz = k as f64 - cz as f64;
                let r = (dx*dx + dy*dy + dz*dz).sqrt();
                
                if r <= radius * 2.0 {
                    let pressure = amplitude * (-0.5 * (r / radius).powi(2)).exp();
                    fields[[0, i, j, k]] = pressure; // Pressure field is at index 0
                }
            }
        }
    }
    
    println!("  Initial conditions set successfully");
    Ok(())
}

/// Analyze simulation results
fn analyze_results(results: &SimulationResults) {
    println!("\n📊 Simulation Results Analysis:");
    println!("  Total simulation time: {:.2} seconds", results.total_time());
    println!("  Maximum pressure reached: {:.2e} Pa", results.max_pressure());
    
    let timestep_data = results.timestep_data();
    println!("  Recorded {} timesteps", timestep_data.len());
    
    if timestep_data.len() >= 2 {
        let initial_pressure = timestep_data[0].max_pressure;
        let final_pressure = timestep_data.last().unwrap().max_pressure;
        let pressure_ratio = final_pressure / initial_pressure;
        
        println!("  Pressure evolution: {:.2e} → {:.2e} Pa (ratio: {:.2})", 
                 initial_pressure, final_pressure, pressure_ratio);
        
        if pressure_ratio > 10.0 {
            println!("  ⚠️  Pressure growth detected - may indicate numerical instability");
        } else if pressure_ratio < 0.1 {
            println!("  ℹ️  Pressure decay observed - normal for dissipative systems");
        } else {
            println!("  ✅ Pressure levels stable");
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_simple_simulation_config() {
        let config = create_simulation_config();
        assert_eq!(config.grid.nx, 24);
        assert_eq!(config.time.num_steps, 200);
    }
    
    #[test]
    fn test_simulation_creation() {
        let config = create_simulation_config();
        let builder = SimulationFactory::create_simulation(config).unwrap();
        let simulation = builder.build().unwrap();
        
        let summary = simulation.get_summary();
        assert_eq!(summary.get("grid_dimensions").unwrap(), "24x24x24");
    }
    
    #[test]
    fn test_initial_conditions() {
        use kwavers::Grid;
        use ndarray::Array4;
        
        let grid = Grid::new(8, 8, 8, 1e-4, 1e-4, 1e-4);
        let mut fields = Array4::<f64>::zeros((8, 8, 8, 8));
        
        let result = set_gaussian_pressure_ball(&mut fields, &grid, (4, 4, 4), 1.0, 1e6);
        assert!(result.is_ok());
        
        // Check that pressure was set at center
        assert!(fields[[0, 4, 4, 4]] > 0.0);
    }
}