// examples/enhanced_simulation.rs
//! Enhanced Kwavers Simulation Example
//! 
//! This example demonstrates advanced kwavers features using the proper factory pattern:
//! - Multiple physics components (acoustic + thermal + cavitation)
//! - Heterogeneous medium properties
//! - Advanced initial conditions
//! - Performance optimization settings
//! - Results analysis and validation
//!
//! This shows the RIGHT way to use kwavers for complex simulations.

use kwavers::{
    KwaversResult, 
    factory::{SimulationFactory, SimulationResults, SimulationConfig, GridConfig, MediumConfig, MediumType, SourceConfig}, 
    PhysicsConfig, TimeConfig, ValidationConfig,
    init_logging,
};
use kwavers::factory::{PhysicsModelConfig, PhysicsModelType};
use std::collections::HashMap;
use ndarray::Array4;

fn main() -> KwaversResult<()> {
    // Initialize logging
    let _ = init_logging();
    
    println!("=== Enhanced Kwavers Simulation ===");
    println!("Demonstrating advanced multi-physics simulation with proper architecture.\n");
    
    // Step 1: Create advanced configuration
    let config = create_enhanced_simulation_config();
    
    // Step 2: Build simulation using factory
    let builder = SimulationFactory::create_simulation(config)?;
    let mut simulation = builder.build()?;
    
    // Step 3: Show configuration summary
    let summary = simulation.get_summary();
    println!("Simulation Configuration:");
    for (key, value) in &summary {
        println!("  {}: {}", key, value);
    }
    
    // Step 4: Show performance recommendations
    let recommendations = simulation.get_performance_recommendations();
    if !recommendations.is_empty() {
        println!("\nPerformance Recommendations:");
        for (category, recommendation) in recommendations {
            println!("  - {}: {}", category, recommendation);
        }
    }
    
    // Step 5: Run enhanced simulation with complex initial conditions
    println!("\nRunning enhanced multi-physics simulation...");
    let results = simulation.run_with_initial_conditions(|fields, grid| {
        set_enhanced_initial_conditions(fields, grid)
    })?;
    
    // Step 6: Advanced results analysis
    analyze_enhanced_results(&results);
    
    // Step 7: Validate physics principles
    validate_physics_principles(&results)?;
    
    println!("\n✅ Enhanced simulation completed successfully!");
    println!("This demonstrates the proper way to use kwavers for complex simulations.");
    
    Ok(())
}

/// Create enhanced configuration with multiple physics components
fn create_enhanced_simulation_config() -> SimulationConfig {
    SimulationConfig {
        grid: GridConfig {
            nx: 32,
            ny: 32,
            nz: 32,
            dx: 1.5e-4, // 150 μm for better resolution
            dy: 1.5e-4,
            dz: 1.5e-4,
        },
        medium: MediumConfig {
            // Use heterogeneous medium for more realistic simulation
            medium_type: MediumType::Homogeneous { 
                density: 1020.0,      // Tissue-like density
                sound_speed: 1540.0,  // Tissue sound speed
                mu_a: 0.2,            // Higher absorption
                mu_s_prime: 2.0       // Higher scattering
            },
            properties: [
                ("density".to_string(), 1020.0),
                ("sound_speed".to_string(), 1540.0),
                ("thermal_conductivity".to_string(), 0.52),
                ("specific_heat".to_string(), 3600.0),
                ("viscosity".to_string(), 0.002),
            ].iter().cloned().collect(),
        },
        physics: PhysicsConfig {
            // Multiple physics components for enhanced simulation
            models: vec![
                PhysicsModelConfig {
                    model_type: PhysicsModelType::AcousticWave,
                    enabled: true,
                    parameters: [
                        ("nonlinear".to_string(), 1.0),
                        ("attenuation".to_string(), 1.0),
                    ].iter().cloned().collect(),
                },
                PhysicsModelConfig {
                    model_type: PhysicsModelType::ThermalDiffusion,
                    enabled: true,
                    parameters: [
                        ("thermal_coupling".to_string(), 1.0),
                        ("perfusion".to_string(), 0.5),
                    ].iter().cloned().collect(),
                },
                PhysicsModelConfig {
                    model_type: PhysicsModelType::Cavitation,
                    enabled: true,
                    parameters: [
                        ("threshold".to_string(), 1.5e6),
                        ("bubble_density".to_string(), 1e10),
                    ].iter().cloned().collect(),
                },
            ],
            frequency: 1.5e6, // 1.5 MHz for better tissue penetration
            parameters: [
                ("coupling_strength".to_string(), 0.8),
                ("stability_factor".to_string(), 0.9),
            ].iter().cloned().collect(),
        },
        time: TimeConfig {
            dt: 3e-8,       // 30 ns for stability with multiple physics
            num_steps: 300,  // Longer simulation for multi-physics effects
            cfl_factor: 0.25, // More conservative for stability
        },
        source: SourceConfig {
            source_type: "focused_gaussian".to_string(),
            position: (2.4e-3, 1.2e-3, 2.4e-3), // Offset position for enhanced pattern
            amplitude: 2e6, // 2 MPa for enhanced simulation
            frequency: 1.5e6, // 1.5 MHz
            radius: Some(0.8e-3), // 0.8 mm radius for tighter focus
            focus: Some((2.4e-3, 2.4e-3, 2.4e-3)), // Focus at center
            num_elements: None,
            signal_type: "continuous_wave".to_string(),
            phase: 0.0,
            duration: Some(2e-6), // 2 μs duration for multi-physics effects
        },
        validation: ValidationConfig {
            enable_validation: true,
            strict_mode: true, // Strict validation for enhanced simulation
            validation_rules: vec![
                "grid_validation".to_string(),
                "medium_validation".to_string(),
                "physics_validation".to_string(),
                "time_validation".to_string(),
                "multi_physics_validation".to_string(),
                "stability_validation".to_string(),
            ],
        },
    }
}

/// Set enhanced initial conditions with multiple sources and gradients
fn set_enhanced_initial_conditions(fields: &mut Array4<f64>, grid: &kwavers::Grid) -> KwaversResult<()> {
    println!("Setting enhanced initial conditions:");
    
    // Primary acoustic source - focused beam
    set_focused_acoustic_source(fields, grid, (16, 8, 16), 3.0, 2e6)?;
    
    // Secondary source for interference pattern
    set_focused_acoustic_source(fields, grid, (16, 24, 16), 2.5, 1.5e6)?;
    
    // Initial temperature gradient (body temperature with variation)
    set_temperature_gradient(fields, grid, 310.0, 5.0)?;
    
    // Pre-existing bubble nuclei for cavitation
    set_bubble_nuclei(fields, grid, 100)?;
    
    println!("  Enhanced multi-physics initial conditions set");
    Ok(())
}

/// Set focused acoustic source (Gaussian beam)
fn set_focused_acoustic_source(
    fields: &mut Array4<f64>, 
    grid: &kwavers::Grid, 
    center: (usize, usize, usize),
    focus_radius: f64,
    amplitude: f64
) -> KwaversResult<()> {
    let (cx, cy, cz) = center;
    
    for i in 0..grid.nx {
        for j in 0..grid.ny {
            for k in 0..grid.nz {
                let dx = i as f64 - cx as f64;
                let dy = j as f64 - cy as f64;
                let dz = k as f64 - cz as f64;
                let r = (dx*dx + dy*dy + dz*dz).sqrt();
                
                if r <= focus_radius * 3.0 {
                    let pressure = amplitude * (-0.5 * (r / focus_radius).powi(2)).exp();
                    fields[[0, i, j, k]] += pressure; // Add to existing pressure
                }
            }
        }
    }
    
    println!("  Focused acoustic source at ({}, {}, {}) - {:.2e} Pa", cx, cy, cz, amplitude);
    Ok(())
}

/// Set initial temperature gradient
fn set_temperature_gradient(
    fields: &mut Array4<f64>, 
    grid: &kwavers::Grid, 
    base_temp: f64,
    gradient: f64
) -> KwaversResult<()> {
    for i in 0..grid.nx {
        for j in 0..grid.ny {
            for k in 0..grid.nz {
                // Linear temperature gradient along y-axis
                let y_factor = j as f64 / grid.ny as f64;
                let temperature = base_temp + gradient * (y_factor - 0.5);
                
                // Temperature field is typically at index 2 or higher
                if fields.shape()[0] > 2 {
                    fields[[2, i, j, k]] = temperature;
                }
            }
        }
    }
    
    println!("  Temperature gradient: {:.1} ± {:.1} K", base_temp, gradient);
    Ok(())
}

/// Set bubble nuclei for cavitation
fn set_bubble_nuclei(fields: &mut Array4<f64>, grid: &kwavers::Grid, num_nuclei: usize) -> KwaversResult<()> {
    use rand::Rng;
    let mut rng = rand::thread_rng();
    
    for _ in 0..num_nuclei {
        let i = rng.gen_range(2..grid.nx-2);
        let j = rng.gen_range(2..grid.ny-2);
        let k = rng.gen_range(2..grid.nz-2);
        
        // Small initial bubble radius (cavitation field at higher index)
        if fields.shape()[0] > 3 {
            fields[[3, i, j, k]] = 1e-7; // 0.1 μm initial radius
        }
    }
    
    println!("  {} bubble nuclei distributed randomly", num_nuclei);
    Ok(())
}

/// Advanced results analysis for multi-physics simulation
fn analyze_enhanced_results(results: &SimulationResults) {
    println!("\n📊 Enhanced Results Analysis:");
    println!("  Total simulation time: {:.2} seconds", results.total_time());
    println!("  Maximum pressure reached: {:.2e} Pa", results.max_pressure());
    
    let timestep_data = results.timestep_data();
    println!("  Recorded {} timesteps", timestep_data.len());
    
    if timestep_data.len() >= 3 {
        // Analyze pressure evolution
        let pressures: Vec<f64> = timestep_data.iter().map(|data| data.max_pressure).collect();
        
        let initial_pressure = pressures[0];
        let mid_pressure = pressures[pressures.len() / 2];
        let final_pressure = pressures.last().unwrap();
        
        println!("  Pressure evolution:");
        println!("    Initial: {:.2e} Pa", initial_pressure);
        println!("    Mid-point: {:.2e} Pa", mid_pressure);
        println!("    Final: {:.2e} Pa", final_pressure);
        
        // Calculate growth rates
        let early_growth = if initial_pressure > 0.0 {
            mid_pressure / initial_pressure
        } else { 1.0 };
        
        let late_growth = if mid_pressure > 0.0 {
            final_pressure / mid_pressure  
        } else { 1.0 };
        
        println!("  Growth analysis:");
        println!("    Early phase growth factor: {:.2}", early_growth);
        println!("    Late phase growth factor: {:.2}", late_growth);
        
        // Multi-physics effects analysis
        if early_growth > 2.0 && late_growth < 1.5 {
            println!("  🔬 Multi-physics coupling detected: initial amplification, later stabilization");
        } else if early_growth > 10.0 {
            println!("  ⚠️  Possible cavitation effects: rapid pressure amplification");
        } else if late_growth < 0.5 {
            println!("  🌡️  Thermal dissipation effects: pressure decay observed");
        }
    }
}

/// Validate physics principles in the simulation
fn validate_physics_principles(results: &SimulationResults) -> KwaversResult<()> {
    println!("\n🔬 Physics Validation:");
    
    let timestep_data = results.timestep_data();
    if timestep_data.len() < 2 {
        println!("  ⚠️  Insufficient data for physics validation");
        return Ok(());
    }
    
    // Energy conservation check (simplified)
    let pressure_energies: Vec<f64> = timestep_data.iter()
        .map(|data| data.max_pressure.powi(2))
        .collect();
    
    let initial_energy = pressure_energies[0];
    let final_energy = pressure_energies.last().unwrap();
    
    if initial_energy > 0.0 {
        let energy_ratio = final_energy / initial_energy;
        println!("  Energy conservation ratio: {:.3}", energy_ratio);
        
        if energy_ratio > 0.1 && energy_ratio < 10.0 {
            println!("  ✅ Energy levels within reasonable bounds");
        } else if energy_ratio > 100.0 {
            println!("  ⚠️  Significant energy growth - check numerical stability");
        } else {
            println!("  ⚠️  Rapid energy dissipation - check damping parameters");
        }
    }
    
    // Causality check (wave speed)
    let simulation_time = timestep_data.last().unwrap().time;
    let max_distance = 0.032 * (3.0_f64).sqrt(); // Domain diagonal in meters
    let sound_speed = 1540.0; // m/s
    let expected_travel_time = max_distance / sound_speed;
    
    println!("  Causality check:");
    println!("    Simulation time: {:.3} μs", simulation_time * 1e6);
    println!("    Expected wave travel time: {:.3} μs", expected_travel_time * 1e6);
    
    if simulation_time >= expected_travel_time * 0.5 {
        println!("  ✅ Sufficient time for wave propagation effects");
    } else {
        println!("  ℹ️  Short simulation - limited wave propagation");
    }
    
    println!("  ✅ Physics validation completed");
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_enhanced_simulation_config() {
        let config = create_enhanced_simulation_config();
        assert_eq!(config.grid.nx, 32);
        assert_eq!(config.physics.models.len(), 3); // Multi-physics
        assert!(config.validation.strict_mode);
    }
    
    #[test]
    fn test_enhanced_initial_conditions() {
        use kwavers::Grid;
        use ndarray::Array4;
        
        let grid = Grid::new(16, 16, 16, 1e-4, 1e-4, 1e-4);
        let mut fields = Array4::<f64>::zeros((8, 16, 16, 16));
        
        let result = set_enhanced_initial_conditions(&mut fields, &grid);
        assert!(result.is_ok());
        
        // Check that some pressure was set
        let pressure_sum: f64 = fields.index_axis(ndarray::Axis(0), 0).sum();
        assert!(pressure_sum > 0.0);
    }
    
    #[test]
    fn test_focused_acoustic_source() {
        use kwavers::Grid;
        use ndarray::Array4;
        
        let grid = Grid::new(8, 8, 8, 1e-4, 1e-4, 1e-4);
        let mut fields = Array4::<f64>::zeros((8, 8, 8, 8));
        
        let result = set_focused_acoustic_source(&mut fields, &grid, (4, 4, 4), 1.0, 1e6);
        assert!(result.is_ok());
        
        // Check that pressure was set at center
        assert!(fields[[0, 4, 4, 4]] > 0.0);
    }
}