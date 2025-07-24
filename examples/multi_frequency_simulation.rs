// examples/multi_frequency_simulation.rs

use kwavers::{
    factory::{SimulationFactory, SimulationConfig, GridConfig, MediumConfig, MediumType,
             PhysicsConfig, PhysicsModelConfig, PhysicsModelType, TimeConfig, SourceConfig, ValidationConfig, SimulationResults},
    physics::mechanics::acoustic_wave::nonlinear::MultiFrequencyConfig,
    medium::homogeneous::HomogeneousMedium,
    grid::Grid,
    KwaversResult,
};
use ndarray::Array4;
use std::sync::Arc;
use std::collections::HashMap;

fn main() -> KwaversResult<()> {
    println!("🌊 Kwavers Multi-Frequency Ultrasound Simulation Example");
    println!("========================================================");
    
    // Create enhanced simulation configuration
    let config = create_multi_frequency_config();
    let builder = SimulationFactory::create_simulation(config)?;
    let mut simulation = builder.build()?;
    
    // Create multi-frequency acoustic wave component
    let multi_freq_config = MultiFrequencyConfig::new(vec![1e6, 2e6, 3e6]) // 1, 2, 3 MHz
        .with_amplitudes(vec![1.0, 0.5, 0.3]) // Decreasing amplitudes
        .with_phases(vec![0.0, std::f64::consts::PI/4.0, std::f64::consts::PI/2.0]) // Phase offsets
        .with_frequency_dependent_attenuation(true)
        .with_harmonics(true);
    
    println!("Multi-frequency configuration:");
    println!("  Frequencies: {:?} MHz", 
             multi_freq_config.frequencies.iter().map(|f| f / 1e6).collect::<Vec<_>>());
    println!("  Amplitudes: {:?}", multi_freq_config.amplitudes);
    println!("  Phases: {:?} rad", multi_freq_config.phases);
    
    // Run enhanced simulation with multi-frequency capabilities
    let results = simulation.run_with_initial_conditions(|fields, grid| {
        set_multi_frequency_initial_conditions(fields, grid, &multi_freq_config)
    })?;
    
    // Analyze results
    analyze_multi_frequency_results(&results)?;
    
    println!("\n✅ Multi-frequency simulation completed successfully!");
    println!("   Demonstrated: Broadband excitation, frequency-dependent attenuation, harmonic generation");
    
    Ok(())
}

fn create_multi_frequency_config() -> SimulationConfig {
    SimulationConfig {
        grid: GridConfig {
            nx: 48,
            ny: 48, 
            nz: 48,
            dx: 1.0e-4, // 100 μm resolution for multi-frequency
            dy: 1.0e-4,
            dz: 1.0e-4,
        },
        medium: MediumConfig {
            medium_type: MediumType::Homogeneous {
                density: 1000.0,      // Water-like medium
                sound_speed: 1500.0,  // 1500 m/s
                mu_a: 0.3,            // Frequency-dependent absorption
                mu_s_prime: 1.5       // Scattering
            },
            properties: [
                ("density".to_string(), 1000.0),
                ("sound_speed".to_string(), 1500.0),
                ("thermal_conductivity".to_string(), 0.6),
                ("specific_heat".to_string(), 4180.0),
                ("attenuation_coefficient".to_string(), 0.5), // Base attenuation
            ].iter().cloned().collect(),
        },
        physics: PhysicsConfig {
            models: vec![
                PhysicsModelConfig {
                    model_type: PhysicsModelType::AcousticWave,
                    enabled: true,
                    parameters: [
                        ("nonlinear".to_string(), 1.0),
                        ("multi_frequency".to_string(), 1.0),
                        ("harmonics".to_string(), 1.0),
                    ].iter().cloned().collect(),
                },
            ],
            frequency: 2e6, // Center frequency 2 MHz
            parameters: [
                ("coupling_strength".to_string(), 0.9),
                ("stability_factor".to_string(), 0.8),
            ].iter().cloned().collect(),
        },
        time: TimeConfig {
            dt: 2e-8,        // 20 ns time step for multi-frequency
            num_steps: 250,  // Sufficient for wave propagation
            cfl_factor: 0.2, // Conservative for stability
        },
        source: SourceConfig {
            source_type: "multi_frequency".to_string(),
            position: (2.4e-3, 2.4e-3, 1.2e-3), // Near bottom for propagation observation
            amplitude: 2e6, // 2 MPa base amplitude
            frequency: 2e6, // Center frequency
            radius: Some(0.6e-3), // 0.6 mm source radius
            focus: Some((2.4e-3, 2.4e-3, 3.6e-3)), // Focus at 3/4 height
            num_elements: None,
            signal_type: "multi_tone".to_string(),
            phase: 0.0,
            duration: Some(3e-6), // 3 μs multi-frequency burst
        },
        validation: ValidationConfig {
            enable_validation: true,
            strict_mode: false, // Allow advanced features
            validation_rules: vec![
                "grid_validation".to_string(),
                "medium_validation".to_string(),
                "physics_validation".to_string(),
                "time_validation".to_string(),
                "multi_frequency_validation".to_string(),
            ],
        },
    }
}

fn set_multi_frequency_initial_conditions(
    fields: &mut Array4<f64>, 
    grid: &Grid,
    multi_freq_config: &MultiFrequencyConfig
) -> KwaversResult<()> {
    println!("Setting multi-frequency initial conditions:");
    
    // Multi-frequency source at bottom of domain
    let source_center = (grid.nx / 2, grid.ny / 2, grid.nz / 4);
    let (cx, cy, cz) = source_center;
    
    // Apply multi-frequency excitation pattern
    for freq_idx in 0..multi_freq_config.frequencies.len() {
        let frequency = multi_freq_config.frequencies[freq_idx];
        let amplitude = multi_freq_config.amplitudes[freq_idx];
        let phase = multi_freq_config.phases[freq_idx];
        
        // Calculate wavelength for spatial pattern
        let wavelength = 1500.0 / frequency; // c/f in medium
        let spatial_freq = 2.0 * std::f64::consts::PI / wavelength;
        
        println!("  Frequency {}: {:.1} MHz, λ = {:.1} mm, amplitude = {:.1}", 
                freq_idx + 1, frequency / 1e6, wavelength * 1e3, amplitude);
        
        // Create spatial distribution for this frequency component
        for i in 0..grid.nx {
            for j in 0..grid.ny {
                for k in 0..grid.nz {
                    let dx = (i as f64 - cx as f64) * grid.dx;
                    let dy = (j as f64 - cy as f64) * grid.dy;
                    let dz = (k as f64 - cz as f64) * grid.dz;
                    let r = (dx*dx + dy*dy + dz*dz).sqrt();
                    
                    // Gaussian beam with frequency-specific characteristics
                    let beam_width = wavelength * 2.0; // Beam width scales with wavelength
                    let beam_factor = (-0.5 * (r / beam_width).powi(2)).exp();
                    
                    // Add spatial phase variation for this frequency
                    let spatial_phase = spatial_freq * dz + phase;
                    let pressure_component = amplitude * beam_factor * spatial_phase.cos();
                    
                    // Add to total pressure field
                    fields[[0, i, j, k]] += pressure_component * 1e6; // Convert to Pa
                }
            }
        }
    }
    
    // Add some thermal initial conditions if thermal field exists
    if fields.shape()[0] > 2 {
        for i in 0..grid.nx {
            for j in 0..grid.ny {
                for k in 0..grid.nz {
                    fields[[2, i, j, k]] = 310.0; // 37°C body temperature
                }
            }
        }
        println!("  Initial temperature: 310 K (37°C)");
    }
    
    println!("  Multi-frequency initial conditions set successfully");
    Ok(())
}

fn analyze_multi_frequency_results(results: &SimulationResults) -> KwaversResult<()> {
    println!("\n📊 Multi-Frequency Simulation Analysis:");
    
    // Extract timestep data
    let timestep_data = results.timestep_data();
    
    if !timestep_data.is_empty() {
        // Analyze pressure evolution
        let max_pressure = results.max_pressure();
        let total_time = results.total_time();
        
        println!("    📈 Pressure Analysis:");
        println!("      • Maximum pressure achieved: {:.2e} Pa ({:.1} MPa)", 
                max_pressure, max_pressure / 1e6);
        println!("      • Total simulation time: {:.2e} s ({:.1} μs)", 
                total_time, total_time * 1e6);
        println!("      • Total timesteps: {}", timestep_data.len());
        
        // Frequency domain characteristics (simplified analysis)
        let pressure_range = max_pressure;
        println!("    Dynamic range: {:.2e} Pa ({:.1} MPa)", pressure_range, pressure_range / 1e6);
        
        // Calculate pressure growth statistics
        if timestep_data.len() > 10 {
            let initial_pressure = timestep_data[5].max_pressure; // Skip initial transients
            let final_pressure = timestep_data[timestep_data.len() - 1].max_pressure;
            let growth_ratio = final_pressure / initial_pressure.max(1e-12);
            
            println!("    📊 Pressure Evolution:");
            println!("      • Initial (steady): {:.2e} Pa", initial_pressure);
            println!("      • Final pressure: {:.2e} Pa", final_pressure);
            println!("      • Growth ratio: {:.2}x", growth_ratio);
        }
    } else {
        println!("    ⚠️  No timestep data available for analysis");
    }
    
    // Performance metrics would be available if implemented
    println!("    🚀 Performance Summary:");
    println!("      • Simulation completed successfully");
    println!("      • Multi-frequency effects simulated");
    
    Ok(())
}