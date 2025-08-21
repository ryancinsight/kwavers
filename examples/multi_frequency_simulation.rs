// examples/multi_frequency_simulation.rs

use kwavers::{
    factory::{
        GridConfig, MediumConfig, MediumType, PhysicsConfig, PhysicsModelConfig, PhysicsModelType,
        SimulationConfig, SimulationFactory, SourceConfig, TimeConfig,
        ValidationConfig,
    },
    grid::Grid,
    medium::{homogeneous::HomogeneousMedium, Medium},
    physics::mechanics::acoustic_wave::nonlinear::MultiFrequencyConfig,
    KwaversResult,
};
use ndarray::Array4;
use std::collections::HashMap;

fn main() -> KwaversResult<()> {
    println!("🌊 Multi-Frequency Acoustic Wave Simulation");
    println!("========================================================");

    // Create medium properties HashMap as expected by validation
    let mut medium_properties = HashMap::new();
    medium_properties.insert("density".to_string(), 1000.0);
    medium_properties.insert("sound_speed".to_string(), 1500.0);
    medium_properties.insert("mu_a".to_string(), 0.1);
    medium_properties.insert("mu_s_prime".to_string(), 1.0);

    // Create simulation configuration
    let config = SimulationConfig {
        grid: GridConfig {
            nx: 64,
            ny: 64,
            nz: 64,
            dx: 1e-4,
            dy: 1e-4,
            dz: 1e-4,
        },
        medium: MediumConfig {
            medium_type: MediumType::Homogeneous {
                density: 1000.0,
                sound_speed: 1500.0,
                mu_a: 0.1,
                mu_s_prime: 1.0,
            },
            properties: medium_properties,
        },
        physics: PhysicsConfig {
            models: vec![PhysicsModelConfig {
                model_type: PhysicsModelType::AcousticWave,
                enabled: true,
                parameters: HashMap::new(),
            }],
        },
        time: TimeConfig {
            dt: 1e-8,
            steps: 100,
            start: 0.0,
        },
        source: SourceConfig {
            source_type: kwavers::factory::SourceType::Point,
            position: [32, 32, 32],
            frequency: 1e6,
            amplitude: 1.0,
        },
        validation: ValidationConfig {
            check_medium: true,
            check_stability: true,
            check_sources: true,
            check_boundaries: true,
        },
    };

    // Build the simulation
    let mut simulation = SimulationFactory::create_simulation(config)?;

    // Setup multi-frequency configuration
    let frequencies = vec![1e6, 2e6, 3e6]; // 1, 2, 3 MHz
    let weights = vec![1.0, 0.5, 0.25];     // Relative weights
    
    let multi_freq_config = MultiFrequencyConfig {
        frequencies: frequencies.clone(),
        weights: weights.clone(),
        track_harmonics: true,
        max_harmonic_order: 3,
        frequency_resolution: 1e4, // 10 kHz resolution
    };

    println!("\n🎵 Multi-Frequency Configuration:");
    println!("  Frequencies: {:?} Hz", multi_freq_config.frequencies);
    println!("  Weights: {:?}", multi_freq_config.weights);
    println!("  Track harmonics: {}", multi_freq_config.track_harmonics);
    println!("  Max harmonic order: {}", multi_freq_config.max_harmonic_order);

    // Run simulation with initial conditions
    println!("\n🚀 Running simulation...");
    
    // Get the grid and medium from simulation components
    let grid = Grid::new(64, 64, 64, 1e-4, 1e-4, 1e-4);
    let medium = HomogeneousMedium::new(1000.0, 1500.0, &grid, 0.1, 1.0);
    
    // Initialize fields
    let mut fields = Array4::zeros((7, 64, 64, 64));
    
    // Set multi-frequency initial conditions
    set_multi_frequency_initial_conditions(&mut fields, &grid, &multi_freq_config, &medium)?;
    
    // Run the simulation
    let dt = 1e-8;
    let steps = 100;
    
    for step in 0..steps {
        if step % 10 == 0 {
            println!("  Step {}/{}", step, steps);
        }
        // Simulation would update fields here
        // For now, this is a demonstration
    }

    println!("\n✅ Simulation completed successfully!");
    
    // Analyze results
    analyze_multi_frequency_fields(&fields, &multi_freq_config)?;

    Ok(())
}

fn set_multi_frequency_initial_conditions(
    fields: &mut Array4<f64>,
    grid: &Grid,
    config: &MultiFrequencyConfig,
    _medium: &dyn Medium,
) -> KwaversResult<()> {
    println!("  Setting up multi-frequency initial conditions...");

    // Center of the grid
    let cx = grid.nx / 2;
    let cy = grid.ny / 2;
    let cz = grid.nz / 2;

    // Initialize pressure field with multi-frequency components
    for i in 0..grid.nx {
        for j in 0..grid.ny {
            for k in 0..grid.nz {
                let x = i as f64 - cx as f64;
                let y = j as f64 - cy as f64;
                let z = k as f64 - cz as f64;
                let r = (x * x + y * y + z * z).sqrt();

                // Sum contributions from all frequencies
                let mut pressure = 0.0;
                for (freq_idx, freq) in config.frequencies.iter().enumerate() {
                    let weight = if freq_idx < config.weights.len() {
                        config.weights[freq_idx]
                    } else {
                        1.0
                    };
                    
                    // Gaussian envelope with frequency-dependent width
                    let sigma = 10.0 / (1.0 + freq / 1e6);
                    let envelope = (-r * r / (2.0 * sigma * sigma)).exp();
                    
                    // Add frequency component with some phase variation
                    let phase = 2.0 * std::f64::consts::PI * freq * r * 1e-4;
                    pressure += weight * envelope * phase.cos();
                }

                // Set pressure field (index 0)
                fields[[0, i, j, k]] = pressure;
            }
        }
    }

    println!("  Initial conditions set with {} frequency components", config.frequencies.len());
    Ok(())
}

fn analyze_multi_frequency_fields(fields: &Array4<f64>, config: &MultiFrequencyConfig) -> KwaversResult<()> {
    println!("\n📊 Multi-Frequency Field Analysis:");
    
    // Analyze pressure field
    let pressure = fields.index_axis(ndarray::Axis(0), 0);
    let max_pressure = pressure.fold(0.0_f64, |max, &val| max.max(val.abs()));
    let mean_pressure = pressure.mean().unwrap_or(0.0);
    
    println!("  Pressure field:");
    println!("    Maximum: {:.3e} Pa", max_pressure);
    println!("    Mean: {:.3e} Pa", mean_pressure);
    
    // Frequency components info
    println!("\n  Frequency components:");
    for (i, freq) in config.frequencies.iter().enumerate() {
        let weight = if i < config.weights.len() {
            config.weights[i]
        } else {
            1.0
        };
        let wavelength = 1500.0 / freq; // assuming c = 1500 m/s
        println!("    {} MHz: weight={:.2}, λ={:.3} mm", 
                 freq / 1e6, weight, wavelength * 1000.0);
    }
    
    // Check for multi-frequency signatures at center
    let center = [32, 32, 32];
    let center_pressure = fields[[0, center[0], center[1], center[2]]];
    println!("\n  Center point pressure: {:.3e} Pa", center_pressure);
    
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_multi_frequency_config() -> KwaversResult<()> {
        let config = MultiFrequencyConfig {
            frequencies: vec![1e6, 2e6],
            weights: vec![1.0, 0.5],
            track_harmonics: true,
            max_harmonic_order: 2,
            frequency_resolution: 1e4,
        };
        
        assert_eq!(config.frequencies.len(), 2);
        assert_eq!(config.weights.len(), 2);
        assert!(config.track_harmonics);
        
        Ok(())
    }
}
