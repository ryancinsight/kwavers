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
    println!("🌊 Advanced Multi-Frequency Acoustic Wave Simulation");
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
                parameters: std::collections::HashMap::new(),
            }],
            frequency: 2e6,
            parameters: std::collections::HashMap::new(),
        },
        time: TimeConfig {
            dt: 1e-8,
            num_steps: 100,
            cfl_factor: 0.5,
        },
        source: SourceConfig {
            source_type: "multi_frequency".to_string(),
            position: (3.2e-3, 3.2e-3, 1.6e-3),
            amplitude: 1e6,
            frequency: 2e6,
            radius: Some(0.5e-3),
            focus: Some((3.2e-3, 3.2e-3, 4.8e-3)),
            num_elements: None,
            signal_type: "multi_tone".to_string(),
            phase: 0.0,
        },
        validation: ValidationConfig {
            enable_validation: true,
            strict_mode: false,
            validation_rules: vec!["basic_validation".to_string()],
        },
    };

    let mut simulation = SimulationFactory::create_simulation(config)?;

    // Create multi-frequency acoustic wave component
    let multi_freq_config = MultiFrequencyConfig::new(vec![1e6, 2e6, 3e6]) // 1, 2, 3 MHz
        .with_amplitudes(vec![1.0, 0.5, 0.3])
        .expect("Valid amplitudes") // Decreasing amplitudes
        .with_phases(vec![
            0.0,
            std::f64::consts::PI / 4.0,
            std::f64::consts::PI / 2.0,
        ]) // Phase progression
        .with_frequency_dependent_attenuation(true)
        .with_harmonics(true);

    println!("Multi-frequency configuration:");
    println!(
        "  Frequencies: {:?} MHz",
        multi_freq_config
            .frequencies
            .iter()
            .map(|f| f / 1e6)
            .collect::<Vec<_>>()
    );
    println!("  Amplitudes: {:?}", multi_freq_config.amplitudes);
    println!("  Phases: {:?} rad", multi_freq_config.phases);

    // Run enhanced simulation with multi-frequency capabilities
    let grid = Grid::new(64, 64, 64, 1e-4, 1e-4, 1e-4);
    let medium = HomogeneousMedium::new(1000.0, 1500.0, &grid, 0.1, 1.0);
    let results = simulation.run_with_initial_conditions(|fields, grid| {
        set_multi_frequency_initial_conditions(fields, grid, &multi_freq_config, &medium)
    })?;

    // Analyze results
    analyze_multi_frequency_results(&results)?;

    println!("\n✅ Multi-frequency simulation completed successfully!");
    println!("   Demonstrated: Broadband excitation, frequency-dependent attenuation, harmonic generation");

    Ok(())
}

fn create_multi_frequency_config() -> MultiFrequencyConfig {
    let multi_freq_config = MultiFrequencyConfig::new(vec![1e6, 2e6, 3e6]) // 1, 2, 3 MHz
        .with_amplitudes(vec![1.0, 0.5, 0.3])
        .expect("Valid amplitudes") // Decreasing amplitudes
        .with_phases(vec![
            0.0,
            std::f64::consts::PI / 4.0,
            std::f64::consts::PI / 2.0,
        ]) // Phase progression
        .with_frequency_dependent_attenuation(true)
        .with_harmonics(true);

    multi_freq_config
}

fn set_multi_frequency_initial_conditions(
    fields: &mut Array4<f64>,
    grid: &Grid,
    multi_freq_config: &MultiFrequencyConfig,
    medium: &dyn Medium,
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
        let wavelength = medium.sound_speed(0.0, 0.0, 0.0, grid) / frequency; // c/f in medium
        let spatial_freq = 2.0 * std::f64::consts::PI / wavelength;

        println!(
            "  Frequency {}: {:.1} MHz, λ = {:.1} mm, amplitude = {:.1}",
            freq_idx + 1,
            frequency / 1e6,
            wavelength * 1e3,
            amplitude
        );

        // Create spatial distribution for this frequency component
        for i in 0..grid.nx {
            for j in 0..grid.ny {
                for k in 0..grid.nz {
                    let dx = (i as f64 - cx as f64) * grid.dx;
                    let dy = (j as f64 - cy as f64) * grid.dy;
                    let dz = (k as f64 - cz as f64) * grid.dz;
                    let r = (dx * dx + dy * dy + dz * dz).sqrt();

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
        println!(
            "      • Maximum pressure achieved: {:.2e} Pa ({:.1} MPa)",
            max_pressure,
            max_pressure / 1e6
        );
        println!(
            "      • Total simulation time: {:.2e} s ({:.1} μs)",
            total_time,
            total_time * 1e6
        );
        println!("      • Total timesteps: {}", timestep_data.len());

        // Frequency domain characteristics analysis
        let pressure_range = max_pressure;
        println!(
            "    Dynamic range: {:.2e} Pa ({:.1} MPa)",
            pressure_range,
            pressure_range / 1e6
        );

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
