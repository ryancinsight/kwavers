//! Comparison of Bremsstrahlung vs Cherenkov Sonoluminescence Hypotheses
//!
//! This example demonstrates the key differences between two proposed mechanisms
//! for sonoluminescence light emission:
//!
//! 1. **Bremsstrahlung Hypothesis**: Free-free emission from decelerating electrons
//!    in hot, ionized gas (conventional thermal plasma emission)
//!
//! 2. **Cherenkov Hypothesis**: Coherent radiation from relativistic particles
//!    moving faster than light in the compressed medium
//!
//! ## Key Differences
//!
//! | Aspect | Bremsstrahlung | Cherenkov |
//! |--------|---------------|-----------|
//! | **Mechanism** | Incoherent thermal emission | Coherent threshold-based emission |
//! | **Spectrum** | Broadband thermal (Planck-like) | Broadband 1/ω (UV/blue bias) |
//! | **Threshold** | Temperature > 5000K | Velocity > c/n (relativistic) |
//! | **Efficiency** | Depends on ionization fraction | Depends on particle velocity |
//! | **Polarization** | Unpolarized | Polarized (directional) |
//! | **Pulse Structure** | Follows temperature profile | Sharp threshold behavior |
//!
//! ## Simulation Results
//!
//! The simulation shows:
//! - Bremsstrahlung dominates at high temperatures (>10,000K)
//! - Cherenkov requires extreme compression and relativistic velocities
//! - Cherenkov produces bluer, more directional emission
//! - Combined emission shows complex interplay of both mechanisms

use kwavers::grid::Grid;
use kwavers::physics::bubble_dynamics::bubble_state::BubbleParameters;
use kwavers::physics::optics::sonoluminescence::{EmissionParameters, IntegratedSonoluminescence};
use ndarray::Array3;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("🔬 Sonoluminescence Hypothesis Comparison");
    println!("========================================\n");

    // Setup simulation parameters
    let grid = Grid::new(32, 8, 8, 0.001, 0.001, 0.001).unwrap();
    let bubble_params = BubbleParameters {
        r0: 4e-6,                       // 4 μm initial radius
        t0: 300.0,                      // 300 K ambient temperature
        gamma: 1.667, // Argon polytropic index (Monatomic gas is essential for SBSL)
        initial_gas_pressure: 101325.0, // 1 atm
        ..Default::default()
    };

    // Test different emission scenarios
    run_bremsstrahlung_dominant_scenario(&grid, &bubble_params)?;
    run_cherenkov_dominant_scenario(&grid, &bubble_params)?;
    run_combined_emission_scenario(&grid, &bubble_params)?;

    println!("\n📊 Summary of Key Differences:");
    println!("------------------------------");
    println!("• Bremsstrahlung: Thermal emission, broadband, temperature-dependent");
    println!("• Cherenkov: Threshold-based, directional, velocity-dependent");
    println!("• Combined: Complex interplay requiring both high T and relativistic velocities");
    println!("• Experimental validation needed to distinguish mechanisms");

    Ok(())
}

/// Scenario 1: Bremsstrahlung-dominant emission (high temperature, moderate compression)
fn run_bremsstrahlung_dominant_scenario(
    grid: &Grid,
    bubble_params: &BubbleParameters,
) -> Result<(), Box<dyn std::error::Error>> {
    println!("🌡️  Scenario 1: Bremsstrahlung-Dominant Emission");
    println!("------------------------------------------------");

    let emission_params = EmissionParameters {
        use_blackbody: false,
        use_bremsstrahlung: true,
        use_cherenkov: false,
        min_temperature: 5000.0,
        ..Default::default()
    };

    let mut simulator =
        IntegratedSonoluminescence::new(grid.dimensions(), bubble_params.clone(), emission_params);

    // Setup moderate acoustic driving (non-relativistic)
    // Negative pressure starts with rarefaction -> growth -> collapse
    let acoustic_pressure = Array3::from_elem(grid.dimensions(), -1.25e5); // -1.25 bar
    simulator.set_acoustic_pressure(acoustic_pressure);

    // Track peak state
    let mut peak_temp = 0.0;
    let mut peak_fields = (
        simulator.temperature_field.clone(),
        simulator.pressure_field.clone(),
        simulator.radius_field.clone(),
        simulator.particle_velocity_field.clone(),
        simulator.charge_density_field.clone(),
        simulator.compression_field.clone(),
    );

    // Simulate bubble dynamics
    println!("Simulating 1 full acoustic cycle (~38 µs) with high precision...");
    for step in 0..800000 {
        simulator.simulate_step(5e-11, step as f64 * 5e-11)?;

        // Check for peak temperature
        let current_max_temp = simulator
            .temperature_field
            .iter()
            .fold(0.0f64, |a, &b| a.max(b));
        if current_max_temp > peak_temp {
            peak_temp = current_max_temp;
            peak_fields = (
                simulator.temperature_field.clone(),
                simulator.pressure_field.clone(),
                simulator.radius_field.clone(),
                simulator.particle_velocity_field.clone(),
                simulator.charge_density_field.clone(),
                simulator.compression_field.clone(),
            );
        }
    }

    // Restore peak state for analysis
    simulator.temperature_field = peak_fields.0;
    simulator.pressure_field = peak_fields.1;
    simulator.radius_field = peak_fields.2;
    simulator.particle_velocity_field = peak_fields.3;
    simulator.charge_density_field = peak_fields.4;
    simulator.compression_field = peak_fields.5;

    // Calculate emission
    let temp_field = &simulator.temperature_field;
    let pressure_field = &simulator.pressure_field;
    let radius_field = &simulator.radius_field;
    let velocity_field = &simulator.particle_velocity_field;
    let charge_density_field = &simulator.charge_density_field;
    let compression_field = &simulator.compression_field;

    simulator.emission.calculate_emission(
        temp_field,
        pressure_field,
        radius_field,
        velocity_field,
        charge_density_field,
        compression_field,
        0.0,
    );

    // Calculate spectrum at peak emission location
    let (i, j, k) = simulator.emission.peak_emission_location();
    let spectrum = simulator.emission.calculate_spectrum_at_point(
        temp_field[[i, j, k]],
        pressure_field[[i, j, k]],
        radius_field[[i, j, k]],
        velocity_field[[i, j, k]],
        charge_density_field[[i, j, k]],
        compression_field[[i, j, k]],
    );

    println!("  Peak Temperature: {:.0} K", temp_field[[i, j, k]]);
    println!("  Peak Pressure: {:.1e} Pa", pressure_field[[i, j, k]]);
    println!("  Compression Ratio: {:.1}", compression_field[[i, j, k]]);
    println!("  Particle Velocity: {:.1e} m/s", velocity_field[[i, j, k]]);
    println!(
        "  Total Emission: {:.2e} W/m³",
        simulator.emission.total_light_output()
    );
    println!(
        "  Peak Wavelength: {:.0} nm",
        spectrum.peak_wavelength() * 1e9
    );
    println!("  Spectral Range: Thermal (broadband)\n");

    Ok(())
}

/// Scenario 2: Cherenkov-dominant emission (extreme compression, relativistic velocities)
fn run_cherenkov_dominant_scenario(
    grid: &Grid,
    bubble_params: &BubbleParameters,
) -> Result<(), Box<dyn std::error::Error>> {
    println!("⚡ Scenario 2: Cherenkov-Dominant Emission");
    println!("-----------------------------------------");

    let emission_params = EmissionParameters {
        use_blackbody: false,
        use_bremsstrahlung: false,
        use_cherenkov: true,
        cherenkov_refractive_index: 1.5, // Higher refractive index for compressed water
        cherenkov_coherence_factor: 500.0, // Enhanced coherence
        min_temperature: 1000.0,         // Lower threshold for Cherenkov
        ..Default::default()
    };

    let mut simulator =
        IntegratedSonoluminescence::new(grid.dimensions(), bubble_params.clone(), emission_params);

    // Setup extreme acoustic driving (potentially relativistic)
    let acoustic_pressure = Array3::from_elem(grid.dimensions(), -1.28e5); // -1.28 bar (reduced from -1.35 to avoid Mach instability)
    simulator.set_acoustic_pressure(acoustic_pressure);

    // Track peak state
    let mut peak_temp = 0.0;
    let mut peak_fields = (
        simulator.temperature_field.clone(),
        simulator.pressure_field.clone(),
        simulator.radius_field.clone(),
        simulator.particle_velocity_field.clone(),
        simulator.charge_density_field.clone(),
        simulator.compression_field.clone(),
    );

    // Simulate with smaller timesteps for stability
    println!("Simulating 1 full acoustic cycle (~38 µs) with high precision...");
    for step in 0..800000 {
        simulator.simulate_step(5e-11, step as f64 * 5e-11)?;

        // Check for peak temperature
        let current_max_temp = simulator
            .temperature_field
            .iter()
            .fold(0.0f64, |a, &b| a.max(b));
        if current_max_temp > peak_temp {
            peak_temp = current_max_temp;
            peak_fields = (
                simulator.temperature_field.clone(),
                simulator.pressure_field.clone(),
                simulator.radius_field.clone(),
                simulator.particle_velocity_field.clone(),
                simulator.charge_density_field.clone(),
                simulator.compression_field.clone(),
            );
        }
    }

    // Restore peak state for analysis
    simulator.temperature_field = peak_fields.0;
    simulator.pressure_field = peak_fields.1;
    simulator.radius_field = peak_fields.2;
    simulator.particle_velocity_field = peak_fields.3;
    simulator.charge_density_field = peak_fields.4;
    simulator.compression_field = peak_fields.5;

    // Calculate emission
    let temp_field = &simulator.temperature_field;
    let pressure_field = &simulator.pressure_field;
    let radius_field = &simulator.radius_field;
    let velocity_field = &simulator.particle_velocity_field;
    let charge_density_field = &simulator.charge_density_field;
    let compression_field = &simulator.compression_field;

    simulator.emission.calculate_emission(
        temp_field,
        pressure_field,
        radius_field,
        velocity_field,
        charge_density_field,
        compression_field,
        0.0,
    );

    // Calculate spectrum at peak emission location
    let (i, j, k) = simulator.emission.peak_emission_location();
    let spectrum = simulator.emission.calculate_spectrum_at_point(
        temp_field[[i, j, k]],
        pressure_field[[i, j, k]],
        radius_field[[i, j, k]],
        velocity_field[[i, j, k]],
        charge_density_field[[i, j, k]],
        compression_field[[i, j, k]],
    );

    // Check Cherenkov threshold
    let cherenkov_model = &simulator.emission.cherenkov;
    let critical_velocity = cherenkov_model.critical_velocity;
    let exceeds_threshold = velocity_field[[i, j, k]] > critical_velocity;

    println!("  Peak Temperature: {:.0} K", temp_field[[i, j, k]]);
    println!("  Peak Pressure: {:.1e} Pa", pressure_field[[i, j, k]]);
    println!("  Compression Ratio: {:.1}", compression_field[[i, j, k]]);
    println!("  Particle Velocity: {:.1e} m/s", velocity_field[[i, j, k]]);
    println!("  Cherenkov Threshold: {:.1e} m/s", critical_velocity);
    println!("  Exceeds Threshold: {}", exceeds_threshold);
    println!(
        "  Total Emission: {:.2e} W/m³",
        simulator.emission.total_light_output()
    );
    if spectrum.peak_wavelength() > 0.0 {
        println!(
            "  Peak Wavelength: {:.0} nm",
            spectrum.peak_wavelength() * 1e9
        );
    }
    println!("  Spectral Range: Cherenkov (UV/blue bias)\n");

    Ok(())
}

/// Scenario 3: Combined emission (both mechanisms active)
fn run_combined_emission_scenario(
    grid: &Grid,
    bubble_params: &BubbleParameters,
) -> Result<(), Box<dyn std::error::Error>> {
    println!("🔬 Scenario 3: Combined Bremsstrahlung + Cherenkov Emission");
    println!("----------------------------------------------------------");

    let emission_params = EmissionParameters {
        use_blackbody: false,
        use_bremsstrahlung: true,
        use_cherenkov: true,
        cherenkov_refractive_index: 1.4,
        cherenkov_coherence_factor: 200.0,
        min_temperature: 3000.0,
        ..Default::default()
    };

    let mut simulator =
        IntegratedSonoluminescence::new(grid.dimensions(), bubble_params.clone(), emission_params);

    // Setup intermediate acoustic driving
    let acoustic_pressure = Array3::from_elem(grid.dimensions(), -1.27e5); // -1.27 bar
    simulator.set_acoustic_pressure(acoustic_pressure);

    // Track peak state
    let mut peak_temp = 0.0;
    let mut peak_fields = (
        simulator.temperature_field.clone(),
        simulator.pressure_field.clone(),
        simulator.radius_field.clone(),
        simulator.particle_velocity_field.clone(),
        simulator.charge_density_field.clone(),
        simulator.compression_field.clone(),
    );

    // Simulate bubble dynamics
    println!("Simulating 1 full acoustic cycle (~38 µs) with high precision...");
    for step in 0..800000 {
        simulator.simulate_step(5e-11, step as f64 * 5e-11)?;

        // Check for peak temperature
        let current_max_temp = simulator
            .temperature_field
            .iter()
            .fold(0.0f64, |a, &b| a.max(b));
        if current_max_temp > peak_temp {
            peak_temp = current_max_temp;
            peak_fields = (
                simulator.temperature_field.clone(),
                simulator.pressure_field.clone(),
                simulator.radius_field.clone(),
                simulator.particle_velocity_field.clone(),
                simulator.charge_density_field.clone(),
                simulator.compression_field.clone(),
            );
        }
    }

    // Restore peak state for analysis
    simulator.temperature_field = peak_fields.0;
    simulator.pressure_field = peak_fields.1;
    simulator.radius_field = peak_fields.2;
    simulator.particle_velocity_field = peak_fields.3;
    simulator.charge_density_field = peak_fields.4;
    simulator.compression_field = peak_fields.5;

    // Calculate emission
    let temp_field = &simulator.temperature_field;
    let pressure_field = &simulator.pressure_field;
    let radius_field = &simulator.radius_field;
    let velocity_field = &simulator.particle_velocity_field;
    let charge_density_field = &simulator.charge_density_field;
    let compression_field = &simulator.compression_field;

    simulator.emission.calculate_emission(
        temp_field,
        pressure_field,
        radius_field,
        velocity_field,
        charge_density_field,
        compression_field,
        0.0,
    );

    // Calculate spectrum at peak emission location
    let (i, j, k) = simulator.emission.peak_emission_location();
    let spectrum = simulator.emission.calculate_spectrum_at_point(
        temp_field[[i, j, k]],
        pressure_field[[i, j, k]],
        radius_field[[i, j, k]],
        velocity_field[[i, j, k]],
        charge_density_field[[i, j, k]],
        compression_field[[i, j, k]],
    );

    println!("  Peak Temperature: {:.0} K", temp_field[[i, j, k]]);
    println!("  Peak Pressure: {:.1e} Pa", pressure_field[[i, j, k]]);
    println!("  Compression Ratio: {:.1}", compression_field[[i, j, k]]);
    println!("  Particle Velocity: {:.1e} m/s", velocity_field[[i, j, k]]);
    println!(
        "  Total Emission: {:.2e} W/m³",
        simulator.emission.total_light_output()
    );
    if spectrum.peak_wavelength() > 0.0 {
        println!(
            "  Peak Wavelength: {:.0} nm",
            spectrum.peak_wavelength() * 1e9
        );
    }
    println!("  Spectral Characteristics: Mixed (thermal + coherent)\n");

    // Calculate full spectral field for analysis
    simulator.emission.calculate_spectral_field(
        &simulator.temperature_field,
        &simulator.pressure_field,
        &simulator.radius_field,
        &simulator.particle_velocity_field,
        &simulator.charge_density_field,
        &simulator.compression_field,
        0.0,
    );

    // Analyze emission components if spectral field is available
    if let Some(spectral_field) = &simulator.emission.spectral_field {
        let peak_spectrum = spectral_field.get_spectrum_at(i, j, k);
        println!("  Spectral Analysis:");
        println!(
            "    Total Intensity: {:.2e} W/(m²·sr)",
            peak_spectrum.total_intensity()
        );

        // Check for Cherenkov signature (shorter wavelengths)
        // Use trapezoidal integration for UV portion to match total_intensity dimensions
        let wavelengths = &spectral_field.wavelengths;
        let intensities = &peak_spectrum.intensities;
        let mut uv_integrated_intensity = 0.0;

        for idx in 0..wavelengths.len().saturating_sub(1) {
            let lambda_avg = (wavelengths[idx] + wavelengths[idx + 1]) * 0.5;

            if lambda_avg < 400e-9 {
                let d_lambda = wavelengths[idx + 1] - wavelengths[idx];
                let i_avg = (intensities[idx] + intensities[idx + 1]) * 0.5;
                uv_integrated_intensity += i_avg * d_lambda;
            }
        }

        let total_int = peak_spectrum.total_intensity();
        let uv_fraction = if total_int > 1e-20 {
            uv_integrated_intensity / total_int
        } else {
            0.0
        };

        println!("    UV/Blue Fraction: {:.1}%", uv_fraction * 100.0);
        println!(
            "    Cherenkov Indicator: {}",
            if uv_fraction > 0.3 { "High" } else { "Low" }
        );
    }

    Ok(())
}
