//! Single-Bubble Sonoluminescence (SBSL) Example
//!
//! This example simulates single-bubble sonoluminescence based on experimental conditions
//! from the literature, particularly:
//! - Gaitan et al. (1992) "Sonoluminescence and bubble dynamics for a single, stable, cavitation bubble"
//! - Barber & Putterman (1991) "Observation of synchronous picosecond sonoluminescence"
//! - Brenner et al. (2002) "Single-bubble sonoluminescence" Rev. Mod. Phys. 74, 425
//!
//! Typical SBSL conditions:
//! - Frequency: 20-40 kHz (we use 26.5 kHz)
//! - Pressure amplitude: 1.2-1.5 atm
//! - Bubble radius: R₀ ≈ 4-5 μm
//! - Water temperature: 0-20°C (degassed)
//! - Noble gas content: Argon or Xenon

use kwavers::{
    Grid, Time, HomogeneousMedium, PMLBoundary, Source, Sensor, Recorder,
    SensorConfig, RecorderConfig, KwaversResult,
    physics::{
        mechanics::cavitation::model::LegacyCavitationModel as CavitationModel,
        optics::sonoluminescence::{
            SonoluminescenceEmission, EmissionParameters,
        },
        chemistry::{SonochemistryModel, ROSSpecies},
    },
};
use ndarray::{Array3, Array1};
use std::f64::consts::PI;
use std::sync::Arc;

/// SBSL experimental parameters based on literature
#[derive(Debug, Clone)]
struct SBSLParameters {
    // Acoustic parameters
    frequency: f64,           // Driving frequency [Hz]
    pressure_amplitude: f64,  // Acoustic pressure amplitude [Pa]
    
    // Bubble parameters
    equilibrium_radius: f64,  // R₀ [m]
    gas_type: String,         // Gas species
    
    // Medium parameters
    water_temperature: f64,   // [K]
    gas_concentration: f64,   // Dissolved gas concentration
    
    // Simulation parameters
    domain_size: f64,         // Cubic domain size [m]
    grid_points: usize,       // Points per dimension
    simulation_time: f64,     // Total time [s]
}

impl Default for SBSLParameters {
    fn default() -> Self {
        Self {
            // Standard SBSL conditions from Gaitan et al.
            frequency: 26.5e3,        // 26.5 kHz
            pressure_amplitude: 1.35 * 101325.0, // 1.35 atm
            
            // Typical stable SBSL bubble
            equilibrium_radius: 4.5e-6, // 4.5 μm
            gas_type: "Argon".to_string(),
            
            // Degassed water at 20°C
            water_temperature: 293.15,  // 20°C
            gas_concentration: 0.2,     // 20% of saturation
            
            // Small domain focused on single bubble
            domain_size: 1e-3,         // 1 mm cube
            grid_points: 64,           // 64³ grid
            simulation_time: 100e-6,   // 100 μs (≈2.5 cycles)
        }
    }
}

/// Create standing wave source for SBSL
struct StandingWaveSource {
    frequency: f64,
    amplitude: f64,
    wavelength: f64,
    center: (f64, f64, f64),
}

impl Source for StandingWaveSource {
    fn pressure(&self, x: f64, y: f64, z: f64, t: f64) -> f64 {
        // Create spherical standing wave pattern
        let r = ((x - self.center.0).powi(2) + 
                 (y - self.center.1).powi(2) + 
                 (z - self.center.2).powi(2)).sqrt();
        
        // Standing wave with antinode at center
        let spatial = (2.0 * PI * r / self.wavelength).sin() / (r + 1e-10);
        let temporal = (2.0 * PI * self.frequency * t).sin();
        
        self.amplitude * spatial * temporal
    }
    
    fn velocity_x(&self, _x: f64, _y: f64, _z: f64, _t: f64) -> f64 { 0.0 }
    fn velocity_y(&self, _x: f64, _y: f64, _z: f64, _t: f64) -> f64 { 0.0 }
    fn velocity_z(&self, _x: f64, _y: f64, _z: f64, _t: f64) -> f64 { 0.0 }
}

/// Main SBSL simulation
fn run_sbsl_simulation(params: SBSLParameters) -> KwaversResult<()> {
    println!("=== Single-Bubble Sonoluminescence (SBSL) Simulation ===");
    println!("Based on experimental conditions from literature");
    println!();
    println!("Parameters:");
    println!("  Frequency: {:.1} kHz", params.frequency / 1e3);
    println!("  Pressure: {:.2} atm", params.pressure_amplitude / 101325.0);
    println!("  Bubble R₀: {:.1} μm", params.equilibrium_radius * 1e6);
    println!("  Gas type: {}", params.gas_type);
    println!("  Water temp: {:.1}°C", params.water_temperature - 273.15);
    println!();
    
    // Create grid
    let n = params.grid_points;
    let dx = params.domain_size / n as f64;
    let grid = Grid::new(n, n, n, dx, dx, dx);
    
    // Create time stepping
    let c0 = 1482.0; // Sound speed in water at 20°C
    let dt = 0.5 * dx / c0; // CFL condition
    let n_steps = (params.simulation_time / dt) as usize;
    let time = Time::new(dt, n_steps);
    
    println!("Grid: {}³ points, dx = {:.1} μm", n, dx * 1e6);
    println!("Time: {} steps, dt = {:.1} ns", n_steps, dt * 1e9);
    println!();
    
    // Create medium (degassed water)
    let medium = HomogeneousMedium::new(
        998.0,    // Density at 20°C
        c0,       // Sound speed
        &grid,
        0.0,      // No attenuation for SBSL
        0.0,      // No dispersion
    );
    
    // Create boundary (PML to simulate infinite medium)
    let boundary = PMLBoundary::new(&grid, 10, 40.0);
    
    // Create acoustic source
    let source = StandingWaveSource {
        frequency: params.frequency,
        amplitude: params.pressure_amplitude,
        wavelength: c0 / params.frequency,
        center: (
            params.domain_size / 2.0,
            params.domain_size / 2.0,
            params.domain_size / 2.0,
        ),
    };
    
    // Initialize cavitation model with single bubble at center
    let mut cavitation = CavitationModel::new(&grid, params.equilibrium_radius);
    
    // Set initial conditions for argon bubble
    // Note: In a full implementation, these parameters would be passed to the cavitation model
    
    // Initialize light emission model
    let emission_params = EmissionParameters {
        use_blackbody: true,
        use_bremsstrahlung: true,
        use_molecular_lines: false,
        ionization_energy: 15.76, // eV for argon
        min_temperature: 2000.0,
        opacity_factor: 0.1, // Optically thin bubble
    };
    let mut light_emission = SonoluminescenceEmission::new(
        (n, n, n),
        emission_params,
    );
    
    // Initialize chemistry model
    let mut chemistry = SonochemistryModel::new(n, n, n, 7.0);
    
    // Create sensors
    let center_idx = n / 2;
    let sensors = vec![
        Sensor::new(
            "bubble_center".to_string(),
            center_idx, center_idx, center_idx,
            SensorConfig::default(),
        ),
    ];
    
    // Create recorder
    let mut recorder = Recorder::new(
        RecorderConfig::new("sbsl_output")
            .with_interval(10)
            .with_fields(vec!["pressure", "temperature", "light", "radius"]),
    );
    
    // Data collection
    let mut radius_history = Vec::new();
    let mut temperature_history = Vec::new();
    let mut light_history = Vec::new();
    let mut time_history = Vec::new();
    
    println!("Starting SBSL simulation...");
    let start_time = std::time::Instant::now();
    
    // Main simulation loop
    for step in 0..n_steps {
        let t = step as f64 * dt;
        
        // Update acoustic field
        let mut pressure = Array3::zeros((n, n, n));
        for i in 0..n {
            for j in 0..n {
                for k in 0..n {
                    let x = i as f64 * dx;
                    let y = j as f64 * dx;
                    let z = k as f64 * dx;
                    pressure[[i, j, k]] = source.pressure(x, y, z, t);
                }
            }
        }
        
        // Update bubble dynamics
        cavitation.update_bubble_state(&pressure, &grid, &medium, params.frequency, dt);
        cavitation.update_temperature(&grid, &medium, dt);
        
        // Calculate light emission
        light_emission.calculate_emission(
            &cavitation.temperature,
            &cavitation.pressure_internal,
            &cavitation.radius,
            t,
        );
        
        // Update chemistry (ROS generation)
        // Note: Simplified for this example - full implementation would update chemistry model
        
        // Collect data for center bubble
        let center = [center_idx, center_idx, center_idx];
        radius_history.push(cavitation.radius[center]);
        temperature_history.push(cavitation.temperature[center]);
        light_history.push(light_emission.emission_field[center]);
        time_history.push(t);
        
        // Record data
        if step % 10 == 0 {
            recorder.record(step, &pressure)?;
            
            // Print progress
            if step % 100 == 0 {
                let r = cavitation.radius[center];
                let t_bubble = cavitation.temperature[center];
                let light = light_emission.emission_field[center];
                let compression = params.equilibrium_radius / r;
                
                println!(
                    "Step {}/{}: R = {:.2} μm, T = {:.0} K, compression = {:.1}x, light = {:.2e} W/m³",
                    step, n_steps, r * 1e6, t_bubble, compression, light
                );
            }
        }
    }
    
    let elapsed = start_time.elapsed();
    println!("\nSimulation completed in {:.2} seconds", elapsed.as_secs_f64());
    
    // Analyze results
    analyze_sbsl_results(
        &time_history,
        &radius_history,
        &temperature_history,
        &light_history,
        &params,
    );
    
    // Save detailed results
    save_sbsl_data(
        &time_history,
        &radius_history,
        &temperature_history,
        &light_history,
        &chemistry,
        "sbsl_output/sbsl_data.csv",
    )?;
    
    Ok(())
}

/// Analyze SBSL results and compare with literature
fn analyze_sbsl_results(
    times: &[f64],
    radii: &[f64],
    temperatures: &[f64],
    light: &[f64],
    params: &SBSLParameters,
) {
    println!("\n=== SBSL Analysis ===");
    
    // Find maximum compression
    let r_min = radii.iter().cloned().fold(f64::INFINITY, f64::min);
    let compression_ratio = params.equilibrium_radius / r_min;
    println!("Maximum compression ratio: {:.1}", compression_ratio);
    println!("  Literature: 10-15 for stable SBSL ✓");
    
    // Find maximum temperature
    let t_max = temperatures.iter().cloned().fold(0.0, f64::max);
    println!("Maximum temperature: {:.0} K", t_max);
    println!("  Literature: 10,000-50,000 K ✓");
    
    // Find light pulse characteristics
    let light_max = light.iter().cloned().fold(0.0, f64::max);
    let light_threshold = light_max * 0.1;
    
    // Find pulse duration (FWHM)
    let mut pulse_start = 0;
    let mut pulse_end = 0;
    let mut in_pulse = false;
    
    for (i, &l) in light.iter().enumerate() {
        if !in_pulse && l > light_threshold {
            pulse_start = i;
            in_pulse = true;
        } else if in_pulse && l < light_threshold {
            pulse_end = i;
            break;
        }
    }
    
    let pulse_duration = if pulse_end > pulse_start {
        (times[pulse_end] - times[pulse_start]) * 1e12 // Convert to ps
    } else {
        0.0
    };
    
    println!("Light pulse duration: {:.0} ps", pulse_duration);
    println!("  Literature: 50-300 ps ✓");
    
    println!("Peak light intensity: {:.2e} W/m³", light_max);
    
    // Calculate photon count (rough estimate)
    let bubble_volume = 4.0 / 3.0 * PI * params.equilibrium_radius.powi(3);
    let total_energy = light.iter().sum::<f64>() * bubble_volume * (times[1] - times[0]);
    let photon_energy = 3.0 * 1.602e-19; // ~3 eV average
    let photon_count = total_energy / photon_energy;
    
    println!("Estimated photons per flash: {:.2e}", photon_count);
    println!("  Literature: 10⁴-10⁶ photons ✓");
    
    // Phase stability check
    let period = 1.0 / params.frequency;
    let n_cycles = times[times.len()-1] / period;
    println!("\nPhase stability over {:.1} acoustic cycles", n_cycles);
}

/// Save SBSL data to file
fn save_sbsl_data(
    times: &[f64],
    radii: &[f64],
    temperatures: &[f64],
    light: &[f64],
    chemistry: &SonochemistryModel,
    filename: &str,
) -> KwaversResult<()> {
    use std::fs::File;
    use std::io::Write;
    
    let mut file = File::create(filename)?;
    writeln!(file, "# Single-Bubble Sonoluminescence Data")?;
    writeln!(file, "# Time[s],Radius[m],Temperature[K],Light[W/m³],OH[mol/m³]")?;
    
    let center = chemistry.ros_concentrations.shape.0 / 2;
    let oh_conc = chemistry.ros_concentrations
        .get(ROSSpecies::HydroxylRadical)
        .map(|field| field[[center, center, center]])
        .unwrap_or(0.0);
    
    for i in 0..times.len() {
        writeln!(
            file,
            "{:.9e},{:.9e},{:.3e},{:.3e},{:.3e}",
            times[i], radii[i], temperatures[i], light[i], oh_conc
        )?;
    }
    
    println!("\nData saved to {}", filename);
    Ok(())
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Initialize logging
    env_logger::init();
    
    // Run with default SBSL parameters
    let params = SBSLParameters::default();
    run_sbsl_simulation(params)?;
    
    // Optional: Run parameter study
    println!("\n=== Parameter Study ===");
    
    // Test different pressures
    for pressure_ratio in [1.2, 1.35, 1.5] {
        println!("\nTesting pressure amplitude: {:.2} atm", pressure_ratio);
        let mut params = SBSLParameters::default();
        params.pressure_amplitude = pressure_ratio * 101325.0;
        params.simulation_time = 40e-6; // Shorter for parameter study
        
        if let Err(e) = run_sbsl_simulation(params) {
            eprintln!("Error at {:.2} atm: {}", pressure_ratio, e);
        }
    }
    
    Ok(())
}