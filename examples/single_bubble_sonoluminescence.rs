//! Single-Bubble Sonoluminescence (SBSL) Example
//!
//! This example demonstrates a single-bubble sonoluminescence simulation based on
//! scientific literature, particularly:
//! - Gaitan et al. (1992) "Sonoluminescence and bubble dynamics for a single, stable, cavitation bubble"
//! - Brenner et al. (2002) "Single-bubble sonoluminescence"
//!
//! The simulation includes:
//! - Acoustic driving at resonance frequency
//! - Bubble dynamics with Keller-Miksis equation
//! - Temperature evolution and shock heating
//! - Light emission from blackbody and bremsstrahlung
//! - ROS generation and chemistry

use kwavers::{
    Grid, Time, HomogeneousMedium, PMLBoundary, Source, Sensor, Recorder,
    SensorConfig, RecorderConfig, KwaversResult,
    physics::{
        bubble_dynamics::{
            BubbleField, BubbleParameters, GasSpecies,
        },
        mechanics::cavitation::{
            CavitationDamage, MaterialProperties, DamageParameters,
        },
        optics::sonoluminescence::{
            SonoluminescenceEmission, EmissionParameters,
        },
        chemistry::{SonochemistryModel, ROSSpecies},
    },
};
use ndarray::{Array3, Array1, s};
use std::f64::consts::PI;
use std::fs::File;
use std::io::Write;

/// SBSL experimental parameters from literature
#[derive(Debug, Clone)]
struct SBSLParameters {
    // Acoustic parameters (Gaitan et al. 1992)
    frequency: f64,           // Driving frequency (Hz)
    pressure_amplitude: f64,  // Acoustic pressure amplitude (Pa)
    ambient_pressure: f64,    // Ambient pressure (Pa)
    
    // Bubble parameters
    equilibrium_radius: f64,  // R0 (m)
    gas_type: GasSpecies,    // Gas inside bubble
    
    // Liquid properties (water at 20°C)
    liquid_density: f64,      // ρ (kg/m³)
    sound_speed: f64,         // c (m/s)
    surface_tension: f64,     // σ (N/m)
    viscosity: f64,           // μ (Pa·s)
    vapor_pressure: f64,      // Pv (Pa)
    
    // Simulation parameters
    domain_size: f64,         // Physical domain size (m)
    grid_points: usize,       // Number of grid points
    simulation_time: f64,     // Total simulation time (s)
    
    // Material for damage assessment
    vessel_material: MaterialProperties,
}

impl Default for SBSLParameters {
    fn default() -> Self {
        Self {
            // Standard SBSL conditions
            frequency: 26.5e3,              // 26.5 kHz
            pressure_amplitude: 1.35 * 101325.0, // 1.35 atm
            ambient_pressure: 101325.0,     // 1 atm
            
            // Argon bubble
            equilibrium_radius: 4.5e-6,     // 4.5 μm
            gas_type: GasSpecies::Argon,
            
            // Water at 20°C
            liquid_density: 998.0,
            sound_speed: 1482.0,
            surface_tension: 0.0728,
            viscosity: 1.002e-3,
            vapor_pressure: 2.33e3,
            
            // Small computational domain
            domain_size: 1e-3,              // 1 mm
            grid_points: 64,
            simulation_time: 100e-6,        // 100 μs (several acoustic cycles)
            
            // Pyrex glass vessel
            vessel_material: MaterialProperties {
                yield_strength: 50e6,
                ultimate_strength: 100e6,
                hardness: 5.5e9,
                density: 2230.0,
                fatigue_exponent: 10.0,
                erosion_resistance: 0.3,
            },
        }
    }
}

/// Standing wave acoustic source for SBSL
struct StandingWaveSource {
    frequency: f64,
    amplitude: f64,
    wavelength: f64,
    center: (f64, f64, f64),
}

impl Source for StandingWaveSource {
    fn pressure(&self, x: f64, y: f64, z: f64, t: f64) -> f64 {
        // Standing wave pattern with antinode at center
        let r = ((x - self.center.0).powi(2) + 
                 (y - self.center.1).powi(2) + 
                 (z - self.center.2).powi(2)).sqrt();
        
        let k = 2.0 * PI / self.wavelength;
        let spatial_pattern = (k * r).cos();
        let temporal_pattern = (2.0 * PI * self.frequency * t).sin();
        
        self.amplitude * spatial_pattern * temporal_pattern
    }
    
    fn velocity_x(&self, _x: f64, _y: f64, _z: f64, _t: f64) -> f64 { 0.0 }
    fn velocity_y(&self, _x: f64, _y: f64, _z: f64, _t: f64) -> f64 { 0.0 }
    fn velocity_z(&self, _x: f64, _y: f64, _z: f64, _t: f64) -> f64 { 0.0 }
}

/// Run SBSL simulation
fn run_sbsl_simulation(params: SBSLParameters) -> KwaversResult<()> {
    println!("=== Single-Bubble Sonoluminescence Simulation ===");
    println!("Based on Gaitan et al. (1992) experimental conditions");
    println!();
    
    // Setup grid
    let n = params.grid_points;
    let dx = params.domain_size / n as f64;
    let grid = Grid::new(n, n, n, dx, dx, dx);
    
    // Time stepping
    let dt = 0.5 * dx / params.sound_speed; // CFL condition
    let n_steps = (params.simulation_time / dt) as usize;
    
    println!("Simulation parameters:");
    println!("  Frequency: {:.1} kHz", params.frequency / 1e3);
    println!("  Pressure: {:.2} atm", params.pressure_amplitude / 101325.0);
    println!("  Bubble R₀: {:.1} μm", params.equilibrium_radius * 1e6);
    println!("  Gas: {:?}", params.gas_type);
    println!("  Grid: {}³ points, dx = {:.1} μm", n, dx * 1e6);
    println!("  Time steps: {}, dt = {:.2} ns", n_steps, dt * 1e9);
    println!();
    
    // Initialize bubble dynamics
    let bubble_params = BubbleParameters {
        r0: params.equilibrium_radius,
        p0: params.ambient_pressure,
        rho_liquid: params.liquid_density,
        c_liquid: params.sound_speed,
        mu_liquid: params.viscosity,
        sigma: params.surface_tension,
        pv: params.vapor_pressure,
        thermal_conductivity: 0.6,  // Water
        specific_heat_liquid: 4182.0,
        accommodation_coeff: 0.04,   // For argon
        gas_species: params.gas_type,
        initial_gas_pressure: params.ambient_pressure,
        use_compressibility: true,
        use_thermal_effects: true,
        use_mass_transfer: true,
    };
    
    let mut bubble_field = BubbleField::new((n, n, n), bubble_params.clone());
    bubble_field.add_center_bubble(&bubble_params);
    
    // Initialize damage model
    let damage_params = DamageParameters::default();
    let mut cavitation_damage = CavitationDamage::new(
        (n, n, n),
        params.vessel_material.clone(),
        damage_params,
    );
    
    // Initialize light emission
    let emission_params = EmissionParameters {
        use_blackbody: true,
        use_bremsstrahlung: true,
        use_molecular_lines: false,
        ionization_energy: 15.76,  // Argon first ionization
        min_temperature: 2000.0,
        opacity_factor: 0.1,
    };
    let mut light_emission = SonoluminescenceEmission::new(
        (n, n, n),
        emission_params,
    );
    
    // Initialize chemistry
    let mut chemistry = SonochemistryModel::new(n, n, n, 7.0); // pH 7
    
    // Create acoustic source
    let source = StandingWaveSource {
        frequency: params.frequency,
        amplitude: params.pressure_amplitude,
        wavelength: params.sound_speed / params.frequency,
        center: (
            params.domain_size / 2.0,
            params.domain_size / 2.0,
            params.domain_size / 2.0,
        ),
    };
    
    // Data collection
    let mut time_data = Vec::new();
    let mut radius_data = Vec::new();
    let mut temperature_data = Vec::new();
    let mut pressure_data = Vec::new();
    let mut light_intensity_data = Vec::new();
    let mut photon_count_data = Vec::new();
    
    println!("Starting simulation...");
    let start_time = std::time::Instant::now();
    
    // Main simulation loop
    for step in 0..n_steps {
        let t = step as f64 * dt;
        
        // Generate acoustic field
        let mut pressure_field = Array3::zeros((n, n, n));
        let mut dp_dt_field = Array3::zeros((n, n, n));
        
        for i in 0..n {
            for j in 0..n {
                for k in 0..n {
                    let x = i as f64 * dx;
                    let y = j as f64 * dx;
                    let z = k as f64 * dx;
                    
                    pressure_field[[i, j, k]] = source.pressure(x, y, z, t);
                    
                    // Approximate pressure time derivative
                    let dt_small = 1e-9;
                    let p_future = source.pressure(x, y, z, t + dt_small);
                    dp_dt_field[[i, j, k]] = (p_future - pressure_field[[i, j, k]]) / dt_small;
                }
            }
        }
        
        // Update bubble dynamics
        bubble_field.update(&pressure_field, &dp_dt_field, dt, t);
        
        // Get bubble states
        let bubble_states = bubble_field.get_state_fields();
        
        // Update damage
        cavitation_damage.update_damage(
            &bubble_states,
            (params.liquid_density, params.sound_speed),
            dt,
        );
        
        // Calculate light emission
        light_emission.calculate_emission(
            &bubble_states.temperature,
            &bubble_states.pressure,
            &bubble_states.radius,
            t,
        );
        
        // Collect data for center bubble
        let center = (n/2, n/2, n/2);
        if let Some(bubble) = bubble_field.bubbles.get(&center) {
            time_data.push(t);
            radius_data.push(bubble.radius);
            temperature_data.push(bubble.temperature);
            pressure_data.push(bubble.pressure_internal);
            
            let intensity = light_emission.emission_field[[center.0, center.1, center.2]];
            light_intensity_data.push(intensity);
            
            // Estimate photon count (simplified)
            let volume = bubble.volume();
            let photon_energy = 3.0 * 1.602e-19; // ~3 eV average
            let photons = intensity * volume * dt / photon_energy;
            photon_count_data.push(photons);
        }
        
        // Progress update
        if step % 1000 == 0 {
            let stats = bubble_field.get_statistics();
            println!(
                "Step {}/{}: T_max = {:.0} K, Compression = {:.1}x",
                step, n_steps,
                stats.max_temperature,
                stats.max_compression
            );
        }
    }
    
    let elapsed = start_time.elapsed();
    println!("\nSimulation completed in {:.2} seconds", elapsed.as_secs_f64());
    
    // Analyze results
    analyze_sbsl_results(
        &time_data,
        &radius_data,
        &temperature_data,
        &pressure_data,
        &light_intensity_data,
        &photon_count_data,
        &params,
    );
    
    // Save data
    save_sbsl_data(
        &time_data,
        &radius_data,
        &temperature_data,
        &light_intensity_data,
        "sbsl_results.csv",
    )?;
    
    Ok(())
}

/// Analyze SBSL results and compare with literature
fn analyze_sbsl_results(
    times: &[f64],
    radii: &[f64],
    temperatures: &[f64],
    pressures: &[f64],
    light_intensities: &[f64],
    photon_counts: &[f64],
    params: &SBSLParameters,
) {
    println!("\n=== SBSL Results Analysis ===");
    
    // Find key metrics
    let r_min = radii.iter().cloned().fold(f64::INFINITY, f64::min);
    let r_max = radii.iter().cloned().fold(0.0, f64::max);
    let t_max = temperatures.iter().cloned().fold(0.0, f64::max);
    let p_max = pressures.iter().cloned().fold(0.0, f64::max);
    
    let compression_ratio = params.equilibrium_radius / r_min;
    
    // Find light pulse characteristics
    let light_max = light_intensities.iter().cloned().fold(0.0, f64::max);
    let mut pulse_width = 0.0;
    let threshold = light_max * 0.1; // 10% of peak
    
    let mut in_pulse = false;
    let mut pulse_start = 0.0;
    for (i, &intensity) in light_intensities.iter().enumerate() {
        if !in_pulse && intensity > threshold {
            in_pulse = true;
            pulse_start = times[i];
        } else if in_pulse && intensity < threshold {
            pulse_width = times[i] - pulse_start;
            break;
        }
    }
    
    // Total photon count per flash
    let total_photons: f64 = photon_counts.iter().sum();
    
    println!("\nBubble Dynamics:");
    println!("  R_min: {:.2} μm", r_min * 1e6);
    println!("  R_max: {:.2} μm", r_max * 1e6);
    println!("  Compression ratio: {:.1}", compression_ratio);
    println!("  Max temperature: {:.0} K", t_max);
    println!("  Max pressure: {:.0} MPa", p_max / 1e6);
    
    println!("\nLight Emission:");
    println!("  Peak intensity: {:.2e} W/m³", light_max);
    println!("  Pulse width: {:.1} ps", pulse_width * 1e12);
    println!("  Photons per flash: {:.2e}", total_photons);
    
    println!("\nComparison with Literature:");
    println!("  Parameter              | Simulation | Literature (Gaitan et al.)");
    println!("  --------------------- | ---------- | -------------------------");
    println!("  Compression ratio     | {:.1}      | 10-15", compression_ratio);
    println!("  Max temperature       | {:.0} K    | 10,000-50,000 K", t_max);
    println!("  Pulse width           | {:.0} ps   | 50-300 ps", pulse_width * 1e12);
    println!("  Photons per flash     | {:.1e}     | 10⁴-10⁶", total_photons);
    
    // Validation checks
    let compression_ok = compression_ratio >= 10.0 && compression_ratio <= 15.0;
    let temperature_ok = t_max >= 10000.0 && t_max <= 50000.0;
    let pulse_width_ok = pulse_width >= 50e-12 && pulse_width <= 300e-12;
    let photons_ok = total_photons >= 1e4 && total_photons <= 1e6;
    
    println!("\nValidation:");
    println!("  Compression: {}", if compression_ok { "✓ PASS" } else { "✗ FAIL" });
    println!("  Temperature: {}", if temperature_ok { "✓ PASS" } else { "✗ FAIL" });
    println!("  Pulse width: {}", if pulse_width_ok { "✓ PASS" } else { "✗ FAIL" });
    println!("  Photon count: {}", if photons_ok { "✓ PASS" } else { "✗ FAIL" });
}

/// Save SBSL data to CSV file
fn save_sbsl_data(
    times: &[f64],
    radii: &[f64],
    temperatures: &[f64],
    light_intensities: &[f64],
    filename: &str,
) -> KwaversResult<()> {
    let mut file = File::create(filename)?;
    
    // Write header
    writeln!(file, "time_us,radius_um,temperature_K,light_intensity_W_m3")?;
    
    // Write data
    for i in 0..times.len() {
        writeln!(
            file,
            "{:.3},{:.3},{:.0},{:.3e}",
            times[i] * 1e6,
            radii[i] * 1e6,
            temperatures[i],
            light_intensities[i]
        )?;
    }
    
    println!("\nData saved to {}", filename);
    Ok(())
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    env_logger::init();
    
    // Run with default SBSL parameters
    let params = SBSLParameters::default();
    run_sbsl_simulation(params)?;
    
    // Optional: Parameter study
    println!("\n=== Parameter Study ===");
    
    // Test different gases
    for gas in [GasSpecies::Argon, GasSpecies::Xenon, GasSpecies::Air] {
        println!("\nTesting gas: {:?}", gas);
        let mut params = SBSLParameters::default();
        params.gas_type = gas;
        params.simulation_time = 20e-6; // Shorter for parameter study
        
        if let Err(e) = run_sbsl_simulation(params) {
            eprintln!("Error with {:?}: {}", gas, e);
        }
    }
    
    Ok(())
}