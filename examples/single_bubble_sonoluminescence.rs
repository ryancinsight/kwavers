//! Single-Bubble Sonoluminescence (SBSL) Example
//!
//! This example demonstrates the proper separation of concerns:
//! - Bubble dynamics (core physics)
//! - Mechanical damage (cavitation erosion)
//! - Light emission (sonoluminescence)
//! - Chemistry (ROS generation)

use kwavers::{
    Grid, Time, HomogeneousMedium, PMLBoundary, Source, Sensor, Recorder,
    SensorConfig, RecorderConfig, KwaversResult, signal::Signal,
    physics::{
        // Core bubble dynamics
        bubble_dynamics::{
            BubbleField, BubbleParameters, BubbleState, GasSpecies,
        },
        // Mechanical damage from cavitation
        mechanics::cavitation::{
            CavitationDamage, MaterialProperties, DamageParameters,
        },
        // Light emission
        optics::sonoluminescence::{
            SonoluminescenceEmission, EmissionParameters,
        },
        // Chemistry and ROS
        chemistry::{
            SonochemistryModel, ROSSpecies,
        },
    },
};
use ndarray::{Array3, Array1};
use std::f64::consts::PI;

/// SBSL simulation parameters
#[derive(Debug, Clone)]
struct SBSLConfig {
    // Acoustic driving
    frequency: f64,
    pressure_amplitude: f64,
    
    // Bubble parameters
    initial_radius: f64,
    gas_type: GasSpecies,
    
    // Simulation domain
    grid_size: usize,
    domain_size: f64,
    simulation_time: f64,
    
    // Material for damage calculation
    wall_material: MaterialProperties,
}

impl Default for SBSLConfig {
    fn default() -> Self {
        Self {
            // Standard SBSL conditions
            frequency: 26.5e3,
            pressure_amplitude: 1.35 * 101325.0,
            initial_radius: 4.5e-6,
            gas_type: GasSpecies::Argon,
            
            // Small focused domain
            grid_size: 64,
            domain_size: 1e-3,
            simulation_time: 100e-6,
            
            // Stainless steel vessel
            wall_material: MaterialProperties::default(),
        }
    }
}

/// Acoustic standing wave source
#[derive(Debug)]
struct StandingWaveSource {
    frequency: f64,
    amplitude: f64,
    center: (f64, f64, f64),
}

impl Source for StandingWaveSource {
    fn get_source_term(&self, t: f64, x: f64, y: f64, z: f64, _grid: &Grid) -> f64 {
        let dx = x - self.center.0;
        let dy = y - self.center.1;
        let dz = z - self.center.2;
        let r = (dx*dx + dy*dy + dz*dz).sqrt();
        
        // Standing wave pattern
        let k = 2.0 * PI * self.frequency / 1500.0; // Wave number
        let spatial = (k * r).sin() / (k * r + 1e-10);
        let temporal = (2.0 * PI * self.frequency * t).sin();
        
        self.amplitude * spatial * temporal
    }
    
    fn positions(&self) -> Vec<(f64, f64, f64)> {
        vec![self.center]
    }
    
    fn signal(&self) -> &dyn Signal {
        panic!("StandingWaveSource doesn't use a separate signal")
    }
}

/// Run integrated SBSL simulation
fn run_sbsl_simulation(config: SBSLConfig) -> KwaversResult<()> {
    println!("=== Single-Bubble Sonoluminescence Simulation (v2) ===");
    println!("Demonstrating proper physics module separation");
    println!();
    
    // Setup grid
    let n = config.grid_size;
    let dx = config.domain_size / n as f64;
    let grid = Grid::new(n, n, n, dx, dx, dx);
    
    // Time stepping
    let c0 = 1482.0; // Sound speed in water
    let dt = 0.5 * dx / c0;
    let n_steps = (config.simulation_time / dt) as usize;
    
    println!("Configuration:");
    println!("  Frequency: {:.1} kHz", config.frequency / 1e3);
    println!("  Pressure: {:.2} atm", config.pressure_amplitude / 101325.0);
    println!("  Bubble R₀: {:.1} μm", config.initial_radius * 1e6);
    println!("  Gas: {:?}", config.gas_type);
    println!("  Grid: {}³ points, dx = {:.1} μm", n, dx * 1e6);
    println!("  Time steps: {}, dt = {:.2} ns", n_steps, dt * 1e9);
    println!();
    
    // Initialize physics modules
    
    // 1. Bubble dynamics (core physics)
    let bubble_params = BubbleParameters {
        r0: config.initial_radius,
        p0: 101325.0,
        rho_liquid: 998.0,
        c_liquid: c0,
        mu_liquid: 1.002e-3,
        sigma: 0.0728,
        pv: 2.33e3,
        thermal_conductivity: 0.6,
        specific_heat_liquid: 4182.0,
        accommodation_coeff: 0.04,
        gas_species: config.gas_type,
        initial_gas_pressure: 101325.0,
        use_compressibility: true,
        use_thermal_effects: true,
        use_mass_transfer: true,
    };
    
    let mut bubble_field = BubbleField::new((n, n, n), bubble_params.clone());
    bubble_field.add_center_bubble(&bubble_params);
    
    // 2. Mechanical damage model
    let damage_params = DamageParameters::default();
    let mut cavitation_damage = CavitationDamage::new(
        (n, n, n),
        config.wall_material.clone(),
        damage_params,
    );
    
    // 3. Light emission model
    let emission_params = EmissionParameters {
        use_blackbody: true,
        use_bremsstrahlung: true,
        use_molecular_lines: false,
        ionization_energy: 15.76, // Argon
        min_temperature: 2000.0,
        opacity_factor: 0.1,
    };
    let mut light_emission = SonoluminescenceEmission::new(
        (n, n, n),
        emission_params,
    );
    
    // 4. Chemistry model
    let mut chemistry = SonochemistryModel::new(n, n, n, 7.0);
    
    // Create acoustic source
    let source = StandingWaveSource {
        frequency: config.frequency,
        amplitude: config.pressure_amplitude,
        center: (
            config.domain_size / 2.0,
            config.domain_size / 2.0,
            config.domain_size / 2.0,
        ),
    };
    
    // Data collection
    let mut time_history = Vec::new();
    let mut radius_history = Vec::new();
    let mut temperature_history = Vec::new();
    let mut light_history = Vec::new();
    let mut damage_history = Vec::new();
    
    println!("Starting simulation...");
    let start_time = std::time::Instant::now();
    
    // Main simulation loop
    for step in 0..n_steps {
        let t = step as f64 * dt;
        
        // Generate acoustic field
        let mut pressure = grid.zeros_array();
        let mut dp_dt = grid.zeros_array();
        
        // Update pressure field with standing wave
        let time = step as f64 * dt;
        let omega = 2.0 * std::f64::consts::PI * config.frequency;
        let k_wave = 2.0 * PI * config.frequency / 1500.0; // Wave number
        
        // Use iterators for field updates
        pressure.indexed_iter_mut()
            .for_each(|((i, j, k), p)| {
                let x = i as f64 * dx;
                let y = j as f64 * dx;
                let z = k as f64 * dx;
                *p = config.pressure_amplitude * (omega * time).sin() * 
                     (k_wave * x).sin() * (k_wave * y).sin() * (k_wave * z).sin();
            });
        
        // Approximate time derivative
        dp_dt.indexed_iter_mut()
            .for_each(|((i, j, k), dp)| {
                *dp = -config.pressure_amplitude * 2.0 * PI * config.frequency
                    * (2.0 * PI * config.frequency * time).cos();
            });
        
        // Update bubble dynamics
        bubble_field.update(&pressure, &dp_dt, dt, t);
        
        // Get bubble states for other physics modules
        let bubble_states = bubble_field.get_state_fields();
        
        // Update mechanical damage
        cavitation_damage.update_damage(
            &bubble_states,
            (bubble_params.rho_liquid, bubble_params.c_liquid),
            dt,
        );
        
        // Calculate light emission
        light_emission.calculate_emission(
            &bubble_states.temperature,
            &bubble_states.pressure,
            &bubble_states.radius,
            t,
        );
        
        // Update chemistry (simplified for example)
        // In full implementation, would convert bubble states to chemistry format
        
        // Collect data for center bubble
        let center = (n/2, n/2, n/2);
        if let Some(bubble) = bubble_field.bubbles.get(&center) {
            radius_history.push(bubble.radius);
            temperature_history.push(bubble.temperature);
        }
        
        light_history.push(light_emission.emission_field[center]);
        damage_history.push(cavitation_damage.damage_field[center]);
        time_history.push(t);
        
        // Progress report
        if step % 1000 == 0 {
            let stats = bubble_field.get_statistics();
            println!(
                "Step {}/{}: Max T = {:.0} K, Max compression = {:.1}x, Total damage = {:.2e}",
                step, n_steps, 
                stats.max_temperature, 
                stats.max_compression,
                cavitation_damage.total_damage()
            );
        }
    }
    
    let elapsed = start_time.elapsed();
    println!("\nSimulation completed in {:.2} seconds", elapsed.as_secs_f64());
    
    // Analysis
    analyze_results(
        &time_history,
        &radius_history,
        &temperature_history,
        &light_history,
        &damage_history,
        &config,
        &cavitation_damage,
    );
    
    Ok(())
}

/// Analyze simulation results
fn analyze_results(
    times: &[f64],
    radii: &[f64],
    temperatures: &[f64],
    light: &[f64],
    damage: &[f64],
    config: &SBSLConfig,
    damage_model: &CavitationDamage,
) {
    println!("\n=== Results Analysis ===");
    
    // Bubble dynamics
    let r_min = radii.iter().cloned().fold(f64::INFINITY, f64::min);
    let r_max = radii.iter().cloned().fold(0.0, f64::max);
    let compression = config.initial_radius / r_min;
    println!("\nBubble Dynamics:");
    println!("  R_min: {:.2} μm", r_min * 1e6);
    println!("  R_max: {:.2} μm", r_max * 1e6);
    println!("  Max compression: {:.1}x", compression);
    
    // Temperature
    let t_max = temperatures.iter().cloned().fold(0.0, f64::max);
    println!("\nThermal:");
    println!("  Max temperature: {:.0} K", t_max);
    
    // Light emission
    let light_max = light.iter().cloned().fold(0.0, f64::max);
    let light_total: f64 = light.iter().sum();
    println!("\nLight Emission:");
    println!("  Peak intensity: {:.2e} W/m³", light_max);
    println!("  Total emission: {:.2e} J/m³", light_total * times[1]);
    
    // Mechanical damage
    let damage_max = damage.iter().cloned().fold(0.0, f64::max);
    let (i, j, k) = damage_model.max_damage_location();
    let mttf = damage_model.mean_time_to_failure(times[1]);
    
    println!("\nMechanical Damage:");
    println!("  Max damage parameter: {:.2e}", damage_max);
    println!("  Damage location: ({}, {}, {})", i, j, k);
    println!("  Mean time to failure: {:.2e} s", mttf);
    println!("  Total impacts: {}", damage_model.impact_count.sum());
    
    // Validation against literature
    println!("\nValidation:");
    println!("  Compression ratio: {} (Literature: 10-15 ✓)", 
        if compression > 10.0 && compression < 15.0 { "PASS" } else { "FAIL" });
    println!("  Max temperature: {} (Literature: 10,000-50,000 K ✓)", 
        if t_max > 10000.0 && t_max < 50000.0 { "PASS" } else { "FAIL" });
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    env_logger::init();
    
    // Run with default configuration
    let config = SBSLConfig::default();
    run_sbsl_simulation(config)?;
    
    // Optional: Parameter study
    println!("\n=== Parameter Study ===");
    
    // Test different materials
    for (name, material) in [
        ("Stainless Steel", MaterialProperties::default()),
        ("Aluminum", MaterialProperties {
            yield_strength: 270e6,
            ultimate_strength: 310e6,
            hardness: 1.2e9,
            density: 2700.0,
            fatigue_exponent: 3.5,
            erosion_resistance: 0.8,
        }),
        ("Glass", MaterialProperties {
            yield_strength: 50e6,
            ultimate_strength: 100e6,
            hardness: 5.5e9,
            density: 2500.0,
            fatigue_exponent: 10.0,
            erosion_resistance: 0.3,
        }),
    ] {
        println!("\nTesting material: {}", name);
        let mut config = SBSLConfig::default();
        config.wall_material = material;
        config.simulation_time = 20e-6; // Shorter for study
        
        if let Err(e) = run_sbsl_simulation(config) {
            eprintln!("Error with {}: {}", name, e);
        }
    }
    
    Ok(())
}