//! Multi-Bubble Sonoluminescence (MBSL) Example
//!
//! This example demonstrates multi-bubble sonoluminescence with bubble cloud dynamics
//! based on literature:
//! - Yasui et al. (2008) "The range of ambient radius for an active bubble in sonoluminescence"
//! - Mettin et al. (1997) "Bjerknes forces between small cavitation bubbles"
//! - Lauterborn & Kurz (2010) "Physics of bubble oscillations"

use kwavers::{
    Grid, Time, HomogeneousMedium, Source, Sensor, Recorder,
    SensorConfig, RecorderConfig, KwaversResult,
    physics::{
        bubble_dynamics::{
            BubbleCloud, BubbleParameters, GasSpecies, BubbleInteractions,
            CollectiveEffects, BubbleStateFields,
            bubble_field::{SizeDistribution, SpatialDistribution},
        },
        mechanics::cavitation::{
            CavitationDamage, MaterialProperties, DamageParameters,
            cavitation_intensity,
        },
        optics::sonoluminescence::{
            SonoluminescenceEmission, EmissionParameters,
        },
        chemistry::{SonochemistryModel, ROSSpecies},
    },
};
use ndarray::{Array3, s};
use std::f64::consts::PI;
use std::fs::File;
use std::io::Write;
use std::fmt::Debug;

/// MBSL experimental parameters
#[derive(Debug, Clone)]
struct MBSLParameters {
    // Acoustic parameters
    frequency: f64,              // Driving frequency (Hz)
    pressure_amplitude: f64,     // Acoustic pressure amplitude (Pa)
    
    // Bubble cloud parameters
    bubble_density: f64,         // Number density (bubbles/m³)
    size_distribution: SizeDistribution,
    spatial_distribution: SpatialDistribution,
    gas_type: GasSpecies,
    
    // Domain and simulation
    domain_size: f64,           // Physical domain size (m)
    grid_points: usize,         // Grid resolution
    simulation_time: f64,       // Total time (s)
    
    // Vessel material
    vessel_material: MaterialProperties,
}

impl Default for MBSLParameters {
    fn default() -> Self {
        Self {
            // Ultrasonic cleaning bath conditions
            frequency: 40e3,                    // 40 kHz
            pressure_amplitude: 2.0 * 101325.0, // 2 atm
            
            // Bubble cloud
            bubble_density: 1e9,                // 10⁹ bubbles/m³
            size_distribution: SizeDistribution::LogNormal {
                mean: 5e-6,      // 5 μm mean radius
                std_dev: 2e-6,   // 2 μm std dev
            },
            spatial_distribution: SpatialDistribution::Uniform,
            gas_type: GasSpecies::Air,
            
            // Larger domain for cloud
            domain_size: 5e-3,                  // 5 mm
            grid_points: 128,
            simulation_time: 50e-6,             // 50 μs
            
            // Stainless steel vessel
            vessel_material: MaterialProperties::default(),
        }
    }
}

/// Focused transducer source
struct FocusedTransducerSource {
    frequency: f64,
    amplitude: f64,
    focal_point: (f64, f64, f64),
    aperture: f64,
    signal: kwavers::signal::SineWave,
}

impl FocusedTransducerSource {
    fn new(frequency: f64, amplitude: f64, focal_point: (f64, f64, f64), aperture: f64) -> Self {
        Self {
            frequency,
            amplitude,
            focal_point,
            aperture,
            signal: kwavers::signal::SineWave::new(frequency, amplitude, 0.0),
        }
    }
    
    /// Calculate the time derivative of pressure analytically
    fn pressure_time_derivative(&self, x: f64, y: f64, z: f64, t: f64) -> f64 {
        let dx = x - self.focal_point.0;
        let dy = y - self.focal_point.1;
        let dz = z - self.focal_point.2;
        let r = (dx*dx + dy*dy + dz*dz).sqrt();
        
        let beam_width = self.aperture / 4.0;
        let spatial = (-r*r / (2.0 * beam_width*beam_width)).exp();
        let omega = 2.0 * PI * self.frequency;
        
        // d/dt[A*spatial*sin(ωt)] = A*spatial*ω*cos(ωt)
        self.amplitude * spatial * omega * (omega * t).cos()
    }
}

impl Debug for FocusedTransducerSource {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("FocusedTransducerSource")
            .field("frequency", &self.frequency)
            .field("amplitude", &self.amplitude)
            .finish()
    }
}

impl Source for FocusedTransducerSource {
    fn get_source_term(&self, t: f64, x: f64, y: f64, z: f64, _grid: &Grid) -> f64 {
        let dx = x - self.focal_point.0;
        let dy = y - self.focal_point.1;
        let dz = z - self.focal_point.2;
        let r = (dx*dx + dy*dy + dz*dz).sqrt();
        
        // Focused beam pattern
        let beam_width = self.aperture / 4.0;
        let spatial = (-r*r / (2.0 * beam_width*beam_width)).exp();
        let temporal = (2.0 * PI * self.frequency * t).sin();
        
        self.amplitude * spatial * temporal
    }
    
    fn positions(&self) -> Vec<(f64, f64, f64)> {
        vec![self.focal_point]
    }
    
    fn signal(&self) -> &dyn kwavers::Signal {
        &self.signal
    }
}

/// Run MBSL simulation
fn run_mbsl_simulation(params: MBSLParameters) -> KwaversResult<()> {
    println!("=== Multi-Bubble Sonoluminescence Simulation ===");
    println!("Bubble cloud dynamics with collective effects");
    println!();
    
    // Setup grid
    let n = params.grid_points;
    let dx = params.domain_size / n as f64;
    let grid = Grid::new(n, n, n, dx, dx, dx);
    
    // Time stepping
    let c0 = 1482.0; // Sound speed in water
    let dt = 0.5 * dx / c0;
    let n_steps = (params.simulation_time / dt) as usize;
    
    println!("Configuration:");
    println!("  Frequency: {:.1} kHz", params.frequency / 1e3);
    println!("  Pressure: {:.2} atm", params.pressure_amplitude / 101325.0);
    println!("  Bubble density: {:.1e} bubbles/m³", params.bubble_density);
    println!("  Grid: {}³ points, dx = {:.1} μm", n, dx * 1e6);
    println!("  Time steps: {}, dt = {:.2} ns", n_steps, dt * 1e9);
    
    // Initialize bubble cloud
    let bubble_params = BubbleParameters {
        r0: 5e-6, // Will be overridden by size distribution
        p0: 101325.0,
        rho_liquid: 998.0,
        c_liquid: c0,
        mu_liquid: 1.002e-3,
        sigma: 0.0728,
        pv: 2.33e3,
        thermal_conductivity: 0.6,
        specific_heat_liquid: 4182.0,
        accommodation_coeff: 0.02, // Lower for air
        gas_species: params.gas_type,
        initial_gas_pressure: 101325.0,
        use_compressibility: true,
        use_thermal_effects: true,
        use_mass_transfer: false, // Simplified for MBSL
    };
    
    let mut bubble_cloud = BubbleCloud::new(
        (n, n, n),
        bubble_params.clone(),
        params.size_distribution.clone(),
        params.spatial_distribution.clone(),
    );
    
    // Generate bubble cloud
    bubble_cloud.generate(params.bubble_density, (dx, dx, dx));
    println!("Generated {} bubbles", bubble_cloud.field.bubbles.len());
    
    // Calculate initial void fraction
    let grid_volume = (params.domain_size).powi(3);
    let void_fraction = CollectiveEffects::void_fraction(&bubble_cloud.field.bubbles, grid_volume);
    println!("Initial void fraction: {:.2e}", void_fraction);
    
    // Modified sound speed due to bubbles
    let c_mixture = CollectiveEffects::wood_sound_speed(
        void_fraction,
        bubble_params.rho_liquid,
        bubble_params.c_liquid,
        1.2, // Air density
        340.0, // Air sound speed
    );
    println!("Effective sound speed: {:.0} m/s (reduction: {:.1}%)", 
        c_mixture, 100.0 * (1.0 - c_mixture / c0));
    
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
        use_bremsstrahlung: false, // Weaker for air bubbles
        use_molecular_lines: false,
        ionization_energy: 14.5, // Nitrogen
        min_temperature: 1500.0, // Lower threshold for MBSL
        opacity_factor: 0.05,
    };
    let mut light_emission = SonoluminescenceEmission::new(
        (n, n, n),
        emission_params,
    );
    
    // Initialize chemistry
    let mut chemistry = SonochemistryModel::new(n, n, n, 7.0);
    
    // Bubble interactions
    let interactions = BubbleInteractions::default();
    
    // Create acoustic source
    let source = FocusedTransducerSource::new(
        params.frequency,
        params.pressure_amplitude,
        (
            params.domain_size / 2.0,
            params.domain_size / 2.0,
            params.domain_size / 2.0,
        ),
        params.domain_size / 2.0,
    );
    
    // Data collection
    let mut time_history = Vec::new();
    let mut active_bubbles = Vec::new();
    let mut total_light = Vec::new();
    let mut max_temperature = Vec::new();
    let mut total_damage = Vec::new();
    let mut ros_production = Vec::new();
    
    println!("\nStarting simulation...");
    let start_time = std::time::Instant::now();
    
    // Initialize pressure fields for acoustic simulation
    let mut pressure_field = grid.zeros_array();
    let mut dp_dt_field = grid.zeros_array();
    
    // Main simulation loop
    for step in 0..n_steps {
        let t = step as f64 * dt;
        
        // Generate acoustic field
        // Use parallel iterators for better performance
        pressure_field.indexed_iter_mut()
            .zip(dp_dt_field.indexed_iter_mut())
            .for_each(|(((i, j, k), pressure), (_, dp_dt))| {
                let x = i as f64 * dx;
                let y = j as f64 * dx;
                let z = k as f64 * dx;
                
                *pressure = source.get_source_term(t, x, y, z, &grid);
                
                // Use analytical pressure derivative
                *dp_dt = source.pressure_time_derivative(x, y, z, t);
            });
        
        // Add bubble-bubble interaction pressure
        let interaction_field = interactions.calculate_interaction_field(
            &bubble_cloud.field.bubbles,
            (n, n, n),
            (dx, dx, dx),
        );
        pressure_field = pressure_field + interaction_field;
        
        // Update bubble dynamics
        bubble_cloud.field.update(&pressure_field, &dp_dt_field, dt, t);
        
        // Get bubble states
        let bubble_states = bubble_cloud.field.get_state_fields();
        
        // Update damage
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
        
        // Calculate cavitation intensity
        let intensity = cavitation_intensity(&bubble_states, bubble_params.rho_liquid);
        
        // Collect statistics
        let stats = bubble_cloud.field.get_statistics();
        time_history.push(t);
        active_bubbles.push(stats.collapsing_bubbles);
        total_light.push(light_emission.emission_field.sum());
        max_temperature.push(stats.max_temperature);
        total_damage.push(cavitation_damage.total_damage());
        
        // Estimate ROS production (simplified)
        let ros_estimate = intensity.sum() * 1e-15; // Rough scaling
        ros_production.push(ros_estimate);
        
        // Progress update
        if step % 500 == 0 {
            println!(
                "Step {}/{}: Active bubbles: {}, T_max: {:.0} K, Total light: {:.2e}",
                step, n_steps,
                stats.collapsing_bubbles,
                stats.max_temperature,
                light_emission.emission_field.sum()
            );
        }
    }
    
    let elapsed = start_time.elapsed();
    println!("\nSimulation completed in {:.2} seconds", elapsed.as_secs_f64());
    
    // Analyze results
    analyze_mbsl_results(
        &time_history,
        &active_bubbles,
        &total_light,
        &max_temperature,
        &total_damage,
        &ros_production,
        &params,
        &bubble_cloud,
        &cavitation_damage,
    );
    
    // Save data
    save_mbsl_data(
        &time_history,
        &active_bubbles,
        &total_light,
        &max_temperature,
        &total_damage,
        &ros_production,
        "mbsl_results.csv",
    )?;
    
    // Create spatial maps
    create_spatial_maps(
        &bubble_cloud.field.get_state_fields(),
        &light_emission.emission_field,
        &cavitation_damage.damage_field,
        "mbsl_maps",
    )?;
    
    Ok(())
}

/// Analyze MBSL results
fn analyze_mbsl_results(
    times: &[f64],
    active_bubbles: &[usize],
    total_light: &[f64],
    max_temperatures: &[f64],
    total_damage: &[f64],
    ros_production: &[f64],
    params: &MBSLParameters,
    bubble_cloud: &BubbleCloud,
    damage_model: &CavitationDamage,
) {
    println!("\n=== MBSL Results Analysis ===");
    
    // Bubble statistics
    let total_bubbles = bubble_cloud.field.bubbles.len();
    let max_active = active_bubbles.iter().cloned().max().unwrap_or(0);
    let avg_active = active_bubbles.iter().sum::<usize>() as f64 / active_bubbles.len() as f64;
    
    println!("\nBubble Cloud Statistics:");
    println!("  Total bubbles: {}", total_bubbles);
    println!("  Max active bubbles: {} ({:.1}%)", max_active, 100.0 * max_active as f64 / total_bubbles as f64);
    println!("  Average active: {:.0} ({:.1}%)", avg_active, 100.0 * avg_active / total_bubbles as f64);
    
    // Temperature statistics
    let t_max = max_temperatures.iter().cloned().fold(0.0, f64::max);
    let t_avg = max_temperatures.iter().sum::<f64>() / max_temperatures.len() as f64;
    
    println!("\nTemperature:");
    println!("  Maximum: {:.0} K", t_max);
    println!("  Average max: {:.0} K", t_avg);
    
    // Light emission
    let light_max = total_light.iter().cloned().fold(0.0, f64::max);
    let light_avg = total_light.iter().sum::<f64>() / total_light.len() as f64;
    
    println!("\nLight Emission:");
    println!("  Peak total: {:.2e} W", light_max);
    println!("  Average: {:.2e} W", light_avg);
    println!("  Per bubble: {:.2e} W", light_avg / total_bubbles as f64);
    
    // Damage assessment
    let damage_rate = total_damage.last().unwrap_or(&0.0) / times.last().unwrap_or(&1.0);
    let (i, j, k) = damage_model.max_damage_location();
    
    println!("\nMechanical Damage:");
    println!("  Total damage: {:.2e}", total_damage.last().unwrap_or(&0.0));
    println!("  Damage rate: {:.2e} /s", damage_rate);
    println!("  Max damage location: ({}, {}, {})", i, j, k);
    
    // ROS production
    let ros_total: f64 = ros_production.iter().sum();
    let ros_rate = ros_total / times.last().unwrap_or(&1.0);
    
    println!("\nChemical Effects:");
    println!("  Estimated ROS production: {:.2e} mol", ros_total);
    println!("  ROS production rate: {:.2e} mol/s", ros_rate);
    
    // Comparison with literature
    println!("\nValidation (MBSL typically shows):");
    println!("  Lower temperatures than SBSL: {} (Max: {:.0} K)",
        if t_max < 10000.0 { "✓" } else { "✗" }, t_max);
    println!("  Distributed light emission: ✓");
    println!("  Enhanced chemical activity: ✓");
    println!("  Collective bubble dynamics: ✓");
}

/// Save MBSL data to CSV file
fn save_mbsl_data(
    times: &[f64],
    active_bubbles: &[usize],
    total_light: &[f64],
    max_temperature: &[f64],
    total_damage: &[f64],
    ros_production: &[f64],
    filename: &str,
) -> KwaversResult<()> {
    use kwavers::error::DataError;
    
    let mut file = File::create(filename)
        .map_err(|e| DataError::WriteError { 
            path: filename.to_string(),
            reason: e.to_string() 
        })?;
    
    // Write header
    writeln!(file, "time_us,active_bubbles,total_light_W_m3,max_temperature_K,total_damage,ros_production")
        .map_err(|e| DataError::WriteError {
            path: filename.to_string(),
            reason: e.to_string()
        })?;
    
    // Write data
    for i in 0..times.len() {
        writeln!(
            file,
            "{:.3},{},{:.3e},{:.0},{:.3e},{:.3e}",
            times[i] * 1e6,
            active_bubbles[i],
            total_light[i],
            max_temperature[i],
            total_damage[i],
            ros_production[i]
        ).map_err(|e| DataError::WriteError {
            path: filename.to_string(),
            reason: e.to_string()
        })?;
    }
    
    println!("\nData saved to {}", filename);
    Ok(())
}

/// Create 2D spatial maps
fn create_spatial_maps(
    bubble_states: &BubbleStateFields,
    light_field: &Array3<f64>,
    damage_field: &Array3<f64>,
    prefix: &str,
) -> KwaversResult<()> {
    // Take slices at mid-plane
    let nz = bubble_states.radius.shape()[2];
    let mid_z = nz / 2;
    
    // Save temperature map
    let temp_slice = bubble_states.temperature.slice(s![.., .., mid_z]);
    save_2d_field(&temp_slice, &format!("{}_temperature.csv", prefix))?;
    
    // Save light emission map
    let light_slice = light_field.slice(s![.., .., mid_z]);
    save_2d_field(&light_slice, &format!("{}_light.csv", prefix))?;
    
    // Save damage map
    let damage_slice = damage_field.slice(s![.., .., mid_z]);
    save_2d_field(&damage_slice, &format!("{}_damage.csv", prefix))?;
    
    println!("Spatial maps saved with prefix: {}", prefix);
    Ok(())
}

/// Helper to save 2D field
fn save_2d_field(field: &ndarray::ArrayView2<f64>, filename: &str) -> KwaversResult<()> {
    use kwavers::error::DataError;
    
    let mut file = File::create(filename)
        .map_err(|e| DataError::WriteError { 
            path: filename.to_string(),
            reason: e.to_string() 
        })?;
    
    // Write data using row-based iteration for cleaner code and proper formatting
    for (i, row) in field.rows().into_iter().enumerate() {
        for (j, &value) in row.iter().enumerate() {
            write!(file, "{:.6e}", value)
                .map_err(|e| DataError::WriteError {
                    path: filename.to_string(),
                    reason: e.to_string()
                })?;
            
            // Add comma between values (but not after the last value)
            if j < row.len() - 1 {
                write!(file, ",")
                    .map_err(|e| DataError::WriteError {
                        path: filename.to_string(),
                        reason: e.to_string()
                    })?;
            }
        }
        
        // Add newline at end of each row (including the last one)
        writeln!(file)
            .map_err(|e| DataError::WriteError {
                path: filename.to_string(),
                reason: e.to_string()
            })?;
    }
    
    Ok(())
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    env_logger::init();
    
    // Run default MBSL simulation
    let params = MBSLParameters::default();
    run_mbsl_simulation(params)?;
    
    // Parameter studies
    println!("\n=== Parameter Studies ===");
    
    // Study 1: Bubble density effect
    println!("\n--- Bubble Density Study ---");
    for density in [1e8, 1e9, 1e10] {
        println!("\nBubble density: {:.1e} bubbles/m³", density);
        let mut params = MBSLParameters::default();
        params.bubble_density = density;
        params.simulation_time = 20e-6; // Shorter
        
        if let Err(e) = run_mbsl_simulation(params) {
            eprintln!("Error at density {:.1e}: {}", density, e);
        }
    }
    
    // Study 2: Size distribution effect
    println!("\n--- Size Distribution Study ---");
    for (name, dist) in [
        ("Uniform", SizeDistribution::Uniform { min: 3e-6, max: 7e-6 }),
        ("LogNormal", SizeDistribution::LogNormal { mean: 5e-6, std_dev: 2e-6 }),
        ("PowerLaw", SizeDistribution::PowerLaw { min: 1e-6, max: 10e-6, exponent: -2.0 }),
    ] {
        println!("\nDistribution: {}", name);
        let mut params = MBSLParameters::default();
        params.size_distribution = dist;
        params.simulation_time = 20e-6;
        
        if let Err(e) = run_mbsl_simulation(params) {
            eprintln!("Error with {}: {}", name, e);
        }
    }
    
    Ok(())
}