//! Multi-Bubble Sonoluminescence (MBSL) Example
//!
//! This example simulates multi-bubble sonoluminescence in cavitation clouds
//! based on experimental conditions from the literature:
//! - Yasui et al. (2008) "The range of ambient radius for an active bubble in sonoluminescence"
//! - Lauterborn & Kurz (2010) "Physics of bubble oscillations" Rep. Prog. Phys. 73, 106501
//! - Mettin et al. (1997) "Acoustic cavitation structures and simulations"
//!
//! MBSL characteristics:
//! - Higher frequency: 100-500 kHz typical
//! - Lower pressure threshold than SBSL
//! - Bubble clouds with interactions
//! - Broader emission spectrum
//! - Applications in sonochemistry

use kwavers::{
    Grid, Time, HomogeneousMedium, AbsorbingBoundary, Source, Sensor, Recorder,
    SensorConfig, RecorderConfig, KwaversResult,
    physics::{
        mechanics::cavitation::model::LegacyCavitationModel as CavitationModel,
        optics::sonoluminescence::{
            SonoluminescenceEmission, EmissionParameters,
        },
        chemistry::{SonochemistryModel, ROSSpecies},
    },
};
use ndarray::{Array3, Array1, Array2};
use rand::prelude::*;
use std::f64::consts::PI;

/// MBSL experimental parameters
#[derive(Debug, Clone)]
struct MBSLParameters {
    // Acoustic parameters
    frequency: f64,              // Driving frequency [Hz]
    pressure_amplitude: f64,     // Acoustic pressure amplitude [Pa]
    
    // Bubble cloud parameters
    bubble_density: f64,         // Bubbles per unit volume [m⁻³]
    size_distribution: String,   // Size distribution type
    mean_radius: f64,           // Mean bubble radius [m]
    radius_std_dev: f64,        // Standard deviation of radius [m]
    
    // Medium parameters
    water_temperature: f64,      // [K]
    dissolved_oxygen: f64,       // O₂ concentration [mg/L]
    
    // Simulation parameters
    domain_size: (f64, f64, f64), // Domain dimensions [m]
    grid_points: (usize, usize, usize), // Grid resolution
    simulation_time: f64,        // Total time [s]
}

impl Default for MBSLParameters {
    fn default() -> Self {
        Self {
            // Typical MBSL conditions
            frequency: 200e3,           // 200 kHz
            pressure_amplitude: 2.0 * 101325.0, // 2 atm
            
            // Bubble cloud
            bubble_density: 1e9,        // 10⁹ bubbles/m³
            size_distribution: "lognormal".to_string(),
            mean_radius: 10e-6,         // 10 μm mean
            radius_std_dev: 5e-6,       // 5 μm std dev
            
            // Aerated water at 25°C
            water_temperature: 298.15,   // 25°C
            dissolved_oxygen: 8.0,       // mg/L (saturated)
            
            // Larger domain for bubble cloud
            domain_size: (5e-3, 5e-3, 5e-3), // 5 mm cube
            grid_points: (128, 128, 128),     // 128³ grid
            simulation_time: 50e-6,           // 50 μs (10 cycles)
        }
    }
}

/// Focused transducer source for MBSL
struct FocusedTransducerSource {
    frequency: f64,
    amplitude: f64,
    focal_point: (f64, f64, f64),
    aperture: f64,
    f_number: f64,
}

impl Source for FocusedTransducerSource {
    fn pressure(&self, x: f64, y: f64, z: f64, t: f64) -> f64 {
        // Distance from focal point
        let dx = x - self.focal_point.0;
        let dy = y - self.focal_point.1;
        let dz = z - self.focal_point.2;
        let r = (dx*dx + dy*dy + dz*dz).sqrt();
        
        // Focused beam pattern (simplified)
        let focal_length = self.f_number * self.aperture;
        let beam_width = self.aperture * r / focal_length;
        let radial_distance = (dx*dx + dy*dy).sqrt();
        
        let spatial = if radial_distance < beam_width / 2.0 {
            1.0 / (1.0 + (r / focal_length).powi(2))
        } else {
            0.0
        };
        
        let temporal = (2.0 * PI * self.frequency * t).sin();
        
        self.amplitude * spatial * temporal
    }
    
    fn velocity_x(&self, _x: f64, _y: f64, _z: f64, _t: f64) -> f64 { 0.0 }
    fn velocity_y(&self, _x: f64, _y: f64, _z: f64, _t: f64) -> f64 { 0.0 }
    fn velocity_z(&self, _x: f64, _y: f64, _z: f64, _t: f64) -> f64 { 0.0 }
}

/// Generate bubble cloud with specified distribution
fn generate_bubble_cloud(
    grid: &Grid,
    params: &MBSLParameters,
) -> Array3<f64> {
    let mut rng = thread_rng();
    let mut bubble_field = Array3::zeros((grid.nx, grid.ny, grid.nz));
    
    // Calculate number of bubbles to place
    let volume = params.domain_size.0 * params.domain_size.1 * params.domain_size.2;
    let n_bubbles = (params.bubble_density * volume) as usize;
    
    println!("Generating bubble cloud with {} bubbles", n_bubbles);
    
    // Place bubbles randomly
    for _ in 0..n_bubbles {
        // Random position
        let i = rng.gen_range(10..grid.nx-10);
        let j = rng.gen_range(10..grid.ny-10);
        let k = rng.gen_range(10..grid.nz-10);
        
        // Random size from distribution
        let radius = match params.size_distribution.as_str() {
            "lognormal" => {
                let normal = rng.gen::<f64>() * params.radius_std_dev + params.mean_radius;
                normal.max(1e-6)
            }
            "uniform" => {
                rng.gen_range(
                    (params.mean_radius - params.radius_std_dev).max(1e-6)
                    ..(params.mean_radius + params.radius_std_dev)
                )
            }
            _ => params.mean_radius,
        };
        
        // Set bubble radius
        bubble_field[[i, j, k]] = radius;
    }
    
    // Smooth field to avoid discontinuities
    smooth_bubble_field(&mut bubble_field);
    
    bubble_field
}

/// Smooth bubble field using simple averaging
fn smooth_bubble_field(field: &mut Array3<f64>) {
    let shape = field.shape();
    let mut smoothed = field.clone();
    
    for i in 1..shape[0]-1 {
        for j in 1..shape[1]-1 {
            for k in 1..shape[2]-1 {
                if field[[i, j, k]] == 0.0 {
                    // Average non-zero neighbors
                    let mut sum = 0.0;
                    let mut count = 0;
                    
                    for di in -1..=1 {
                        for dj in -1..=1 {
                            for dk in -1..=1 {
                                let ii = (i as i32 + di) as usize;
                                let jj = (j as i32 + dj) as usize;
                                let kk = (k as i32 + dk) as usize;
                                
                                if field[[ii, jj, kk]] > 0.0 {
                                    sum += field[[ii, jj, kk]];
                                    count += 1;
                                }
                            }
                        }
                    }
                    
                    if count > 0 {
                        smoothed[[i, j, k]] = sum / count as f64 * 0.1; // Dilute
                    }
                }
            }
        }
    }
    
    *field = smoothed;
}

/// Main MBSL simulation
fn run_mbsl_simulation(params: MBSLParameters) -> KwaversResult<()> {
    println!("=== Multi-Bubble Sonoluminescence (MBSL) Simulation ===");
    println!("Simulating cavitation cloud dynamics and light emission");
    println!();
    println!("Parameters:");
    println!("  Frequency: {:.0} kHz", params.frequency / 1e3);
    println!("  Pressure: {:.2} atm", params.pressure_amplitude / 101325.0);
    println!("  Bubble density: {:.2e} m⁻³", params.bubble_density);
    println!("  Mean radius: {:.1} μm", params.mean_radius * 1e6);
    println!("  Water temp: {:.1}°C", params.water_temperature - 273.15);
    println!();
    
    // Create grid
    let (nx, ny, nz) = params.grid_points;
    let dx = params.domain_size.0 / nx as f64;
    let dy = params.domain_size.1 / ny as f64;
    let dz = params.domain_size.2 / nz as f64;
    let grid = Grid::new(nx, ny, nz, dx, dy, dz);
    
    // Create time stepping
    let c0 = 1500.0; // Sound speed in aerated water
    let dt = 0.5 * dx.min(dy).min(dz) / c0;
    let n_steps = (params.simulation_time / dt) as usize;
    let time = Time::new(dt, n_steps);
    
    println!("Grid: {}×{}×{} points", nx, ny, nz);
    println!("Resolution: {:.1} μm", dx * 1e6);
    println!("Time: {} steps, dt = {:.2} ns", n_steps, dt * 1e9);
    println!();
    
    // Create medium (aerated water)
    let medium = HomogeneousMedium::new(
        998.0,    // Density
        c0,       // Sound speed
        &grid,
        0.2,      // Some attenuation due to bubbles
        0.0,      // No dispersion
    );
    
    // Create boundary
    let boundary = AbsorbingBoundary::new(&grid, 20);
    
    // Create focused transducer source
    let source = FocusedTransducerSource {
        frequency: params.frequency,
        amplitude: params.pressure_amplitude,
        focal_point: (
            params.domain_size.0 / 2.0,
            params.domain_size.1 / 2.0,
            params.domain_size.2 / 2.0,
        ),
        aperture: params.domain_size.0 * 0.8,
        f_number: 1.0,
    };
    
    // Generate bubble cloud
    let initial_bubble_field = generate_bubble_cloud(&grid, &params);
    
    // Initialize cavitation model with bubble cloud
    let mut cavitation = CavitationModel::new(&grid, params.mean_radius);
    
    // Set initial bubble sizes
    for i in 0..nx {
        for j in 0..ny {
            for k in 0..nz {
                if initial_bubble_field[[i, j, k]] > 0.0 {
                    cavitation.radius[[i, j, k]] = initial_bubble_field[[i, j, k]];
                    cavitation.r0[[i, j, k]] = initial_bubble_field[[i, j, k]];
                }
            }
        }
    }
    
    // Initialize light emission model
    let emission_params = EmissionParameters {
        use_blackbody: true,
        use_bremsstrahlung: true,
        use_molecular_lines: false,
        ionization_energy: 13.6, // eV for air/water vapor
        min_temperature: 1500.0,  // Lower threshold for MBSL
        opacity_factor: 0.5,      // Some opacity in cloud
    };
    let mut light_emission = SonoluminescenceEmission::new(
        (nx, ny, nz),
        emission_params,
    );
    
    // Initialize chemistry model
    let mut chemistry = SonochemistryModel::new(nx, ny, nz, 7.0);
    
    // Create sensors at different locations
    let sensors = vec![
        Sensor::new(
            "focal_point".to_string(),
            nx/2, ny/2, nz/2,
            SensorConfig::default(),
        ),
        Sensor::new(
            "off_axis".to_string(),
            nx/4, ny/2, nz/2,
            SensorConfig::default(),
        ),
    ];
    
    // Create recorder
    let mut recorder = Recorder::new(
        RecorderConfig::new("mbsl_output")
            .with_interval(20)
            .with_fields(vec!["pressure", "light", "cavitation"]),
    );
    
    // Data collection
    let mut total_light_history = Vec::new();
    let mut active_bubbles_history = Vec::new();
    let mut ros_yield_history = Vec::new();
    let mut time_history = Vec::new();
    
    println!("Starting MBSL simulation...");
    let start_time = std::time::Instant::now();
    
    // Main simulation loop
    for step in 0..n_steps {
        let t = step as f64 * dt;
        
        // Update acoustic field
        let mut pressure = Array3::zeros((nx, ny, nz));
        for i in 0..nx {
            for j in 0..ny {
                for k in 0..nz {
                    let x = i as f64 * dx;
                    let y = j as f64 * dy;
                    let z = k as f64 * dz;
                    pressure[[i, j, k]] = source.pressure(x, y, z, t);
                }
            }
        }
        
        // Add bubble-bubble interaction effects (simplified)
        apply_bubble_interactions(&mut pressure, &cavitation.radius, &grid);
        
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
        
        // Update chemistry - simplified for this example
        // In full implementation would update ROS generation based on bubble states
        
        // Collect statistics
        let total_light = light_emission.emission_field.sum();
        let active_bubbles = cavitation.radius.iter()
            .filter(|&&r| r > 1e-9 && r < 100e-6)
            .count();
        let total_ros = chemistry.ros_concentrations.total_ros.sum();
        
        total_light_history.push(total_light);
        active_bubbles_history.push(active_bubbles);
        ros_yield_history.push(total_ros);
        time_history.push(t);
        
        // Record data
        if step % 20 == 0 {
            recorder.record(step, &pressure)?;
            
            // Print progress
            if step % 100 == 0 {
                println!(
                    "Step {}/{}: Active bubbles = {}, Total light = {:.2e} W/m³, ROS = {:.2e} mol/m³",
                    step, n_steps, active_bubbles, total_light, total_ros
                );
            }
        }
    }
    
    let elapsed = start_time.elapsed();
    println!("\nSimulation completed in {:.2} seconds", elapsed.as_secs_f64());
    
    // Analyze results
    analyze_mbsl_results(
        &time_history,
        &total_light_history,
        &active_bubbles_history,
        &ros_yield_history,
        &params,
    );
    
    // Save results
    save_mbsl_data(
        &time_history,
        &total_light_history,
        &active_bubbles_history,
        &ros_yield_history,
        &chemistry,
        "mbsl_output/mbsl_data.csv",
    )?;
    
    // Create spatial maps
    create_spatial_maps(&cavitation, &light_emission, &chemistry, "mbsl_output")?;
    
    Ok(())
}

/// Apply simplified bubble-bubble interactions
fn apply_bubble_interactions(
    pressure: &mut Array3<f64>,
    bubble_radii: &Array3<f64>,
    grid: &Grid,
) {
    let shape = pressure.shape();
    let interaction_range = 5; // Grid points
    
    for i in interaction_range..shape[0]-interaction_range {
        for j in interaction_range..shape[1]-interaction_range {
            for k in interaction_range..shape[2]-interaction_range {
                if bubble_radii[[i, j, k]] > 1e-9 {
                    // Sum contributions from nearby bubbles
                    let mut interaction = 0.0;
                    
                    for di in -interaction_range..=interaction_range {
                        for dj in -interaction_range..=interaction_range {
                            for dk in -interaction_range..=interaction_range {
                                if di == 0 && dj == 0 && dk == 0 { continue; }
                                
                                let ii = (i as i32 + di) as usize;
                                let jj = (j as i32 + dj) as usize;
                                let kk = (k as i32 + dk) as usize;
                                
                                if bubble_radii[[ii, jj, kk]] > 1e-9 {
                                    let distance = ((di*di + dj*dj + dk*dk) as f64).sqrt() * grid.dx;
                                    let r_other = bubble_radii[[ii, jj, kk]];
                                    
                                    // Bjerknes force approximation
                                    interaction += r_other.powi(3) / distance.powi(2);
                                }
                            }
                        }
                    }
                    
                    // Modify pressure based on interactions
                    pressure[[i, j, k]] *= 1.0 + 0.1 * interaction.tanh();
                }
            }
        }
    }
}

/// Analyze MBSL results
fn analyze_mbsl_results(
    times: &[f64],
    total_light: &[f64],
    active_bubbles: &[usize],
    ros_yield: &[f64],
    params: &MBSLParameters,
) {
    println!("\n=== MBSL Analysis ===");
    
    // Average values
    let avg_light = total_light.iter().sum::<f64>() / total_light.len() as f64;
    let avg_bubbles = active_bubbles.iter().sum::<usize>() / active_bubbles.len();
    let max_ros = ros_yield.iter().cloned().fold(0.0, f64::max);
    
    println!("Average total light emission: {:.2e} W/m³", avg_light);
    println!("Average active bubbles: {}", avg_bubbles);
    println!("Peak ROS concentration: {:.2e} mol/m³", max_ros);
    
    // Light emission per bubble
    let light_per_bubble = avg_light / avg_bubbles as f64;
    println!("Light per active bubble: {:.2e} W/m³", light_per_bubble);
    
    // Sonochemical efficiency
    let acoustic_power = params.pressure_amplitude.powi(2) / (2.0 * 998.0 * 1500.0);
    let chemical_efficiency = max_ros / acoustic_power;
    println!("Sonochemical efficiency: {:.2e} mol/J", chemical_efficiency);
    
    // Temporal characteristics
    let period = 1.0 / params.frequency;
    let n_cycles = times[times.len()-1] / period;
    println!("\nSimulated {:.1} acoustic cycles", n_cycles);
    
    // Find periodicity in light emission
    let samples_per_cycle = (period / (times[1] - times[0])) as usize;
    if samples_per_cycle < total_light.len() {
        let mut cycle_correlation = 0.0;
        for i in samples_per_cycle..total_light.len() {
            cycle_correlation += total_light[i] * total_light[i - samples_per_cycle];
        }
        cycle_correlation /= (total_light.len() - samples_per_cycle) as f64;
        println!("Cycle-to-cycle correlation: {:.3}", cycle_correlation);
    }
}

/// Save MBSL data
fn save_mbsl_data(
    times: &[f64],
    total_light: &[f64],
    active_bubbles: &[usize],
    ros_yield: &[f64],
    chemistry: &SonochemistryModel,
    filename: &str,
) -> KwaversResult<()> {
    use std::fs::File;
    use std::io::Write;
    
    let mut file = File::create(filename)?;
    writeln!(file, "# Multi-Bubble Sonoluminescence Data")?;
    writeln!(file, "# Time[s],TotalLight[W/m³],ActiveBubbles,TotalROS[mol/m³],OH[mol/m³],H2O2[mol/m³]")?;
    
    // Get average ROS concentrations
    let oh_avg = chemistry.ros_concentrations
        .get(ROSSpecies::HydroxylRadical)
        .map(|field| field.mean().unwrap_or(0.0))
        .unwrap_or(0.0);
    
    let h2o2_avg = chemistry.ros_concentrations
        .get(ROSSpecies::HydrogenPeroxide)
        .map(|field| field.mean().unwrap_or(0.0))
        .unwrap_or(0.0);
    
    for i in 0..times.len() {
        writeln!(
            file,
            "{:.9e},{:.3e},{},{:.3e},{:.3e},{:.3e}",
            times[i], total_light[i], active_bubbles[i], ros_yield[i], oh_avg, h2o2_avg
        )?;
    }
    
    println!("\nData saved to {}", filename);
    Ok(())
}

/// Create spatial distribution maps
fn create_spatial_maps(
    cavitation: &CavitationModel,
    light_emission: &SonoluminescenceEmission,
    chemistry: &SonochemistryModel,
    output_dir: &str,
) -> KwaversResult<()> {
    use std::fs::{File, create_dir_all};
    use std::io::Write;
    
    create_dir_all(output_dir)?;
    
    // Save 2D slices at center
    let shape = cavitation.radius.shape();
    let center_z = shape[2] / 2;
    
    // Bubble size distribution
    let mut file = File::create(format!("{}/bubble_distribution.csv", output_dir))?;
    writeln!(file, "# X[idx],Y[idx],Radius[m],Temperature[K],Light[W/m³]")?;
    
    for i in 0..shape[0] {
        for j in 0..shape[1] {
            let r = cavitation.radius[[i, j, center_z]];
            if r > 1e-9 {
                let t = cavitation.temperature[[i, j, center_z]];
                let l = light_emission.emission_field[[i, j, center_z]];
                writeln!(file, "{},{},{:.6e},{:.1f},{:.3e}", i, j, r, t, l)?;
            }
        }
    }
    
    // ROS distribution
    let mut file = File::create(format!("{}/ros_distribution.csv", output_dir))?;
    writeln!(file, "# X[idx],Y[idx],TotalROS[mol/m³],OxidativeStress")?;
    
    let oxidative_stress = chemistry.oxidative_stress();
    
    for i in 0..shape[0] {
        for j in 0..shape[1] {
            let ros = chemistry.ros_concentrations.total_ros[[i, j, center_z]];
            let stress = oxidative_stress[[i, j, center_z]];
            if ros > 0.0 {
                writeln!(file, "{},{},{:.6e},{:.3e}", i, j, ros, stress)?;
            }
        }
    }
    
    println!("Spatial maps saved to {}/", output_dir);
    Ok(())
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Initialize logging
    env_logger::init();
    
    // Run with default MBSL parameters
    let params = MBSLParameters::default();
    run_mbsl_simulation(params)?;
    
    // Optional: Test different conditions
    println!("\n=== Testing Different Conditions ===");
    
    // High frequency MBSL
    println!("\nHigh frequency test (500 kHz):");
    let mut params = MBSLParameters::default();
    params.frequency = 500e3;
    params.mean_radius = 5e-6; // Smaller bubbles for higher frequency
    params.simulation_time = 20e-6; // Shorter time
    run_mbsl_simulation(params)?;
    
    // High bubble density
    println!("\nHigh density test:");
    let mut params = MBSLParameters::default();
    params.bubble_density = 1e10; // 10¹⁰ bubbles/m³
    params.simulation_time = 20e-6;
    run_mbsl_simulation(params)?;
    
    Ok(())
}