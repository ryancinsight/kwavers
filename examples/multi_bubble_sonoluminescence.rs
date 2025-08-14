//! Multi-bubble sonoluminescence (MBSL) simulation
//! 
//! This example demonstrates multi-bubble sonoluminescence using:
//! - Proper bubble cloud dynamics
//! - Complete sonoluminescence detection from library
//! - Proper acoustic solver from library
//! - Collective bubble effects
//! - Full data acquisition with cavitation and thermal monitoring

use kwavers::{
    Grid, KwaversResult,
    recorder::{Recorder, RecorderConfig},
    sensor::Sensor,
    time::Time,
    source::{Source, BowlTransducer},
    signal::{Signal, SineWave},
    physics::{
        bubble_dynamics::{
            BubbleCloud, BubbleCloudConfig,
            BubbleInteractions, CollectiveEffects,
            BubbleStateFields,
        },
        sonoluminescence_detector::{
            SonoluminescenceDetector, DetectorConfig,
            SonoluminescenceEvent, SonoluminescenceStatistics,
        },
    },
    solver::{
        pstd::{PSTDSolver, PSTDConfig},
        Solver,
    },
    config::BowlConfig,
    constants::PI,
};
use ndarray::{Array3, Array4, Axis, s};
use std::sync::Arc;
use std::fs::File;
use std::io::Write;

/// MBSL simulation parameters
#[derive(Debug, Clone)]
struct MBSLParameters {
    // Acoustic parameters
    frequency: f64,
    pressure_amplitude: f64,
    
    // Transducer parameters
    transducer_diameter: f64,
    focal_length: f64,
    
    // Bubble cloud parameters
    mean_radius: f64,
    size_std_dev: f64,
    bubble_density: f64,
    cloud_radius: f64,
    
    // Medium properties
    sound_speed: f64,
    density: f64,
    temperature: f64,
    
    // Simulation parameters
    grid_size: usize,
    grid_spacing: f64,
    simulation_time: f64,
    dt: f64,
    
    // Data acquisition
    enable_full_monitoring: bool,
}

impl Default for MBSLParameters {
    fn default() -> Self {
        Self {
            // 26.5 kHz for MBSL (Yasui et al.)
            frequency: 26.5e3,
            pressure_amplitude: 1.5e5, // 1.5 bar
            
            // Bowl transducer
            transducer_diameter: 0.1, // 10 cm
            focal_length: 0.15, // 15 cm focal length
            
            // Bubble cloud
            mean_radius: 5e-6, // 5 μm mean radius
            size_std_dev: 2e-6, // 2 μm standard deviation
            bubble_density: 1e9, // 10^9 bubbles/m³
            cloud_radius: 0.01, // 1 cm cloud radius
            
            // Water properties
            sound_speed: 1500.0,
            density: 1000.0,
            temperature: 293.15, // 20°C
            
            // Grid parameters
            grid_size: 128,
            grid_spacing: 1e-3, // 1 mm
            simulation_time: 1e-3, // 1 ms
            dt: 1e-7, // 100 ns time step
            
            // Enable comprehensive monitoring
            enable_full_monitoring: true,
        }
    }
}

/// Create focused transducer source
fn create_focused_source(params: &MBSLParameters, grid: &Grid) -> BowlTransducer {
    let signal = Arc::new(SineWave::new(
        params.frequency,
        params.pressure_amplitude,
        0.0,
    ));
    
    let config = BowlConfig {
        position: (
            grid.nx as f64 * grid.dx / 2.0,
            grid.ny as f64 * grid.dy / 2.0,
            0.0, // Bottom of grid
        ),
        direction: (0.0, 0.0, 1.0), // Pointing up
        diameter: params.transducer_diameter,
        focal_length: params.focal_length,
        frequency: params.frequency,
    };
    
    BowlTransducer::new(config, signal)
}

/// Run MBSL simulation with full data acquisition
fn run_mbsl_simulation(params: MBSLParameters) -> KwaversResult<()> {
    println!("Multi-Bubble Sonoluminescence Simulation");
    println!("=========================================");
    println!("Frequency: {} kHz", params.frequency / 1e3);
    println!("Pressure: {} bar", params.pressure_amplitude / 1e5);
    println!("Bubble density: {:.2e} bubbles/m³", params.bubble_density);
    println!("Full monitoring: {}", if params.enable_full_monitoring { "✓" } else { "✗" });
    println!();
    
    // Create grid
    let n = params.grid_size;
    let grid = Grid::new(n, n, n, params.grid_spacing, params.grid_spacing, params.grid_spacing);
    
    // Create time configuration
    let time = Time::new(0.0, params.simulation_time, params.dt);
    
    // Setup sensors for data acquisition
    let sensor_positions = vec![
        // Focus region monitoring
        (n/2, n/2, n/2),       // Center
        (n/2, n/2, 3*n/4),     // Above center
        (n/2, n/2, n/4),       // Below center
        // Cloud boundary monitoring
        (n/2 + n/8, n/2, n/2),
        (n/2 - n/8, n/2, n/2),
        (n/2, n/2 + n/8, n/2),
        (n/2, n/2 - n/8, n/2),
    ];
    
    let sensor = Sensor::new(&grid, &time, sensor_positions);
    
    // Configure comprehensive data acquisition
    let sl_detector_config = DetectorConfig {
        spectral_analysis: true,
        time_resolved: true,
        temperature_threshold: 5000.0, // 5000 K for SL
        pressure_threshold: 1e6, // 1 MPa
        compression_threshold: 5.0, // 5x compression
        spatial_resolution: params.grid_spacing,
        time_resolution: params.dt * 10.0,
    };
    
    let recorder_config = RecorderConfig::new("mbsl_output")
        .with_pressure_recording(true)
        .with_light_recording(true)  // Records general light field
        .with_temperature_recording(params.enable_full_monitoring)
        .with_cavitation_detection(params.enable_full_monitoring, -0.5e5) // -0.5 bar threshold
        .with_sonoluminescence_detection(true, Some(sl_detector_config)) // Specific SL events
        .with_snapshot_interval(100);
    
    let mut recorder = Recorder::from_config(sensor, &time, &recorder_config);
    
    // Initialize bubble cloud
    let cloud_config = BubbleCloudConfig {
        mean_radius: params.mean_radius,
        size_std_dev: params.size_std_dev,
        bubble_density: params.bubble_density,
        cloud_radius: params.cloud_radius,
        ambient_pressure: 101325.0,
        surface_tension: 0.072,
        viscosity: 1e-3,
        polytropic_index: 1.4,
    };
    
    let cloud_center = (
        grid.nx as f64 * grid.dx / 2.0,
        grid.ny as f64 * grid.dy / 2.0,
        grid.nz as f64 * grid.dz / 2.0,
    );
    
    let mut bubble_cloud = BubbleCloud::new(cloud_config, cloud_center, &grid)?;
    
    // Store initial bubble radii for compression ratio calculation
    let initial_radius = bubble_cloud.get_state_fields().radius.clone();
    
    // Create acoustic source
    let source = create_focused_source(&params, &grid);
    
    // Initialize PSTD solver with proper configuration
    let solver_config = PSTDConfig {
        cfl_number: 0.5,
        pml_thickness: 10,
        pml_alpha: 2.0,
        enable_nonlinear: true, // Enable for MBSL
        enable_absorption: true,
        enable_dispersion: false,
    };
    
    let mut solver = PSTDSolver::new(solver_config, &grid)?;
    
    // Initialize fields (4D array: [field_type, nx, ny, nz])
    // Field indices: 0=pressure, 1=light, 2=temperature, 3=bubble_radius
    let mut fields = Array4::zeros((4, n, n, n));
    
    // Initialize velocity fields for solver
    let mut velocity_x = Array3::zeros((n, n, n));
    let mut velocity_y = Array3::zeros((n, n, n));
    let mut velocity_z = Array3::zeros((n, n, n));
    
    // Simulation variables
    let num_steps = (params.simulation_time / params.dt) as usize;
    let output_interval = num_steps / 100; // Output 100 times
    
    let c0 = params.sound_speed;
    let dt = params.dt;
    
    println!("Starting simulation with {} time steps...", num_steps);
    println!("Data acquisition:");
    println!("  - Pressure: ✓");
    println!("  - Light field: ✓ (continuous optical field)");
    println!("  - SL detection: ✓ (discrete bubble collapse events)");
    if params.enable_full_monitoring {
        println!("  - Cavitation: ✓");
        println!("  - Temperature: ✓");
        println!("  - Thermal dose: ✓");
    }
    println!();
    
    // Main simulation loop
    for step in 0..num_steps {
        let t = step as f64 * dt;
        
        // Extract current pressure field
        let mut pressure = fields.index_axis_mut(Axis(0), 0);
        
        // Add source contribution
        for i in 0..n {
            for j in 0..n {
                for k in 0..n {
                    let x = i as f64 * grid.dx;
                    let y = j as f64 * grid.dy;
                    let z = k as f64 * grid.dz;
                    pressure[[i, j, k]] += source.get_source_term(t, x, y, z, &grid) * dt;
                }
            }
        }
        drop(pressure);
        
        // Update bubble cloud with acoustic pressure
        let interactions = BubbleInteractions::default();
        let collective = CollectiveEffects::new(params.bubble_density);
        
        // Get current pressure for bubble dynamics
        let pressure = fields.index_axis(Axis(0), 0).to_owned();
        
        // Get bubble state fields for collective effects
        let state_fields = bubble_cloud.get_state_fields();
        
        // Calculate collective pressure modification
        let p_collective = collective.compute_collective_pressure(
            &state_fields.radius,
            &state_fields.velocity,
            &pressure,
            c0,
        );
        
        // Update bubble dynamics
        bubble_cloud.update(
            &pressure,
            &p_collective,
            dt,
            &grid,
            &interactions,
        );
        
        // Get updated bubble states
        let bubble_states = bubble_cloud.get_state_fields();
        
        // Update temperature and bubble radius fields
        fields.index_axis_mut(Axis(0), 2).assign(&bubble_states.temperature);
        fields.index_axis_mut(Axis(0), 3).assign(&bubble_states.radius);
        
        // Calculate light emission from sonoluminescence
        // This is the CONTINUOUS light field that results from SL events
        let mut light_field = fields.index_axis_mut(Axis(0), 1);
        for i in 0..n {
            for j in 0..n {
                for k in 0..n {
                    let temp = bubble_states.temperature[[i, j, k]];
                    let radius = bubble_states.radius[[i, j, k]];
                    
                    // If conditions for SL are met, add light to the field
                    if temp > 5000.0 && radius > 0.0 {
                        // Stefan-Boltzmann radiation for total power
                        let sigma = 5.67e-8;
                        let surface_area = 4.0 * std::f64::consts::PI * radius.powi(2);
                        let power = sigma * surface_area * temp.powi(4);
                        
                        // Add to light field (this accumulates over time)
                        light_field[[i, j, k]] += power * dt;
                    }
                }
            }
        }
        drop(light_field);
        
        // Record all data (includes both light field AND SL event detection)
        recorder.record(&fields, step, t);
        
        // Update acoustic field using proper PSTD solver
        let mut pressure = fields.index_axis(Axis(0), 0).to_owned();
        solver.step(
            &mut pressure,
            &mut velocity_x,
            &mut velocity_y,
            &mut velocity_z,
            dt,
        )?;
        
        // Apply bubble-induced pressure modifications
        let bubble_pressure = bubble_cloud.compute_scattered_pressure(&grid, c0);
        pressure = pressure + bubble_pressure;
        
        // Store updated pressure back
        fields.index_axis_mut(Axis(0), 0).assign(&pressure);
        
        // Output progress
        if step % output_interval == 0 {
            let progress = (step as f64 / num_steps as f64) * 100.0;
            let stats = &recorder.statistics;
            println!(
                "Progress: {:.1}%, Time: {:.3} ms",
                progress,
                t * 1e3,
            );
            println!(
                "  SL Events: {} (discrete collapse events)",
                stats.total_sl_events,
            );
            println!(
                "  Total Photons: {:.2e}",
                stats.total_sl_photons,
            );
            println!(
                "  Max Light Intensity: {:.2e} W/m² (continuous field)",
                stats.max_light_intensity,
            );
            if params.enable_full_monitoring {
                println!(
                    "  Cavitation Events: {}",
                    stats.total_cavitation_events,
                );
                println!(
                    "  Max Temperature: {:.0} K",
                    stats.max_temperature,
                );
            }
            println!();
        }
    }
    
    // Final statistics
    println!("Simulation Complete!");
    println!("====================");
    
    let final_stats = &recorder.statistics;
    println!("Sonoluminescence Statistics:");
    println!("  Total SL Events: {} (discrete bubble collapses)", final_stats.total_sl_events);
    println!("  Total Photons: {:.2e}", final_stats.total_sl_photons);
    println!("  Total Energy: {:.2e} J", final_stats.total_sl_energy);
    println!();
    
    println!("Light Field Statistics:");
    println!("  Max Light Intensity: {:.2e} W/m² (continuous field)", final_stats.max_light_intensity);
    println!();
    
    println!("Acoustic Statistics:");
    println!("  Max Pressure: {:.2e} Pa", final_stats.max_pressure);
    println!("  Min Pressure: {:.2e} Pa", final_stats.min_pressure);
    
    if params.enable_full_monitoring {
        println!();
        println!("Additional Monitoring:");
        println!("  Total Cavitation Events: {}", final_stats.total_cavitation_events);
        println!("  Max Temperature: {:.0} K", final_stats.max_temperature);
    }
    
    // Save all recorded data
    recorder.save()?;
    println!("\nData saved to mbsl_output files:");
    println!("  - mbsl_output.csv: Sensor time series");
    println!("  - mbsl_output_sonoluminescence.csv: SL events");
    if params.enable_full_monitoring {
        println!("  - mbsl_output_cavitation.csv: Cavitation events");
        println!("  - mbsl_output_thermal.csv: Thermal events");
    }
    
    // Export specialized maps
    if let Some(sl_map) = recorder.sonoluminescence_intensity_map() {
        let max_intensity = sl_map.iter().fold(0.0, |a, &b| a.max(b));
        println!("\nSL Intensity Map:");
        println!("  Max cumulative photons at single location: {:.2e}", max_intensity);
        
        // Find hotspots
        let mut hotspots = Vec::new();
        for i in 0..n {
            for j in 0..n {
                for k in 0..n {
                    if sl_map[[i, j, k]] > max_intensity * 0.5 {
                        hotspots.push((i, j, k, sl_map[[i, j, k]]));
                    }
                }
            }
        }
        println!("  Number of SL hotspots (>50% max): {}", hotspots.len());
    }
    
    if params.enable_full_monitoring {
        if let Some(cavitation_map) = recorder.cavitation_map() {
            let max_events = cavitation_map.iter().fold(0.0, |a, &b| a.max(b));
            println!("\nCavitation Map:");
            println!("  Max events at single location: {:.0}", max_events);
        }
        
        if let Some(thermal_dose) = recorder.thermal_dose_map() {
            let max_dose = thermal_dose.iter().fold(0.0, |a, &b| a.max(b));
            println!("\nThermal Dose Map:");
            println!("  Max CEM43: {:.2e} min", max_dose);
        }
    }
    
    // Analyze spatial distribution of SL
    analyze_sl_distribution(&recorder, &grid);
    
    Ok(())
}

/// Analyze spatial distribution of sonoluminescence events
fn analyze_sl_distribution(recorder: &Recorder, grid: &Grid) {
    println!("\nSpatial Analysis of Sonoluminescence:");
    println!("=====================================");
    
    if recorder.sl_events.is_empty() {
        println!("No SL events detected.");
        return;
    }
    
    // Calculate center of mass of SL events
    let mut x_sum = 0.0;
    let mut y_sum = 0.0;
    let mut z_sum = 0.0;
    let mut total_photons = 0.0;
    
    for event in &recorder.sl_events {
        let weight = event.photon_count;
        x_sum += event.physical_position.0 * weight;
        y_sum += event.physical_position.1 * weight;
        z_sum += event.physical_position.2 * weight;
        total_photons += weight;
    }
    
    if total_photons > 0.0 {
        let x_center = x_sum / total_photons;
        let y_center = y_sum / total_photons;
        let z_center = z_sum / total_photons;
        
        println!("  SL emission center of mass (weighted by photons):");
        println!("    X: {:.3} mm", x_center * 1e3);
        println!("    Y: {:.3} mm", y_center * 1e3);
        println!("    Z: {:.3} mm", z_center * 1e3);
        
        // Calculate spread
        let mut r_sum = 0.0;
        for event in &recorder.sl_events {
            let dx = event.physical_position.0 - x_center;
            let dy = event.physical_position.1 - y_center;
            let dz = event.physical_position.2 - z_center;
            let r = (dx*dx + dy*dy + dz*dz).sqrt();
            r_sum += r * event.photon_count;
        }
        
        let r_avg = r_sum / total_photons;
        println!("  Average distance from center: {:.3} mm", r_avg * 1e3);
    }
    
    // Temporal distribution
    if !recorder.sl_events.is_empty() {
        let first_event = recorder.sl_events.iter().map(|e| e.time).fold(f64::INFINITY, f64::min);
        let last_event = recorder.sl_events.iter().map(|e| e.time).fold(0.0, f64::max);
        let duration = last_event - first_event;
        
        println!("\n  Temporal distribution:");
        println!("    First event: {:.3} μs", first_event * 1e6);
        println!("    Last event: {:.3} μs", last_event * 1e6);
        println!("    Active duration: {:.3} μs", duration * 1e6);
        
        if duration > 0.0 {
            let event_rate = recorder.sl_events.len() as f64 / duration;
            println!("    Event rate: {:.2e} events/s", event_rate);
        }
    }
}

fn main() -> KwaversResult<()> {
    // Initialize logging
    env_logger::init();
    
    // Run with default parameters
    let params = MBSLParameters::default();
    run_mbsl_simulation(params)?;
    
    println!("\n✓ MBSL simulation with full data acquisition complete!");
    
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_mbsl_parameters() {
        let params = MBSLParameters::default();
        assert!(params.frequency > 0.0);
        assert!(params.pressure_amplitude > 0.0);
        assert!(params.bubble_density > 0.0);
        assert!(params.mean_radius > 0.0);
    }
}