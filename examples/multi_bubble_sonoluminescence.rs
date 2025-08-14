//! Multi-bubble sonoluminescence (MBSL) simulation
//! 
//! This example demonstrates multi-bubble sonoluminescence using:
//! - Proper bubble cloud dynamics
//! - Complete sonoluminescence detection from library
//! - Proper acoustic solver from library
//! - Collective bubble effects

use kwavers::{
    Grid, KwaversResult,
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
use ndarray::{Array3, s};
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

/// Run MBSL simulation
fn run_mbsl_simulation(params: MBSLParameters) -> KwaversResult<()> {
    println!("Multi-Bubble Sonoluminescence Simulation");
    println!("=========================================");
    println!("Frequency: {} kHz", params.frequency / 1e3);
    println!("Pressure: {} bar", params.pressure_amplitude / 1e5);
    println!("Bubble density: {:.2e} bubbles/m³", params.bubble_density);
    println!();
    
    // Create grid
    let n = params.grid_size;
    let grid = Grid::new(n, n, n, params.grid_spacing, params.grid_spacing, params.grid_spacing);
    
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
    
    // Initialize fields
    let mut pressure = Array3::zeros((n, n, n));
    let mut velocity_x = Array3::zeros((n, n, n));
    let mut velocity_y = Array3::zeros((n, n, n));
    let mut velocity_z = Array3::zeros((n, n, n));
    
    // Initialize sonoluminescence detector
    let detector_config = DetectorConfig {
        spectral_analysis: true,
        time_resolved: true,
        temperature_threshold: 5000.0, // 5000 K threshold
        pressure_threshold: 1e6, // 1 MPa threshold
        compression_threshold: 5.0, // 5x compression
        spatial_resolution: params.grid_spacing,
        time_resolution: params.dt * 10.0,
    };
    
    let mut sl_detector = SonoluminescenceDetector::new(
        (n, n, n),
        (params.grid_spacing, params.grid_spacing, params.grid_spacing),
        detector_config,
    );
    
    // Create plugin manager and register detector
    // Note: Would register sl_detector as plugin here if PluginManager supported it
    
    // Simulation variables
    let num_steps = (params.simulation_time / params.dt) as usize;
    let output_interval = num_steps / 100; // Output 100 times
    
    let c0 = params.sound_speed;
    let dt = params.dt;
    
    println!("Starting simulation with {} time steps...", num_steps);
    
    // Main simulation loop
    for step in 0..num_steps {
        let t = step as f64 * dt;
        
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
        
        // Update bubble cloud with acoustic pressure
        let interactions = BubbleInteractions::default();
        let collective = CollectiveEffects::new(params.bubble_density);
        
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
        
        // Detect sonoluminescence events using complete detector
        let bubble_states = bubble_cloud.get_state_fields();
        let sl_events = sl_detector.detect_events(&bubble_states, &pressure, &initial_radius, dt);
        
        // Update acoustic field using proper PSTD solver
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
        
        // Output progress
        if step % output_interval == 0 {
            let progress = (step as f64 / num_steps as f64) * 100.0;
            let stats = sl_detector.get_statistics();
            println!(
                "Progress: {:.1}%, Time: {:.3} ms, SL Events: {}, Total Photons: {:.2e}",
                progress,
                t * 1e3,
                stats.total_events,
                stats.total_photons,
            );
        }
    }
    
    // Final statistics
    println!("\nSimulation Complete!");
    println!("====================");
    
    let final_stats = sl_detector.get_statistics();
    println!("Total SL Events: {}", final_stats.total_events);
    println!("Total Photons: {:.2e}", final_stats.total_photons);
    println!("Total Energy: {:.2e} J", final_stats.total_energy);
    println!("Max Temperature: {:.0} K", final_stats.max_temperature);
    println!("Avg Temperature: {:.0} K", final_stats.avg_temperature);
    println!("Event Rate: {:.2e} events/s", final_stats.event_rate);
    
    // Save results
    save_results(&sl_detector, &bubble_cloud, &params)?;
    
    Ok(())
}

/// Save simulation results
fn save_results(
    detector: &SonoluminescenceDetector,
    bubble_cloud: &BubbleCloud,
    params: &MBSLParameters,
) -> KwaversResult<()> {
    // Save event data
    let mut file = File::create("mbsl_events.csv")?;
    writeln!(file, "time,x,y,z,temperature,pressure,photons,wavelength,energy")?;
    
    for event in detector.get_events() {
        writeln!(
            file,
            "{},{},{},{},{},{},{},{},{}",
            event.time,
            event.physical_position.0,
            event.physical_position.1,
            event.physical_position.2,
            event.peak_temperature,
            event.peak_pressure,
            event.photon_count,
            event.peak_wavelength,
            event.energy,
        )?;
    }
    
    println!("\nResults saved to mbsl_events.csv");
    
    Ok(())
}

fn main() -> KwaversResult<()> {
    // Initialize logging
    env_logger::init();
    
    // Run with default parameters
    let params = MBSLParameters::default();
    run_mbsl_simulation(params)?;
    
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