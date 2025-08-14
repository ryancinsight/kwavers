//! Multi-Bubble Sonoluminescence (MBSL) Simulation
//!
//! This example demonstrates multi-bubble cavitation and sonoluminescence
//! in a bubble cloud, including collective effects and bubble-bubble interactions.
//!
//! Physics modeled:
//! - Bubble cloud dynamics with size distribution
//! - Collective oscillation effects  
//! - Bubble-bubble interactions via Bjerknes forces
//! - Multiple sonoluminescence sites
//! - Cavitation damage potential
//!
//! References:
//! - Yasui et al. (2008) "Bubble dynamics and sonoluminescence"
//! - Mettin (2007) "Bubble structures in acoustic cavitation"
//! - Lauterborn & Kurz (2010) "Physics of bubble oscillations"

use kwavers::{
    Grid, KwaversResult, KwaversError,
    source::{Source, BowlTransducer, BowlConfig},
    signal::{Signal, SineWave},
    physics::{
        bubble_dynamics::{
            BubbleParameters, BubbleCloud, BubbleStateFields,
            GasSpecies, BubbleInteractions, CollectiveEffects,
        },
        state::PhysicsState,
    },
    solver::pstd::{PstdSolver, PstdConfig},
    recorder::Recorder,
    constants::PI,
};
use ndarray::{Array3, s};
use std::sync::Arc;
use std::f64::consts::PI as PI2;
use std::fs::File;
use std::io::Write;

/// MBSL simulation parameters
#[derive(Debug, Clone)]
struct MBSLParameters {
    // Acoustic field
    frequency: f64,              // Driving frequency (Hz)
    pressure_amplitude: f64,     // Pressure amplitude (Pa)
    
    // Bubble cloud parameters
    bubble_density: f64,         // Number density (bubbles/m³)
    mean_radius: f64,            // Mean bubble radius (m)
    size_std_dev: f64,           // Size distribution std dev (m)
    gas_type: GasSpecies,
    
    // Domain and simulation
    domain_size: f64,            // Cubic domain size (m)
    grid_points: usize,          // Grid points per dimension
    simulation_time: f64,        // Total simulation time (s)
    
    // Transducer parameters (for BowlTransducer)
    transducer_diameter: f64,
    focal_length: f64,
}

impl Default for MBSLParameters {
    fn default() -> Self {
        Self {
            // 26.5 kHz for strong MBSL
            frequency: 26.5e3,
            pressure_amplitude: 1.5 * 101325.0, // 1.5 atm
            
            // Bubble cloud
            bubble_density: 1e9,                // 10⁹ bubbles/m³
            mean_radius: 5e-6,                  // 5 μm mean radius
            size_std_dev: 2e-6,                 // 2 μm std dev
            gas_type: GasSpecies::Air,
            
            // Larger domain for cloud
            domain_size: 0.01,                  // 1 cm
            grid_points: 64,
            simulation_time: 100e-6,            // 100 μs
            
            // Transducer parameters
            transducer_diameter: 0.01,          // 1 cm diameter
            focal_length: 0.005,                 // 5 mm focal length
        }
    }
}

/// Create focused transducer source using library's BowlTransducer
fn create_focused_source(params: &MBSLParameters, grid: &Grid) -> Box<dyn Source> {
    let config = BowlConfig {
        diameter: params.transducer_diameter,
        frequency: params.frequency,
        source_strength: params.pressure_amplitude,
        focal_length: Some(params.focal_length),
        position: (grid.nx as f64 * grid.dx / 2.0, 0.0, grid.nz as f64 * grid.dz / 2.0),
        direction: (0.0, 1.0, 0.0),
    };
    
    Box::new(BowlTransducer::new(config, Arc::new(SineWave::new(params.frequency, params.pressure_amplitude, 0.0))))
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
        r0: params.mean_radius,
        p0: 101325.0,
        rho_liquid: 998.0,
        c_liquid: c0,
        mu_liquid: 1.002e-3,
        sigma: 0.0728,
        pv: 2.33e3,
        thermal_conductivity: 0.6,
        specific_heat_liquid: 4182.0,
        accommodation_coeff: 0.02,
        gas_species: params.gas_type,
        initial_gas_pressure: 101325.0,
        use_compressibility: true,
        use_thermal_effects: true,
        use_mass_transfer: false,
    };
    
    // Create bubble cloud with size distribution
    let mut bubble_cloud = BubbleCloud::new(
        (n, n, n),
        bubble_params.clone(),
    );
    
    // Generate bubble distribution
    bubble_cloud.generate(params.bubble_density, (dx, dx, dx));
    
    // Initialize fields
    let mut pressure = Array3::zeros((n, n, n));
    let mut velocity_x = Array3::zeros((n, n, n));
    let mut velocity_y = Array3::zeros((n, n, n));
    let mut velocity_z = Array3::zeros((n, n, n));
    
    // Create focused source using library implementation
    let source = create_focused_source(&params, &grid);
    
    // Create solver
    let solver_config = PstdConfig {
        cfl_number: 0.3,
        pml_size: 10,
        pml_alpha: 2.0,
        use_staggered_grid: true,
    };
    let mut solver = PstdSolver::new(solver_config);
    
    // Create recorder
    let mut recorder = Recorder::new();
    recorder.add_scalar_field("pressure".to_string(), pressure.clone());
    
    // Data collection
    let mut time_history = Vec::new();
    let mut max_pressures = Vec::new();
    let mut mean_radii = Vec::new();
    let mut luminescence_events = Vec::new();
    
    println!("\nRunning simulation...");
    let mut last_progress = 0;
    
    for step in 0..n_steps {
        let t = step as f64 * dt;
        
        // Progress indicator
        let progress = (step * 100) / n_steps;
        if progress > last_progress + 10 {
            println!("  Progress: {}%", progress);
            last_progress = progress;
        }
        
        // Add source term
        for i in 0..n {
            for j in 0..n {
                for k in 0..n {
                    let x = i as f64 * dx;
                    let y = j as f64 * dx;
                    let z = k as f64 * dx;
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
        
        // Detect sonoluminescence events (simplified)
        let bubble_states = bubble_cloud.get_state_fields();
        for i in 0..n {
            for j in 0..n {
                for k in 0..n {
                    let r = bubble_states.radius[[i, j, k]];
                    let t_gas = bubble_states.temperature[[i, j, k]];
                    
                    // Simple SL detection: high temperature during collapse
                    if t_gas > 5000.0 && r < params.mean_radius * 0.1 {
                        luminescence_events.push((t, i, j, k, t_gas));
                    }
                }
            }
        }
        
        // Update acoustic field (simplified - would use proper solver)
        solver.step(
            &mut pressure,
            &mut velocity_x,
            &mut velocity_y,
            &mut velocity_z,
            &grid,
            dt,
        )?;
        
        // Record data
        time_history.push(t);
        max_pressures.push(pressure.iter().fold(0.0f64, |a, &b| a.max(b.abs())));
        
        let stats = bubble_cloud.get_statistics();
        mean_radii.push(stats.mean_radius);
        
        // Record fields periodically
        if step % 100 == 0 {
            recorder.record(step as f64);
        }
    }
    
    // Analysis
    println!("\n=== Simulation Results ===");
    println!("Total SL events: {}", luminescence_events.len());
    
    if !luminescence_events.is_empty() {
        let max_temp = luminescence_events.iter()
            .map(|(_, _, _, _, t)| *t)
            .fold(0.0f64, |a, b| a.max(b));
        println!("Maximum gas temperature: {:.0} K", max_temp);
        
        // Spatial distribution of SL events
        let mut spatial_dist = Array3::zeros((n, n, n));
        for (_, i, j, k, _) in &luminescence_events {
            spatial_dist[[*i, *j, *k]] += 1.0;
        }
        
        let max_events = spatial_dist.iter().fold(0.0f64, |a, &b| a.max(b));
        if max_events > 0.0 {
            println!("Maximum SL events at single location: {:.0}", max_events);
        }
    }
    
    // Save results
    save_mbsl_results(&params, &time_history, &max_pressures, &mean_radii, &luminescence_events)?;
    
    println!("\n✅ MBSL simulation completed successfully!");
    
    Ok(())
}

/// Save MBSL results to file
fn save_mbsl_results(
    params: &MBSLParameters,
    time_history: &[f64],
    max_pressures: &[f64],
    mean_radii: &[f64],
    luminescence_events: &[(f64, usize, usize, usize, f64)],
) -> KwaversResult<()> {
    let mut file = File::create("mbsl_results.dat")
        .map_err(|e| KwaversError::Io(format!("Failed to create file: {}", e)))?;
    
    writeln!(file, "# Multi-Bubble Sonoluminescence Results")
        .map_err(|e| KwaversError::Io(format!("Write error: {}", e)))?;
    writeln!(file, "# Frequency: {} kHz", params.frequency / 1e3)
        .map_err(|e| KwaversError::Io(format!("Write error: {}", e)))?;
    writeln!(file, "# Pressure: {} atm", params.pressure_amplitude / 101325.0)
        .map_err(|e| KwaversError::Io(format!("Write error: {}", e)))?;
    writeln!(file, "# Bubble density: {} bubbles/m³", params.bubble_density)
        .map_err(|e| KwaversError::Io(format!("Write error: {}", e)))?;
    writeln!(file, "#")
        .map_err(|e| KwaversError::Io(format!("Write error: {}", e)))?;
    writeln!(file, "# Time(s) MaxPressure(Pa) MeanRadius(m)")
        .map_err(|e| KwaversError::Io(format!("Write error: {}", e)))?;
    
    for i in 0..time_history.len() {
        writeln!(file, "{:.6e} {:.6e} {:.6e}", 
                time_history[i], max_pressures[i], mean_radii[i])
            .map_err(|e| KwaversError::Io(format!("Write error: {}", e)))?;
    }
    
    // Save SL events separately
    if !luminescence_events.is_empty() {
        let mut sl_file = File::create("mbsl_sl_events.dat")
            .map_err(|e| KwaversError::Io(format!("Failed to create file: {}", e)))?;
        
        writeln!(sl_file, "# Sonoluminescence Events")
            .map_err(|e| KwaversError::Io(format!("Write error: {}", e)))?;
        writeln!(sl_file, "# Time(s) i j k Temperature(K)")
            .map_err(|e| KwaversError::Io(format!("Write error: {}", e)))?;
        
        for (t, i, j, k, temp) in luminescence_events {
            writeln!(sl_file, "{:.6e} {} {} {} {:.0}", t, i, j, k, temp)
                .map_err(|e| KwaversError::Io(format!("Write error: {}", e)))?;
        }
    }
    
    Ok(())
}

/// Analyze field statistics
fn analyze_field(field: &Array3<f64>, name: &str) {
    let min = field.iter().fold(f64::INFINITY, |a, &b| a.min(b));
    let max = field.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));
    let mean = field.iter().sum::<f64>() / field.len() as f64;
    
    println!("  {}: min={:.3e}, max={:.3e}, mean={:.3e}", name, min, max, mean);
}

fn main() -> KwaversResult<()> {
    env_logger::init();
    
    println!("Multi-Bubble Sonoluminescence (MBSL) Simulation");
    println!("================================================");
    println!();
    
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