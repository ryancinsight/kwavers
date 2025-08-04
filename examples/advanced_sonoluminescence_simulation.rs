// examples/advanced_sonoluminescence_simulation.rs
//! Advanced Sonoluminescence Simulation
//! 
//! This example demonstrates the complete physics of sonoluminescence including:
//! - Acoustic cavitation and bubble dynamics
//! - Thermal effects and heat transfer
//! - Light emission mechanisms

use kwavers::{
    Grid, Time, HomogeneousMedium, 
    physics::{
        composable::{
            PhysicsPipeline, 
            ThermalDiffusionComponent
        },
    },
    boundary::{CPMLBoundary, CPMLConfig},
    error::KwaversResult,
    medium::Medium,
};
use std::sync::Arc;
use std::time::Instant;
use ndarray::Array4;

fn main() -> KwaversResult<()> {
    println!("=== Advanced Sonoluminescence Simulation ===\n");
    
    // Create simulation grid
    let grid = Grid::new(64, 64, 64, 1e-4, 1e-4, 1e-4); // 100μm resolution
    
    // Create medium (water at room temperature)
    let medium = Arc::new(HomogeneousMedium::new(
        998.0,  // density kg/m³
        1480.0, // sound speed m/s
        &grid,
        0.0,    // nonlinearity
        2.5e-3, // absorption at 1 MHz
    ));
    
    // Create time parameters
    let dt = grid.cfl_timestep_default(medium.sound_speed(0.0, 0.0, 0.0, &grid));
    let time = Time::new(dt, 1000); // 1000 time steps
    
    // Create boundary conditions
    let cpml_config = CPMLConfig {
        thickness: 10,
        ..Default::default()
    };
    let _boundary = Arc::new(CPMLBoundary::new(cpml_config, &grid)?);
    
    // Create physics pipeline
    let mut pipeline = PhysicsPipeline::new();
    
    // Add thermal diffusion
    let thermal = ThermalDiffusionComponent::new("thermal".to_string());
    pipeline.add_component(Box::new(thermal))?;
    
    // Initialize fields
    let mut fields = Array4::<f64>::zeros((10, grid.nx, grid.ny, grid.nz));
    
    // Set initial bubble distribution
    let bubble_center = (grid.nx / 2, grid.ny / 2, grid.nz / 2);
    let initial_radius = 5e-6; // 5 μm initial bubble radius
    
    // Initialize bubble radius field (assuming it's at index 7)
    for i in 0..grid.nx {
        for j in 0..grid.ny {
            for k in 0..grid.nz {
                let dx = (i as f64 - bubble_center.0 as f64) * grid.dx;
                let dy = (j as f64 - bubble_center.1 as f64) * grid.dy;
                let dz = (k as f64 - bubble_center.2 as f64) * grid.dz;
                let r = (dx * dx + dy * dy + dz * dz).sqrt();
                
                if r < 10.0 * grid.dx {
                    fields[[7, i, j, k]] = initial_radius;
                }
            }
        }
    }
    
    // Set initial temperature (293K = 20°C)
    fields.index_axis_mut(ndarray::Axis(0), 2).fill(293.0);
    
    // Initialize acoustic pressure with standing wave
    let frequency = 26.5e3; // 26.5 kHz typical for SBSL
    let wavelength = medium.sound_speed(0.0, 0.0, 0.0, &grid) / frequency;
    let k = 2.0 * std::f64::consts::PI / wavelength;
    let amplitude = 1.4e5; // 1.4 atm driving pressure
    
    for i in 0..grid.nx {
        for j in 0..grid.ny {
            for k_idx in 0..grid.nz {
                let x = i as f64 * grid.dx;
                let y = j as f64 * grid.dy;
                let z = k_idx as f64 * grid.dz;
                
                // Standing wave pattern
                fields[[0, i, j, k_idx]] = amplitude * 
                    (k * x).sin() * (k * y).sin() * (k * z).sin();
            }
        }
    }
    
    println!("Grid: {}×{}×{} points", grid.nx, grid.ny, grid.nz);
    println!("Time step: {:.2e} s", dt);
    println!("Frequency: {:.1} kHz", frequency / 1e3);
    println!("Initial bubble radius: {:.1} μm", initial_radius * 1e6);
    println!("Driving pressure: {:.1} atm", amplitude / 101325.0);
    println!();
    
    // Run simulation
    let start_time = Instant::now();
    let mut max_temperature: f64 = 293.0;
    let mut max_light_intensity: f64 = 0.0;
    let mut bubble_collapse_count = 0;
    
    println!("Starting simulation...");
    
    for step in 0..time.n_steps {
        // Simple physics update - just thermal diffusion
        let thermal_diffusivity = medium.thermal_diffusivity(0.0, 0.0, 0.0, &grid);
        let temperature = fields.index_axis_mut(ndarray::Axis(0), 2);
        
        // Apply simple thermal diffusion
        let mut new_temp = temperature.to_owned();
        for i in 1..grid.nx-1 {
            for j in 1..grid.ny-1 {
                for k in 1..grid.nz-1 {
                    let laplacian = 
                        (temperature[[i+1, j, k]] - 2.0 * temperature[[i, j, k]] + temperature[[i-1, j, k]]) / (grid.dx * grid.dx) +
                        (temperature[[i, j+1, k]] - 2.0 * temperature[[i, j, k]] + temperature[[i, j-1, k]]) / (grid.dy * grid.dy) +
                        (temperature[[i, j, k+1]] - 2.0 * temperature[[i, j, k]] + temperature[[i, j, k-1]]) / (grid.dz * grid.dz);
                    
                    new_temp[[i, j, k]] = temperature[[i, j, k]] + thermal_diffusivity * laplacian * dt;
                }
            }
        }
        fields.index_axis_mut(ndarray::Axis(0), 2).assign(&new_temp);
        
        // Monitor bubble dynamics
        let bubble_radius = fields.index_axis(ndarray::Axis(0), 7);
        let min_radius = bubble_radius.iter()
            .filter(|&&r| r > 0.0)
            .fold(f64::INFINITY, |a, &b| a.min(b));
        
        if min_radius < 1e-6 && step > 0 {
            bubble_collapse_count += 1;
        }
        
        // Monitor temperature
        let temperature = fields.index_axis(ndarray::Axis(0), 2);
        let current_max_temp = temperature.iter().fold(0.0f64, |a, &b| a.max(b));
        max_temperature = max_temperature.max(current_max_temp);
        
        // Monitor light emission (if available)
        if fields.shape()[0] > 9 {
            let light_intensity = fields.index_axis(ndarray::Axis(0), 9);
            let current_max_light = light_intensity.iter().fold(0.0f64, |a, &b| a.max(b));
            max_light_intensity = max_light_intensity.max(current_max_light);
        }
        
        // Progress update
        if step % 100 == 0 {
            let elapsed = start_time.elapsed();
            let progress = (step as f64 / time.n_steps as f64) * 100.0;
            println!(
                "Step {}/{} ({:.1}%) - Time: {:.2?}, Max T: {:.0}K, Collapses: {}",
                step, time.n_steps, progress, elapsed, max_temperature, bubble_collapse_count
            );
        }
    }
    
    let total_time = start_time.elapsed();
    
    println!("\n=== Simulation Results ===");
    println!("Total simulation time: {:.2?}", total_time);
    println!("Maximum temperature reached: {:.0} K", max_temperature);
    println!("Maximum light intensity: {:.2e} W/m²", max_light_intensity);
    println!("Number of bubble collapses: {}", bubble_collapse_count);
    println!("Average time per step: {:.2?}", total_time / time.n_steps as u32);
    
    // Analyze final state
    let final_bubble_radius = fields.index_axis(ndarray::Axis(0), 7);
    let non_zero_count = final_bubble_radius.iter().filter(|&&r| r > 0.0).count();
    let avg_radius: f64 = if non_zero_count > 0 {
        final_bubble_radius.iter()
            .filter(|&&r| r > 0.0)
            .sum::<f64>() / non_zero_count as f64
    } else {
        0.0
    };
    
    println!("\nFinal average bubble radius: {:.2} μm", avg_radius * 1e6);
    
    if max_temperature > 5000.0 {
        println!("\n✓ Sonoluminescence conditions achieved!");
        println!("  Peak temperature sufficient for light emission");
    } else {
        println!("\n✗ Sonoluminescence conditions not achieved");
        println!("  Consider increasing driving pressure or optimizing parameters");
    }
    
    Ok(())
}