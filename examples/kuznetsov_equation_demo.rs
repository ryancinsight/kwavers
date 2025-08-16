//! Demonstration of the Full Kuznetsov Equation Implementation
//! 
//! This example shows how to use the Kuznetsov equation solver for
//! complete nonlinear acoustic simulations including all second-order
//! nonlinear terms and acoustic diffusivity.

use kwavers::*;
use kwavers::physics::mechanics::{KuznetsovWave, KuznetsovConfig, TimeIntegrationScheme, NonlinearWave};
// use kwavers::physics::{KuznetsovWaveComponent, PhysicsPipeline, PhysicsContext};
use kwavers::medium::HomogeneousMedium;
use kwavers::source::{Source, NullSource};
use ndarray::{Array3, Array4, ArrayView3, Axis};
use std::error::Error;

fn main() -> Result<(), Box<dyn Error>> {
    // Initialize logging
    init_logging();
    
    println!("=== Kuznetsov Equation Demonstration ===\n");
    
    // Create computational grid
    let grid = Grid::new(
        256, 128, 128,  // Grid dimensions
        0.1e-3, 0.2e-3, 0.2e-3  // Grid spacing (0.1mm x 0.2mm x 0.2mm)
    );
    
    println!("Grid created: {}x{}x{} points", grid.nx, grid.ny, grid.nz);
    println!("Domain size: {:.1}mm x {:.1}mm x {:.1}mm\n", 
        grid.nx as f64 * grid.dx * 1000.0,
        grid.ny as f64 * grid.dy * 1000.0,
        grid.nz as f64 * grid.dz * 1000.0
    );
    
    // Create medium with realistic tissue properties
    let mut medium = HomogeneousMedium::new(
        1050.0,  // Density (kg/m³) - soft tissue
        1540.0,  // Sound speed (m/s) - soft tissue
        &grid,   // Grid reference
        0.5,     // Optical absorption coefficient (1/m)
        0.0      // Optical scattering coefficient (1/m)
    );
    medium.b_a = 7.0;  // Nonlinearity parameter B/A for tissue
    
    println!("Medium properties:");
    println!("  Density: {} kg/m³", medium.density);
    println!("  Sound speed: {} m/s", medium.sound_speed);
    println!("  Absorption: {} dB/cm/MHz", medium.alpha0);
    println!("  Nonlinearity B/A: {}\n", medium.b_a);
    
    // Demonstrate three different ways to use Kuznetsov equation
    
    println!("1. Direct KuznetsovWave solver:");
    demonstrate_direct_solver(&grid, &medium)?;
    
    // Component-based solver implementation
    // println!("\n2. KuznetsovWaveComponent in physics pipeline:");
    // demonstrate_component(&grid, &medium)?;
    
    println!("\n3. Enhanced NonlinearWave with Kuznetsov terms:");
    demonstrate_nonlinear(&grid, &medium)?;
    
    println!("\n=== Demonstration Complete ===");
    Ok(())
}

/// Demonstrate direct use of KuznetsovWave solver
fn demonstrate_direct_solver(grid: &Grid, medium: &HomogeneousMedium) -> Result<(), Box<dyn Error>> {
    println!("Using KuznetsovWave solver directly:");
    
    // Create configuration
    let config = KuznetsovConfig {
        enable_nonlinearity: true,
        enable_diffusivity: true,
        nonlinearity_scaling: 1.0,
        spatial_order: 4,
        time_scheme: TimeIntegrationScheme::RK4,
        ..Default::default()
    };
    
    println!("  Configuration:");
    println!("    - Nonlinearity: enabled");
    println!("    - Diffusivity: enabled");
    println!("    - Spatial order: 4");
    println!("    - Time scheme: RK4");
    
    // Create solver
    let mut solver = KuznetsovWave::new(grid, config).expect("Failed to create Kuznetsov solver");
    
    // Initialize fields array (standard field layout)
    let mut fields = Array4::zeros((13, grid.nx, grid.ny, grid.nz));
    
    // Create focused beam initial condition
    let beam = create_focused_beam(grid, 3e6, 0.01);
    fields.index_axis_mut(Axis(0), 0).assign(&beam); // Pressure at index 0
    
    // Create a null source for this demo
    let source = NullSource;
    let prev_pressure = fields.index_axis(Axis(0), 0).to_owned();
    
    // Time stepping
    let dt = 0.5 * grid.dx / medium.sound_speed; // CFL = 0.5
    let steps = 50;
    
    println!("\n  Simulating {} time steps...", steps);
    
    for step in 0..steps {
        let t = step as f64 * dt;
        solver.update_wave(&mut fields, &prev_pressure, &source, grid, medium, dt, t);
        
        if step % 10 == 0 {
            let pressure = fields.index_axis(Axis(0), 0);
            let max_p = pressure.iter().fold(0.0_f64, |a, &b| a.max(b.abs()));
            println!("    Step {}: max pressure = {:.2e} Pa", step, max_p);
        }
    }
    
    // Report performance
    solver.report_performance();
    
    Ok(())
}

/// Demonstrate KuznetsovWaveComponent in physics pipeline
#[allow(dead_code)]
fn demonstrate_component(_grid: &Grid, _medium: &HomogeneousMedium) -> Result<(), Box<dyn Error>> {
    /*
    println!("Using KuznetsovWaveComponent in physics pipeline:");
    
    // Create physics pipeline
    let mut pipeline = PhysicsPipeline::new();
    
    // Add Kuznetsov wave component
    let kuznetsov = KuznetsovWaveComponent::new("kuznetsov".to_string(), grid)
        .with_nonlinearity(true, 1.0)
        .with_diffusivity(true);
    
    pipeline.add_component(Box::new(kuznetsov))?;
    
    println!("  Pipeline components:");
    println!("    - KuznetsovWaveComponent (nonlinearity + diffusivity)");
    
    // Initialize fields
    let mut fields = Array4::zeros((13, grid.nx, grid.ny, grid.nz));
    
    // Create Gaussian pulse
    let pulse = create_gaussian_pulse(grid, 2e6, grid.nx/2, grid.ny/2, grid.nz/2, 0.01);
    fields.index_axis_mut(Axis(0), 0).assign(&pulse);
    
    // Create context
    let mut context = PhysicsContext::new(1e6); // 1 MHz reference frequency
    
    // Run pipeline
    let dt = 0.5 * grid.dx / medium.sound_speed;
    let steps = 50;
    
    println!("\n  Running pipeline for {} steps...", steps);
    
    for step in 0..steps {
        let t = step as f64 * dt;
        pipeline.execute(&mut fields, grid, medium, dt, t, &mut context)?;
        
        if step % 10 == 0 {
            let pressure = fields.index_axis(Axis(0), 0);
            let max_p = pressure.iter().fold(0.0_f64, |a, &b| a.max(b.abs()));
            println!("    Step {}: max pressure = {:.2e} Pa", step, max_p);
        }
    }
    
    println!("\n  Pipeline execution complete.");
    */
    
    Ok(())
}

/// Demonstrate enhanced NonlinearWave with Kuznetsov terms
fn demonstrate_nonlinear(grid: &Grid, medium: &HomogeneousMedium) -> Result<(), Box<dyn Error>> {
    println!("Using enhanced NonlinearWave with Kuznetsov terms:");
    
    // Create enhanced nonlinear solver
    let mut solver = NonlinearWave::new(grid);
    solver.enable_kuznetsov_terms(true);
    solver.enable_diffusivity(true);
    solver.set_nonlinearity_scaling(1.0);
    
    println!("  Configuration:");
    println!("    - Kuznetsov terms: enabled");
    println!("    - Diffusivity: enabled");
    println!("    - Nonlinearity scaling: 1.0");
    
    // Initialize fields
    let mut fields = Array4::zeros((13, grid.nx, grid.ny, grid.nz));
    
    // Create focused beam
    let beam = create_focused_beam(grid, 2.5e6, 0.008);
    fields.index_axis_mut(Axis(0), 0).assign(&beam);
    
    // Create null source
    let source = NullSource;
    let prev_pressure = fields.index_axis(Axis(0), 0).to_owned();
    
    // Time stepping
    let dt = 0.5 * grid.dx / medium.sound_speed;
    let steps = 50;
    
    println!("\n  Simulating {} time steps...", steps);
    
    for step in 0..steps {
        let t = step as f64 * dt;
        solver.update_wave(&mut fields, &prev_pressure, &source, grid, medium, dt, t);
        
        if step % 10 == 0 {
            let pressure = fields.index_axis(Axis(0), 0);
            let max_p = pressure.iter().fold(0.0_f64, |a, &b| a.max(b.abs()));
            let grad_max = compute_max_gradient(&pressure, grid);
            println!("    Step {}: max pressure = {:.2e} Pa, max gradient = {:.2e} Pa/m", 
                step, max_p, grad_max);
        }
    }
    
    solver.report_performance();
    
    Ok(())
}

// Helper functions

/// Create a focused Gaussian beam
fn create_focused_beam(grid: &Grid, amplitude: f64, beam_width: f64) -> Array3<f64> {
    let mut field = grid.create_field();
    
    let center_y = grid.ny as f64 / 2.0;
    let center_z = grid.nz as f64 / 2.0;
    
    // Use iterator-based approach for better performance
    field.indexed_iter_mut()
        .for_each(|((i, j, k), value)| {
            let x = i as f64 * grid.dx;
            let y = (j as f64 - center_y) * grid.dy;
            let z = (k as f64 - center_z) * grid.dz;
            
            // Gaussian beam profile
            let r2 = y*y + z*z;
            let envelope = (-r2 / (beam_width * beam_width)).exp();
            
            // Initial pressure profile
            if x < 0.02 { // 20mm extent
                *value = amplitude * envelope;
            }
        });
    
    field
}

/// Create a Gaussian pulse
fn create_gaussian_pulse(grid: &Grid, amplitude: f64, cx: usize, cy: usize, cz: usize, width: f64) -> Array3<f64> {
    let mut field = grid.create_field();
    
    // Use iterator-based approach for better performance
    field.indexed_iter_mut()
        .for_each(|((i, j, k), value)| {
            let dx = (i as f64 - cx as f64) * grid.dx;
            let dy = (j as f64 - cy as f64) * grid.dy;
            let dz = (k as f64 - cz as f64) * grid.dz;
            
            let r2 = dx*dx + dy*dy + dz*dz;
            *value = amplitude * (-r2 / (width * width)).exp();
        });
    
    field
}

/// Compute maximum gradient magnitude
fn compute_max_gradient(field: &ArrayView3<f64>, grid: &Grid) -> f64 {
    let mut max_grad: f64 = 0.0;
    
    for i in 1..grid.nx-1 {
        for j in 1..grid.ny-1 {
            for k in 1..grid.nz-1 {
                let grad_x = (field[[i+1, j, k]] - field[[i-1, j, k]]) / (2.0 * grid.dx);
                let grad_y = (field[[i, j+1, k]] - field[[i, j-1, k]]) / (2.0 * grid.dy);
                let grad_z = (field[[i, j, k+1]] - field[[i, j, k-1]]) / (2.0 * grid.dz);
                
                let grad_mag = (grad_x*grad_x + grad_y*grad_y + grad_z*grad_z).sqrt();
                max_grad = max_grad.max(grad_mag);
            }
        }
    }
    
    max_grad
}