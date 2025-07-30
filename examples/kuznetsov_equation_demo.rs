//! Demonstration of the Full Kuznetsov Equation Implementation
//! 
//! This example shows how to use the complete nonlinear acoustic model
//! with all second-order terms and acoustic diffusivity for accurate
//! finite-amplitude sound propagation simulation.

use kwavers::*;
use kwavers::physics::mechanics::{KuznetsovWave, KuznetsovConfig, TimeIntegrationScheme};
use kwavers::physics::{KuznetsovWaveComponent, PhysicsPipeline, PhysicsContext};
use ndarray::Array3;
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
        0.5      // Absorption coefficient (dB/cm/MHz)
    );
    medium.b_a = 7.0;  // Nonlinearity parameter B/A for tissue
    
    println!("Medium properties:");
    println!("  Density: {} kg/m³", medium.rho);
    println!("  Sound speed: {} m/s", medium.c);
    println!("  Absorption: {} dB/cm/MHz", medium.alpha);
    println!("  Nonlinearity B/A: {}\n", medium.b_a);
    
    // Demonstrate three different ways to use Kuznetsov equation
    
    println!("1. Direct KuznetsovWave solver:");
    demonstrate_direct_solver(&grid, &medium)?;
    
    println!("\n2. KuznetsovWaveComponent in physics pipeline:");
    demonstrate_component(&grid, &medium)?;
    
    println!("\n3. Enhanced NonlinearWave with Kuznetsov terms:");
    demonstrate_enhanced_nonlinear(&grid, &medium)?;
    
    println!("\n=== Demonstration Complete ===");
    Ok(())
}

/// Demonstrate direct use of KuznetsovWave solver
fn demonstrate_direct_solver(grid: &Grid, medium: &HomogeneousMedium) -> Result<(), Box<dyn Error>> {
    // Configure Kuznetsov solver
    let config = KuznetsovConfig {
        enable_nonlinearity: true,
        enable_diffusivity: true,
        nonlinearity_scaling: 1.0,
        max_pressure: 10e6,  // 10 MPa max
        time_scheme: TimeIntegrationScheme::RK4,
        spatial_order: 4,
        adaptive_timestep: false,
        cfl_factor: 0.3,
    };
    
    println!("  Configuration:");
    println!("    - Full nonlinearity: enabled");
    println!("    - Acoustic diffusivity: enabled");
    println!("    - Spatial order: 4th");
    println!("    - Time integration: RK4");
    
    // Create solver
    let mut solver = KuznetsovWave::new(grid, config);
    
    // Initialize pressure field with focused beam
    let mut pressure = create_focused_beam(grid, 2e6, 0.005);  // 2 MPa, 5mm beam width
    let mut velocity = Array4::zeros((3, grid.nx, grid.ny, grid.nz));
    let source = Array3::zeros((grid.nx, grid.ny, grid.nz));
    
    // Time stepping
    let dt = 5e-8;  // 50 ns
    let steps = 100;
    
    println!("  Simulating {} steps with dt = {} s", steps, dt);
    
    // Run simulation
    for step in 0..steps {
        let t = step as f64 * dt;
        solver.update_wave(&mut pressure, &mut velocity, &source, grid, medium, dt, t)?;
        
        if step % 20 == 0 {
            let max_p = pressure.iter().fold(0.0_f64, |a, &b| a.max(b.abs()));
            println!("    Step {}: max pressure = {:.2} MPa", step, max_p / 1e6);
        }
    }
    
    // Get performance metrics
    let metrics = solver.get_performance_metrics();
    println!("  Performance metrics:");
    for (key, value) in metrics.iter() {
        println!("    - {}: {:.3e}", key, value);
    }
    
    Ok(())
}

/// Demonstrate KuznetsovWaveComponent in physics pipeline
fn demonstrate_component(grid: &Grid, medium: &HomogeneousMedium) -> Result<(), Box<dyn Error>> {
    // Create physics pipeline
    let mut pipeline = PhysicsPipeline::new();
    
    // Add Kuznetsov wave component
    let kuznetsov = KuznetsovWaveComponent::new("kuznetsov".to_string(), grid)
        .with_nonlinearity(true, 1.0)
        .with_diffusivity(true);
    
    pipeline.add_component(Box::new(kuznetsov))?;
    
    println!("  Pipeline created with KuznetsovWaveComponent");
    
    // Add thermal diffusion for complete physics
    let thermal = ThermalDiffusionComponent::new("thermal".to_string());
    pipeline.add_component(Box::new(thermal))?;
    
    println!("  Added thermal diffusion component");
    
    // Initialize fields
    let mut fields = Array4::zeros((8, grid.nx, grid.ny, grid.nz));
    
    // Set initial pressure
    let pressure_field = create_focused_beam(grid, 1.5e6, 0.004);  // 1.5 MPa, 4mm beam
    fields.index_axis_mut(Axis(0), 0).assign(&pressure_field);
    
    // Create context
    let mut context = PhysicsContext::new(1e6);  // 1 MHz
    
    // Time stepping
    let dt = 5e-8;
    let steps = 50;
    
    println!("  Running pipeline for {} steps", steps);
    
    for step in 0..steps {
        let t = step as f64 * dt;
        pipeline.execute(&mut fields, grid, medium, dt, t, &mut context)?;
        
        if step % 10 == 0 {
            let pressure = fields.index_axis(Axis(0), 0);
            let temperature = fields.index_axis(Axis(0), 2);
            let max_p = pressure.iter().fold(0.0_f64, |a, &b| a.max(b.abs()));
            let max_t = temperature.iter().fold(0.0_f64, |a, &b| a.max(b));
            println!("    Step {}: pressure = {:.2} MPa, temp rise = {:.3} K", 
                step, max_p / 1e6, max_t);
        }
    }
    
    // Validate pipeline
    let validation = pipeline.validate(&context);
    println!("  Pipeline validation: {}", if validation.is_valid { "PASSED" } else { "FAILED" });
    
    Ok(())
}

/// Demonstrate enhanced NonlinearWave with Kuznetsov terms
fn demonstrate_enhanced_nonlinear(grid: &Grid, medium: &HomogeneousMedium) -> Result<(), Box<dyn Error>> {
    use kwavers::physics::mechanics::NonlinearWave;
    
    // Create NonlinearWave solver
    let mut solver = NonlinearWave::new(grid);
    
    // Enable Kuznetsov terms
    solver.enable_kuznetsov_terms(true);
    solver.enable_diffusivity(true);
    solver.set_nonlinearity_scaling(1.0);
    solver.set_k_space_correction_order(4);
    
    println!("  NonlinearWave configured with:");
    println!("    - Kuznetsov terms: enabled");
    println!("    - Acoustic diffusivity: enabled");
    println!("    - K-space correction: 4th order");
    
    // Initialize fields
    let mut fields = Array4::zeros((8, grid.nx, grid.ny, grid.nz));
    let pressure_field = create_gaussian_pulse(grid, 3e6, 0.003);  // 3 MPa, 3mm pulse
    fields.index_axis_mut(Axis(0), 0).assign(&pressure_field);
    
    // Create source
    let source = PointSource::new(0.0, 0.0, 0.0, 1e6, 0.0);  // 1 MHz at origin
    
    // Time stepping
    let dt = 5e-8;
    let steps = 80;
    
    println!("  Simulating high-amplitude pulse propagation");
    
    for step in 0..steps {
        let t = step as f64 * dt;
        
        // Get pressure for history update
        let pressure = fields.index_axis(Axis(0), 0).to_owned();
        
        solver.update_wave(&mut fields, &pressure, &source, grid, medium, dt, t);
        
        if step % 20 == 0 {
            let p_field = fields.index_axis(Axis(0), 0);
            let max_p = p_field.iter().fold(0.0_f64, |a, &b| a.max(b.abs()));
            
            // Check for shock formation
            let grad_max = compute_max_gradient(p_field, grid);
            
            println!("    Step {}: pressure = {:.2} MPa, max gradient = {:.2e} Pa/m", 
                step, max_p / 1e6, grad_max);
        }
    }
    
    Ok(())
}

/// Create a focused Gaussian beam
fn create_focused_beam(grid: &Grid, amplitude: f64, beam_width: f64) -> Array3<f64> {
    let mut field = Array3::zeros((grid.nx, grid.ny, grid.nz));
    
    for i in 0..grid.nx {
        for j in 0..grid.ny {
            for k in 0..grid.nz {
                let x = i as f64 * grid.dx;
                let y = (j as f64 - grid.ny as f64 / 2.0) * grid.dy;
                let z = (k as f64 - grid.nz as f64 / 2.0) * grid.dz;
                
                // Gaussian beam profile
                let r2 = y*y + z*z;
                let envelope = (-r2 / (beam_width * beam_width)).exp();
                
                // Sinusoidal carrier
                if x < 0.02 {  // 20mm extent
                    let wavelength = 0.0015;  // 1.5mm (1 MHz in tissue)
                    let k_wave = 2.0 * std::f64::consts::PI / wavelength;
                    field[[i, j, k]] = amplitude * envelope * (k_wave * x).sin();
                }
            }
        }
    }
    
    field
}

/// Create a Gaussian pulse
fn create_gaussian_pulse(grid: &Grid, amplitude: f64, width: f64) -> Array3<f64> {
    let mut field = Array3::zeros((grid.nx, grid.ny, grid.nz));
    
    let center_x = grid.nx as f64 / 4.0;
    let center_y = grid.ny as f64 / 2.0;
    let center_z = grid.nz as f64 / 2.0;
    
    for i in 0..grid.nx {
        for j in 0..grid.ny {
            for k in 0..grid.nz {
                let dx = (i as f64 - center_x) * grid.dx;
                let dy = (j as f64 - center_y) * grid.dy;
                let dz = (k as f64 - center_z) * grid.dz;
                
                let r2 = dx*dx + dy*dy + dz*dz;
                field[[i, j, k]] = amplitude * (-r2 / (width * width)).exp();
            }
        }
    }
    
    field
}

/// Compute maximum spatial gradient
fn compute_max_gradient(field: &ArrayView3<f64>, grid: &Grid) -> f64 {
    let mut max_grad = 0.0;
    
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

use ndarray::{Array4, ArrayView3, Axis};