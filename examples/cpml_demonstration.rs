//! Demonstration of Convolutional PML (C-PML) for Enhanced Boundary Absorption
//! 
//! This example shows the superior performance of C-PML compared to standard PML,
//! especially for grazing angle incidence and evanescent waves.

use kwavers::*;
use kwavers::boundary::{CPMLBoundary, CPMLConfig, PMLBoundary, PMLConfig};
use kwavers::solver::cpml_integration::CPMLSolver;
use ndarray::{Array3, Array4};
use std::error::Error;
use std::f64::consts::PI;

fn main() -> Result<(), Box<dyn Error>> {
    // Initialize logging
    init_logging();
    
    println!("=== Convolutional PML (C-PML) Demonstration ===\n");
    
    // Create computational grid
    let grid = Grid::new(
        300, 300, 100,  // Grid dimensions
        0.5e-3, 0.5e-3, 0.5e-3  // Grid spacing (0.5mm)
    );
    
    println!("Grid created: {}x{}x{} points", grid.nx, grid.ny, grid.nz);
    println!("Domain size: {:.1}mm x {:.1}mm x {:.1}mm\n", 
        grid.nx as f64 * grid.dx * 1000.0,
        grid.ny as f64 * grid.dy * 1000.0,
        grid.nz as f64 * grid.dz * 1000.0
    );
    
    // Create medium
    let medium = HomogeneousMedium::new(
        1000.0,  // Density (kg/m³) - water
        1500.0,  // Sound speed (m/s) - water
        grid,    // Grid reference
        0.0,     // No absorption (mu_a)
        0.0      // No scattering (mu_s_prime)
    );
    
    // Demonstrate different aspects of C-PML
    println!("1. Comparing Standard PML vs C-PML for normal incidence:");
    compare_normal_incidence(&grid, &medium)?;
    
    println!("\n2. Comparing Standard PML vs C-PML for grazing angles:");
    compare_grazing_angles(&grid, &medium)?;
    
    println!("\n3. Demonstrating C-PML solver integration:");
    demonstrate_cpml_solver(&grid, &medium)?;
    
    println!("\n4. Analyzing reflection coefficients:");
    analyze_reflection_coefficients()?;
    
    println!("\n=== Demonstration Complete ===");
    Ok(())
}

/// Compare standard PML and C-PML for normal incidence
fn compare_normal_incidence(grid: &Grid, medium: &HomogeneousMedium) -> Result<(), Box<dyn Error>> {
    // Standard PML configuration
    let pml_config = PMLConfig {
        thickness: 20,
        sigma_max_acoustic: 2.0,
        target_reflection: Some(1e-6),
        ..Default::default()
    };
    let mut standard_pml = PMLBoundary::new(pml_config)?;
    
    // C-PML configuration
    let cpml_config = CPMLConfig {
        thickness: 20,
        polynomial_order: 3.0,
        target_reflection: 1e-6,
        ..Default::default()
    };
    let mut cpml = CPMLBoundary::new(cpml_config, grid)?;
    
    // Create plane wave at normal incidence
    let mut field_pml = create_plane_wave(grid, 0.0, 1e6);
    let mut field_cpml = field_pml.clone();
    let initial_energy = compute_field_energy(&field_pml);
    
    // Apply boundaries
    for step in 0..100 {
        standard_pml.apply_acoustic(&mut field_pml, grid, step)?;
        cpml.apply_acoustic(&mut field_cpml, grid, step)?;
    }
    
    let pml_energy = compute_field_energy(&field_pml);
    let cpml_energy = compute_field_energy(&field_cpml);
    
    println!("  Initial energy: {:.2e}", initial_energy);
    println!("  Standard PML final energy: {:.2e} (reflection: {:.2e})", 
        pml_energy, pml_energy / initial_energy);
    println!("  C-PML final energy: {:.2e} (reflection: {:.2e})", 
        cpml_energy, cpml_energy / initial_energy);
    
    Ok(())
}

/// Compare standard PML and C-PML for grazing angles
fn compare_grazing_angles(grid: &Grid, medium: &HomogeneousMedium) -> Result<(), Box<dyn Error>> {
    // Test multiple angles
    let angles = vec![60.0, 75.0, 85.0, 89.0];
    
    println!("  Angle | Standard PML | C-PML (default) | C-PML (grazing)");
    println!("  ------|--------------|-----------------|----------------");
    
    for angle in angles {
        // Standard PML
        let pml_config = PMLConfig::default();
        let mut standard_pml = PMLBoundary::new(pml_config)?;
        
        // C-PML default
        let cpml_config = CPMLConfig::default();
        let mut cpml_default = CPMLBoundary::new(cpml_config, grid)?;
        
        // C-PML optimized for grazing
        let cpml_grazing_config = CPMLConfig::for_grazing_angles();
        let mut cpml_grazing = CPMLBoundary::new(cpml_grazing_config, grid)?;
        
        // Create plane wave at angle
        let mut field_pml = create_plane_wave(grid, angle, 1e6);
        let mut field_cpml_default = field_pml.clone();
        let mut field_cpml_grazing = field_pml.clone();
        let initial_energy = compute_field_energy(&field_pml);
        
        // Apply boundaries
        for step in 0..50 {
            standard_pml.apply_acoustic(&mut field_pml, grid, step)?;
            cpml_default.apply_acoustic(&mut field_cpml_default, grid, step)?;
            cpml_grazing.apply_acoustic(&mut field_cpml_grazing, grid, step)?;
        }
        
        let pml_reflection = compute_field_energy(&field_pml) / initial_energy;
        let cpml_default_reflection = compute_field_energy(&field_cpml_default) / initial_energy;
        let cpml_grazing_reflection = compute_field_energy(&field_cpml_grazing) / initial_energy;
        
        println!("  {:>4.0}° | {:>12.2e} | {:>15.2e} | {:>15.2e}", 
            angle, pml_reflection, cpml_default_reflection, cpml_grazing_reflection);
    }
    
    Ok(())
}

/// Demonstrate C-PML solver integration
fn demonstrate_cpml_solver(grid: &Grid, medium: &HomogeneousMedium) -> Result<(), Box<dyn Error>> {
    // Create C-PML solver with optimized configuration
    let config = CPMLConfig {
        thickness: 30,
        polynomial_order: 4.0,
        kappa_max: 20.0,
        alpha_max: 0.3,
        target_reflection: 1e-8,
        enhanced_grazing: true,
        ..Default::default()
    };
    
    let mut cpml_solver = CPMLSolver::new(config, grid)?;
    
    println!("  C-PML solver created with enhanced configuration:");
    println!("    - Thickness: 30 cells");
    println!("    - Polynomial order: 4.0");
    println!("    - κ_max: 20.0 (high for grazing angles)");
    println!("    - α_max: 0.3 (frequency shifting)");
    
    // Initialize fields with focused beam
    let mut pressure = create_focused_beam(grid, 3e6, 0.01);
    let mut velocity = Array4::zeros((3, grid.nx, grid.ny, grid.nz));
    
    let initial_max = pressure.iter().fold(0.0_f64, |a, &b| a.max(b.abs()));
    println!("\n  Initial maximum pressure: {:.2e} Pa", initial_max);
    
    // Time stepping
    let dt = 1e-7;
    let steps = 200;
    
    println!("  Running {} time steps with dt = {} s", steps, dt);
    
    for step in 0..steps {
        // Apply C-PML boundary to pressure field
        cpml.apply_acoustic(&mut pressure, grid, step)?;
        
        if step % 50 == 0 {
            let max_p = pressure.iter().fold(0.0_f64, |a, &b| a.max(b.abs()));
            let center_p = pressure[[grid.nx/2, grid.ny/2, grid.nz/2]];
            println!("    Step {:>3}: max pressure = {:.2e} Pa, center = {:.2e} Pa", 
                step, max_p, center_p);
        }
    }
    
    let final_max = pressure.iter().fold(0.0_f64, |a, &b| a.max(b.abs()));
    println!("  Final maximum pressure: {:.2e} Pa", final_max);
    println!("  Attenuation: {:.1} dB", 20.0 * (initial_max / final_max).log10());
    
    Ok(())
}

/// Analyze reflection coefficients at various angles
fn analyze_reflection_coefficients() -> Result<(), Box<dyn Error>> {
    let grid = Grid::new(64, 64, 64, 1e-3, 1e-3, 1e-3);
    
    // Different C-PML configurations
    let configs = vec![
        ("Default", CPMLConfig::default()),
        ("Grazing-optimized", CPMLConfig::for_grazing_angles()),
        ("Custom high-κ", CPMLConfig {
            kappa_max: 30.0,
            polynomial_order: 4.0,
            enhanced_grazing: true,
            ..Default::default()
        }),
    ];
    
    println!("  Configuration comparison:");
    println!("  Angle | Default | Grazing-opt | High-κ");
    println!("  ------|---------|-------------|--------");
    
    for angle in (0..=90).step_by(15) {
        print!("  {:>4}° |", angle);
        
        for (_, config) in &configs {
            let cpml = CPMLBoundary::new(config.clone(), &grid)?;
            let reflection = cpml.estimate_reflection(angle as f64);
            print!(" {:>7.1e} |", reflection);
        }
        println!();
    }
    
    println!("\n  Note: These are theoretical estimates. Actual performance");
    println!("  depends on wave frequency, grid resolution, and other factors.");
    
    Ok(())
}

// Helper functions

/// Create a plane wave at given angle
fn create_plane_wave(grid: &Grid, angle_degrees: f64, frequency: f64) -> Array3<f64> {
    let mut field = Array3::zeros((grid.nx, grid.ny, grid.nz));
    let angle_rad = angle_degrees * PI / 180.0;
    let k = 2.0 * PI * frequency / 1500.0; // Wave number
    
    for i in 0..grid.nx {
        for j in 0..grid.ny {
            for k_idx in 0..grid.nz {
                let x = i as f64 * grid.dx;
                let y = j as f64 * grid.dy;
                
                // Plane wave propagating at angle from x-axis
                let phase = k * (x * angle_rad.cos() + y * angle_rad.sin());
                field[[i, j, k_idx]] = phase.sin();
            }
        }
    }
    
    field
}

/// Create a focused Gaussian beam
fn create_focused_beam(grid: &Grid, amplitude: f64, beam_width: f64) -> Array3<f64> {
    let mut field = Array3::zeros((grid.nx, grid.ny, grid.nz));
    
    let center_y = grid.ny as f64 / 2.0;
    let center_z = grid.nz as f64 / 2.0;
    
    for i in 0..grid.nx {
        for j in 0..grid.ny {
            for k in 0..grid.nz {
                let x = i as f64 * grid.dx;
                let y = (j as f64 - center_y) * grid.dy;
                let z = (k as f64 - center_z) * grid.dz;
                
                // Gaussian beam profile
                let r2 = y*y + z*z;
                let envelope = (-r2 / (beam_width * beam_width)).exp();
                
                // Sinusoidal carrier
                if x < 0.05 { // 50mm extent
                    let wavelength = 0.0015; // 1.5mm (1 MHz in water)
                    let k_wave = 2.0 * PI / wavelength;
                    field[[i, j, k]] = amplitude * envelope * (k_wave * x).sin();
                }
            }
        }
    }
    
    field
}

/// Compute total field energy
fn compute_field_energy(field: &Array3<f64>) -> f64 {
    field.iter().map(|&v| v * v).sum()
}