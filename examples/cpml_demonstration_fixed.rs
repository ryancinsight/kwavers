//! Example demonstrating Convolutional PML (C-PML) boundary conditions
//! 
//! This example shows the effectiveness of C-PML for absorbing acoustic waves
//! at boundaries with minimal reflections.

use kwavers::{
    KwaversResult,
    Grid,
    medium::homogeneous::HomogeneousMedium,
    boundary::{
        pml::{PMLBoundary, PMLConfig},
        cpml::{CPMLBoundary, CPMLConfig},
        Boundary,
    },
};
use ndarray::{Array3, s};
use std::error::Error;

fn main() -> Result<(), Box<dyn Error>> {
    env_logger::init();
    
    println!("=== Kwavers C-PML Demonstration ===\n");
    
    // Create grid
    let grid = Grid::new(200, 200, 200, 1e-3, 1e-3, 1e-3);
    println!("Grid: {}x{}x{} cells", grid.nx, grid.ny, grid.nz);
    println!("Domain: {:.1}x{:.1}x{:.1} mm", 
             grid.nx as f64 * grid.dx * 1000.0,
             grid.ny as f64 * grid.dy * 1000.0,
             grid.nz as f64 * grid.dz * 1000.0
    );
    
    // Create medium
    let medium = HomogeneousMedium::new(
        1000.0,  // Density (kg/m³) - water
        1500.0,  // Sound speed (m/s) - water
        &grid,   // Grid reference
        0.0,     // No absorption (mu_a)
        0.0      // No scattering (mu_s_prime)
    );
    
    // Compare standard PML vs C-PML
    println!("\n1. Comparing Standard PML vs C-PML:");
    compare_pml_cpml(&grid)?;
    
    // Demonstrate C-PML with different configurations
    println!("\n2. C-PML Configuration Effects:");
    demonstrate_cpml_configs(&grid)?;
    
    Ok(())
}

/// Compare standard PML and C-PML performance
fn compare_pml_cpml(grid: &Grid) -> KwaversResult<()> {
    // Standard PML configuration
    let pml_config = PMLConfig {
        size: 20,
        max_value: 2.0,
        alpha_max: 0.0,
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
    
    // Create plane wave
    let mut field_pml = create_plane_wave(grid, 0.0);
    let mut field_cpml = field_pml.clone();
    let initial_energy = compute_field_energy(&field_pml);
    
    // Apply boundaries multiple times
    let steps = 100;
    for step in 0..steps {
        standard_pml.apply_acoustic(&mut field_pml, grid, step)?;
        cpml.apply_acoustic(&mut field_cpml, grid, step)?;
    }
    
    // Compare reflections
    let pml_energy = compute_field_energy(&field_pml);
    let cpml_energy = compute_field_energy(&field_cpml);
    
    println!("  Initial energy: {:.3e}", initial_energy);
    println!("  Standard PML - Final energy: {:.3e} (reflection: {:.2}%)", 
             pml_energy, (pml_energy / initial_energy) * 100.0);
    println!("  C-PML - Final energy: {:.3e} (reflection: {:.2}%)", 
             cpml_energy, (cpml_energy / initial_energy) * 100.0);
    
    Ok(())
}

/// Demonstrate different C-PML configurations
fn demonstrate_cpml_configs(grid: &Grid) -> KwaversResult<()> {
    let configs = vec![
        ("Standard", CPMLConfig {
            thickness: 10,
            polynomial_order: 2.0,
            target_reflection: 1e-3,
            ..Default::default()
        }),
        ("High-order", CPMLConfig {
            thickness: 10,
            polynomial_order: 4.0,
            target_reflection: 1e-3,
            ..Default::default()
        }),
        ("Thick layer", CPMLConfig {
            thickness: 30,
            polynomial_order: 3.0,
            target_reflection: 1e-3,
            ..Default::default()
        }),
        ("Ultra-low reflection", CPMLConfig {
            thickness: 20,
            polynomial_order: 3.5,
            target_reflection: 1e-8,
            ..Default::default()
        }),
    ];
    
    for (name, config) in configs {
        println!("\n  Testing {} configuration:", name);
        println!("    Thickness: {} cells", config.thickness);
        println!("    Polynomial order: {}", config.polynomial_order);
        println!("    Target reflection: {:.1e}", config.target_reflection);
        
        let mut cpml = CPMLBoundary::new(config, grid)?;
        let mut field = create_gaussian_pulse(grid);
        let initial_max = field.iter().fold(0.0f64, |a, &b| a.max(b.abs()));
        
        // Apply boundary
        for step in 0..50 {
            cpml.apply_acoustic(&mut field, grid, step)?;
        }
        
        let final_max = field.iter().fold(0.0f64, |a, &b| a.max(b.abs()));
        println!("    Amplitude reduction: {:.1}%", 
                 (1.0 - final_max / initial_max) * 100.0);
    }
    
    Ok(())
}

/// Create a plane wave propagating in the x-direction
fn create_plane_wave(grid: &Grid, angle: f64) -> Array3<f64> {
    let mut field = Array3::zeros((grid.nx, grid.ny, grid.nz));
    let k = 2.0 * std::f64::consts::PI / (10.0 * grid.dx); // 10 cells per wavelength
    
    for i in 0..grid.nx {
        let x = i as f64 * grid.dx;
        let value = (k * x * angle.cos()).sin();
        field.slice_mut(s![i, .., ..]).fill(value);
    }
    
    field
}

/// Create a Gaussian pulse at the center
fn create_gaussian_pulse(grid: &Grid) -> Array3<f64> {
    let mut field = Array3::zeros((grid.nx, grid.ny, grid.nz));
    let center_x = grid.nx / 2;
    let center_y = grid.ny / 2;
    let center_z = grid.nz / 2;
    let sigma = 5.0; // 5 cells standard deviation
    
    for i in 0..grid.nx {
        for j in 0..grid.ny {
            for k in 0..grid.nz {
                let dx = (i as f64 - center_x as f64) * grid.dx;
                let dy = (j as f64 - center_y as f64) * grid.dy;
                let dz = (k as f64 - center_z as f64) * grid.dz;
                let r2 = dx * dx + dy * dy + dz * dz;
                field[[i, j, k]] = (-r2 / (2.0 * sigma * sigma * grid.dx * grid.dx)).exp();
            }
        }
    }
    
    field
}

/// Compute total field energy
fn compute_field_energy(field: &Array3<f64>) -> f64 {
    field.iter().map(|&v| v * v).sum::<f64>()
}