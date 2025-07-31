//! Comparison of PSTD and FDTD solvers
//! 
//! This example demonstrates the usage of both Pseudo-Spectral Time Domain (PSTD)
//! and Finite-Difference Time Domain (FDTD) solvers, comparing their accuracy,
//! performance, and numerical dispersion characteristics.

use kwavers::*;
use kwavers::solver::pstd::{PstdConfig, PstdPlugin};
use kwavers::solver::fdtd::{FdtdConfig, FdtdPlugin};
use kwavers::physics::plugin::{PluginManager, PluginContext};
use ndarray::{Array3, Array4, Axis};
use std::f64::consts::PI;
use std::time::Instant;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Initialize logging
    env_logger::init();
    
    println!("=== PSTD vs FDTD Solver Comparison ===\n");
    
    // Simulation parameters
    let frequency = 1e6; // 1 MHz
    let wavelength = 1500.0 / frequency; // Speed of sound in water / frequency
    let domain_size = 10.0 * wavelength; // 10 wavelengths
    let grid_points_per_wavelength = 8; // For PSTD (can use fewer)
    let grid_points = (domain_size / wavelength * grid_points_per_wavelength as f64) as usize;
    
    // Ensure power of 2 for optimal FFT performance
    let grid_points = grid_points.next_power_of_two();
    let dx = domain_size / grid_points as f64;
    
    println!("Simulation parameters:");
    println!("  Frequency: {} MHz", frequency / 1e6);
    println!("  Wavelength: {:.3} mm", wavelength * 1000.0);
    println!("  Domain size: {:.1} wavelengths", domain_size / wavelength);
    println!("  Grid points: {}³", grid_points);
    println!("  Grid spacing: {:.3} mm", dx * 1000.0);
    println!("  Points per wavelength: {:.1}", wavelength / dx);
    
    // Create grid
    let grid = Grid::new(grid_points, grid_points, grid_points, dx, dx, dx);
    
    // Create medium (homogeneous water)
    let medium = HomogeneousMedium::new(1000.0, 1500.0, &grid, 0.0, 1.0);
    
    // Time parameters
    let c_max = 1500.0;
    let simulation_time = 5.0 * domain_size / c_max; // 5 domain traversals
    
    // Run PSTD simulation
    println!("\n--- Running PSTD Simulation ---");
    let pstd_result = run_pstd_simulation(&grid, &medium, frequency, simulation_time)?;
    
    // Run FDTD simulation
    println!("\n--- Running FDTD Simulation ---");
    let fdtd_result = run_fdtd_simulation(&grid, &medium, frequency, simulation_time)?;
    
    // Compare results
    println!("\n--- Comparison Results ---");
    compare_results(&pstd_result, &fdtd_result, &grid);
    
    // Analyze numerical dispersion
    println!("\n--- Numerical Dispersion Analysis ---");
    analyze_dispersion(&pstd_result, &fdtd_result, &grid, wavelength);
    
    Ok(())
}

/// Result structure for storing simulation outputs
struct SimulationResult {
    final_pressure: Array3<f64>,
    computation_time: f64,
    time_steps: usize,
    max_pressure: f64,
    method_name: String,
}

/// Run PSTD simulation
fn run_pstd_simulation(
    grid: &Grid,
    medium: &dyn Medium,
    frequency: f64,
    simulation_time: f64,
) -> KwaversResult<SimulationResult> {
    let start_time = Instant::now();
    
    // Configure PSTD solver
    let pstd_config = PstdConfig {
        k_space_correction: true,
        k_space_order: 4,
        anti_aliasing: true,
        pml_stencil_size: 4,
        cfl_factor: 0.3, // PSTD typically needs smaller CFL
    };
    
    // Create plugin
    let pstd_plugin = PstdPlugin::new(pstd_config.clone(), grid)?;
    
    // Set up plugin manager
    let mut plugin_manager = PluginManager::new();
    plugin_manager.register(Box::new(pstd_plugin))?;
    
    // Initialize fields
    let mut fields = Array4::zeros((13, grid.nx, grid.ny, grid.nz));
    
    // Add initial Gaussian pulse
    let center = (grid.nx / 2, grid.ny / 2, grid.nz / 2);
    let sigma = grid.nx as f64 * grid.dx / 10.0; // Width of Gaussian
    
    for i in 0..grid.nx {
        for j in 0..grid.ny {
            for k in 0..grid.nz {
                let x = (i as f64 - center.0 as f64) * grid.dx;
                let y = (j as f64 - center.1 as f64) * grid.dy;
                let z = (k as f64 - center.2 as f64) * grid.dz;
                let r2 = x * x + y * y + z * z;
                
                // Gaussian pulse
                fields[[0, i, j, k]] = (-r2 / (2.0 * sigma * sigma)).exp();
            }
        }
    }
    
    // Time stepping
    let c_max = 1500.0;
    let dt = pstd_config.cfl_factor * grid.dx.min(grid.dy).min(grid.dz) / c_max;
    let n_steps = (simulation_time / dt).ceil() as usize;
    
    println!("  Time step: {:.3} μs", dt * 1e6);
    println!("  Number of steps: {}", n_steps);
    println!("  CFL number: {:.3}", c_max * dt / grid.dx);
    
    let mut max_pressure: f64 = 0.0;
    
    // Main time loop
    for step in 0..n_steps {
        // Create plugin context
        let context = PluginContext::new(step, n_steps, 1e6);
        
        // Process with plugins
        plugin_manager.update_all(&mut fields, grid, medium, dt, step as f64 * dt, &context)?;
        
        // Track maximum pressure
        let pressure = fields.index_axis(Axis(0), 0);
        let current_max = pressure.iter().map(|&p| p.abs()).fold(0.0, f64::max);
        max_pressure = max_pressure.max(current_max);
        
        // Progress indicator
        if step % (n_steps / 10) == 0 {
            print!(".");
            use std::io::{self, Write};
            io::stdout().flush().unwrap();
        }
    }
    println!(" Done!");
    
    let computation_time = start_time.elapsed().as_secs_f64();
    println!("  Computation time: {:.2} seconds", computation_time);
    println!("  Time per step: {:.2} ms", computation_time * 1000.0 / n_steps as f64);
    
    Ok(SimulationResult {
        final_pressure: fields.index_axis(Axis(0), 0).to_owned(),
        computation_time,
        time_steps: n_steps,
        max_pressure,
        method_name: "PSTD".to_string(),
    })
}

/// Run FDTD simulation
fn run_fdtd_simulation(
    grid: &Grid,
    medium: &dyn Medium,
    frequency: f64,
    simulation_time: f64,
) -> KwaversResult<SimulationResult> {
    let start_time = Instant::now();
    
    // Configure FDTD solver
    let fdtd_config = FdtdConfig {
        spatial_order: 4,
        staggered_grid: true,
        cfl_factor: 0.95, // FDTD can use larger CFL
        subgridding: false,
        subgrid_factor: 2,
    };
    
    // Create plugin
    let fdtd_plugin = FdtdPlugin::new(fdtd_config.clone(), grid)?;
    
    // Set up plugin manager
    let mut plugin_manager = PluginManager::new();
    plugin_manager.register(Box::new(fdtd_plugin))?;
    
    // Initialize fields (same as PSTD)
    let mut fields = Array4::zeros((13, grid.nx, grid.ny, grid.nz));
    
    // Add initial Gaussian pulse
    let center = (grid.nx / 2, grid.ny / 2, grid.nz / 2);
    let sigma = grid.nx as f64 * grid.dx / 10.0;
    
    for i in 0..grid.nx {
        for j in 0..grid.ny {
            for k in 0..grid.nz {
                let x = (i as f64 - center.0 as f64) * grid.dx;
                let y = (j as f64 - center.1 as f64) * grid.dy;
                let z = (k as f64 - center.2 as f64) * grid.dz;
                let r2 = x * x + y * y + z * z;
                
                fields[[0, i, j, k]] = (-r2 / (2.0 * sigma * sigma)).exp();
            }
        }
    }
    
    // Time stepping
    let c_max = 1500.0;
    let cfl_limit = match fdtd_config.spatial_order {
        2 => 1.0 / 3.0_f64.sqrt(),
        4 => 0.5,
        6 => 0.4,
        _ => 0.5,
    };
    let dt = fdtd_config.cfl_factor * cfl_limit * grid.dx.min(grid.dy).min(grid.dz) / c_max;
    let n_steps = (simulation_time / dt).ceil() as usize;
    
    println!("  Time step: {:.3} μs", dt * 1e6);
    println!("  Number of steps: {}", n_steps);
    println!("  CFL number: {:.3}", c_max * dt / grid.dx);
    
    let mut max_pressure: f64 = 0.0;
    
    // Main time loop
    for step in 0..n_steps {
        // Create plugin context
        let context = PluginContext::new(step, n_steps, 1e6);
        
        // Process with plugins
        plugin_manager.update_all(&mut fields, grid, medium, dt, step as f64 * dt, &context)?;
        
        // Track maximum pressure
        let pressure = fields.index_axis(Axis(0), 0);
        let current_max = pressure.iter().map(|&p| p.abs()).fold(0.0, f64::max);
        max_pressure = max_pressure.max(current_max);
        
        // Progress indicator
        if step % (n_steps / 10) == 0 {
            print!(".");
            use std::io::{self, Write};
            io::stdout().flush().unwrap();
        }
    }
    println!(" Done!");
    
    let computation_time = start_time.elapsed().as_secs_f64();
    println!("  Computation time: {:.2} seconds", computation_time);
    println!("  Time per step: {:.2} ms", computation_time * 1000.0 / n_steps as f64);
    
    Ok(SimulationResult {
        final_pressure: fields.index_axis(Axis(0), 0).to_owned(),
        computation_time,
        time_steps: n_steps,
        max_pressure,
        method_name: "FDTD".to_string(),
    })
}

/// Compare results from PSTD and FDTD
fn compare_results(pstd: &SimulationResult, fdtd: &SimulationResult, _grid: &Grid) {
    // Compute difference
    let diff = &pstd.final_pressure - &fdtd.final_pressure;
    let max_diff = diff.iter().map(|&d| d.abs()).fold(0.0, f64::max);
    let rms_diff = (diff.iter().map(|&d| d * d).sum::<f64>() / diff.len() as f64).sqrt();
    
    // Normalize by maximum pressure
    let normalized_max_diff = max_diff / pstd.max_pressure.max(fdtd.max_pressure);
    let normalized_rms_diff = rms_diff / pstd.max_pressure.max(fdtd.max_pressure);
    
    println!("Accuracy comparison:");
    println!("  Maximum difference: {:.2e} ({:.2}%)", max_diff, normalized_max_diff * 100.0);
    println!("  RMS difference: {:.2e} ({:.2}%)", rms_diff, normalized_rms_diff * 100.0);
    
    println!("\nPerformance comparison:");
    println!("  PSTD computation time: {:.2} s", pstd.computation_time);
    println!("  FDTD computation time: {:.2} s", fdtd.computation_time);
    println!("  Speedup factor: {:.2}x", fdtd.computation_time / pstd.computation_time);
    
    println!("\nTime stepping:");
    println!("  PSTD time steps: {}", pstd.time_steps);
    println!("  FDTD time steps: {}", fdtd.time_steps);
    println!("  Step ratio: {:.2}x", fdtd.time_steps as f64 / pstd.time_steps as f64);
}

/// Analyze numerical dispersion
fn analyze_dispersion(
    pstd: &SimulationResult,
    fdtd: &SimulationResult,
    grid: &Grid,
    wavelength: f64,
) {
    // Extract central slice
    let z_slice = grid.nz / 2;
    let pstd_slice = pstd.final_pressure.slice(ndarray::s![.., .., z_slice]);
    let fdtd_slice = fdtd.final_pressure.slice(ndarray::s![.., .., z_slice]);
    
    // Find wave peaks along x-axis at center
    let y_center = grid.ny / 2;
    let mut pstd_peaks = Vec::new();
    let mut fdtd_peaks = Vec::new();
    
    // Simple peak detection
    for i in 1..grid.nx-1 {
        let pstd_val = pstd_slice[[i, y_center]];
        let fdtd_val = fdtd_slice[[i, y_center]];
        
        // Check if local maximum
        if pstd_val > pstd_slice[[i-1, y_center]] && pstd_val > pstd_slice[[i+1, y_center]] && pstd_val > 0.1 * pstd.max_pressure {
            pstd_peaks.push(i);
        }
        if fdtd_val > fdtd_slice[[i-1, y_center]] && fdtd_val > fdtd_slice[[i+1, y_center]] && fdtd_val > 0.1 * fdtd.max_pressure {
            fdtd_peaks.push(i);
        }
    }
    
    // Calculate average wavelength from peak spacing
    let calc_avg_wavelength = |peaks: &[usize]| -> Option<f64> {
        if peaks.len() < 2 {
            return None;
        }
        let spacings: Vec<f64> = peaks.windows(2)
            .map(|w| (w[1] - w[0]) as f64 * grid.dx)
            .collect();
        Some(spacings.iter().sum::<f64>() / spacings.len() as f64)
    };
    
    if let (Some(pstd_wl), Some(fdtd_wl)) = (calc_avg_wavelength(&pstd_peaks), calc_avg_wavelength(&fdtd_peaks)) {
        println!("Numerical dispersion analysis:");
        println!("  Theoretical wavelength: {:.3} mm", wavelength * 1000.0);
        println!("  PSTD measured wavelength: {:.3} mm ({:.2}% error)", 
                 pstd_wl * 1000.0, (pstd_wl - wavelength) / wavelength * 100.0);
        println!("  FDTD measured wavelength: {:.3} mm ({:.2}% error)", 
                 fdtd_wl * 1000.0, (fdtd_wl - wavelength) / wavelength * 100.0);
    } else {
        println!("Numerical dispersion analysis: Insufficient peaks detected for analysis");
    }
    
    // Phase velocity error estimate
    let ppw = wavelength / grid.dx; // Points per wavelength
    let pstd_phase_error = estimate_pstd_phase_error(ppw);
    let fdtd_phase_error = estimate_fdtd_phase_error(ppw, 4); // 4th order
    
    println!("\nTheoretical phase velocity error:");
    println!("  PSTD: ~{:.2e} (spectral accuracy)", pstd_phase_error);
    println!("  FDTD (4th order): ~{:.2}%", fdtd_phase_error * 100.0);
}

/// Estimate PSTD phase velocity error
fn estimate_pstd_phase_error(points_per_wavelength: f64) -> f64 {
    // PSTD has spectral accuracy, error decreases exponentially with PPW
    (-points_per_wavelength).exp()
}

/// Estimate FDTD phase velocity error
fn estimate_fdtd_phase_error(points_per_wavelength: f64, order: usize) -> f64 {
    // Approximate phase velocity error for FDTD
    let k_dx = 2.0 * PI / points_per_wavelength;
    match order {
        2 => 1.0 - (k_dx).sin() / k_dx,
        4 => 1.0 - (8.0 * (k_dx).sin() - (2.0 * k_dx).sin()) / (6.0 * k_dx),
        6 => 1.0 - (45.0 * (k_dx).sin() - 9.0 * (2.0 * k_dx).sin() + (3.0 * k_dx).sin()) / (30.0 * k_dx),
        _ => 0.01, // Default estimate
    }
}