//! Simple tests to verify FDTD and PSTD solvers are working correctly

use kwavers::*;
use kwavers::solver::fdtd::{FdtdConfig, FdtdPlugin};
use kwavers::solver::pstd::{PstdConfig, PstdPlugin};
use kwavers::physics::plugin::{PluginManager, PluginContext};
use ndarray::Array4;

/// Test that FDTD solver doesn't crash and produces non-zero output
#[test]
fn test_fdtd_basic() {
    let grid = Grid::new(32, 32, 32, 1e-3, 1e-3, 1e-3);
    let medium = HomogeneousMedium::water(&grid);
    
    // Simple initial condition
    let mut initial_pressure = grid.zeros_array();
    initial_pressure[[16, 16, 16]] = 1e6; // Point source
    
    let config = FdtdConfig {
        spatial_order: 2, // Use 2nd order for simplicity
        staggered_grid: true,
        cfl_factor: 0.5,
        subgridding: false,
        subgrid_factor: 2,
    };
    
    let mut plugin_manager = PluginManager::new();
    plugin_manager.register(Box::new(FdtdPlugin::new(config, &grid).unwrap())).unwrap();
    
    // Initialize fields
    let mut fields = Array4::zeros((7, grid.nx, grid.ny, grid.nz));
    fields.slice_mut(ndarray::s![0, .., .., ..]).assign(&initial_pressure);
    
    // Time stepping
    let c = 1500.0;
    let dt = config.cfl_factor * grid.dx / c;
    
    plugin_manager.initialize_all(&grid, &medium).unwrap();
    
    // Run for just a few steps
    for step in 0..10 {
        let context = PluginContext::new(step, 10, 1e6);
        plugin_manager.update_all(&mut fields, &grid, &medium, dt, step as f64 * dt, &context).unwrap();
    }
    
    // Check that we have non-zero output
    let final_pressure = fields.slice(ndarray::s![0, .., .., ..]);
    let max_pressure = final_pressure.fold(0.0f64, |a, &b| a.max(b.abs()));
    
    assert!(max_pressure > 0.0, "FDTD produced zero output");
    println!("FDTD max pressure after 10 steps: {}", max_pressure);
}

/// Test that PSTD solver doesn't crash and produces non-zero output
#[test]
fn test_pstd_basic() {
    let grid = Grid::new(32, 32, 32, 1e-3, 1e-3, 1e-3);
    let medium = HomogeneousMedium::water(&grid);
    
    // Simple initial condition
    let mut initial_pressure = grid.zeros_array();
    initial_pressure[[16, 16, 16]] = 1e6; // Point source
    
    let config = PstdConfig {
        k_space_correction: true,
        k_space_order: 2,
        anti_aliasing: false, // Disable for simplicity
        pml_stencil_size: 4,
        cfl_factor: 0.3,
    };
    
    let mut plugin_manager = PluginManager::new();
    plugin_manager.register(Box::new(PstdPlugin::new(config, &grid).unwrap())).unwrap();
    
    // Initialize fields
    let mut fields = Array4::zeros((7, grid.nx, grid.ny, grid.nz));
    fields.slice_mut(ndarray::s![0, .., .., ..]).assign(&initial_pressure);
    
    // Time stepping
    let c = 1500.0;
    let dt = config.cfl_factor * grid.dx / c;
    
    plugin_manager.initialize_all(&grid, &medium).unwrap();
    
    // Run for just a few steps
    for step in 0..10 {
        let context = PluginContext::new(step, 10, 1e6);
        plugin_manager.update_all(&mut fields, &grid, &medium, dt, step as f64 * dt, &context).unwrap();
    }
    
    // Check that we have non-zero output
    let final_pressure = fields.slice(ndarray::s![0, .., .., ..]);
    let max_pressure = final_pressure.fold(0.0f64, |a, &b| a.max(b.abs()));
    
    assert!(max_pressure > 0.0, "PSTD produced zero output");
    println!("PSTD max pressure after 10 steps: {}", max_pressure);
}

/// Test that both solvers propagate waves (pressure should spread from source)
#[test]
fn test_wave_propagation() {
    let grid = Grid::new(64, 64, 64, 1e-3, 1e-3, 1e-3);
    let medium = HomogeneousMedium::water(&grid);
    
    // Gaussian initial condition
    let mut initial_pressure = grid.zeros_array();
    let center = 32;
    let sigma: f64 = 3.0;
    
    for i in 0..grid.nx {
        for j in 0..grid.ny {
            for k in 0..grid.nz {
                let r2 = ((i as f64 - center as f64).powi(2) + 
                         (j as f64 - center as f64).powi(2) + 
                         (k as f64 - center as f64).powi(2)) / sigma.powi(2);
                initial_pressure[[i, j, k]] = 1e6 * (-r2).exp();
            }
        }
    }
    
    // Test FDTD
    {
        let config = FdtdConfig {
            spatial_order: 2,
            staggered_grid: true,
            cfl_factor: 0.5,
            subgridding: false,
            subgrid_factor: 2,
        };
        
        let mut plugin_manager = PluginManager::new();
        plugin_manager.register(Box::new(FdtdPlugin::new(config, &grid).unwrap())).unwrap();
        
        let mut fields = Array4::zeros((7, grid.nx, grid.ny, grid.nz));
        fields.slice_mut(ndarray::s![0, .., .., ..]).assign(&initial_pressure);
        
        let c = 1500.0;
        let dt = config.cfl_factor * grid.dx / c;
        
        plugin_manager.initialize_all(&grid, &medium).unwrap();
        
        // Record initial energy
        let initial_energy = fields.slice(ndarray::s![0, .., .., ..])
            .fold(0.0f64, |a, &b| a + b.powi(2));
        
        // Run for 20 steps
        for step in 0..20 {
            let context = PluginContext::new(step, 20, 1e6);
            plugin_manager.update_all(&mut fields, &grid, &medium, dt, step as f64 * dt, &context).unwrap();
        }
        
        // Check that wave has spread (pressure at center should decrease)
        let center_pressure = fields[[0, center, center, center]].abs();
        let initial_center = initial_pressure[[center, center, center]];
        
        assert!(center_pressure < initial_center * 0.9, 
            "FDTD: Wave should spread from center. Initial: {}, Final: {}", 
            initial_center, center_pressure);
        
        println!("FDTD: Initial center pressure: {}, Final: {}", initial_center, center_pressure);
    }
    
    // Test PSTD
    {
        let config = PstdConfig {
            k_space_correction: true,
            k_space_order: 2,
            anti_aliasing: false,
            pml_stencil_size: 4,
            cfl_factor: 0.3,
        };
        
        let mut plugin_manager = PluginManager::new();
        plugin_manager.register(Box::new(PstdPlugin::new(config, &grid).unwrap())).unwrap();
        
        let mut fields = Array4::zeros((7, grid.nx, grid.ny, grid.nz));
        fields.slice_mut(ndarray::s![0, .., .., ..]).assign(&initial_pressure);
        
        let c = 1500.0;
        let dt = config.cfl_factor * grid.dx / c;
        
        plugin_manager.initialize_all(&grid, &medium).unwrap();
        
        // Run for 20 steps
        for step in 0..20 {
            let context = PluginContext::new(step, 20, 1e6);
            plugin_manager.update_all(&mut fields, &grid, &medium, dt, step as f64 * dt, &context).unwrap();
        }
        
        // Check that wave has spread
        let center_pressure = fields[[0, center, center, center]].abs();
        let initial_center = initial_pressure[[center, center, center]];
        
        assert!(center_pressure < initial_center * 0.9, 
            "PSTD: Wave should spread from center. Initial: {}, Final: {}", 
            initial_center, center_pressure);
        
        println!("PSTD: Initial center pressure: {}, Final: {}", initial_center, center_pressure);
    }
}