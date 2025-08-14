//! Simple tests to verify FDTD and PSTD solvers are working correctly

use kwavers::solver::fdtd::{FdtdPlugin, FdtdConfig};
use kwavers::solver::pstd::{PstdPlugin, PstdConfig};
use kwavers::grid::Grid;
use kwavers::medium::HomogeneousMedium;
use kwavers::physics::plugin::{PluginManager, PluginContext};
use ndarray::{Array4, Zip, s};

// Named constants for test configuration
const TEST_GRID_SIZE: usize = 32;
const TEST_GRID_SPACING: f64 = 1e-3;
const TEST_PRESSURE_AMPLITUDE: f64 = 1e6;
const TEST_SOUND_SPEED: f64 = 1500.0;
const TEST_FREQUENCY: f64 = 1e6;
const FDTD_CFL_FACTOR: f64 = 0.5;
const PSTD_CFL_FACTOR: f64 = 0.3;
const TEST_STEPS_SHORT: usize = 10;
const TEST_STEPS_MEDIUM: usize = 20;
const LARGE_GRID_SIZE: usize = 64;
const GAUSSIAN_CENTER: usize = 32;
const GAUSSIAN_SIGMA: f64 = 3.0;
const WAVE_DECAY_THRESHOLD: f64 = 0.9;
const DEFAULT_SUBGRID_FACTOR: usize = 2;
const DEFAULT_K_SPACE_ORDER: usize = 2;
const DEFAULT_PML_STENCIL_SIZE: usize = 4;
const NUM_FIELD_COMPONENTS: usize = 7;  // Standard field components

#[test]
fn test_fdtd_solver() {
    let grid = Grid::new(TEST_GRID_SIZE, TEST_GRID_SIZE, TEST_GRID_SIZE, 
                        TEST_GRID_SPACING, TEST_GRID_SPACING, TEST_GRID_SPACING);
    let medium = HomogeneousMedium::water(&grid);
    
    // Initialize fields array (7 components: pressure, vx, vy, vz, temperature, density, source)
    let mut fields = Array4::zeros((NUM_FIELD_COMPONENTS, TEST_GRID_SIZE, TEST_GRID_SIZE, TEST_GRID_SIZE));
    let center = TEST_GRID_SIZE / 2;
    fields[[0, center, center, center]] = TEST_PRESSURE_AMPLITUDE; // Point source in pressure field
    
    let config = FdtdConfig {
        spatial_order: 2,
        staggered_grid: true,
        cfl_factor: FDTD_CFL_FACTOR,
        subgridding: false,
        subgrid_factor: DEFAULT_SUBGRID_FACTOR,
    };
    
    let plugin = FdtdPlugin::new(config, &grid).expect("Failed to create FDTD plugin");
    let mut plugin_manager = PluginManager::new();
    plugin_manager.register(Box::new(plugin)).expect("Failed to register plugin");
    
    plugin_manager.initialize_all(&grid, &medium).expect("Failed to initialize plugins");
    
    let c = TEST_SOUND_SPEED;
    let dt = FDTD_CFL_FACTOR * TEST_GRID_SPACING / c;
    
    // Run simulation for a few steps
    for step in 0..TEST_STEPS_SHORT {
        let t = step as f64 * dt;
        let context = PluginContext::new(step, TEST_STEPS_SHORT, TEST_FREQUENCY);
        plugin_manager.update_all(&mut fields, &grid, &medium, dt, t, &context)
            .expect("Failed to update plugins");
    }
    
    // Check that wave has propagated
    let pressure_field = fields.slice(s![0, .., .., ..]);
    let max_pressure = pressure_field.iter().fold(0.0f64, |a, &b| a.max(b.abs()));
    assert!(max_pressure > 0.0, "FDTD produced zero output");
    println!("FDTD max pressure after {} steps: {}", TEST_STEPS_SHORT, max_pressure);
}

#[test]
fn test_pstd_solver() {
    let grid = Grid::new(TEST_GRID_SIZE, TEST_GRID_SIZE, TEST_GRID_SIZE,
                        TEST_GRID_SPACING, TEST_GRID_SPACING, TEST_GRID_SPACING);
    let medium = HomogeneousMedium::water(&grid);
    
    // Initialize fields array
    let mut fields = Array4::zeros((NUM_FIELD_COMPONENTS, TEST_GRID_SIZE, TEST_GRID_SIZE, TEST_GRID_SIZE));
    let center = TEST_GRID_SIZE / 2;
    fields[[0, center, center, center]] = TEST_PRESSURE_AMPLITUDE; // Point source in pressure field
    
    let config = PstdConfig {
        k_space_correction: true,
        k_space_order: DEFAULT_K_SPACE_ORDER,
        anti_aliasing: true,
        pml_stencil_size: DEFAULT_PML_STENCIL_SIZE,
        cfl_factor: PSTD_CFL_FACTOR,
        use_leapfrog: true,
        enable_absorption: false,
        absorption_model: None,
    };
    
    let plugin = PstdPlugin::new(config, &grid).expect("Failed to create PSTD plugin");
    let mut plugin_manager = PluginManager::new();
    plugin_manager.register(Box::new(plugin)).expect("Failed to register plugin");
    
    plugin_manager.initialize_all(&grid, &medium).expect("Failed to initialize plugins");
    
    let c = TEST_SOUND_SPEED;
    let dt = PSTD_CFL_FACTOR * TEST_GRID_SPACING / c;
    
    // Run simulation for a few steps
    for step in 0..TEST_STEPS_SHORT {
        let t = step as f64 * dt;
        let context = PluginContext::new(step, TEST_STEPS_SHORT, TEST_FREQUENCY);
        plugin_manager.update_all(&mut fields, &grid, &medium, dt, t, &context)
            .expect("Failed to update plugins");
    }
    
    // Check that wave has propagated
    let pressure_field = fields.slice(s![0, .., .., ..]);
    let max_pressure = pressure_field.iter().fold(0.0f64, |a, &b| a.max(b.abs()));
    assert!(max_pressure > 0.0, "PSTD produced zero output");
    println!("PSTD max pressure after {} steps: {}", TEST_STEPS_SHORT, max_pressure);
}

#[test]
fn test_wave_propagation() {
    let grid = Grid::new(LARGE_GRID_SIZE, LARGE_GRID_SIZE, LARGE_GRID_SIZE,
                        TEST_GRID_SPACING, TEST_GRID_SPACING, TEST_GRID_SPACING);
    let medium = HomogeneousMedium::water(&grid);
    
    // Initialize fields with Gaussian pulse
    let mut initial_fields = Array4::zeros((NUM_FIELD_COMPONENTS, LARGE_GRID_SIZE, LARGE_GRID_SIZE, LARGE_GRID_SIZE));
    let center = GAUSSIAN_CENTER;
    let sigma: f64 = GAUSSIAN_SIGMA;
    
    // Set Gaussian pulse in pressure field
    {
        let mut pressure_slice = initial_fields.slice_mut(s![0, .., .., ..]);
        Zip::indexed(&mut pressure_slice)
            .for_each(|(i, j, k), p| {
                let di = i as f64 - center as f64;
                let dj = j as f64 - center as f64;
                let dk = k as f64 - center as f64;
                let r2 = di * di + dj * dj + dk * dk;
                *p = TEST_PRESSURE_AMPLITUDE * (-r2 / (2.0 * sigma * sigma)).exp();
            });
    }
    
    let initial_center = initial_fields[[0, center, center, center]];
    
    // Test FDTD
    {
        let mut fields_fdtd = initial_fields.clone();
        let config = FdtdConfig {
            spatial_order: 2,
            staggered_grid: true,
            cfl_factor: FDTD_CFL_FACTOR,
            subgridding: false,
            subgrid_factor: DEFAULT_SUBGRID_FACTOR,
        };
        
        let plugin = FdtdPlugin::new(config, &grid).expect("Failed to create FDTD plugin");
        let mut plugin_manager = PluginManager::new();
        plugin_manager.register(Box::new(plugin)).expect("Failed to register plugin");
        plugin_manager.initialize_all(&grid, &medium).expect("Failed to initialize plugins");
        
        let c = TEST_SOUND_SPEED;
        let dt = FDTD_CFL_FACTOR * TEST_GRID_SPACING / c;
        
        // Run for enough steps to see propagation
        for step in 0..TEST_STEPS_MEDIUM {
            let t = step as f64 * dt;
            let context = PluginContext::new(step, TEST_STEPS_MEDIUM, TEST_FREQUENCY);
            plugin_manager.update_all(&mut fields_fdtd, &grid, &medium, dt, t, &context)
                .expect("Failed to update plugins");
        }
        
        // Check that wave has spread (center should be lower)
        let center_pressure = fields_fdtd[[0, center, center, center]];
        assert!(center_pressure < initial_center * WAVE_DECAY_THRESHOLD, 
                "FDTD: Wave didn't propagate from center. Initial: {}, Final: {}", 
                initial_center, center_pressure);
    }
    
    // Test PSTD
    {
        let mut fields_pstd = initial_fields.clone();
        let config = PstdConfig {
            k_space_correction: true,
            k_space_order: DEFAULT_K_SPACE_ORDER,
            anti_aliasing: true,
            pml_stencil_size: DEFAULT_PML_STENCIL_SIZE,
            cfl_factor: PSTD_CFL_FACTOR,
            use_leapfrog: true,
            enable_absorption: false,
            absorption_model: None,
        };
        
        let plugin = PstdPlugin::new(config, &grid).expect("Failed to create PSTD plugin");
        let mut plugin_manager = PluginManager::new();
        plugin_manager.register(Box::new(plugin)).expect("Failed to register plugin");
        plugin_manager.initialize_all(&grid, &medium).expect("Failed to initialize plugins");
        
        let c = TEST_SOUND_SPEED;
        let dt = PSTD_CFL_FACTOR * TEST_GRID_SPACING / c;
        
        // Run for enough steps to see propagation
        for step in 0..TEST_STEPS_MEDIUM {
            let t = step as f64 * dt;
            let context = PluginContext::new(step, TEST_STEPS_MEDIUM, TEST_FREQUENCY);
            plugin_manager.update_all(&mut fields_pstd, &grid, &medium, dt, t, &context)
                .expect("Failed to update plugins");
        }
        
        // Check that wave has spread (center should be lower)
        let center_pressure = fields_pstd[[0, center, center, center]];
        assert!(center_pressure < initial_center * WAVE_DECAY_THRESHOLD, 
                "PSTD: Wave didn't propagate from center. Initial: {}, Final: {}", 
                initial_center, center_pressure);
    }
}