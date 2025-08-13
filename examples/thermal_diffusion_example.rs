//! Example demonstrating the thermal diffusion solver with bioheat equation
//! 
//! This example shows:
//! - Standard heat diffusion
//! - Pennes bioheat equation with blood perfusion
//! - Thermal dose (CEM43) calculation
//! - Integration with acoustic heating
//! - Plugin-based usage

use kwavers::{
    error::KwaversResult,
    grid::Grid,
    medium::HomogeneousMedium,
    physics::{
        // composable::{ThermalDiffusionComponent, AcousticWaveComponent, PhysicsComponent},
        plugin::{PluginManager, PluginContext},
    },
    solver::thermal_diffusion::{ThermalDiffusionConfig, ThermalDiffusionPlugin},
};
use ndarray::{Array3, Array4, s};

fn main() -> KwaversResult<()> {
    // Initialize logging
    env_logger::init();
    
    println!("=== Thermal Diffusion Solver Example ===\n");
    
    // Create computational grid
    let grid = Grid::new(128, 128, 128, 1e-3, 1e-3, 1e-3); // 128mm cube, 1mm resolution
    println!("Grid: {}x{}x{} ({:.1}mm x {:.1}mm x {:.1}mm)",
        grid.nx, grid.ny, grid.nz,
        grid.nx as f64 * grid.dx * 1000.0,
        grid.ny as f64 * grid.dy * 1000.0,
        grid.nz as f64 * grid.dz * 1000.0
    );
    
    // Create medium with thermal properties
    let medium = HomogeneousMedium::new(
        1000.0,  // density [kg/m³]
        1500.0,  // sound speed [m/s]
        &grid,
        0.5,     // absorption coefficient [Np/m]
        2.0      // nonlinearity parameter
    );
    
    // Example 1: Standard thermal diffusion
    println!("\n1. Standard Thermal Diffusion");
    println!("-----------------------------");
    standard_thermal_diffusion(&grid, &medium)?;
    
    // Example 2: Bioheat equation with perfusion
    println!("\n2. Pennes Bioheat Equation");
    println!("--------------------------");
    bioheat_equation(&grid, &medium)?;
    
    // Example 3: Thermal dose tracking
    println!("\n3. Thermal Dose (CEM43) Tracking");
    println!("---------------------------------");
    thermal_dose_tracking(&grid, &medium)?;
    
    // Example 4: Plugin-based usage with acoustic heating
    println!("\n4. Plugin-Based Thermal Simulation");
    println!("-----------------------------------");
    plugin_based_simulation(&grid, &medium)?;
    
    // Example 5: Using composable component
    // TODO: Enable when composable module is implemented
    // println!("\n5. Composable Component Usage");
    // println!("-----------------------------");
    // composable_component_usage(&grid, &medium)?;
    
    Ok(())
}

/// Example 1: Standard thermal diffusion without bioheat terms
fn standard_thermal_diffusion(grid: &Grid, medium: &dyn kwavers::medium::Medium) -> KwaversResult<()> {
    use kwavers::solver::thermal_diffusion::ThermalDiffusionSolver;
    
    // Configure for standard diffusion
    let mut config = ThermalDiffusionConfig::default();
    config.enable_bioheat = false;
    config.track_thermal_dose = false;
    
    // Create solver
    let mut solver = ThermalDiffusionSolver::new(config, grid)?;
    
    // Set initial temperature distribution (hot spot in center)
    let mut initial_temp = Array3::from_elem((grid.nx, grid.ny, grid.nz), 310.15); // 37°C
    let center = (grid.nx / 2, grid.ny / 2, grid.nz / 2);
    
    // Create a Gaussian hot spot
    for i in 0..grid.nx {
        for j in 0..grid.ny {
            for k in 0..grid.nz {
                let r2 = ((i as f64 - center.0 as f64).powi(2) +
                         (j as f64 - center.1 as f64).powi(2) +
                         (k as f64 - center.2 as f64).powi(2)) * grid.dx.powi(2);
                let sigma2 = (10e-3_f64).powi(2); // 10mm standard deviation
                initial_temp[[i, j, k]] = 310.15 + 10.0 * (-r2 / (2.0 * sigma2)).exp(); // Up to 47°C at center
            }
        }
    }
    
    solver.set_temperature(initial_temp)?;
    
    // No external heat source
    let heat_source = grid.zeros_array();
    
    // Simulate for 10 seconds
    let dt = 1e-3; // 1ms time step
    let n_steps = 10_000;
    
    println!("Initial max temperature: {:.1}°C", 
        solver.temperature().iter().fold(f64::NEG_INFINITY, |a: f64, &b| a.max(b)) - 273.15);
    
    for step in 0..n_steps {
        solver.update(&heat_source, grid, medium, dt)?;
        
        if step % 1000 == 999 {
            let max_temp = solver.temperature().iter().fold(f64::NEG_INFINITY, |a: f64, &b| a.max(b));
            println!("t = {:.1}s: Max temperature = {:.1}°C", 
                (step + 1) as f64 * dt, max_temp - 273.15);
        }
    }
    
    Ok(())
}

/// Example 2: Pennes bioheat equation with blood perfusion
fn bioheat_equation(grid: &Grid, medium: &dyn kwavers::medium::Medium) -> KwaversResult<()> {
    use kwavers::solver::thermal_diffusion::ThermalDiffusionSolver;
    
    // Configure with bioheat equation
    let mut config = ThermalDiffusionConfig::default();
    config.enable_bioheat = true;
    config.perfusion_rate = 0.5e-3; // 0.5 mL/g/min typical muscle perfusion
    config.track_thermal_dose = false;
    
    let mut solver = ThermalDiffusionSolver::new(config, grid)?;
    
    // Uniform elevated temperature
    let initial_temp = Array3::from_elem((grid.nx, grid.ny, grid.nz), 315.15); // 42°C
    solver.set_temperature(initial_temp)?;
    
    // No external heating
    let heat_source = grid.zeros_array();
    
    // Simulate cooling due to perfusion
    let dt = 1e-2; // 10ms time step
    let n_steps = 1000; // 10 seconds
    
    println!("Initial temperature: 42.0°C");
    println!("Blood perfusion rate: 0.5 mL/g/min");
    
    for step in 0..n_steps {
        solver.update(&heat_source, grid, medium, dt)?;
        
        if step % 100 == 99 {
            let avg_temp = solver.temperature().mean().unwrap();
            println!("t = {:.1}s: Average temperature = {:.1}°C", 
                (step + 1) as f64 * dt, avg_temp - 273.15);
        }
    }
    
    Ok(())
}

/// Example 3: Thermal dose calculation for hyperthermia treatment
fn thermal_dose_tracking(grid: &Grid, medium: &dyn kwavers::medium::Medium) -> KwaversResult<()> {
    use kwavers::solver::thermal_diffusion::ThermalDiffusionSolver;
    
    // Configure with dose tracking
    let config = ThermalDiffusionConfig {
        enable_bioheat: true,
        perfusion_rate: 0.5e-3,
        track_thermal_dose: true,
        dose_reference_temp: 43.0, // CEM43 reference
        ..Default::default()
    };
    
    let mut solver = ThermalDiffusionSolver::new(config, grid)?;
    
    // Set temperature to 45°C in a small region (simulating focused heating)
    let mut initial_temp = Array3::from_elem((grid.nx, grid.ny, grid.nz), 310.15); // 37°C
    let center = (grid.nx / 2, grid.ny / 2, grid.nz / 2);
    let radius = 10; // 10mm radius
    
    for i in (center.0 - radius)..(center.0 + radius) {
        for j in (center.1 - radius)..(center.1 + radius) {
            for k in (center.2 - radius)..(center.2 + radius) {
                let r2 = ((i as i32 - center.0 as i32).pow(2) +
                         (j as i32 - center.1 as i32).pow(2) +
                         (k as i32 - center.2 as i32).pow(2)) as f64;
                if r2 < (radius as f64).powi(2) {
                    initial_temp[[i, j, k]] = 318.15; // 45°C
                }
            }
        }
    }
    
    solver.set_temperature(initial_temp)?;
    
    // No additional heating (just diffusion and perfusion)
    let heat_source = Array3::zeros((grid.nx, grid.ny, grid.nz));
    
    // Simulate for 5 minutes
    let dt = 1.0; // 1 second time step
    let n_steps = 300; // 5 minutes
    
    println!("Heating region to 45°C for thermal dose calculation");
    
    for step in 0..n_steps {
        solver.update(&heat_source, grid, medium, dt)?;
        
        if step % 60 == 59 {
            let dose = solver.thermal_dose().unwrap();
            let max_dose = dose.iter().fold(0.0_f64, |a, &b| a.max(b));
            let dose_volume = dose.iter().filter(|&&d| d > 240.0).count(); // 240 CEM43 = complete cell kill
            
            println!("t = {}min: Max dose = {:.1} CEM43, Ablation volume = {} voxels", 
                (step + 1) / 60, max_dose, dose_volume);
        }
    }
    
    Ok(())
}

/// Example 4: Using thermal diffusion as a plugin with acoustic heating
fn plugin_based_simulation(grid: &Grid, medium: &dyn kwavers::medium::Medium) -> KwaversResult<()> {
    // Create plugin manager
    let mut plugin_manager = PluginManager::new();
    
    // Configure thermal diffusion plugin
    let thermal_config = ThermalDiffusionConfig {
        enable_bioheat: true,
        perfusion_rate: 0.5e-3,
        track_thermal_dose: true,
        spatial_order: 4,
        ..Default::default()
    };
    
    // Create and register thermal plugin
    let thermal_plugin = ThermalDiffusionPlugin::new(thermal_config, grid)?;
    plugin_manager.register(Box::new(thermal_plugin))?;
    
    // Initialize fields (pressure and temperature)
    let mut fields = Array4::zeros((13, grid.nx, grid.ny, grid.nz));
    
    // Set initial temperature
    fields.slice_mut(s![2, .., .., ..]).fill(310.15); // 37°C
    
    // Create a pressure field (simulating focused ultrasound)
    let center = (grid.nx / 2, grid.ny / 2, grid.nz / 2);
    for i in 0..grid.nx {
        for j in 0..grid.ny {
            for k in 0..grid.nz {
                let r = (((i as f64 - center.0 as f64).powi(2) +
                         (j as f64 - center.1 as f64).powi(2) +
                         (k as f64 - center.2 as f64).powi(2)) * grid.dx.powi(2)).sqrt();
                
                // Focused beam with 1 MPa peak pressure
                if r < 20e-3 { // 20mm focal region
                    fields[[0, i, j, k]] = 1e6 * (1.0 - r / 20e-3); // Linear taper
                }
            }
        }
    }
    
    println!("Simulating focused ultrasound heating with plugin system");
    
    // Run simulation
    let dt = 1e-3;
    let n_steps = 5000; // 5 seconds
    
    for step in 0..n_steps {
        let context = PluginContext::new(step, n_steps, 1e6); // 1 MHz
        plugin_manager.update_all(&mut fields, grid, medium, dt, step as f64 * dt, &context)?;
        
        if step % 1000 == 999 {
            let max_temp = fields.slice(s![2, .., .., ..]).iter()
                .fold(f64::NEG_INFINITY, |a: f64, &b| a.max(b));
            println!("t = {:.1}s: Max temperature = {:.1}°C", 
                (step + 1) as f64 * dt, max_temp - 273.15);
        }
    }
    
    Ok(())
}

/// Example 5: Using the composable component system
#[allow(dead_code)]
fn composable_component_usage(_grid: &Grid, _medium: &dyn kwavers::medium::Medium) -> KwaversResult<()> {
    // TODO: Implement when composable module is ready
    /*
    use kwavers::physics::composable::PhysicsPipeline;
    
    // Create physics pipeline
    let mut pipeline = PhysicsPipeline::new();
    
    // Add acoustic wave component
    let mut acoustic = AcousticWaveComponent::new("acoustic".to_string());
    acoustic.initialize(grid, medium)?;
    pipeline.add_component(Box::new(acoustic))?;
    
    // Add thermal diffusion component with bioheat
    let mut thermal = ThermalDiffusionComponent::new("thermal".to_string())
        .with_bioheat(0.5e-3)
        .with_thermal_dose();
    thermal.initialize(grid, medium)?;
    pipeline.add_component(Box::new(thermal))?;
    
    // Create physics context
    let mut context = kwavers::physics::composable::PhysicsContext::new(1e6); // 1 MHz
    
    // Create fields
    let mut fields = Array4::zeros((13, grid.nx, grid.ny, grid.nz));
    
    // Set initial conditions
    fields.slice_mut(s![2, .., .., ..]).fill(310.15); // 37°C
    
    // Add a pressure source in the center
    let center = (grid.nx / 2, grid.ny / 2, grid.nz / 2);
    fields[[0, center.0, center.1, center.2]] = 1e6; // 1 MPa
    
    println!("Running composable physics pipeline with acoustic and thermal components");
    
    // Run simulation
    let dt = 1e-4;
    let n_steps = 1000;
    
    for step in 0..n_steps {
        pipeline.execute(&mut fields, grid, medium, dt, step as f64 * dt, &mut context)?;
        
        if step % 200 == 199 {
            println!("t = {:.1}ms: Step completed", 
                (step + 1) as f64 * dt * 1000.0);
        }
    }
    */
    
    Ok(())
}