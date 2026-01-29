//! Factory Pattern Example - Phase 2 Enhancement
//!
//! Demonstrates automatic CFL calculation and grid spacing validation.
//!
//! Run with: cargo run --example phase2_factory

use kwavers::simulation::factory::{
    AccuracyLevel, CFLCalculator, GridSpacingCalculator, SimulationFactory, SimulationPreset,
};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== Kwavers Factory Pattern Examples ===\n");

    // Example 1: Automatic CFL and grid spacing calculation
    println!("1. Automatic Parameter Calculation");
    println!("   Factory automatically calculates:");
    println!("   - Grid spacing from frequency and PPW");
    println!("   - Time step from CFL condition");
    println!("   - Number of time steps from duration\n");

    let config = SimulationFactory::new()
        .frequency(5e6) // 5 MHz
        .domain_size(0.1, 0.1, 0.05)
        .points_per_wavelength(8.0) // Standard accuracy
        .auto_configure()
        .build()?;

    println!("   Configuration created:");
    println!("   - Grid spacing: {:.2e} m", config.grid.dx);
    println!("   - Time step: {:.2e} s", config.simulation.dt);
    println!("   - CFL number: {:.3}", config.simulation.cfl);
    println!(
        "   - Grid points: {}×{}×{}",
        config.grid.nx, config.grid.ny, config.grid.nz
    );
    println!("   - Time steps: {}\n", config.simulation.num_time_steps);

    // Example 2: CFL Calculator
    println!("2. CFL Calculator");
    println!("   Calculate optimal time step for stability\n");

    let cfl_calc = CFLCalculator::new(3, 1500.0, 1e-4); // 3D, water, 0.1mm spacing
    let dt = cfl_calc.calculate_time_step(0.3)?;
    let cfl_actual = cfl_calc.calculate_cfl_number(dt);

    println!("   CFL Calculator results:");
    println!("   - Wave speed: 1500 m/s");
    println!("   - Grid spacing: 0.1 mm");
    println!("   - Desired CFL: 0.3");
    println!("   - Calculated dt: {:.2e} s", dt);
    println!("   - Actual CFL: {:.3}\n", cfl_actual);

    // Example 3: Grid Spacing Calculator
    println!("3. Grid Spacing Calculator");
    println!("   Ensure Nyquist criterion is met\n");

    let spacing_calc = GridSpacingCalculator::new(5e6, 1500.0, 8.0); // 5 MHz, water, 8 PPW
    let dx = spacing_calc.calculate_grid_spacing()?;
    let wavelength = spacing_calc.wavelength();
    let ppw_actual = spacing_calc.actual_ppw(dx);

    println!("   Grid Spacing results:");
    println!("   - Frequency: 5 MHz");
    println!("   - Wavelength: {:.2e} m", wavelength);
    println!("   - Desired PPW: 8.0");
    println!("   - Calculated dx: {:.2e} m", dx);
    println!("   - Actual PPW: {:.2}\n", ppw_actual);

    // Example 4: Accuracy Level Presets
    println!("4. Accuracy Level Presets");
    println!("   Automatic PPW and CFL configuration\n");

    for (name, level) in [
        ("Preview", AccuracyLevel::Preview),
        ("Standard", AccuracyLevel::Standard),
        ("High Accuracy", AccuracyLevel::HighAccuracy),
        ("Research", AccuracyLevel::Research),
    ] {
        let config = SimulationFactory::new()
            .frequency(5e6)
            .domain_size(0.1, 0.1, 0.05)
            .accuracy(level)
            .auto_configure()
            .build()?;

        let wavelength = 1500.0 / 5e6;
        let ppw = wavelength / config.grid.dx;

        println!("   {}:", name);
        println!("     - PPW: {:.1}", ppw);
        println!("     - CFL: {:.3}", config.simulation.cfl);
        println!(
            "     - Grid points: {}×{}×{}",
            config.grid.nx, config.grid.ny, config.grid.nz
        );
        println!(
            "     - Memory: {:.2} GB\n",
            (config.grid.nx * config.grid.ny * config.grid.nz * 4 * 8) as f64 / 1e9
        );
    }

    // Example 5: Simulation Presets
    println!("5. Simulation Presets");
    println!("   Pre-configured for common applications\n");

    SimulationPreset::print_catalog();

    // Example 6: Physics Validation
    println!("6. Physics Validation");
    println!("   Automatic constraint checking\n");

    use kwavers::simulation::factory::PhysicsValidator;

    let dx = 1e-4;
    let dt = 1e-8;
    let frequency = 1e6;
    let sound_speed = 1500.0;
    let domain_size = (0.1, 0.1, 0.05);

    match PhysicsValidator::validate_all(dx, dt, frequency, sound_speed, domain_size) {
        Ok(report) => {
            report.print();
        }
        Err(e) => {
            println!("   Validation failed: {}", e);
        }
    }

    println!("\n=== Factory Pattern Examples Complete ===");
    println!("The factory pattern reduces configuration errors and improves usability!");

    Ok(())
}
