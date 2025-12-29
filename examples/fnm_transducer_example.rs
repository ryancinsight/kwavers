//! Fast Nearfield Method (FNM) Transducer Field Computation Example
//!
//! This example demonstrates the Fast Nearfield Method for computing
//! ultrasound transducer pressure fields with O(n) complexity.

use kwavers::physics::transducer::fast_nearfield::{
    FNMConfig, FastNearfieldSolver, RectangularTransducer,
};
use ndarray::Array2;
use num_complex::Complex;
use std::time::Instant;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("Fast Nearfield Method (FNM) Transducer Example");
    println!("==============================================");

    // Configure FNM solver
    let config = FNMConfig {
        angular_spectrum_size: (128, 128), // Reasonable size for demonstration
        dx: 0.1e-3,                        // 0.1 mm grid spacing
        ..Default::default()
    };

    let mut solver = FastNearfieldSolver::new(config)?;

    // Define transducer geometry
    let transducer = RectangularTransducer {
        width: 5.0e-3,      // 5 mm width
        height: 5.0e-3,     // 5 mm height
        frequency: 2.0e6,   // 2 MHz center frequency
        elements: (16, 16), // 16x16 element array
    };

    println!(
        "Transducer: {}x{} elements, {}x{} mm, {} MHz",
        transducer.elements.0,
        transducer.elements.1,
        transducer.width * 1e3,
        transducer.height * 1e3,
        transducer.frequency * 1e-6
    );

    // Set transducer and medium properties
    solver.set_transducer(transducer);
    solver.set_medium(1500.0, 1000.0); // Water: 1500 m/s speed, 1000 kg/m³ density

    // Precompute angular spectrum factors for z = 25 mm
    let z_distance = 25e-3; // 25 mm
    println!(
        "\nPrecomputing angular spectrum factors for z = {} mm...",
        z_distance * 1e3
    );

    let start = Instant::now();
    solver.precompute_factors(z_distance)?;
    let precompute_time = start.elapsed();

    println!(
        "Precomputation completed in {:.2} ms",
        precompute_time.as_secs_f64() * 1000.0
    );

    // Create uniform velocity distribution (all elements active with unit velocity)
    let velocity = Array2::<Complex<f64>>::from_elem((16, 16), Complex::new(1.0, 0.0));

    // Compute pressure field
    println!("\nComputing pressure field...");
    let start = Instant::now();
    let pressure_field = solver.compute_field(&velocity, z_distance)?;
    let compute_time = start.elapsed();

    println!(
        "Field computation completed in {:.2} ms",
        compute_time.as_secs_f64() * 1000.0
    );

    // Analyze results
    let (nx, ny) = pressure_field.dim();
    println!("\nResults:");
    println!("Pressure field dimensions: {}x{}", nx, ny);

    // Calculate some statistics
    let mut max_pressure = 0.0f64;
    let mut total_energy = 0.0f64;

    for &val in pressure_field.iter() {
        let magnitude: f64 = val.norm();
        max_pressure = max_pressure.max(magnitude);
        total_energy += magnitude * magnitude;
    }

    println!("Maximum pressure magnitude: {:.2e} Pa", max_pressure);
    println!("Total acoustic energy: {:.2e}", total_energy);

    // Memory usage
    let memory_kb = solver.memory_usage() as f64 / 1024.0;
    println!("Memory usage: {:.1} KB", memory_kb);

    // Demonstrate caching by computing at another z-distance
    println!("\nTesting caching with different z-distance...");
    let z_distance2 = 50e-3; // 50 mm

    let start = Instant::now();
    solver.precompute_factors(z_distance2)?;
    let precompute_time2 = start.elapsed();

    println!(
        "Second precomputation completed in {:.2} ms",
        precompute_time2.as_secs_f64() * 1000.0
    );

    let pressure_field2 = solver.compute_field(&velocity, z_distance2)?;
    let max_pressure2 = pressure_field2
        .iter()
        .map(|&val| val.norm())
        .fold(0.0f64, f64::max);

    println!(
        "Pressure at {} mm: {:.2e} Pa",
        z_distance2 * 1e3,
        max_pressure2
    );

    // Show cached distances
    let cached_z = solver.cached_z_distances();
    println!("Cached z-distances: {} locations", cached_z.len());

    println!("\nFNM Example completed successfully!");
    println!(
        "The Fast Nearfield Method provides O(n) complexity vs O(n²) for traditional methods."
    );

    Ok(())
}
