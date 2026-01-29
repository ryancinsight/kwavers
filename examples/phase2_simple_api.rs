//! Simple API Example - Phase 2 Enhancement
//!
//! Demonstrates the new Simple API for quick ultrasound simulations.
//!
//! Run with: cargo run --example phase2_simple_api

use kwavers::api::SimpleAPI;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== Kwavers Simple API Example ===\n");

    // Example 1: B-mode ultrasound imaging (most common use case)
    println!("1. B-mode Ultrasound Imaging");
    println!("   - Frequency: 5 MHz");
    println!("   - Domain: 10cm × 10cm × 5cm (default)");
    println!("   - Configuration: Automatic\n");

    let result = SimpleAPI::ultrasound_imaging()
        .frequency(5e6) // 5 MHz
        .run()?;

    println!("   Simulation complete!");
    println!("   - Time steps: {}", result.statistics.time_steps);
    println!("   - Execution time: {:.3} s", result.execution_time);
    println!("   - Grid size: {:?}\n", result.pressure.dim());

    // Example 2: Quick preview for debugging
    println!("2. Quick Preview (Fast)");
    println!("   - Low accuracy, fast execution");
    println!("   - Ideal for debugging\n");

    let result = SimpleAPI::quick_preview()
        .frequency(2e6)
        .domain_size(0.05, 0.05, 0.03) // Smaller domain
        .run()?;

    println!("   Preview complete!");
    println!("   - Execution time: {:.3} s", result.execution_time);
    println!("   - Backend: {}\n", result.statistics.backend);

    // Example 3: High-resolution research imaging
    println!("3. High-Resolution Imaging");
    println!("   - Research-grade accuracy");
    println!("   - PSTD solver for precision\n");

    let result = SimpleAPI::high_resolution().frequency(7e6).run()?;

    println!("   High-resolution simulation complete!");
    result.statistics.print();

    // Example 4: HIFU therapy simulation
    println!("4. HIFU Therapy Simulation");
    println!("   - Nonlinear propagation");
    println!("   - High spatial resolution\n");

    let result = SimpleAPI::hifu_therapy()
        .frequency(1e6) // 1 MHz for therapy
        .domain_size(0.08, 0.08, 0.06)
        .run()?;

    println!("   HIFU simulation complete!");
    println!("   - Memory used: {:.2} GB", result.statistics.memory_usage);

    println!("\n=== All Examples Complete ===");
    println!("The Simple API makes it easy to get started with ultrasound simulation!");

    Ok(())
}
