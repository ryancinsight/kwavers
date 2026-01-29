//! Backend Abstraction Example - Phase 2 Enhancement
//!
//! Demonstrates transparent CPU/GPU backend selection.
//!
//! Run with: cargo run --example phase2_backend

use kwavers::solver::backend::{BackendContext, BackendSelector, SelectionCriteria};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== Kwavers Backend Abstraction Examples ===\n");

    // Example 1: CPU Backend
    println!("1. CPU Backend");
    println!("   Always available, good for small-medium problems\n");

    let cpu_backend = BackendContext::cpu()?;
    println!("   Backend type: {:?}", cpu_backend.backend_type());
    println!("   Available: {}", cpu_backend.is_available());

    let caps = cpu_backend.capabilities();
    println!("   Capabilities:");
    println!("     - FFT support: {}", caps.supports_fft);
    println!("     - f64 support: {}", caps.supports_f64);
    println!("     - Max parallelism: {} threads", caps.max_parallelism);
    println!("     - Unified memory: {}\n", caps.supports_unified_memory);

    // Example 2: Auto Backend Selection
    println!("2. Automatic Backend Selection");
    println!("   Selects optimal backend based on problem size\n");

    // Small problem - should use CPU
    let small_backend = BackendContext::auto_select((64, 64, 32))?;
    println!("   Small problem (64³):");
    println!("     - Selected: {:?}", small_backend.backend_type());
    println!("     - Reason: Overhead would dominate for small problems\n");

    // Large problem - may use GPU if available
    let large_backend = BackendContext::auto_select((256, 256, 128))?;
    println!("   Large problem (256³):");
    println!("     - Selected: {:?}", large_backend.backend_type());
    if large_backend.backend_type() == kwavers::solver::backend::traits::BackendType::GPU {
        println!("     - Reason: Large enough to benefit from GPU parallelization");
    } else {
        println!("     - Reason: GPU not available or problem below threshold");
    }
    println!();

    // Example 3: Backend Selector with Criteria
    println!("3. Backend Selector with Custom Criteria\n");

    let selector = BackendSelector::new();
    println!("   Available backends: {:?}", selector.available_backends());

    let criteria = SelectionCriteria {
        grid_points: (256, 256, 128),
        prefer_gpu: true,
        min_gpu_size: (128, 128, 64),
    };

    let (recommended, explanation) = selector.recommend_with_explanation(&criteria);
    println!("   Recommended backend: {:?}", recommended);
    println!("   Explanation: {}\n", explanation);

    // Example 4: Performance Estimation
    println!("4. GPU Speedup Estimation\n");

    for (name, size) in [
        ("Tiny", (32, 32, 32)),
        ("Small", (64, 64, 64)),
        ("Medium", (128, 128, 128)),
        ("Large", (256, 256, 256)),
        ("Very Large", (512, 512, 256)),
    ] {
        let speedup = selector.estimate_gpu_speedup(&SelectionCriteria {
            grid_points: size,
            prefer_gpu: true,
            min_gpu_size: (128, 128, 64),
        });

        let total_points = size.0 * size.1 * size.2;
        println!(
            "   {} ({} points): {:.1}x speedup",
            name, total_points, speedup
        );
    }
    println!();

    // Example 5: Selection Report
    println!("5. Detailed Selection Report\n");

    let report_criteria = SelectionCriteria {
        grid_points: (256, 256, 128),
        prefer_gpu: true,
        min_gpu_size: (128, 128, 64),
    };

    selector.print_selection_report(&report_criteria);

    // Example 6: Backend Operations (CPU)
    println!("6. Backend Operations Example\n");

    use ndarray::Array3;
    let cpu = BackendContext::cpu()?;

    let a = Array3::from_elem((4, 4, 4), 2.0);
    let b = Array3::from_elem((4, 4, 4), 3.0);
    let mut out = Array3::zeros((4, 4, 4));

    cpu.element_wise_multiply(&a, &b, &mut out)?;
    println!("   Element-wise multiply: 2.0 * 3.0 = {}", out[[0, 0, 0]]);

    cpu.synchronize()?;
    println!("   Backend synchronized successfully\n");

    println!("=== Backend Abstraction Examples Complete ===");
    println!("Transparent backend selection allows code to run on CPU or GPU without changes!");

    Ok(())
}
