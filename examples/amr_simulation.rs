//! Example demonstrating Adaptive Mesh Refinement (AMR) in Kwavers
//!
//! This example shows how to use AMR for efficient simulation
//! of focused ultrasound with adaptive resolution.

use kwavers::{
    solver::amr::{AMRConfig, AMRManager, InterpolationScheme, WaveletType},
    Grid, KwaversResult,
};
use ndarray::{s, Array3};
use std::time::Instant;

fn main() -> KwaversResult<()> {
    println!("=== Kwavers AMR Simulation Example ===\n");

    // Create grid - start with moderate resolution
    let grid = Grid::new(64, 64, 64, 1e-3, 1e-3, 1e-3); // 1mm spacing

    // Create AMR configuration
    let amr_config = AMRConfig {
        max_level: 3,                           // Up to 3 refinement levels
        min_level: 0,                           // Minimum level (coarsest)
        refine_threshold: 1e4,                  // Refinement threshold
        coarsen_threshold: 1e3,                 // Coarsening threshold
        refinement_ratio: 2,                    // 2:1 refinement
        buffer_cells: 2,                        // 2-cell buffer
        wavelet_type: WaveletType::Daubechies4, // Wavelet for error estimation
        interpolation_scheme: InterpolationScheme::Linear,
    };

    // Create AMR manager
    let mut amr_manager = AMRManager::new(amr_config.clone(), &grid);

    println!("Grid: {}x{}x{} points", grid.nx, grid.ny, grid.nz);
    println!("AMR Configuration:");
    println!("  Max refinement level: {}", amr_config.max_level);
    println!(
        "  Refinement threshold: {:.1e}",
        amr_config.refine_threshold
    );
    println!(
        "  Coarsening threshold: {:.1e}",
        amr_config.coarsen_threshold
    );
    println!("  Refinement ratio: {}", amr_config.refinement_ratio);
    println!();

    // Create a test field with a focused region
    let mut field = Array3::<f64>::zeros((grid.nx, grid.ny, grid.nz));

    // Create a Gaussian focus in the center
    let cx = grid.nx / 2;
    let cy = grid.ny / 2;
    let cz = grid.nz / 2;
    let sigma = 5.0; // Width of the Gaussian

    for i in 0..grid.nx {
        for j in 0..grid.ny {
            for k in 0..grid.nz {
                let dx = (i as f64 - cx as f64) / sigma;
                let dy = (j as f64 - cy as f64) / sigma;
                let dz = (k as f64 - cz as f64) / sigma;
                let r2 = dx * dx + dy * dy + dz * dz;
                field[[i, j, k]] = 1e5 * (-r2).exp(); // Peak amplitude of 100 kPa
            }
        }
    }

    println!("Initial field created with Gaussian focus");
    let max_val = field.iter().fold(0.0f64, |a, &b| a.max(b.abs()));
    println!("Max field value: {:.2e} Pa", max_val);

    let start_time = Instant::now();

    // Demonstrate AMR adaptation
    println!("\nPerforming AMR adaptation...");

    // First adaptation
    let result1 = amr_manager.adapt_mesh(&field, 1e6)?;
    println!("First adaptation:");
    println!("  Cells refined: {}", result1.cells_refined);
    println!("  Cells coarsened: {}", result1.cells_coarsened);
    println!("  Max error: {:.2e}", result1.max_error);

    // Modify the field to simulate wave propagation
    // Shift the Gaussian slightly
    let mut field2 = Array3::<f64>::zeros((grid.nx, grid.ny, grid.nz));
    let shift = 5.0;

    for i in 0..grid.nx {
        for j in 0..grid.ny {
            for k in 0..grid.nz {
                let dx = (i as f64 - cx as f64 - shift) / sigma;
                let dy = (j as f64 - cy as f64) / sigma;
                let dz = (k as f64 - cz as f64) / sigma;
                let r2 = dx * dx + dy * dy + dz * dz;
                field2[[i, j, k]] = 0.8e5 * (-r2).exp(); // Slightly attenuated
            }
        }
    }

    // Second adaptation with modified field
    let result2 = amr_manager.adapt_mesh(&field2, 1e6)?;
    println!("\nSecond adaptation (after field change):");
    println!("  Cells refined: {}", result2.cells_refined);
    println!("  Cells coarsened: {}", result2.cells_coarsened);
    println!("  Max error: {:.2e}", result2.max_error);

    // Demonstrate refinement patterns
    println!("\nRefinement pattern analysis:");

    // Create a more complex field with multiple features
    let mut complex_field = Array3::<f64>::zeros((grid.nx, grid.ny, grid.nz));

    // Add multiple Gaussian peaks at different locations
    let peaks = vec![
        (grid.nx / 4, grid.ny / 4, grid.nz / 2, 1e5),
        (3 * grid.nx / 4, grid.ny / 2, grid.nz / 2, 0.8e5),
        (grid.nx / 2, 3 * grid.ny / 4, grid.nz / 2, 0.6e5),
    ];

    for (px, py, pz, amplitude) in peaks {
        for i in 0..grid.nx {
            for j in 0..grid.ny {
                for k in 0..grid.nz {
                    let dx = (i as f64 - px as f64) / sigma;
                    let dy = (j as f64 - py as f64) / sigma;
                    let dz = (k as f64 - pz as f64) / sigma;
                    let r2 = dx * dx + dy * dy + dz * dz;
                    complex_field[[i, j, k]] += amplitude * (-r2).exp();
                }
            }
        }
    }

    // Third adaptation with complex field
    let result3 = amr_manager.adapt_mesh(&complex_field, 1e6)?;
    println!("\nThird adaptation (complex field with multiple peaks):");
    println!("  Cells refined: {}", result3.cells_refined);
    println!("  Cells coarsened: {}", result3.cells_coarsened);
    println!("  Max error: {:.2e}", result3.max_error);

    let total_time = start_time.elapsed();
    println!("\nAMR demonstration completed in {:.2?}", total_time);

    // Show memory usage estimate
    let base_memory = (grid.nx * grid.ny * grid.nz * 8) as f64 / 1e6; // MB
    let refined_memory = base_memory
        * (1.0 + 0.125 * result3.cells_refined as f64 / (grid.nx * grid.ny * grid.nz) as f64);
    println!("\nMemory usage:");
    println!("  Base grid: {:.1} MB", base_memory);
    println!("  With refinement: {:.1} MB (estimated)", refined_memory);

    Ok(())
}
