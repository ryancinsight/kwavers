//! Boundary Smoothing Example
//!
//! Demonstrates staircase boundary smoothing techniques to reduce grid artifacts
//! at curved boundaries in ultrasound simulations.
//!
//! This example shows:
//! - Creating curved boundary geometry on Cartesian grid
//! - Applying three smoothing methods: Subgrid, Ghost Cell, Immersed Interface
//! - Comparing smoothing effectiveness
//! - Measuring reduction in boundary artifacts
//!
//! # Usage
//!
//! ```bash
//! cargo run --example boundary_smoothing
//! ```

use kwavers::domain::boundary::smoothing::{
    BoundarySmoothing, BoundarySmoothingConfig, GhostCellConfig, IIMConfig, JumpConditionType,
    SmoothingMethod, SubgridConfig,
};
use ndarray::Array3;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("╔════════════════════════════════════════════════════════════╗");
    println!("║      Kwavers: Boundary Smoothing Example                  ║");
    println!("╚════════════════════════════════════════════════════════════╝\n");

    // ========================================================================
    // 1. Create Curved Boundary Geometry (Spherical Transducer)
    // ========================================================================

    println!("🏗️  Creating Curved Boundary:");

    let nx = 64;
    let ny = 64;
    let nz = 64;
    let dx = 0.1e-3; // 0.1 mm grid spacing

    println!("  └─ Grid: {} × {} × {} cells", nx, ny, nz);
    println!("  └─ Grid spacing: {} mm\n", dx * 1e3);

    // Create spherical boundary (focused transducer)
    let radius = 15.0 * dx; // 1.5 mm radius
    let center = (nx / 2, ny / 2, nz / 2);

    let mut geometry = Array3::<f64>::zeros((nx, ny, nz));
    let mut property_original = Array3::<f64>::zeros((nx, ny, nz));

    println!("📐 Boundary Type: Spherical (R = {:.2} mm)", radius * 1e3);
    println!("  └─ Center: ({}, {}, {})", center.0, center.1, center.2);

    for i in 0..nx {
        for j in 0..ny {
            for k in 0..nz {
                let x = (i as f64 - center.0 as f64) * dx;
                let y = (j as f64 - center.1 as f64) * dx;
                let z = (k as f64 - center.2 as f64) * dx;

                let r = (x * x + y * y + z * z).sqrt();

                // Volume fraction (1.0 inside sphere, 0.0 outside)
                geometry[[i, j, k]] = if r < radius {
                    1.0
                } else if r < radius + dx {
                    // Transition zone (partial volume)
                    (radius + dx - r) / dx
                } else {
                    0.0
                };

                // Sound speed property (water inside, tissue outside)
                property_original[[i, j, k]] = if r < radius {
                    1482.0 // Water
                } else {
                    1540.0 // Tissue
                };
            }
        }
    }

    // Count boundary cells
    let mut boundary_cells = 0;
    for geom in geometry.iter() {
        if *geom > 0.01 && *geom < 0.99 {
            boundary_cells += 1;
        }
    }

    println!(
        "  └─ Boundary cells: {} ({:.1}% of total)\n",
        boundary_cells,
        (boundary_cells as f64 / (nx * ny * nz) as f64) * 100.0
    );

    // ========================================================================
    // 2. Measure Original Staircase Artifacts
    // ========================================================================

    println!("📏 Measuring Staircase Artifacts:");

    let original_artifacts = measure_boundary_roughness(&property_original, &geometry);
    println!(
        "  └─ Original boundary roughness: {:.4}\n",
        original_artifacts
    );

    // ========================================================================
    // 3. Apply Subgrid Averaging
    // ========================================================================

    println!("🔷 Method 1: Subgrid Averaging");
    println!("  └─ Algorithm: Volume-weighted averaging");
    println!("  └─ Kernel size: 3×3×3");

    let subgrid_config = BoundarySmoothingConfig {
        method: SmoothingMethod::Subgrid,
        subgrid: Some(SubgridConfig {
            kernel_size: 3,
            harmonic_average: false,
            min_volume_fraction: 1e-6,
        }),
        ghost_cell: None,
        iim: None,
    };

    let subgrid_smoother = BoundarySmoothing::new(subgrid_config);
    let property_subgrid = subgrid_smoother.smooth(&property_original, &geometry)?;

    let subgrid_artifacts = measure_boundary_roughness(&property_subgrid, &geometry);
    let subgrid_improvement =
        ((original_artifacts - subgrid_artifacts) / original_artifacts) * 100.0;

    println!("  └─ Smoothed roughness: {:.4}", subgrid_artifacts);
    println!("  └─ Improvement: {:.1}%\n", subgrid_improvement);

    // ========================================================================
    // 4. Apply Ghost Cell Method
    // ========================================================================

    println!("👻 Method 2: Ghost Cell Extrapolation");
    println!("  └─ Algorithm: Polynomial extrapolation");
    println!("  └─ Order: 2 (quadratic)");

    let ghost_cell_config = BoundarySmoothingConfig {
        method: SmoothingMethod::GhostCell,
        subgrid: None,
        ghost_cell: Some(GhostCellConfig {
            n_layers: 2,
            extrapolation_order: 2,
        }),
        iim: None,
    };

    let ghost_smoother = BoundarySmoothing::new(ghost_cell_config);
    let property_ghost = ghost_smoother.smooth(&property_original, &geometry)?;

    let ghost_artifacts = measure_boundary_roughness(&property_ghost, &geometry);
    let ghost_improvement = ((original_artifacts - ghost_artifacts) / original_artifacts) * 100.0;

    println!("  └─ Smoothed roughness: {:.4}", ghost_artifacts);
    println!("  └─ Improvement: {:.1}%\n", ghost_improvement);

    // ========================================================================
    // 5. Apply Immersed Interface Method
    // ========================================================================

    println!("🔬 Method 3: Immersed Interface Method");
    println!("  └─ Algorithm: Modified finite-difference stencils");
    println!("  └─ Jump condition: Continuous");

    let iim_config = BoundarySmoothingConfig {
        method: SmoothingMethod::ImmersedInterface,
        subgrid: None,
        ghost_cell: None,
        iim: Some(IIMConfig {
            interface_thickness: 1.5,
            jump_type: JumpConditionType::Continuous,
        }),
    };

    let iim_smoother = BoundarySmoothing::new(iim_config);
    let property_iim = iim_smoother.smooth(&property_original, &geometry)?;

    let iim_artifacts = measure_boundary_roughness(&property_iim, &geometry);
    let iim_improvement = ((original_artifacts - iim_artifacts) / original_artifacts) * 100.0;

    println!("  └─ Smoothed roughness: {:.4}", iim_artifacts);
    println!("  └─ Improvement: {:.1}%\n", iim_improvement);

    // ========================================================================
    // 6. Comparison Summary
    // ========================================================================

    println!("📊 Comparison Summary:");
    println!("┌─────────────────────────────┬───────────┬────────────────┐");
    println!("│ Method                      │ Roughness │ Improvement    │");
    println!("├─────────────────────────────┼───────────┼────────────────┤");
    println!(
        "│ Original (Staircase)        │  {:.4}   │      —         │",
        original_artifacts
    );
    println!(
        "│ Subgrid Averaging           │  {:.4}   │   {:>5.1}%      │",
        subgrid_artifacts, subgrid_improvement
    );
    println!(
        "│ Ghost Cell Extrapolation    │  {:.4}   │   {:>5.1}%      │",
        ghost_artifacts, ghost_improvement
    );
    println!(
        "│ Immersed Interface Method   │  {:.4}   │   {:>5.1}%      │",
        iim_artifacts, iim_improvement
    );
    println!("└─────────────────────────────┴───────────┴────────────────┘");

    println!("\n✅ Boundary smoothing complete!");
    println!("\n💡 Application Benefits:");
    println!("  • Reduced spurious reflections from grid edges");
    println!("  • Improved accuracy for curved transducers");
    println!("  • Better convergence in simulations");
    println!("  • Enhanced image quality in ultrasound");

    println!("\n📚 References:");
    println!("  • LeVeque & Li (1994) - Immersed Interface Method");
    println!("  • Mittal & Iaccarino (2005) - Immersed Boundary Methods");
    println!("  • Treeby et al. (2012) - k-Wave smoothing techniques");

    Ok(())
}

/// Measure boundary roughness as variance of property gradient
fn measure_boundary_roughness(property: &Array3<f64>, geometry: &Array3<f64>) -> f64 {
    let (nx, ny, nz) = property.dim();
    let mut gradients = Vec::new();

    for i in 1..(nx - 1) {
        for j in 1..(ny - 1) {
            for k in 1..(nz - 1) {
                let geom = geometry[[i, j, k]];

                // Only measure at boundary cells
                if geom > 0.01 && geom < 0.99 {
                    // Compute gradient magnitude
                    let grad_x = (property[[i + 1, j, k]] - property[[i - 1, j, k]]) / 2.0;
                    let grad_y = (property[[i, j + 1, k]] - property[[i, j - 1, k]]) / 2.0;
                    let grad_z = (property[[i, j, k + 1]] - property[[i, j, k - 1]]) / 2.0;

                    let grad_mag = (grad_x * grad_x + grad_y * grad_y + grad_z * grad_z).sqrt();
                    gradients.push(grad_mag);
                }
            }
        }
    }

    if gradients.is_empty() {
        return 0.0;
    }

    // Return variance of gradients (measure of roughness)
    let mean = gradients.iter().sum::<f64>() / gradients.len() as f64;
    let variance =
        gradients.iter().map(|g| (g - mean).powi(2)).sum::<f64>() / gradients.len() as f64;

    variance.sqrt()
}
