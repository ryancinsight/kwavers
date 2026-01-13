// examples/phantom_builder_demo.rs
//! Clinical Phantom Builder Demonstration
//!
//! This example showcases the phantom builder API for constructing realistic
//! tissue models for photoacoustic and optical imaging simulations.
//!
//! # Demonstrated Features
//!
//! 1. Blood oxygenation phantoms (arterial/venous vessels, tumors)
//! 2. Layered tissue phantoms (skin/fat/muscle stratification)
//! 3. Tumor detection phantoms (lesions in background tissue)
//! 4. Vascular network phantoms (vessel trees)
//! 5. Custom region-based construction
//! 6. Predefined clinical phantoms
//!
//! # Use Cases
//!
//! - Algorithm validation (reconstruction, unmixing, segmentation)
//! - Protocol optimization (wavelength selection, imaging geometry)
//! - Performance benchmarking (solver comparison, runtime profiling)
//! - Educational demonstrations (tissue optics, light transport)

use anyhow::Result;
use kwavers::clinical::imaging::phantoms::{ClinicalPhantoms, PhantomBuilder};
use kwavers::domain::grid::GridDimensions;
use kwavers::domain::medium::properties::OpticalPropertyData;
use kwavers::physics::optics::map_builder::{OpticalPropertyMapBuilder, Region};

fn main() -> Result<()> {
    println!("=== Clinical Phantom Builder Demonstration ===\n");

    // Demo 1: Blood oxygenation phantom
    demo_blood_oxygenation()?;

    // Demo 2: Layered tissue phantom
    demo_layered_tissue()?;

    // Demo 3: Tumor detection phantom
    demo_tumor_detection()?;

    // Demo 4: Vascular network phantom
    demo_vascular_network()?;

    // Demo 5: Custom region-based phantom
    demo_custom_regions()?;

    // Demo 6: Predefined clinical phantoms
    demo_predefined_phantoms()?;

    println!("\n=== Demonstration Complete ===");
    println!("\nNext Steps:");
    println!("1. Use these phantoms with diffusion solver (see diffusion_solver_example.rs)");
    println!("2. Run Monte Carlo simulations (see monte_carlo_validation.rs)");
    println!("3. Perform multi-wavelength spectroscopy (see photoacoustic_blood_oxygenation.rs)");
    println!("4. Create custom phantoms for your specific application");

    Ok(())
}

fn demo_blood_oxygenation() -> Result<()> {
    println!("Demo 1: Blood Oxygenation Phantom");
    println!("----------------------------------");
    println!("Purpose: Validate spectroscopic imaging and sO₂ estimation algorithms\n");

    let dims = GridDimensions::new(40, 40, 40, 0.001, 0.001, 0.001);

    // Build phantom with arterial/venous vessels and hypoxic tumor
    let phantom = PhantomBuilder::blood_oxygenation()
        .dimensions(dims)
        .wavelength(800.0)
        .background(OpticalPropertyData::soft_tissue())
        // Arterial vessel: high oxygenation (sO₂ = 98%)
        .add_artery([0.015, 0.020, 0.020], 0.002, 0.98)
        // Venous vessel: lower oxygenation (sO₂ = 65%)
        .add_vein([0.025, 0.020, 0.020], 0.003, 0.65)
        // Hypoxic tumor: low oxygenation (sO₂ = 55%)
        .add_tumor([0.020, 0.020, 0.028], 0.004, 0.55)
        .build();

    let stats = phantom.absorption_stats();
    println!("  Grid: {}×{}×{} voxels", dims.nx, dims.ny, dims.nz);
    println!(
        "  Spacing: {:.1}×{:.1}×{:.1} mm",
        dims.dx * 1000.0,
        dims.dy * 1000.0,
        dims.dz * 1000.0
    );
    println!("  Volume: {:.2} cm³", phantom.volume() * 1e6);
    println!("  Absorption coefficient:");
    println!("    Mean: {:.2} m⁻¹", stats.mean);
    println!("    Range: [{:.2}, {:.2}] m⁻¹", stats.min, stats.max);
    println!("    Std dev: {:.2} m⁻¹", stats.std_dev);
    println!("\n  Clinical relevance:");
    println!("    - Arterial sO₂ = 98% (normal oxygenated blood)");
    println!("    - Venous sO₂ = 65% (normal deoxygenated blood)");
    println!("    - Tumor sO₂ = 55% (hypoxic, indicator of aggressive tumor)\n");

    Ok(())
}

fn demo_layered_tissue() -> Result<()> {
    println!("Demo 2: Layered Tissue Phantom");
    println!("-------------------------------");
    println!("Purpose: Model stratified media (skin/fat/muscle)\n");

    let dims = GridDimensions::new(30, 30, 50, 0.001, 0.001, 0.001);

    let phantom = PhantomBuilder::layered_tissue()
        .dimensions(dims)
        .wavelength(800.0)
        // Skin epidermis: 1 mm (high absorption due to melanin)
        .add_skin_layer(0.0, 0.001)
        // Dermis: 2 mm (moderate absorption, high scattering)
        .add_dermis_layer(0.001, 0.003)
        // Fat layer: 7 mm (low absorption, moderate scattering)
        .add_fat_layer(0.003, 0.010)
        // Muscle: 40 mm (moderate absorption and scattering)
        .add_muscle_layer(0.010, 0.050)
        .build();

    println!("  Grid: {}×{}×{} voxels", dims.nx, dims.ny, dims.nz);
    println!("  Layers:");
    println!("    0-1 mm:   Epidermis (μ_a ≈ 5.0 m⁻¹)");
    println!("    1-3 mm:   Dermis    (μ_a ≈ 1.0 m⁻¹)");
    println!("    3-10 mm:  Fat       (μ_a ≈ 0.3 m⁻¹)");
    println!("    10-50 mm: Muscle    (μ_a ≈ 0.8 m⁻¹)");
    println!("\n  Use case:");
    println!("    - Depth profiling validation");
    println!("    - Layer boundary detection algorithms");
    println!("    - Penetration depth studies\n");

    // Sample properties at different depths
    println!("  Depth sampling:");
    for (depth_mm, layer_name) in &[
        (0.5, "Epidermis"),
        (2.0, "Dermis"),
        (6.0, "Fat"),
        (25.0, "Muscle"),
    ] {
        let k = (depth_mm / 1000.0 / dims.dz) as usize;
        if let Some(props) = phantom.get(dims.nx / 2, dims.ny / 2, k) {
            println!(
                "    {} ({:.1} mm): μ_a = {:.2} m⁻¹, μ_s = {:.1} m⁻¹",
                layer_name, depth_mm, props.absorption_coefficient, props.scattering_coefficient
            );
        }
    }
    println!();

    Ok(())
}

fn demo_tumor_detection() -> Result<()> {
    println!("Demo 3: Tumor Detection Phantom");
    println!("--------------------------------");
    println!("Purpose: Validate tumor detection and characterization algorithms\n");

    let dims = GridDimensions::new(35, 35, 35, 0.001, 0.001, 0.001);

    let phantom = PhantomBuilder::tumor_detection()
        .dimensions(dims)
        .wavelength(800.0)
        .background(OpticalPropertyData::fat()) // Breast tissue approximation
        // Multiple tumors of varying size and oxygenation
        .add_tumor([0.010, 0.010, 0.015], 0.003, 0.60) // 3 mm, moderately hypoxic
        .add_tumor([0.025, 0.018, 0.020], 0.005, 0.55) // 5 mm, more hypoxic
        .add_tumor([0.018, 0.025, 0.025], 0.002, 0.70) // 2 mm, less hypoxic
        .build();

    println!("  Grid: {}×{}×{} voxels", dims.nx, dims.ny, dims.nz);
    println!("  Background: Fat tissue (simulates breast)");
    println!("  Tumors:");
    println!("    Tumor 1: 3 mm radius, sO₂ = 60% (moderately hypoxic)");
    println!("    Tumor 2: 5 mm radius, sO₂ = 55% (severely hypoxic)");
    println!("    Tumor 3: 2 mm radius, sO₂ = 70% (mildly hypoxic)");
    println!("\n  Clinical interpretation:");
    println!("    - Lower sO₂ correlates with tumor aggressiveness");
    println!("    - Hypoxia (sO₂ < 60%) indicates poor prognosis");
    println!("    - Size and oxygenation are independent prognostic factors\n");

    Ok(())
}

fn demo_vascular_network() -> Result<()> {
    println!("Demo 4: Vascular Network Phantom");
    println!("---------------------------------");
    println!("Purpose: Model complex vessel geometries for angiogenesis studies\n");

    let dims = GridDimensions::new(40, 40, 50, 0.001, 0.001, 0.001);
    let cx = dims.dx * (dims.nx as f64) / 2.0;
    let cy = dims.dy * (dims.ny as f64) / 2.0;

    let phantom = PhantomBuilder::vascular()
        .dimensions(dims)
        .wavelength(800.0)
        .background(OpticalPropertyData::soft_tissue())
        // Main arterial trunk (vertical)
        .add_vessel([cx, cy, 0.0], [cx, cy, 0.050], 0.002, 0.97)
        // Branching arteries
        .add_vessel([cx, cy, 0.020], [cx - 0.008, cy, 0.035], 0.0015, 0.96)
        .add_vessel([cx, cy, 0.020], [cx + 0.008, cy, 0.035], 0.0015, 0.96)
        // Venous return (parallel to arterial)
        .add_vessel([cx + 0.005, cy, 0.0], [cx + 0.005, cy, 0.050], 0.003, 0.68)
        .add_vessel([cx - 0.005, cy, 0.0], [cx - 0.005, cy, 0.050], 0.003, 0.68)
        // Capillary network (small vessels)
        .add_vessel(
            [cx - 0.003, cy - 0.003, 0.010],
            [cx - 0.003, cy + 0.003, 0.010],
            0.0003,
            0.85,
        )
        .add_vessel(
            [cx + 0.003, cy - 0.003, 0.030],
            [cx + 0.003, cy + 0.003, 0.030],
            0.0003,
            0.85,
        )
        .build();

    println!("  Grid: {}×{}×{} voxels", dims.nx, dims.ny, dims.nz);
    println!("  Vessel network:");
    println!("    - 1 main arterial trunk (2 mm diameter, sO₂ = 97%)");
    println!("    - 2 arterial branches (1.5 mm diameter, sO₂ = 96%)");
    println!("    - 2 venous vessels (3 mm diameter, sO₂ = 68%)");
    println!("    - 2 capillaries (0.3 mm diameter, sO₂ = 85%)");
    println!("\n  Applications:");
    println!("    - Angiogenesis quantification");
    println!("    - Perfusion mapping validation");
    println!("    - Vessel segmentation algorithms\n");

    Ok(())
}

fn demo_custom_regions() -> Result<()> {
    println!("Demo 5: Custom Region-Based Phantom");
    println!("------------------------------------");
    println!("Purpose: Demonstrate low-level region API for arbitrary geometries\n");

    let dims = GridDimensions::new(30, 30, 30, 0.001, 0.001, 0.001);
    let mut builder = OpticalPropertyMapBuilder::new(dims);

    // Background
    builder.set_background(OpticalPropertyData::soft_tissue());

    // Add complex geometric regions
    // Sphere: tumor
    builder.add_region(
        Region::sphere([0.015, 0.015, 0.015], 0.005),
        OpticalPropertyData::tumor(),
    );

    // Cylinder: blood vessel
    builder.add_region(
        Region::cylinder([0.010, 0.010, 0.0], [0.010, 0.010, 0.030], 0.001),
        OpticalPropertyData::blood_oxygenated(),
    );

    // Box: bone region
    builder.add_region(
        Region::box_region([0.020, 0.020, 0.020], [0.028, 0.028, 0.028]),
        OpticalPropertyData::bone_cortical(),
    );

    // Ellipsoid: elongated tumor
    builder.add_region(
        Region::ellipsoid([0.008, 0.008, 0.015], [0.002, 0.002, 0.006]),
        OpticalPropertyData::tumor(),
    );

    // Half-space: layering boundary
    builder.add_region(
        Region::half_space([0.0, 0.0, 0.010], [0.0, 0.0, 1.0]),
        OpticalPropertyData::muscle(),
    );

    // Custom predicate: radial gradient
    builder.add_region(
        Region::custom(|p| {
            let r = (p[0] * p[0] + p[1] * p[1]).sqrt();
            r > 0.005 && r < 0.008
        }),
        OpticalPropertyData::liver(),
    );

    let phantom = builder.build();

    println!("  Grid: {}×{}×{} voxels", dims.nx, dims.ny, dims.nz);
    println!("  Custom regions:");
    println!("    - Sphere (tumor)");
    println!("    - Cylinder (blood vessel)");
    println!("    - Box (bone inclusion)");
    println!("    - Ellipsoid (elongated tumor)");
    println!("    - Half-space (tissue boundary)");
    println!("    - Custom predicate (radial gradient)");
    println!("\n  Advantages:");
    println!("    - Full control over geometry");
    println!("    - Combine multiple region types");
    println!("    - Support arbitrary predicates\n");

    Ok(())
}

fn demo_predefined_phantoms() -> Result<()> {
    println!("Demo 6: Predefined Clinical Phantoms");
    println!("-------------------------------------");
    println!("Purpose: Quick-start phantoms for common scenarios\n");

    let dims = GridDimensions::new(35, 35, 35, 0.001, 0.001, 0.001);

    // Standard blood oxygenation phantom
    println!("  6a. Standard Blood Oxygenation Phantom");
    let phantom1 = ClinicalPhantoms::standard_blood_oxygenation(dims);
    println!("      Contains: artery (sO₂=98%), vein (sO₂=65%), tumor (sO₂=55%)");
    println!("      Use: Multi-wavelength spectroscopy validation");
    println!();

    // Skin tissue phantom
    let dims_skin = GridDimensions::new(30, 30, 50, 0.001, 0.001, 0.001);
    println!("  6b. Skin Tissue Phantom");
    let phantom2 = ClinicalPhantoms::skin_tissue(dims_skin);
    println!("      Layers: epidermis/dermis/fat/muscle");
    println!("      Use: Depth profiling and layer detection");
    println!();

    // Breast tumor phantom
    println!("  6c. Breast Tumor Phantom");
    let tumor_center = [0.0175, 0.0175, 0.0175];
    let phantom3 = ClinicalPhantoms::breast_tumor(dims, tumor_center);
    println!("      Background: Fat (breast tissue)");
    println!("      Lesion: 8 mm hypoxic tumor");
    println!("      Use: Tumor detection algorithm validation");
    println!();

    // Vascular network phantom
    println!("  6d. Vascular Network Phantom");
    let phantom4 = ClinicalPhantoms::vascular_network(dims);
    println!("      Contains: Arterial tree and venous drainage");
    println!("      Use: Angiogenesis and perfusion studies");
    println!();

    println!("  Quick start:");
    println!("    let phantom = ClinicalPhantoms::standard_blood_oxygenation(dims);");
    println!("    // Ready to use with diffusion solver, MC, or other physics modules\n");

    Ok(())
}
