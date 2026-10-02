//! Custom region-based and predefined clinical phantom demonstrations.

use anyhow::Result;
use kwavers_grid::GridDimensions;
use kwavers_medium::optical_map::{OpticalPropertyMapBuilder, Region};
use kwavers_medium::properties::OpticalPropertyData;
use kwavers_phantom::ClinicalPhantoms;
use std::io::Write;

pub(crate) fn demo_custom_regions() -> Result<()> {
    let _ = writeln!(
        std::io::stdout().lock(),
        "Demo 5: Custom Region-Based Phantom"
    );
    let _ = writeln!(
        std::io::stdout().lock(),
        "------------------------------------"
    );
    let _ = writeln!(
        std::io::stdout().lock(),
        "Purpose: Demonstrate low-level region API for arbitrary geometries\n"
    );

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

    // Additional box region: tissue boundary layer
    builder.add_region(
        Region::box_region([0.0, 0.0, 0.010], [0.030, 0.030, 0.015]),
        OpticalPropertyData::muscle(),
    );

    // Additional ellipsoid: elongated muscle fiber
    builder.add_region(
        Region::ellipsoid([0.022, 0.022, 0.008], [0.001, 0.001, 0.004]),
        OpticalPropertyData::liver(),
    );

    let _phantom = builder.build();

    let _ = writeln!(
        std::io::stdout().lock(),
        "  Grid: {}×{}×{} voxels",
        dims.nx,
        dims.ny,
        dims.nz
    );
    let _ = writeln!(std::io::stdout().lock(), "  Custom regions:");
    let _ = writeln!(std::io::stdout().lock(), "    - Sphere (tumor)");
    let _ = writeln!(std::io::stdout().lock(), "    - Cylinder (blood vessel)");
    let _ = writeln!(std::io::stdout().lock(), "    - Box (bone inclusion)");
    let _ = writeln!(
        std::io::stdout().lock(),
        "    - Ellipsoid (elongated tumor)"
    );
    let _ = writeln!(
        std::io::stdout().lock(),
        "    - Box (tissue boundary layer)"
    );
    let _ = writeln!(std::io::stdout().lock(), "    - Ellipsoid (muscle fiber)");
    let _ = writeln!(std::io::stdout().lock(), "\n  Advantages:");
    let _ = writeln!(std::io::stdout().lock(), "    - Full control over geometry");
    let _ = writeln!(
        std::io::stdout().lock(),
        "    - Combine multiple region types\n"
    );

    Ok(())
}

pub(crate) fn demo_predefined_phantoms() -> Result<()> {
    let _ = writeln!(
        std::io::stdout().lock(),
        "Demo 6: Predefined Clinical Phantoms"
    );
    let _ = writeln!(
        std::io::stdout().lock(),
        "-------------------------------------"
    );
    let _ = writeln!(
        std::io::stdout().lock(),
        "Purpose: Quick-start phantoms for common scenarios\n"
    );

    let dims = GridDimensions::new(35, 35, 35, 0.001, 0.001, 0.001);

    // Standard blood oxygenation phantom
    let _ = writeln!(
        std::io::stdout().lock(),
        "  6a. Standard Blood Oxygenation Phantom"
    );
    let _phantom1 = ClinicalPhantoms::standard_blood_oxygenation(dims)?;
    let _ = writeln!(
        std::io::stdout().lock(),
        "      Contains: artery (sO₂=98%), vein (sO₂=65%), tumor (sO₂=55%)"
    );
    let _ = writeln!(
        std::io::stdout().lock(),
        "      Use: Multi-wavelength spectroscopy validation"
    );
    let _ = writeln!(std::io::stdout().lock());

    // Skin tissue phantom
    let dims_skin = GridDimensions::new(30, 30, 50, 0.001, 0.001, 0.001);
    let _ = writeln!(std::io::stdout().lock(), "  6b. Skin Tissue Phantom");
    let _phantom2 = ClinicalPhantoms::skin_tissue(dims_skin);
    let _ = writeln!(
        std::io::stdout().lock(),
        "      Layers: epidermis/dermis/fat/muscle"
    );
    let _ = writeln!(
        std::io::stdout().lock(),
        "      Use: Depth profiling and layer detection"
    );
    let _ = writeln!(std::io::stdout().lock());

    // Breast tumor phantom
    let _ = writeln!(std::io::stdout().lock(), "  6c. Breast Tumor Phantom");
    let tumor_center = [0.0175, 0.0175, 0.0175];
    let _phantom3 = ClinicalPhantoms::breast_tumor(dims, tumor_center)?;
    let _ = writeln!(
        std::io::stdout().lock(),
        "      Background: Fat (breast tissue)"
    );
    let _ = writeln!(std::io::stdout().lock(), "      Lesion: 8 mm hypoxic tumor");
    let _ = writeln!(
        std::io::stdout().lock(),
        "      Use: Tumor detection algorithm validation"
    );
    let _ = writeln!(std::io::stdout().lock());

    // Vascular network phantom
    let _ = writeln!(std::io::stdout().lock(), "  6d. Vascular Network Phantom");
    let _phantom4 = ClinicalPhantoms::vascular_network(dims)?;
    let _ = writeln!(
        std::io::stdout().lock(),
        "      Contains: Arterial tree and venous drainage"
    );
    let _ = writeln!(
        std::io::stdout().lock(),
        "      Use: Angiogenesis and perfusion studies"
    );
    let _ = writeln!(std::io::stdout().lock());

    let _ = writeln!(std::io::stdout().lock(), "  Quick start:");
    let _ = writeln!(
        std::io::stdout().lock(),
        "    let phantom = ClinicalPhantoms::standard_blood_oxygenation(dims)?;"
    );
    let _ = writeln!(
        std::io::stdout().lock(),
        "    // Ready to use with diffusion solver, MC, or other physics modules\n"
    );

    Ok(())
}
