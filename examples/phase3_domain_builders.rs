//! Domain Builders Example - Phase 3 Enhancement
//!
//! Demonstrates anatomical models and transducer array builders.
//!
//! Run with: cargo run --example phase3_domain_builders

use kwavers::domain::builders::{AnatomicalModel, DomainBuilder, TransducerArray};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== Kwavers Domain Builders Examples ===\n");

    // ========================================================================
    // Example 1: Transducer Arrays
    // ========================================================================

    println!("1. Transducer Array Builders");
    println!("   Creating standard ultrasound probe configurations\n");

    // Linear array for B-mode imaging
    println!("   a) Linear Array (B-mode imaging):");
    let linear = TransducerArray::linear_array()
        .frequency(5e6)
        .elements(128)
        .pitch(0.3e-3)
        .build()?;

    if let Some(geom) = linear.geometry() {
        println!("      - Elements: {}", geom.num_elements);
        println!("      - Element width: {:.2} mm", geom.element_width * 1e3);
        println!(
            "      - Total aperture: {:.2} mm",
            geom.num_elements as f64 * geom.element_width * 1e3
        );
    }
    println!();

    // Phased array for cardiac imaging
    println!("   b) Phased Array (cardiac imaging):");
    let phased = TransducerArray::phased_array()
        .frequency(3e6)
        .elements(80)
        .pitch(0.295e-3)
        .build()?;

    if let Some(geom) = phased.geometry() {
        println!("      - Elements: {}", geom.num_elements);
        println!("      - Element pitch: {:.3} mm", 0.295);
        println!("      - Ideal for: Cardiac imaging, intercostal windows");
    }
    println!();

    // Convex array for abdominal imaging
    println!("   c) Convex Array (abdominal imaging):");
    let convex = TransducerArray::convex_array()
        .frequency(3.5e6)
        .elements(128)
        .pitch(0.5e-3)
        .build()?;

    if let Some(geom) = convex.geometry() {
        println!("      - Elements: {}", geom.num_elements);
        println!("      - Curved geometry with 50mm radius");
        println!("      - Wider field of view than linear array");
    }
    println!();

    // ========================================================================
    // Example 2: Standard Clinical Transducers
    // ========================================================================

    println!("2. Standard Clinical Transducers");
    println!("   Pre-configured industry-standard probes\n");

    // Philips L12-5 linear array
    println!("   a) Philips L12-5 Linear Array:");
    let l12_5 = TransducerArray::l12_5_philips().build()?;
    println!("      - Frequency: 8.5 MHz (5-12 MHz bandwidth)");
    println!("      - Elements: 192");
    println!("      - Applications: Vascular, small parts, superficial");
    println!();

    // Philips C5-2 convex array
    println!("   b) Philips C5-2 Convex Array:");
    let c5_2 = TransducerArray::c5_2_philips().build()?;
    println!("      - Frequency: 3.5 MHz (2-5 MHz bandwidth)");
    println!("      - Elements: 128");
    println!("      - Applications: Abdominal, OB/GYN");
    println!();

    // Philips P4-2 phased array
    println!("   c) Philips P4-2 Phased Array:");
    let p4_2 = TransducerArray::p4_2_philips().build()?;
    println!("      - Frequency: 3 MHz (2-4 MHz bandwidth)");
    println!("      - Elements: 80");
    println!("      - Applications: Cardiac, thoracic");
    println!();

    // ========================================================================
    // Example 3: Anatomical Models
    // ========================================================================

    println!("3. Anatomical Model Builders");
    println!("   Pre-defined organ models for clinical simulations\n");

    // Adult brain model
    println!("   a) Adult Brain Model:");
    let brain = AnatomicalModel::brain_adult().build()?;
    println!("      - Tissues: White matter, gray matter, skull");
    println!("      - Default size: 15cm × 15cm × 12cm");
    println!("      - Resolution: 1 mm");

    if let Some(geom) = brain.geometry() {
        let shape = geom.tissue_map.shape();
        println!(
            "      - Grid points: {} × {} × {}",
            shape[0], shape[1], shape[2]
        );

        // Count tissue types
        let mut white_matter = 0;
        let mut gray_matter = 0;
        let mut skull = 0;

        for &val in geom.tissue_map.iter() {
            match val {
                0 => white_matter += 1,
                1 => gray_matter += 1,
                2 => skull += 1,
                _ => {}
            }
        }

        println!("      - White matter points: {}", white_matter);
        println!("      - Gray matter points: {}", gray_matter);
        println!("      - Skull points: {}", skull);
    }
    println!();

    // Liver model
    println!("   b) Liver Model:");
    let liver = AnatomicalModel::liver().build()?;
    println!("      - Tissues: Liver parenchyma, blood vessels");
    println!("      - Default size: 20cm × 15cm × 10cm");
    println!("      - Includes simplified vascular structure");

    if let Some(geom) = liver.geometry() {
        let shape = geom.tissue_map.shape();
        println!(
            "      - Grid points: {} × {} × {}",
            shape[0], shape[1], shape[2]
        );
    }
    println!();

    // Kidney model
    println!("   c) Kidney Model:");
    let kidney = AnatomicalModel::kidney().build()?;
    println!("      - Tissues: Cortex, medulla, blood vessels");
    println!("      - Default size: 10cm × 6cm × 12cm");
    println!("      - Resolution: 0.5 mm (finer for detail)");
    println!();

    // ========================================================================
    // Example 4: Tissue Properties
    // ========================================================================

    println!("4. Tissue Acoustic Properties");
    println!("   Pre-defined tissue types with realistic parameters\n");

    use kwavers::domain::builders::TissueType;

    let tissues = [
        ("Water", TissueType::WATER),
        ("Brain (white matter)", TissueType::BRAIN_WHITE_MATTER),
        ("Brain (gray matter)", TissueType::BRAIN_GRAY_MATTER),
        ("Skull", TissueType::SKULL),
        ("Liver", TissueType::LIVER),
        ("Blood", TissueType::BLOOD),
        ("Fat", TissueType::FAT),
        ("Muscle", TissueType::MUSCLE),
    ];

    println!("   Tissue Properties Table:");
    println!(
        "   {:<25} {:>12} {:>12} {:>12}",
        "Tissue", "Speed (m/s)", "Density (kg/m³)", "Atten. (dB/MHz/cm)"
    );
    println!("   {}", "-".repeat(70));

    for (name, tissue) in &tissues {
        println!(
            "   {:<25} {:>12.0} {:>12.0} {:>12.2}",
            name, tissue.sound_speed, tissue.density, tissue.attenuation
        );
    }
    println!();

    // ========================================================================
    // Example 5: Custom Configurations
    // ========================================================================

    println!("5. Custom Configurations");
    println!("   Building customized models\n");

    // Custom transducer
    println!("   a) Custom Research Transducer:");
    let custom_transducer = TransducerArray::linear_array()
        .frequency(10e6) // High frequency
        .elements(256) // Dense array
        .pitch(0.1e-3) // Fine pitch
        .element_width(0.095e-3)
        .element_height(4e-3)
        .build()?;

    println!("      - High-resolution configuration");
    println!("      - 10 MHz center frequency");
    println!("      - 256 elements with 0.1 mm pitch");
    println!();

    // Custom brain with specific dimensions
    println!("   b) Custom Brain Model:");
    let custom_brain = AnatomicalModel::brain_adult()
        .dimensions(0.18, 0.18, 0.14) // Larger brain
        .resolution(0.5e-3) // Finer resolution
        .build()?;

    println!("      - Custom dimensions: 18cm × 18cm × 14cm");
    println!("      - High resolution: 0.5 mm");
    println!();

    // Layered tissue model
    println!("   c) Custom Layered Tissue:");
    let layers = vec![TissueType::FAT, TissueType::MUSCLE, TissueType::LIVER];

    let layered = AnatomicalModel::layered_tissue(layers)
        .dimensions(0.10, 0.10, 0.08)
        .build()?;

    println!("      - Multi-layer propagation model");
    println!("      - Layers: Fat → Muscle → Liver");
    println!("      - Useful for studying refraction and attenuation");
    println!();

    // ========================================================================
    // Example 6: Complete Simulation Setup
    // ========================================================================

    println!("6. Complete Simulation Setup");
    println!("   Combining transducer, anatomy, and configuration\n");

    use kwavers::domain::builders::SimulationDomain;
    use kwavers::domain::grid::Grid;
    use kwavers::domain::medium::HomogeneousMedium;

    // Create components
    let transducer = TransducerArray::l12_5_philips().build()?;
    let anatomy = AnatomicalModel::brain_adult().build()?;

    // Create grid
    let grid = Grid::new(150, 150, 120, 1e-3, 1e-3, 1e-3);

    // Create domain
    let domain = SimulationDomain::new(grid, Box::new(HomogeneousMedium::water()))
        .with_transducer(transducer)
        .with_anatomy(anatomy);

    println!("   Complete simulation domain created:");
    println!("      ✓ Grid: 150 × 150 × 120 points");
    println!("      ✓ Transducer: Philips L12-5 (192 elements)");
    println!("      ✓ Anatomy: Adult brain model");
    println!("      ✓ Ready for simulation!");
    println!();

    println!("=== Domain Builders Examples Complete ===");
    println!("\nThese builders make it easy to set up realistic clinical simulations!");
    println!("Combine with the Simple API for complete end-to-end workflows.");

    Ok(())
}
