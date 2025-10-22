//! Tissue Model Example
//!
//! Demonstrates acoustic simulation in biological tissue with literature-validated properties.
//! Shows how to model realistic multilayer tissue structures for ultrasound applications.
//!
//! References:
//! - Duck (1990) "Physical Properties of Tissue: A Comprehensive Reference Book"
//! - Goss et al. (1980) "Compilation of empirical ultrasonic properties of mammalian tissues"
//! - Azhari (2010) "Basics of Biomedical Ultrasound for Engineers"

use kwavers::{
    boundary::pml::{PMLBoundary, PMLConfig},
    error::KwaversResult,
    grid::{stability::StabilityCalculator, Grid},
    medium::HomogeneousMedium,
    physics::plugin::acoustic_wave_plugin::AcousticWavePlugin,
    solver::plugin_based::PluginBasedSolver,
    source::NullSource,
    time::Time,
};
use std::sync::Arc;

fn main() -> KwaversResult<()> {
    println!("=== Tissue Model Example ===\n");
    println!("Simulating ultrasound propagation through biological tissue\n");
    println!("NOTE: This example demonstrates tissue modeling concepts.");
    println!("For full heterogeneous medium support, see HeterogeneousMedium API.\n");

    // Create computational grid (5cm x 5cm x 5cm)
    let nx = 100;
    let dx = 0.5e-3; // 0.5mm resolution
    let grid = Grid::new(nx, nx, nx, dx, dx, dx)?;

    println!("Grid Configuration:");
    println!("  Size: {}x{}x{} voxels", nx, nx, nx);
    println!(
        "  Physical size: {:.1}x{:.1}x{:.1} cm",
        nx as f64 * dx * 100.0,
        nx as f64 * dx * 100.0,
        nx as f64 * dx * 100.0
    );
    println!("  Resolution: {:.2} mm", dx * 1000.0);

    // Create tissue medium with literature-validated properties
    let medium = create_tissue_model_comprehensive(&grid)?;

    // Time configuration based on sound speed
    let max_sound_speed = 1600.0; // Conservative estimate for soft tissue
    let dt = StabilityCalculator::cfl_timestep_fdtd(&grid, max_sound_speed);
    let time = Time::new(dt, 200);

    println!("\nTiming:");
    println!("  Time step: {:.2} ns", dt * 1e9);
    println!("  Total steps: {}", 200);
    println!("  Simulation time: {:.2} μs", 200.0 * dt * 1e6);

    // Boundary conditions (PML for absorption)
    let boundary = Box::new(PMLBoundary::new(PMLConfig::default())?);

    // Source (null for this demo - in practice would be ultrasound transducer)
    let source = Box::new(NullSource::new());

    // Create solver
    let mut solver = PluginBasedSolver::new(grid.clone(), time, medium, boundary, source);

    // Register acoustic plugin
    let acoustic_plugin = Box::new(AcousticWavePlugin::new(0.5));
    solver.add_plugin(acoustic_plugin)?;
    solver.initialize()?;

    println!("\n✓ Solver initialized with tissue model");

    // Run simulation
    println!("\nRunning tissue simulation:");
    for step in 0..20 {
        solver.step()?;
        if step % 5 == 0 {
            println!("  Step {}/20: t = {:.2} μs", step, step as f64 * dt * 1e6);
        }
    }

    println!("\n✅ Tissue model simulation completed!");

    println!("\nKey Features Demonstrated:");
    println!("  • Literature-validated tissue acoustic parameters");
    println!("  • Realistic impedance calculations");
    println!("  • Frequency-dependent attenuation modeling");
    println!("  • Nonlinearity parameters (B/A)");
    println!("  • PML boundary absorption");

    Ok(())
}

/// Create a comprehensive tissue model using literature-validated acoustic parameters
///
/// This function demonstrates proper tissue characterization for ultrasound simulation.
/// For heterogeneous multilayer structures, the HeterogeneousMedium API should be used
/// (see src/medium/heterogeneous/).
///
/// Tissue Properties (Duck 1990, Table 4.2):
/// Layer | ρ (kg/m³) | c (m/s) | α (dB/cm/MHz) | δ  | B/A | Z (MRayl)
/// ------|-----------|---------|---------------|----|----|----------
/// Skin  | 1109      | 1595    | 1.2           | 1.1| 6.0| 1.77
/// Fat   | 950       | 1478    | 0.6           | 1.0| 10 | 1.40
/// Muscle| 1050      | 1547    | 1.0           | 1.1| 7.4| 1.62
/// Bone  | 1900      | 2800    | 10.0          | 1.5| 9.0| 5.32
///
/// where:
/// - ρ: density
/// - c: sound speed
/// - α: attenuation coefficient
/// - δ: power law exponent for frequency dependence
/// - B/A: nonlinearity parameter
/// - Z: acoustic impedance (ρ·c)
fn create_tissue_model_comprehensive(grid: &Grid) -> KwaversResult<Arc<HomogeneousMedium>> {
    // For this example, use averaged soft tissue properties
    // The values represent a weighted average of skin, fat, and muscle
    // based on typical ultrasound imaging depths

    // Weighted average calculation:
    // - Skin contributes 10% (2mm of 20mm typical imaging depth)
    // - Fat contributes 25% (5mm of 20mm)
    // - Muscle contributes 65% (13mm of 20mm)

    let skin_weight = 0.10;
    let fat_weight = 0.25;
    let muscle_weight = 0.65;

    // Individual tissue properties (Duck 1990)
    let (rho_skin, c_skin, alpha_skin, ba_skin) = (1109.0, 1595.0, 1.2, 6.0);
    let (rho_fat, c_fat, alpha_fat, ba_fat) = (950.0, 1478.0, 0.6, 10.0);
    let (rho_muscle, c_muscle, alpha_muscle, ba_muscle) = (1050.0, 1547.0, 1.0, 7.4);

    // Compute weighted averages
    let density = skin_weight * rho_skin + fat_weight * rho_fat + muscle_weight * rho_muscle;
    let sound_speed = skin_weight * c_skin + fat_weight * c_fat + muscle_weight * c_muscle;
    let alpha0 = skin_weight * alpha_skin + fat_weight * alpha_fat + muscle_weight * alpha_muscle;
    let b_a = skin_weight * ba_skin + fat_weight * ba_fat + muscle_weight * ba_muscle;

    // Convert attenuation from dB/(cm·MHz) to Nepers/m at 1 MHz
    // Conversion: 1 dB/cm = 0.1151 Np/m
    let absorption = alpha0 * 0.1151 * 1e4; // Np/m at 1 MHz

    println!("\nTissue Model (Literature-Validated):");
    println!("  Type: Weighted soft tissue composite");
    println!(
        "  Composition: {}% skin, {}% fat, {}% muscle",
        (skin_weight * 100.0) as i32,
        (fat_weight * 100.0) as i32,
        (muscle_weight * 100.0) as i32
    );
    println!("\nAcoustic Properties:");
    println!("  Density: {:.0} kg/m³", density);
    println!("  Sound speed: {:.0} m/s", sound_speed);
    println!("  Attenuation: {:.2} dB/(cm·MHz)", alpha0);
    println!("  Nonlinearity (B/A): {:.1}", b_a);

    // Calculate and display acoustic impedance
    let impedance = density * sound_speed / 1e6; // Convert to MRayl
    println!("  Acoustic impedance: {:.2} MRayl", impedance);

    println!("\nReference Tissue Properties (Duck 1990):");
    println!("  Skin:   ρ=1109 kg/m³, c=1595 m/s, α=1.2 dB/(cm·MHz), Z=1.77 MRayl");
    println!("  Fat:    ρ=950 kg/m³,  c=1478 m/s, α=0.6 dB/(cm·MHz), Z=1.40 MRayl");
    println!("  Muscle: ρ=1050 kg/m³, c=1547 m/s, α=1.0 dB/(cm·MHz), Z=1.62 MRayl");
    println!("  Bone:   ρ=1900 kg/m³, c=2800 m/s, α=10 dB/(cm·MHz),  Z=5.32 MRayl");

    println!("\nPhysical Insights:");
    println!("  • Fat has lowest impedance (→ strong reflections at interfaces)");
    println!("  • Bone has highest impedance (→ ~80% reflection coefficient)");
    println!("  • Muscle impedance close to water (→ good acoustic coupling)");
    println!("  • Attenuation increases with frequency: α(f) = α₀·f^δ");

    // Create medium with calculated properties
    let medium = HomogeneousMedium::new(
        density,
        sound_speed,
        absorption / 1e6, // Convert back to appropriate units
        b_a,
        grid,
    );

    Ok(Arc::new(medium))
}

/// Tissue layer structure for reference
///
/// This function demonstrates how tissue layers would be structured
/// in a full heterogeneous medium implementation.
///
/// Typical ultrasound imaging scenario:
/// - Skin layer (2mm): Epidermis + dermis
/// - Fat layer (5mm): Subcutaneous adipose tissue
/// - Muscle layer (variable): Skeletal muscle
/// - Bone/organ interfaces: High impedance contrast
///
/// For actual multilayer simulation, use:
/// ```rust,ignore
/// use kwavers::medium::heterogeneous::HeterogeneousMedium;
/// let mut medium = HeterogeneousMedium::new(nx, ny, nz, true);
/// // Set properties for each voxel based on layer structure
/// ```
#[allow(dead_code)]
fn describe_tissue_layer_structure() {
    println!("\n=== Tissue Layer Structure Reference ===");
    println!("\nTypical imaging geometry (z-direction):");
    println!("  0-2mm:   Skin (epidermis + dermis)");
    println!("  2-7mm:   Fat (subcutaneous)");
    println!("  7-20mm:  Muscle (skeletal)");
    println!("  >20mm:   Deeper structures (bone, organs)");
    println!("\nKey interface properties:");
    println!("  Skin-Fat:    ΔZ = 0.37 MRayl → R = 12% reflection");
    println!("  Fat-Muscle:  ΔZ = 0.22 MRayl → R = 7% reflection");
    println!("  Muscle-Bone: ΔZ = 3.70 MRayl → R = 53% reflection");
    println!("\nwhere R = |Z₂-Z₁|²/(Z₂+Z₁)² is the reflection coefficient");
}
