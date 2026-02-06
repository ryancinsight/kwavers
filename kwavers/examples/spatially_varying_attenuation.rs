//! Spatially-Varying Power Law Attenuation Example
//!
//! This example demonstrates the use of heterogeneous absorption models
//! where both α₀ (absorption coefficient) and γ (power law exponent) vary
//! spatially throughout the computational domain.
//!
//! This is critical for realistic tissue modeling, including:
//! - Multi-tissue interfaces (fat-muscle-bone)
//! - Tumor/lesion modeling
//! - Gradient tissues (skin layers, vessel walls)
//! - Temperature-dependent absorption for HIFU simulations

use kwavers::domain::medium::absorption::SpatiallyVaryingAbsorption;
use ndarray::Array3;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== Spatially-Varying Power Law Attenuation Example ===\n");

    // Example 1: Uniform medium (baseline)
    println!("1. Uniform Soft Tissue");
    println!("   Creating 100×100×100 grid with uniform properties...");

    let uniform = SpatiallyVaryingAbsorption::uniform(
        100, 100, 100, 0.75, // α₀ = 0.75 Np/m
        1.1,  // γ = 1.1 (typical soft tissue)
    )?;

    let stats = uniform.statistics();
    println!("   α₀: {:.3} Np/m (uniform)", stats.alpha_0_mean);
    println!("   γ:  {:.2} (uniform)", stats.gamma_mean);

    // Calculate absorption at different frequencies
    let freq_1mhz = uniform.absorption_at_point(50, 50, 50, 1e6);
    let freq_5mhz = uniform.absorption_at_point(50, 50, 50, 5e6);
    println!("   Absorption @ 1 MHz: {:.3} Np/m", freq_1mhz);
    println!("   Absorption @ 5 MHz: {:.3} Np/m", freq_5mhz);
    println!("   Frequency scaling: {:.2}x\n", freq_5mhz / freq_1mhz);

    // Example 2: Tumor embedded in soft tissue
    println!("2. Tumor Lesion Model");
    println!("   Creating tissue with spherical tumor inclusion...");

    let mut tumor_model = SpatiallyVaryingAbsorption::uniform(100, 100, 100, 0.75, 1.1)?;

    // Add tumor at center: higher absorption, different power law
    tumor_model.add_spherical_inclusion(
        (0.05, 0.05, 0.05), // Center at 5 cm
        0.01,               // 1 cm radius
        1.5,                // α₀ = 1.5 Np/m (2x background)
        1.3,                // γ = 1.3 (more frequency-dependent)
        0.001,              // dx = 1 mm
        0.001,              // dy = 1 mm
        0.001,              // dz = 1 mm
    );

    let stats_tumor = tumor_model.statistics();
    println!("   Background α₀: {:.3} Np/m", 0.75);
    println!("   Tumor α₀:      {:.3} Np/m", 1.5);
    println!(
        "   Field α₀ range: [{:.3}, {:.3}] Np/m",
        stats_tumor.alpha_0_min, stats_tumor.alpha_0_max
    );
    println!(
        "   Field γ range:  [{:.2}, {:.2}]\n",
        stats_tumor.gamma_min, stats_tumor.gamma_max
    );

    // Example 3: Multi-layer tissue (skin → fat → muscle)
    println!("3. Multi-Layer Tissue Model");
    println!("   Creating layered structure...");

    let mut multilayer = SpatiallyVaryingAbsorption::uniform(100, 100, 100, 0.5, 1.0)?;

    // Layer 1: Skin (0-10 mm)
    multilayer.set_region(0..10, 0..100, 0..100, 1.2, 1.15)?;

    // Layer 2: Fat (10-40 mm)
    multilayer.set_region(10..40, 0..100, 0..100, 0.63, 1.1)?;

    // Layer 3: Muscle (40-100 mm)
    multilayer.set_region(40..100, 0..100, 0..100, 1.3, 1.1)?;

    let stats_multi = multilayer.statistics();
    println!("   Skin:   α₀ = {:.2} Np/m, γ = {:.2}", 1.2, 1.15);
    println!("   Fat:    α₀ = {:.2} Np/m, γ = {:.2}", 0.63, 1.1);
    println!("   Muscle: α₀ = {:.2} Np/m, γ = {:.2}", 1.3, 1.1);
    println!(
        "   Overall range: α₀ ∈ [{:.2}, {:.2}] Np/m\n",
        stats_multi.alpha_0_min, stats_multi.alpha_0_max
    );

    // Example 4: Smooth gradient tissue interface
    println!("4. Smooth Tissue Gradient");
    println!("   Creating fat-muscle interface with Gaussian transition...");

    let mut gradient = SpatiallyVaryingAbsorption::uniform(100, 100, 100, 0.63, 1.1)?; // Start with fat

    // Add smooth transition to muscle properties
    gradient.add_gaussian_transition(
        (0.05, 0.05, 0.05), // Center of transition
        0.01,               // σ = 1 cm (transition width)
        1.3,                // Target α₀ (muscle)
        1.1,                // Target γ
        0.001,
        0.001,
        0.001,
    );

    println!("   Fat baseline:  α₀ = 0.63 Np/m");
    println!("   Muscle target: α₀ = 1.30 Np/m");
    println!("   Transition width: σ = 1 cm");
    println!("   (Smooth gradient avoids numerical artifacts)\n");

    // Example 5: Temperature-dependent absorption for HIFU
    println!("5. Temperature-Dependent Absorption (HIFU)");
    println!("   Modeling thermal effects during focused ultrasound...");

    let base_absorption = SpatiallyVaryingAbsorption::uniform(80, 80, 80, 0.75, 1.1)?;

    // Create temperature field with hot spot
    let mut temperature = Array3::from_elem((80, 80, 80), 310.15); // 37°C baseline

    // HIFU focal spot at center
    for i in 35..45 {
        for j in 35..45 {
            for k in 35..45 {
                let dist_sq = (i as f64 - 40.0).powi(2)
                    + (j as f64 - 40.0).powi(2)
                    + (k as f64 - 40.0).powi(2);
                let temp_rise = 50.0 * (-dist_sq / 20.0).exp(); // Gaussian heating
                temperature[[i, j, k]] += temp_rise;
            }
        }
    }

    let thermal_absorption = base_absorption.with_temperature_dependence(
        temperature,
        0.01, // 1% increase per Kelvin
    )?;

    let alpha_baseline = thermal_absorption.absorption_at_point(10, 10, 10, 1e6);
    let alpha_hotspot = thermal_absorption.absorption_at_point(40, 40, 40, 1e6);

    println!("   Baseline temp:  37°C");
    println!("   Hot spot temp:  ~87°C");
    println!("   Baseline α:     {:.3} Np/m", alpha_baseline);
    println!("   Hot spot α:     {:.3} Np/m", alpha_hotspot);
    println!(
        "   Thermal enhancement: {:.1}%\n",
        (alpha_hotspot / alpha_baseline - 1.0) * 100.0
    );

    // Example 6: Realistic breast tissue heterogeneity
    println!("6. Realistic Breast Tissue Heterogeneity");
    println!("   Modeling mixed glandular and adipose tissue...");

    let mut breast = SpatiallyVaryingAbsorption::uniform(120, 120, 80, 0.7, 1.1)?; // Mixed baseline

    // Add several glandular inclusions
    breast.add_spherical_inclusion((0.03, 0.03, 0.02), 0.008, 0.85, 1.2, 0.001, 0.001, 0.001);
    breast.add_spherical_inclusion((0.07, 0.04, 0.03), 0.012, 0.85, 1.2, 0.001, 0.001, 0.001);
    breast.add_spherical_inclusion((0.05, 0.08, 0.025), 0.01, 0.85, 1.2, 0.001, 0.001, 0.001);

    // Add fat regions
    breast.add_spherical_inclusion((0.02, 0.07, 0.04), 0.015, 0.48, 1.1, 0.001, 0.001, 0.001);
    breast.add_spherical_inclusion((0.09, 0.09, 0.05), 0.018, 0.48, 1.1, 0.001, 0.001, 0.001);

    let stats_breast = breast.statistics();
    println!("   Adipose (fat):   α₀ ≈ 0.48 Np/m, γ = 1.1");
    println!("   Glandular:       α₀ ≈ 0.85 Np/m, γ = 1.2");
    println!("   Field statistics:");
    println!(
        "     α₀ ∈ [{:.2}, {:.2}] Np/m",
        stats_breast.alpha_0_min, stats_breast.alpha_0_max
    );
    println!(
        "     γ ∈ [{:.2}, {:.2}]",
        stats_breast.gamma_min, stats_breast.gamma_max
    );
    println!("     Mean α₀: {:.2} Np/m\n", stats_breast.alpha_0_mean);

    // Example 7: Compute full 3D absorption field at specific frequency
    println!("7. Full 3D Absorption Field Computation");
    println!("   Computing absorption at 3.5 MHz for tumor model...");

    let absorption_field_3p5mhz = tumor_model.compute_absorption_field(3.5e6);

    let min_alpha = absorption_field_3p5mhz
        .iter()
        .cloned()
        .fold(f64::INFINITY, f64::min);
    let max_alpha = absorption_field_3p5mhz
        .iter()
        .cloned()
        .fold(f64::NEG_INFINITY, f64::max);
    let mean_alpha = absorption_field_3p5mhz.mean().unwrap();

    println!("   Frequency: 3.5 MHz");
    println!("   Field size: 100×100×100");
    println!("   α range: [{:.3}, {:.3}] Np/m", min_alpha, max_alpha);
    println!("   α mean:  {:.3} Np/m", mean_alpha);
    println!(
        "   Contrast: {:.2}x (tumor vs background)\n",
        max_alpha / min_alpha
    );

    // Validation
    println!("8. Physical Validation");
    println!("   Checking all models for physical consistency...");

    uniform.validate()?;
    tumor_model.validate()?;
    multilayer.validate()?;
    gradient.validate()?;
    thermal_absorption.validate()?;
    breast.validate()?;

    println!("   ✓ All absorption fields are physically valid");
    println!("   ✓ No negative values");
    println!("   ✓ Power law exponents in valid range [0, 3]");
    println!("   ✓ All values finite\n");

    println!("=== Example Complete ===");
    println!("\nKey Takeaways:");
    println!("• Spatially-varying absorption enables realistic tissue modeling");
    println!("• Both α₀ and γ can vary independently in 3D");
    println!("• Temperature dependence captures HIFU thermal effects");
    println!("• Smooth transitions avoid numerical artifacts");
    println!("• Essential for accurate ultrasound propagation simulations");

    Ok(())
}
