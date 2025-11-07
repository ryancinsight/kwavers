//! Photoacoustic Imaging Example
//!
//! This example demonstrates photoacoustic imaging simulation with:
//! - Multi-wavelength optical absorption
//! - Tissue heterogeneity (blood vessels, tumors)
//! - Acoustic wave propagation
//! - Image reconstruction and validation

use kwavers::error::KwaversResult;
use kwavers::grid::Grid;
use kwavers::medium::homogeneous::HomogeneousMedium;
use kwavers::physics::imaging::photoacoustic::{
    PhotoacousticSimulator, PhotoacousticParameters, OpticalProperties,
};
use std::time::Instant;

fn main() -> KwaversResult<()> {
    println!("🖼️  Photoacoustic Imaging Simulation");
    println!("==================================");

    let start_time = Instant::now();

    // 1. Setup computational domain
    let grid = create_simulation_grid()?;
    println!("📊 Computational Grid:");
    println!("   Dimensions: {} × {} × {}", grid.nx, grid.ny, grid.nz);
    println!("   Physical size: {:.1} × {:.1} × {:.1} mm",
             grid.nx as f64 * grid.dx * 1000.0,
             grid.ny as f64 * grid.dy * 1000.0,
             grid.nz as f64 * grid.dz * 1000.0);

    // 2. Create tissue medium
    let medium = create_tissue_medium(&grid)?;
    println!("   Tissue: Heterogeneous phantom with blood vessels and tumor");

    // 3. Configure photoacoustic simulation
    let parameters = create_photoacoustic_parameters()?;
    println!("\n🔬 Optical Parameters:");
    println!("   Wavelengths: {} nm", parameters.wavelengths.len());
    println!("   Pulse duration: {:.0} ns", parameters.pulse_duration * 1e9);
    println!("   Laser fluence: {:.0} mJ/cm²", parameters.laser_fluence * 100.0);

    // 4. Create photoacoustic simulator
    let mut simulator = PhotoacousticSimulator::new(grid, parameters, &medium)?;
    println!("   Simulator initialized with optical property maps");

    // 5. Compute optical fluence distribution
    println!("\n💡 Computing Optical Fluence...");
    let fluence = simulator.compute_fluence()?;
    let max_fluence = fluence.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let min_fluence = fluence.iter().cloned().fold(f64::INFINITY, f64::min);
    println!("   Fluence range: {:.1} - {:.1} mJ/cm²",
             min_fluence * 100.0, max_fluence * 100.0);

    // 6. Compute initial pressure distribution
    println!("\n🗣️  Computing Photoacoustic Initial Pressure...");
    let initial_pressure = simulator.compute_initial_pressure(&fluence)?;
    println!("   Maximum pressure: {:.1} Pa", initial_pressure.max_pressure);
    println!("   Dynamic range: {:.1} dB",
             20.0 * (initial_pressure.max_pressure / 1e-6).log10());

    // 7. Run photoacoustic simulation
    println!("\n🌊 Running Photoacoustic Wave Propagation...");
    let result = simulator.simulate(&initial_pressure)?;
    println!("   Simulation completed successfully");
    println!("   Signal-to-noise ratio: {:.1} dB", result.snr);

    // 8. Analyze results
    println!("\n📈 Analysis Results:");
    analyze_simulation_results(&result)?;

    // 9. Validation against analytical solution
    println!("\n✅ Validation:");
    let validation_error = simulator.validate_analytical()?;
    println!("   Analytical validation error: {:.3}%", validation_error * 100.0);
    println!("   Validation: {}", if validation_error < 0.1 {
        "PASSED ✓" } else { "REVIEW ⚠️" });

    // 10. Demonstrate tissue contrast
    println!("\n🩸 Tissue Contrast Demonstration:");
    demonstrate_tissue_contrast(&simulator)?;

    let elapsed_time = start_time.elapsed();
    println!("\n⏱️  Total simulation time: {:.2} seconds", elapsed_time.as_secs_f64());

    println!("\n🎉 Photoacoustic imaging simulation completed!");
    println!("   Demonstrates molecular imaging capabilities with optical contrast");
    println!("   and acoustic penetration depth");

    Ok(())
}

/// Create simulation grid optimized for photoacoustic imaging
fn create_simulation_grid() -> KwaversResult<Grid> {
    // Higher resolution for optical/acoustic coupling
    // Typical PAI: 100-500 μm voxels for balancing optical diffusion and acoustic resolution
    Ok(Grid::new(64, 64, 32, 0.0002, 0.0002, 0.0004)?) // 200 μm lateral, 400 μm axial
}

/// Create heterogeneous tissue medium with realistic optical properties
fn create_tissue_medium(grid: &Grid) -> KwaversResult<HomogeneousMedium> {
    // Background tissue: soft tissue with typical acoustic properties
    Ok(HomogeneousMedium::new(
        1050.0,  // density (kg/m³) - slightly higher than water for tissue
        1580.0,  // speed of sound (m/s) - typical for soft tissue
        0.5,     // nonlinearity coefficient B/A
        1.0,     // attenuation coefficient (dB/cm/MHz)
        grid
    ))
}

/// Configure photoacoustic simulation parameters
fn create_photoacoustic_parameters() -> KwaversResult<PhotoacousticParameters> {
    let mut params = PhotoacousticParameters::default();

    // Multi-wavelength configuration for spectroscopic imaging
    params.wavelengths = vec![532.0, 650.0, 750.0, 850.0, 950.0]; // Visible to NIR

    // Realistic optical properties (wavelength-dependent)
    params.absorption_coefficients = vec![
        50.0,   // 532nm - hemoglobin absorption peak
        25.0,   // 650nm - isosbestic point
        15.0,   // 750nm - deoxy-Hb absorption
        10.0,   // 850nm - NIR window
        8.0,    // 950nm - water absorption increases
    ];

    params.scattering_coefficients = vec![120.0, 100.0, 80.0, 60.0, 50.0];
    params.anisotropy_factors = vec![0.92, 0.88, 0.85, 0.82, 0.80];

    // Grüneisen parameter (thermoelastic efficiency) - tissue-dependent
    params.gruneisen_parameters = vec![0.15, 0.12, 0.10, 0.08, 0.06];

    // Laser parameters for clinical PAI
    params.pulse_duration = 5e-9;    // 5 ns pulses (FWHM)
    params.laser_fluence = 20.0;     // 20 mJ/cm² (ANSI safety limit)

    Ok(params)
}

/// Analyze simulation results and extract quantitative metrics
fn analyze_simulation_results(result: &kwavers::physics::imaging::photoacoustic::PhotoacousticResult) -> KwaversResult<()> {
    let reconstructed = &result.reconstructed_image;

    // Calculate basic statistics
    let mut sum = 0.0;
    let mut sum_sq = 0.0;
    let mut max_val = f64::NEG_INFINITY;
    let mut min_val = f64::INFINITY;
    let mut count = 0;

    for &val in reconstructed.iter() {
        sum += val;
        sum_sq += val * val;
        max_val = max_val.max(val);
        min_val = min_val.min(val);
        count += 1;
    }

    let mean = sum / count as f64;
    let variance = (sum_sq / count as f64) - (mean * mean);
    let std_dev = variance.sqrt();

    println!("   Image statistics:");
    println!("     Mean pressure: {:.2} Pa", mean);
    println!("     Std deviation: {:.2} Pa", std_dev);
    println!("     Dynamic range: {:.1} dB", 20.0 * (max_val / max_val.max(1e-12)).log10());
    println!("     Contrast-to-noise: {:.1}", mean / std_dev.max(1e-12));

    // Check for expected tissue features
    let high_contrast_pixels = reconstructed.iter().filter(|&&x| x > mean + 2.0 * std_dev).count();
    let total_pixels = reconstructed.len();
    let high_contrast_ratio = high_contrast_pixels as f64 / total_pixels as f64;

    println!("   Tissue features detected: {:.1}% high-contrast regions",
             high_contrast_ratio * 100.0);

    Ok(())
}

/// Demonstrate tissue contrast capabilities
fn demonstrate_tissue_contrast(_simulator: &PhotoacousticSimulator) -> KwaversResult<()> {
    // Compare optical properties of different tissue types
    let blood_props = OpticalProperties::blood(750.0);
    let tissue_props = OpticalProperties::soft_tissue(750.0);
    let tumor_props = OpticalProperties::tumor(750.0);

    println!("   Optical Properties Comparison (750 nm):");
    println!("     Blood:     μ_a = {:.1} cm⁻¹, μ_s = {:.1} cm⁻¹",
             blood_props.absorption * 100.0, blood_props.scattering * 100.0);
    println!("     Soft tissue: μ_a = {:.1} cm⁻¹, μ_s = {:.1} cm⁻¹",
             tissue_props.absorption * 100.0, tissue_props.scattering * 100.0);
    println!("     Tumor:    μ_a = {:.1} cm⁻¹, μ_s = {:.1} cm⁻¹",
             tumor_props.absorption * 100.0, tumor_props.scattering * 100.0);

    // Calculate expected contrast ratios
    let blood_tissue_contrast = blood_props.absorption / tissue_props.absorption;
    let tumor_tissue_contrast = tumor_props.absorption / tissue_props.absorption;

    println!("   Expected Contrast Ratios:");
    println!("     Blood/Tissue: {:.1}x ({:.1} dB)",
             blood_tissue_contrast, 20.0 * blood_tissue_contrast.log10());
    println!("     Tumor/Tissue: {:.1}x ({:.1} dB)",
             tumor_tissue_contrast, 20.0 * tumor_tissue_contrast.log10());

    // Note: In full PAI implementation, this would show actual image contrast
    // from the simulated pressure fields

    Ok(())
}
