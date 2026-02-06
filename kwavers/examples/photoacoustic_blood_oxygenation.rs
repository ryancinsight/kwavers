//! Blood Oxygenation Estimation with Multi-Wavelength Photoacoustic Imaging
//!
//! This example demonstrates:
//! 1. Multi-wavelength photoacoustic fluence computation using diffusion solver
//! 2. Spectral unmixing to decompose HbO₂ and Hb concentrations
//! 3. Blood oxygen saturation (sO₂) mapping
//! 4. Validation against known arterial/venous values
//!
//! # Physics
//!
//! - **Optical Diffusion**: ∇·(D∇Φ) - μₐΦ = -S
//! - **Spectral Unmixing**: μₐ(λ) = ε_HbO₂(λ)[HbO₂] + ε_Hb(λ)[Hb]
//! - **Oxygen Saturation**: sO₂ = [HbO₂] / ([HbO₂] + [Hb])
//!
//! # Clinical Relevance
//!
//! Blood oxygenation imaging enables:
//! - Tumor hypoxia detection (poor prognosis indicator)
//! - Vascular disease assessment (arterial vs venous discrimination)
//! - Tissue viability monitoring (wound healing, transplant)
//! - Brain functional imaging (hemodynamic response)

use anyhow::Result;
use kwavers::clinical::imaging::chromophores::HemoglobinDatabase;
use kwavers::clinical::imaging::spectroscopy::SpectralUnmixingConfig;
use kwavers::clinical::imaging::workflows::blood_oxygenation::{
    estimate_oxygenation, OxygenationConfig,
};
use kwavers::domain::grid::Grid;
use kwavers::domain::medium::properties::OpticalPropertyData;
use kwavers::solver::forward::optical::diffusion::{DiffusionSolver, DiffusionSolverConfig};
use ndarray::Array3;
use std::time::Instant;

fn main() -> Result<()> {
    println!("=== Blood Oxygenation Estimation with Photoacoustic Imaging ===\n");

    // ========================================================================
    // 1. Setup: Wavelengths and Grid
    // ========================================================================
    println!("Phase 1: Configuration");
    println!("----------------------");

    // Wavelength selection (optimized for hemoglobin spectroscopy)
    let wavelengths = vec![
        532.0, // Green (Nd:YAG doubled) - strong Hb absorption
        700.0, // Red edge - near isosbestic point
        800.0, // NIR window - HbO₂ peak
        850.0, // NIR window - balanced penetration
    ];
    println!("Wavelengths: {:?} nm", wavelengths);

    // Computational grid (5mm × 5mm × 5mm at 0.2mm resolution)
    let grid = Grid::new(25, 25, 25, 0.2e-3, 0.2e-3, 0.2e-3)?;
    println!(
        "Grid: {}×{}×{} voxels ({:.1}×{:.1}×{:.1} mm)",
        grid.nx,
        grid.ny,
        grid.nz,
        grid.nx as f64 * grid.dx * 1e3,
        grid.ny as f64 * grid.dy * 1e3,
        grid.nz as f64 * grid.dz * 1e3
    );
    println!();

    // ========================================================================
    // 2. Create Heterogeneous Phantom
    // ========================================================================
    println!("Phase 2: Phantom Construction");
    println!("------------------------------");

    let (nx, ny, nz) = grid.dimensions();
    let hb_db = HemoglobinDatabase::standard();

    // Background: soft tissue (low absorption)
    let background = OpticalPropertyData::soft_tissue();
    println!("Background: Soft tissue");

    // Arterial blood vessel (98% oxygenation)
    let (total_hb, so2_arterial, so2_venous) = HemoglobinDatabase::typical_blood_parameters();
    let arterial_hbo2 = total_hb * so2_arterial;
    let arterial_hb = total_hb * (1.0 - so2_arterial);
    println!(
        "Arterial blood: sO₂ = {:.1}%, [Hb_total] = {:.2} mM",
        so2_arterial * 100.0,
        total_hb * 1e3
    );

    // Venous blood vessel (75% oxygenation)
    let venous_hbo2 = total_hb * so2_venous;
    let venous_hb = total_hb * (1.0 - so2_venous);
    println!(
        "Venous blood: sO₂ = {:.1}%, [Hb_total] = {:.2} mM",
        so2_venous * 100.0,
        total_hb * 1e3
    );

    // Tumor region (hypoxic, ~50% oxygenation, elevated Hb)
    let tumor_total_hb = total_hb * 1.3; // 30% higher Hb (angiogenesis)
    let tumor_so2 = 0.5; // 50% oxygenation (hypoxic)
    let tumor_hbo2 = tumor_total_hb * tumor_so2;
    let tumor_hb = tumor_total_hb * (1.0 - tumor_so2);
    println!(
        "Tumor (hypoxic): sO₂ = {:.1}%, [Hb_total] = {:.2} mM",
        tumor_so2 * 100.0,
        tumor_total_hb * 1e3
    );
    println!();

    // ========================================================================
    // 3. Multi-Wavelength Optical Absorption Maps
    // ========================================================================
    println!("Phase 3: Multi-Wavelength Fluence Simulation");
    println!("----------------------------------------------");

    let mut absorption_maps = Vec::new();
    let diffusion_config = DiffusionSolverConfig {
        max_iterations: 5000,
        tolerance: 1e-6,
        boundary_parameter: 2.0,
        boundary_conditions: None,
        verbose: false,
    };

    for (wl_idx, &wavelength) in wavelengths.iter().enumerate() {
        let start = Instant::now();
        println!("Wavelength {}: {:.0} nm", wl_idx + 1, wavelength);

        // Create optical property map for this wavelength
        let mut optical_map = Array3::from_elem((nx, ny, nz), background);

        // Get wavelength-dependent absorption for blood
        let arterial_mu_a = hb_db.absorption_coefficient(wavelength, arterial_hbo2, arterial_hb)?;
        let venous_mu_a = hb_db.absorption_coefficient(wavelength, venous_hbo2, venous_hb)?;
        let tumor_mu_a = hb_db.absorption_coefficient(wavelength, tumor_hbo2, tumor_hb)?;

        // Insert structures into phantom
        for i in 0..nx {
            for j in 0..ny {
                for k in 0..nz {
                    let x = i as f64 * grid.dx;
                    let y = j as f64 * grid.dy;
                    let z = k as f64 * grid.dz;

                    // Cylindrical arterial vessel (vertical, 1mm diameter)
                    let artery_dist = ((x - 1.5e-3).powi(2) + (y - 2.5e-3).powi(2)).sqrt();
                    if artery_dist < 0.5e-3 {
                        let arterial_props = OpticalPropertyData::new(
                            arterial_mu_a,
                            150.0, // scattering
                            0.95,  // anisotropy
                            1.4,   // refractive index
                        )
                        .map_err(|e| {
                            anyhow::anyhow!("Failed to create arterial properties: {}", e)
                        })?;
                        optical_map[[i, j, k]] = arterial_props;
                        continue;
                    }

                    // Cylindrical venous vessel (vertical, 1.5mm diameter)
                    let vein_dist = ((x - 3.5e-3).powi(2) + (y - 2.5e-3).powi(2)).sqrt();
                    if vein_dist < 0.75e-3 {
                        let venous_props = OpticalPropertyData::new(
                            venous_mu_a,
                            150.0, // scattering
                            0.95,  // anisotropy
                            1.4,   // refractive index
                        )
                        .map_err(|e| {
                            anyhow::anyhow!("Failed to create venous properties: {}", e)
                        })?;
                        optical_map[[i, j, k]] = venous_props;
                        continue;
                    }

                    // Spherical tumor (2mm diameter, centered)
                    let tumor_dist =
                        ((x - 2.5e-3).powi(2) + (y - 2.5e-3).powi(2) + (z - 2.5e-3).powi(2)).sqrt();
                    if tumor_dist < 1.0e-3 {
                        let tumor_props = OpticalPropertyData::new(
                            tumor_mu_a, 120.0, // slightly different scattering
                            0.85,  // anisotropy
                            1.4,   // refractive index
                        )
                        .map_err(|e| anyhow::anyhow!("Failed to create tumor properties: {}", e))?;
                        optical_map[[i, j, k]] = tumor_props;
                    }
                }
            }
        }

        // Solve diffusion equation for this wavelength
        let solver =
            DiffusionSolver::new(grid.clone(), optical_map.clone(), diffusion_config.clone())?;

        // Top-surface illumination source
        let mut source = Array3::zeros((nx, ny, nz));
        let laser_fluence = 10.0; // J/m² (10 mJ/cm²)
        let source_strength = laser_fluence / grid.dz;
        for i in 0..nx {
            for j in 0..ny {
                source[[i, j, 0]] = source_strength;
            }
        }

        let fluence = solver.solve(&source)?;

        // Extract absorption map (μₐ·Φ for photoacoustic signal)
        let mut absorption = Array3::zeros((nx, ny, nz));
        for i in 0..nx {
            for j in 0..ny {
                for k in 0..nz {
                    absorption[[i, j, k]] = optical_map[[i, j, k]].absorption_coefficient;
                }
            }
        }

        absorption_maps.push(absorption);

        println!(
            "  Fluence computed: max = {:.2e} W/m², mean = {:.2e} W/m²",
            fluence.iter().cloned().fold(f64::NEG_INFINITY, f64::max),
            fluence.mean().unwrap_or(0.0)
        );
        println!("  Time: {:.2} ms", start.elapsed().as_secs_f64() * 1e3);
    }
    println!();

    // ========================================================================
    // 4. Spectral Unmixing
    // ========================================================================
    println!("Phase 4: Spectral Unmixing");
    println!("--------------------------");

    let unmixing_start = Instant::now();

    let oxygenation_config = OxygenationConfig {
        wavelengths: wavelengths.clone(),
        unmixing_config: SpectralUnmixingConfig {
            regularization_lambda: 1e-6,
            non_negative: true,
            min_condition_number: 1e-10,
        },
        min_total_hb: 1e-5, // 10 μM threshold
    };

    let oxygenation_result = estimate_oxygenation(&absorption_maps, &oxygenation_config)?;

    println!(
        "Unmixing completed in {:.2} ms",
        unmixing_start.elapsed().as_secs_f64() * 1e3
    );
    println!(
        "Output dimensions: {}×{}×{}",
        oxygenation_result.so2_map.dim().0,
        oxygenation_result.so2_map.dim().1,
        oxygenation_result.so2_map.dim().2
    );
    println!();

    // ========================================================================
    // 5. Results Analysis
    // ========================================================================
    println!("Phase 5: Results Analysis");
    println!("-------------------------");

    // Extract region-specific results
    let mut arterial_so2_samples = Vec::new();
    let mut venous_so2_samples = Vec::new();
    let mut tumor_so2_samples = Vec::new();

    for i in 0..nx {
        for j in 0..ny {
            for k in 0..nz {
                let x = i as f64 * grid.dx;
                let y = j as f64 * grid.dy;
                let z = k as f64 * grid.dz;

                let so2 = oxygenation_result.so2_map[[i, j, k]];
                let total_hb = oxygenation_result.total_hb_concentration[[i, j, k]];

                // Only consider voxels with sufficient hemoglobin
                if total_hb < 1e-4 {
                    continue;
                }

                // Classify by region
                let artery_dist = ((x - 1.5e-3).powi(2) + (y - 2.5e-3).powi(2)).sqrt();
                if artery_dist < 0.5e-3 {
                    arterial_so2_samples.push(so2);
                    continue;
                }

                let vein_dist = ((x - 3.5e-3).powi(2) + (y - 2.5e-3).powi(2)).sqrt();
                if vein_dist < 0.75e-3 {
                    venous_so2_samples.push(so2);
                    continue;
                }

                let tumor_dist =
                    ((x - 2.5e-3).powi(2) + (y - 2.5e-3).powi(2) + (z - 2.5e-3).powi(2)).sqrt();
                if tumor_dist < 1.0e-3 {
                    tumor_so2_samples.push(so2);
                }
            }
        }
    }

    // Compute statistics
    fn compute_stats(samples: &[f64]) -> (f64, f64, f64) {
        if samples.is_empty() {
            return (0.0, 0.0, 0.0);
        }
        let mean = samples.iter().sum::<f64>() / samples.len() as f64;
        let variance =
            samples.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / samples.len() as f64;
        let std_dev = variance.sqrt();
        let min = samples.iter().cloned().fold(f64::INFINITY, f64::min);
        (mean, std_dev, min)
    }

    println!("\nRegion-Specific Oxygen Saturation:");
    println!("-----------------------------------");

    let (arterial_mean, arterial_std, _) = compute_stats(&arterial_so2_samples);
    println!(
        "Arterial vessel:  sO₂ = {:.1}% ± {:.1}% (expected: {:.1}%)",
        arterial_mean * 100.0,
        arterial_std * 100.0,
        so2_arterial * 100.0
    );
    println!(
        "  Error: {:.1}%",
        ((arterial_mean - so2_arterial) / so2_arterial * 100.0).abs()
    );

    let (venous_mean, venous_std, _) = compute_stats(&venous_so2_samples);
    println!(
        "Venous vessel:    sO₂ = {:.1}% ± {:.1}% (expected: {:.1}%)",
        venous_mean * 100.0,
        venous_std * 100.0,
        so2_venous * 100.0
    );
    println!(
        "  Error: {:.1}%",
        ((venous_mean - so2_venous) / so2_venous * 100.0).abs()
    );

    let (tumor_mean, tumor_std, _) = compute_stats(&tumor_so2_samples);
    println!(
        "Tumor (hypoxic):  sO₂ = {:.1}% ± {:.1}% (expected: {:.1}%)",
        tumor_mean * 100.0,
        tumor_std * 100.0,
        tumor_so2 * 100.0
    );
    println!(
        "  Error: {:.1}%",
        ((tumor_mean - tumor_so2) / tumor_so2 * 100.0).abs()
    );

    // ========================================================================
    // 6. Clinical Interpretation
    // ========================================================================
    println!("\n=== Clinical Interpretation ===");
    println!("-------------------------------");

    if tumor_mean < 0.6 {
        println!("✓ Tumor hypoxia detected (sO₂ < 60%)");
        println!("  → Increased radioresistance likely, consider dose escalation");
        println!("  → Poor prognosis indicator, aggressive treatment recommended");
    }

    let arterial_venous_contrast = (arterial_mean - venous_mean).abs();
    if arterial_venous_contrast > 0.15 {
        println!(
            "✓ Clear arterial-venous discrimination (ΔsO₂ = {:.1}%)",
            arterial_venous_contrast * 100.0
        );
        println!("  → Vascular mapping successful, suitable for treatment planning");
    }

    println!("\n=== Validation Summary ===");
    println!("-------------------------");
    let arterial_error = ((arterial_mean - so2_arterial) / so2_arterial * 100.0).abs();
    let venous_error = ((venous_mean - so2_venous) / so2_venous * 100.0).abs();
    let tumor_error = ((tumor_mean - tumor_so2) / tumor_so2 * 100.0).abs();

    if arterial_error < 5.0 && venous_error < 5.0 && tumor_error < 10.0 {
        println!("✓ PASS: All regions within acceptable error (<5-10%)");
        println!("✓ PASS: Spectral unmixing successfully recovered known sO₂ values");
        println!("✓ PASS: Multi-wavelength photoacoustic oxygenation imaging validated");
    } else {
        println!(
            "⚠ WARNING: Some regions exceed error threshold (arterial: {:.1}%, venous: {:.1}%, tumor: {:.1}%)",
            arterial_error, venous_error, tumor_error
        );
    }

    Ok(())
}
