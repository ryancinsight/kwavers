// examples/monte_carlo_validation.rs
//! Monte Carlo vs. Diffusion Solver Validation
//!
//! This example compares Monte Carlo photon transport with the diffusion approximation
//! to validate both solvers and understand their respective domains of validity.
//!
//! # Theoretical Background
//!
//! ## Diffusion Approximation Validity
//!
//! The diffusion approximation is valid when:
//! 1. μ_s' >> μ_a (scattering dominates absorption)
//! 2. Source-detector distance >> 1/μ_s' (multiple scattering regime)
//! 3. No void regions or sharp discontinuities
//!
//! ## Expected Results
//!
//! - **High scattering regime**: MC and diffusion should agree well
//! - **Low scattering regime**: MC accurate, diffusion overestimates penetration
//! - **Near source**: MC accurate, diffusion breaks down (ballistic regime)
//!
//! # Metrics
//!
//! - Fluence distribution comparison (spatial profiles)
//! - Penetration depth validation
//! - Relative error maps
//! - Runtime performance comparison

use anyhow::Result;
use kwavers::clinical::imaging::phantoms::PhantomBuilder;
use kwavers::domain::grid::{Grid3D, GridDimensions};
use kwavers::domain::medium::properties::OpticalPropertyData;
use kwavers::physics::optics::map_builder::OpticalPropertyMap;
use kwavers::physics::optics::monte_carlo::{MonteCarloSolver, PhotonSource, SimulationConfig};
use kwavers::solver::forward::optical::diffusion::{DiffusionSolver, DiffusionSolverConfig};
use ndarray::Array3;
use std::time::Instant;

fn main() -> Result<()> {
    println!("=== Monte Carlo vs. Diffusion Solver Validation ===\n");

    // Test 1: High scattering regime (diffusion should be accurate)
    println!("Test 1: High Scattering Regime (μ_s' >> μ_a)");
    println!("Expected: Good agreement between MC and diffusion\n");
    validate_high_scattering()?;

    // Test 2: Low scattering regime (diffusion breaks down)
    println!("\nTest 2: Low Scattering Regime (μ_s' ~ μ_a)");
    println!("Expected: Diffusion overestimates penetration\n");
    validate_low_scattering()?;

    // Test 3: Layered tissue phantom
    println!("\nTest 3: Layered Tissue Phantom");
    println!("Expected: Both methods capture layer boundaries\n");
    validate_layered_tissue()?;

    // Test 4: Blood vessel phantom
    println!("\nTest 4: Blood Vessel Phantom");
    println!("Expected: MC captures fine structure, diffusion smooths\n");
    validate_blood_vessel()?;

    println!("\n=== Validation Complete ===");
    println!("\nKey Findings:");
    println!("1. Diffusion approximation is accurate in high-scattering regime");
    println!("2. Monte Carlo is required for low-scattering or ballistic regimes");
    println!("3. MC provides higher fidelity at ~100x computational cost");
    println!("4. Use diffusion for real-time applications, MC for validation");

    Ok(())
}

fn validate_high_scattering() -> Result<()> {
    // Small grid for fast computation
    let dims = GridDimensions::new(30, 30, 30, 0.001, 0.001, 0.001);
    let grid = Grid3D::new(30, 30, 30, 0.001, 0.001, 0.001)?;

    // High scattering tissue: μ_s' >> μ_a
    let mut builder = kwavers::physics::optics::map_builder::OpticalPropertyMapBuilder::new(dims);
    builder.set_background(OpticalPropertyData::soft_tissue()); // μ_a=0.5, μ_s=100, g=0.9
    let optical_map = builder.build();

    let cx = 0.015;
    let cy = 0.015;
    let cz = 0.0;

    // Monte Carlo simulation
    println!("  Running Monte Carlo (1M photons)...");
    let mc_solver = MonteCarloSolver::new(grid.clone(), optical_map.clone());
    let mc_source = PhotonSource::pencil_beam([cx, cy, cz], [0.0, 0.0, 1.0]);
    let mc_config = SimulationConfig::default()
        .num_photons(1_000_000)
        .max_steps(10_000);

    let mc_start = Instant::now();
    let mc_result = mc_solver.simulate(&mc_source, &mc_config)?;
    let mc_time = mc_start.elapsed();

    println!("  Running Diffusion solver...");
    let diff_start = Instant::now();
    let diff_fluence = solve_diffusion_fluence(&grid, &optical_map, [cx, cy, cz])?;
    let diff_time = diff_start.elapsed();

    // Compare results
    let mc_fluence = mc_result.normalized_fluence();
    let (mean_error, max_error, correlation) = compare_fluence(&mc_fluence, &diff_fluence, dims);

    println!("  Results:");
    println!("    MC runtime:        {:?}", mc_time);
    println!("    Diffusion runtime: {:?}", diff_time);
    println!("    Mean rel. error:   {:.2}%", mean_error * 100.0);
    println!("    Max rel. error:    {:.2}%", max_error * 100.0);
    println!("    Correlation:       {:.4}", correlation);

    if mean_error < 0.10 && correlation > 0.90 {
        println!("  ✓ PASS: Good agreement in high scattering regime");
    } else {
        println!("  ⚠ WARNING: Unexpected deviation in high scattering regime");
    }

    Ok(())
}

fn validate_low_scattering() -> Result<()> {
    let dims = GridDimensions::new(30, 30, 30, 0.001, 0.001, 0.001);
    let grid = Grid3D::new(30, 30, 30, 0.001, 0.001, 0.001)?;

    // Low scattering tissue: μ_s' ~ μ_a (diffusion breaks down)
    let low_scatter = OpticalPropertyData::new(
        5.0,  // μ_a
        20.0, // μ_s
        0.5,  // g (low anisotropy)
        1.4,  // n
    )
    .map_err(anyhow::Error::msg)?;

    let mut builder = kwavers::physics::optics::map_builder::OpticalPropertyMapBuilder::new(dims);
    builder.set_background(low_scatter);
    let optical_map = builder.build();

    let cx = 0.015;
    let cy = 0.015;
    let cz = 0.0;

    // Monte Carlo (ground truth)
    println!("  Running Monte Carlo (500k photons)...");
    let mc_solver = MonteCarloSolver::new(grid.clone(), optical_map.clone());
    let mc_source = PhotonSource::pencil_beam([cx, cy, cz], [0.0, 0.0, 1.0]);
    let mc_config = SimulationConfig::default().num_photons(500_000);

    let mc_start = Instant::now();
    let mc_result = mc_solver.simulate(&mc_source, &mc_config)?;
    let mc_time = mc_start.elapsed();

    println!("  Running Diffusion solver...");
    let diff_start = Instant::now();
    let diff_fluence = solve_diffusion_fluence(&grid, &optical_map, [cx, cy, cz])?;
    let diff_time = diff_start.elapsed();

    // Compare results
    let mc_fluence = mc_result.normalized_fluence();
    let (mean_error, max_error, correlation) = compare_fluence(&mc_fluence, &diff_fluence, dims);

    println!("  Results:");
    println!("    MC runtime:        {:?}", mc_time);
    println!("    Diffusion runtime: {:?}", diff_time);
    println!("    Mean rel. error:   {:.2}%", mean_error * 100.0);
    println!("    Max rel. error:    {:.2}%", max_error * 100.0);
    println!("    Correlation:       {:.4}", correlation);

    // Validation: Low scattering should show larger error
    if mean_error > 0.20 {
        println!("  ✓ PASS: Expected deviation in low scattering regime");
        println!("         (Diffusion approximation breaks down)");
    } else {
        println!("  ⚠ WARNING: Unexpectedly good agreement (MC may need more photons)");
    }

    Ok(())
}

fn validate_layered_tissue() -> Result<()> {
    let dims = GridDimensions::new(25, 25, 40, 0.001, 0.001, 0.001);
    let grid = Grid3D::new(25, 25, 40, 0.001, 0.001, 0.001)?;

    // Create layered phantom (skin/fat/muscle)
    println!("  Building layered tissue phantom...");
    let phantom = PhantomBuilder::layered_tissue()
        .dimensions(dims)
        .wavelength(800.0)
        .add_skin_layer(0.0, 0.002)
        .add_fat_layer(0.002, 0.010)
        .add_muscle_layer(0.010, 0.040)
        .build();

    let cx = dims.dx * (dims.nx as f64) / 2.0;
    let cy = dims.dy * (dims.ny as f64) / 2.0;

    // Monte Carlo
    println!("  Running Monte Carlo (500k photons)...");
    let mc_solver = MonteCarloSolver::new(grid.clone(), phantom.clone());
    let mc_source = PhotonSource::pencil_beam([cx, cy, 0.0], [0.0, 0.0, 1.0]);
    let mc_config = SimulationConfig::default().num_photons(500_000);

    let mc_start = Instant::now();
    let mc_result = mc_solver.simulate(&mc_source, &mc_config)?;
    let mc_time = mc_start.elapsed();

    // Diffusion
    println!("  Running Diffusion solver...");
    let diff_start = Instant::now();
    let diff_fluence = solve_diffusion_fluence(&grid, &phantom, [cx, cy, 0.0])?;
    let diff_time = diff_start.elapsed();

    let mc_fluence = mc_result.normalized_fluence();
    let (mean_error, max_error, correlation) = compare_fluence(&mc_fluence, &diff_fluence, dims);

    println!("  Results:");
    println!("    MC runtime:        {:?}", mc_time);
    println!("    Diffusion runtime: {:?}", diff_time);
    println!("    Mean rel. error:   {:.2}%", mean_error * 100.0);
    println!("    Max rel. error:    {:.2}%", max_error * 100.0);
    println!("    Correlation:       {:.4}", correlation);

    // Analyze depth profile
    analyze_depth_profile(&mc_fluence, &diff_fluence, dims, "Layered Tissue");

    if correlation > 0.80 {
        println!("  ✓ PASS: Both methods capture layered structure");
    } else {
        println!("  ⚠ WARNING: Poor correlation in layered tissue");
    }

    Ok(())
}

fn validate_blood_vessel() -> Result<()> {
    let dims = GridDimensions::new(30, 30, 30, 0.001, 0.001, 0.001);
    let grid = Grid3D::new(30, 30, 30, 0.001, 0.001, 0.001)?;

    // Create vascular phantom
    println!("  Building blood vessel phantom...");
    let cx = dims.dx * (dims.nx as f64) / 2.0;
    let cy = dims.dy * (dims.ny as f64) / 2.0;

    let phantom = PhantomBuilder::blood_oxygenation()
        .dimensions(dims)
        .wavelength(800.0)
        .add_artery([cx, cy, 0.010], 0.002, 0.98)
        .add_vein([cx, cy, 0.020], 0.003, 0.65)
        .build();

    // Monte Carlo
    println!("  Running Monte Carlo (500k photons)...");
    let mc_solver = MonteCarloSolver::new(grid.clone(), phantom.clone());
    let mc_source = PhotonSource::pencil_beam([cx, cy, 0.0], [0.0, 0.0, 1.0]);
    let mc_config = SimulationConfig::default().num_photons(500_000);

    let mc_start = Instant::now();
    let mc_result = mc_solver.simulate(&mc_source, &mc_config)?;
    let mc_time = mc_start.elapsed();

    // Diffusion
    println!("  Running Diffusion solver...");
    let diff_start = Instant::now();
    let diff_fluence = solve_diffusion_fluence(&grid, &phantom, [cx, cy, 0.0])?;
    let diff_time = diff_start.elapsed();

    let mc_fluence = mc_result.normalized_fluence();
    let (mean_error, max_error, correlation) = compare_fluence(&mc_fluence, &diff_fluence, dims);

    println!("  Results:");
    println!("    MC runtime:        {:?}", mc_time);
    println!("    Diffusion runtime: {:?}", diff_time);
    println!("    Mean rel. error:   {:.2}%", mean_error * 100.0);
    println!("    Max rel. error:    {:.2}%", max_error * 100.0);
    println!("    Correlation:       {:.4}", correlation);

    println!("  Interpretation:");
    println!("    MC captures sharp vessel boundaries and absorption hotspots");
    println!("    Diffusion smooths small-scale heterogeneity (typical for diffusive process)");

    if correlation > 0.75 {
        println!("  ✓ PASS: Reasonable agreement for vascular phantom");
    } else {
        println!("  ⚠ INFO: Lower correlation expected for heterogeneous structures");
    }

    Ok(())
}

/// Compare two fluence distributions
fn compare_fluence(
    mc_fluence: &[f64],
    diff_fluence: &[f64],
    dims: GridDimensions,
) -> (f64, f64, f64) {
    assert_eq!(mc_fluence.len(), diff_fluence.len());

    let n = mc_fluence.len();
    let mut sum_rel_error = 0.0;
    let mut max_rel_error: f64 = 0.0;
    let mut count = 0;

    // Relative error calculation
    for i in 0..n {
        let mc = mc_fluence[i];
        let diff = diff_fluence[i];

        // Only compare where fluence is significant
        let threshold = mc_fluence.iter().cloned().fold(0.0, f64::max) * 1e-3;
        if mc > threshold && diff > threshold {
            let rel_error = ((mc - diff) / mc.max(diff)).abs();
            sum_rel_error += rel_error;
            max_rel_error = max_rel_error.max(rel_error);
            count += 1;
        }
    }

    let mean_rel_error = if count > 0 {
        sum_rel_error / count as f64
    } else {
        0.0
    };

    // Compute correlation coefficient
    let correlation = compute_correlation(mc_fluence, diff_fluence);

    (mean_rel_error, max_rel_error, correlation)
}

/// Compute Pearson correlation coefficient
fn compute_correlation(x: &[f64], y: &[f64]) -> f64 {
    assert_eq!(x.len(), y.len());
    let n = x.len() as f64;

    let mean_x = x.iter().sum::<f64>() / n;
    let mean_y = y.iter().sum::<f64>() / n;

    let mut cov = 0.0;
    let mut var_x = 0.0;
    let mut var_y = 0.0;

    for i in 0..x.len() {
        let dx = x[i] - mean_x;
        let dy = y[i] - mean_y;
        cov += dx * dy;
        var_x += dx * dx;
        var_y += dy * dy;
    }

    if var_x < 1e-12 || var_y < 1e-12 {
        return 0.0;
    }

    cov / (var_x * var_y).sqrt()
}

/// Analyze depth profile (central axis)
fn analyze_depth_profile(
    mc_fluence: &[f64],
    diff_fluence: &[f64],
    dims: GridDimensions,
    label: &str,
) {
    println!("  Depth Profile Analysis ({}):", label);

    let cx = dims.nx / 2;
    let cy = dims.ny / 2;

    println!("    z (mm) | MC Fluence | Diff Fluence | Rel. Error");
    println!("    -------|------------|--------------|------------");

    for k in (0..dims.nz).step_by(5) {
        let idx = k * (dims.nx * dims.ny) + cy * dims.nx + cx;
        let mc = mc_fluence[idx];
        let diff = diff_fluence[idx];
        let rel_err = if mc > 1e-12 {
            ((mc - diff) / mc).abs() * 100.0
        } else {
            0.0
        };

        let z_mm = (k as f64 + 0.5) * dims.dz * 1000.0;
        println!(
            "    {:.1}    | {:.3e}   | {:.3e}   | {:.1}%",
            z_mm, mc, diff, rel_err
        );
    }
}

fn solve_diffusion_fluence(
    grid: &Grid3D,
    optical_map: &OpticalPropertyMap,
    source_position: [f64; 3],
) -> Result<Vec<f64>> {
    let config = DiffusionSolverConfig::default();
    let optical_properties = optical_property_map_to_array3(optical_map);
    let solver = DiffusionSolver::new(grid.clone(), optical_properties, config)?;

    let (nx, ny, nz) = grid.dimensions();
    let mut source = Array3::<f64>::zeros((nx, ny, nz));
    if let Some((i, j, k)) = grid.coordinates_to_indices(
        source_position[0].max(0.0),
        source_position[1].max(0.0),
        source_position[2].max(0.0),
    ) {
        source[[i, j, k]] = 1e6;
    }

    let fluence = solver.solve(&source)?;
    Ok(flatten_kji(&fluence))
}

fn optical_property_map_to_array3(map: &OpticalPropertyMap) -> Array3<OpticalPropertyData> {
    let nx = map.dimensions.nx;
    let ny = map.dimensions.ny;
    let nz = map.dimensions.nz;

    let background = map
        .data
        .first()
        .cloned()
        .unwrap_or_else(OpticalPropertyData::soft_tissue);
    let mut out = Array3::from_elem((nx, ny, nz), background);

    for k in 0..nz {
        for j in 0..ny {
            for i in 0..nx {
                if let Some(props) = map.get(i, j, k) {
                    out[[i, j, k]] = *props;
                }
            }
        }
    }

    out
}

fn flatten_kji(field: &Array3<f64>) -> Vec<f64> {
    let (nx, ny, nz) = field.dim();
    let mut out = Vec::with_capacity(nx * ny * nz);
    for k in 0..nz {
        for j in 0..ny {
            for i in 0..nx {
                out.push(field[[i, j, k]]);
            }
        }
    }
    out
}
