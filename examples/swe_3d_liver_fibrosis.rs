//! 3D Shear Wave Elastography Liver Fibrosis Assessment Example
//!
//! This example demonstrates a complete clinical workflow for 3D SWE assessment
//! of liver fibrosis using multi-directional shear waves and volumetric analysis.
//!
//! ## Clinical Scenario
//!
//! A 55-year-old patient with suspected liver fibrosis undergoes 3D SWE examination.
//! The system generates shear waves from multiple directions, tracks their propagation
//! through the liver volume, and reconstructs 3D elasticity maps for fibrosis staging.
//!
//! ## Workflow Steps
//!
//! 1. Patient positioning and ROI selection
//! 2. Multi-directional ARFI push pulse generation
//! 3. Volumetric shear wave propagation simulation
//! 4. 3D wave front tracking and arrival time estimation
//! 5. Multi-directional elasticity reconstruction
//! 6. Volumetric statistical analysis
//! 7. Clinical decision support and fibrosis staging
//! 8. Multi-planar visualization and reporting
//!
//! ## Expected Results
//!
//! - Mean liver stiffness: ~8-12 kPa (healthy to moderate fibrosis)
//! - Volume coverage: >90% of ROI
//! - Quality score: >0.8
//! - Fibrosis stage: F2-F3 (moderate fibrosis)
//!
//! ## References
//!
//! - Ferraioli, G., et al. (2018). "Guidelines and good clinical practice recommendations
//!   for contrast enhanced ultrasound (CEUS) in the liver." *Ultrasound in Medicine & Biology*
//! - Barr, R. G., et al. (2019). "Elastography assessment of liver fibrosis." *Abdominal Radiology*

use kwavers::clinical::swe_3d_workflows::{
    VolumetricROI, ElasticityMap3D, ClinicalDecisionSupport, MultiPlanarReconstruction,
    VolumetricStatistics, LiverFibrosisStage, ClassificationConfidence,
};
use kwavers::physics::imaging::elastography::{
    ElasticWaveSolver, VolumetricWaveConfig, WaveFrontTracker,
    AcousticRadiationForce, MultiDirectionalPush,
};
use kwavers::grid::Grid;
use kwavers::medium::heterogeneous::HeterogeneousMedium;
use ndarray::Array3;
use std::collections::HashMap;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("🫀 3D SWE Liver Fibrosis Assessment Example");
    println!("==========================================\n");

    // Step 1: Patient setup and ROI definition
    println!("📋 Step 1: Patient Setup and ROI Definition");
    let grid = Grid::new(80, 80, 60, 0.001, 0.001, 0.0015)?; // 8x8x9cm volume
    println!("   Computational grid: {}x{}x{} ({}x{}x{}cm)",
             grid.nx, grid.ny, grid.nz,
             grid.nx as f64 * grid.dx * 100.0,
             grid.ny as f64 * grid.dy * 100.0,
             grid.nz as f64 * grid.dz * 100.0);

    // Define liver ROI for fibrosis assessment
    let liver_roi = VolumetricROI::liver_roi([0.04, 0.04, 0.045]); // Center of volume
    println!("   Liver ROI: {:.1}x{:.1}x{:.1}cm volume",
             liver_roi.size[0] * 100.0, liver_roi.size[1] * 100.0, liver_roi.size[2] * 100.0);
    println!("   Quality threshold: {:.1}, Depth range: {:.1}-{:.1}cm\n",
             liver_roi.quality_threshold,
             liver_roi.min_depth * 100.0, liver_roi.max_depth * 100.0);

    // Step 2: Create heterogeneous liver medium with fibrosis
    println!("🫘 Step 2: Liver Tissue Model Creation");
    let liver_medium = create_fibrotic_liver_medium(&grid);
    println!("   Created heterogeneous liver medium with fibrotic regions");
    println!("   Background stiffness: 5.5 kPa (healthy liver)");
    println!("   Fibrotic regions: 12-18 kPa (F3-F4 fibrosis)\n");

    // Step 3: Multi-directional shear wave generation
    println!("🌊 Step 3: Multi-Directional Shear Wave Generation");
    let arf = AcousticRadiationForce::new(&grid, &liver_medium)?;

    // Create orthogonal push pattern for comprehensive 3D coverage
    let push_pattern = MultiDirectionalPush::orthogonal_pattern(
        [0.04, 0.04, 0.03], // Push location (3cm depth)
        0.015 // 1.5cm spacing between pushes
    );
    println!("   Generated {} orthogonal push pulses", push_pattern.pushes.len());
    println!("   Push sequence duration: {:.1} μs", push_pattern.sequence_duration * 1e6);
    println!("   Time delays: {:.1} to {:.1} μs\n",
             push_pattern.time_delays.iter().cloned().fold(f64::INFINITY, f64::min) * 1e6,
             push_pattern.time_delays.iter().cloned().fold(0.0, f64::max) * 1e6);

    // Step 4: Volumetric wave propagation simulation
    println!("🔬 Step 4: Volumetric Wave Propagation Simulation");
    let mut solver = ElasticWaveSolver::new(&grid, &liver_medium,
                                          Default::default())?;

    // Configure volumetric features
    let volumetric_config = VolumetricWaveConfig {
        volumetric_boundaries: true,
        interference_tracking: true,
        volumetric_attenuation: true,
        dispersion_correction: true,
        front_tracking_resolution: 0.0005, // 0.5mm resolution
        ..Default::default()
    };
    solver.set_volumetric_config(volumetric_config);

    // Generate initial displacement field from multi-directional pushes
    let initial_displacement = arf.apply_multi_directional_push(&push_pattern)?;
    println!("   Initial displacement field generated");
    println!("   Maximum displacement: {:.2} μm",
             initial_displacement.iter().cloned().fold(0.0_f64, |a, b| a.max(b.abs())) * 1e6);

    // Propagate waves with volumetric tracking
    let push_times: Vec<f64> = push_pattern.time_delays.clone();
    let (displacement_history, tracker) = solver.propagate_volumetric_waves(
        &[initial_displacement],
        &push_times,
    )?;
    println!("   Simulated {} time steps of wave propagation", displacement_history.len());
    println!("   Wave front tracking completed\n");

    // Step 5: 3D elasticity reconstruction
    println!("🧮 Step 5: 3D Elasticity Reconstruction");
    let mut elasticity_map = ElasticityMap3D::new(&grid);

    // Perform multi-directional time-of-flight inversion
    perform_3d_elasticity_reconstruction(
        &tracker,
        &push_pattern,
        &mut elasticity_map,
        &grid,
    )?;
    println!("   3D elasticity map reconstructed");
    println!("   Mean stiffness: {:.1} kPa",
             calculate_mean_stiffness(&elasticity_map, &liver_roi) / 1000.0);
    println!("   Reliable voxels: {:.1}%\n",
             calculate_reliability_percentage(&elasticity_map, &liver_roi) * 100.0);

    // Step 6: Volumetric statistical analysis
    println!("📊 Step 6: Volumetric Statistical Analysis");
    let stats = elasticity_map.volumetric_statistics(&liver_roi);
    println!("   Analysis of {}x{}x{}cm ROI", 8, 8, 6);
    println!("   Valid voxels: {} ({:.1}%)",
             stats.valid_voxels,
             stats.volume_coverage * 100.0);
    println!("   Mean Young's modulus: {:.1} ± {:.1} kPa",
             stats.mean_modulus / 1000.0,
             stats.std_modulus / 1000.0);
    println!("   Range: {:.1} - {:.1} kPa",
             stats.min_modulus / 1000.0,
             stats.max_modulus / 1000.0);
    println!("   Mean shear speed: {:.1} m/s", stats.mean_speed);
    println!("   Quality metrics - Confidence: {:.2}, Quality: {:.2}\n",
             stats.mean_confidence, stats.mean_quality);

    // Step 7: Clinical decision support
    println!("🏥 Step 7: Clinical Decision Support");
    let cds = ClinicalDecisionSupport::default();
    let fibrosis_classification = cds.classify_liver_fibrosis(&stats);

    println!("   Liver Fibrosis Assessment (METAVIR Classification):");
    println!("   Stage: {:?}", fibrosis_classification.stage);
    println!("   Mean stiffness: {:.1} kPa", fibrosis_classification.mean_stiffness_kpa);
    println!("   Confidence: {:?}", fibrosis_classification.confidence);
    println!("   Quality score: {:.2}\n", fibrosis_classification.quality_score);

    // Step 8: Multi-planar visualization
    println!("📈 Step 8: Multi-Planar Visualization");
    let mpr = MultiPlanarReconstruction::from_elasticity_map(&elasticity_map, 0.005); // 5mm slices
    println!("   Generated {} axial slices", mpr.axial_slices.len());
    println!("   Generated {} sagittal slices", mpr.sagittal_slices.len());
    println!("   Generated {} coronal slices", mpr.coronal_slices.len());

    // Display slice statistics
    display_slice_statistics(&mpr);
    println!();

    // Step 9: Clinical report generation
    println!("📄 Step 9: Clinical Report Generation");
    let report = cds.generate_report("liver", &stats);
    println!("   Generated comprehensive clinical report");
    println!("   Report length: {} characters\n", report.len());

    // Step 10: Validation and quality assessment
    println!("✅ Step 10: Validation and Quality Assessment");
    let quality_metrics = solver.calculate_volumetric_quality(&tracker);
    println!("   Volumetric quality assessment:");
    println!("   Coverage: {:.1}%", quality_metrics.coverage * 100.0);
    println!("   Average quality: {:.2}", quality_metrics.average_quality);
    println!("   Maximum interference: {:.0}", quality_metrics.max_interference);
    println!("   Valid tracking points: {}\n", quality_metrics.valid_tracking_points);

    // Final summary
    println!("🎯 Examination Summary");
    println!("====================");
    println!("Patient: Adult male, suspected liver fibrosis");
    println!("Modality: 3D Shear Wave Elastography");
    println!("ROI: Liver parenchyma (8x8x6cm)");
    println!("Fibrosis Stage: {:?}", fibrosis_classification.stage);
    println!("Mean Stiffness: {:.1} kPa", fibrosis_classification.mean_stiffness_kpa);
    println!("Quality: {:?}", fibrosis_classification.confidence);
    println!("Recommendation: Further evaluation with biopsy correlation");

    Ok(())
}

/// Create heterogeneous liver medium with fibrotic regions
fn create_fibrotic_liver_medium(grid: &Grid) -> HeterogeneousMedium {
    let mut stiffness_map = Array3::<f64>::from_elem((grid.nx, grid.ny, grid.nz), 5500.0); // 5.5 kPa background

    // Add fibrotic regions (higher stiffness)
    let center_x = grid.nx / 2;
    let center_y = grid.ny / 2;
    let center_z = grid.nz / 2;

    // Create multiple fibrotic nodules
    let fibrotic_regions = vec![
        (center_x - 10, center_y - 5, center_z, 8, 12000.0),  // F3 fibrosis
        (center_x + 5, center_y + 8, center_z - 5, 6, 18000.0), // F4 fibrosis
        (center_x - 5, center_y - 10, center_z + 8, 5, 15000.0), // F3-F4 fibrosis
    ];

    for (cx, cy, cz, radius, stiffness) in fibrotic_regions {
        for k in (cz - radius)..=(cz + radius) {
            for j in (cy - radius)..=(cy + radius) {
                for i in (cx - radius)..=(cx + radius) {
                    if i >= 0 && i < grid.nx && j >= 0 && j < grid.ny && k >= 0 && k < grid.nz {
                        let dx = (i - cx) as f64;
                        let dy = (j - cy) as f64;
                        let dz = (k - cz) as f64;
                        let distance = (dx * dx + dy * dy + dz * dz).sqrt();

                        if distance <= radius as f64 {
                            // Smooth transition at boundaries
                            let weight = 1.0 - (distance / radius as f64).powi(2);
                            stiffness_map[[i, j, k]] = 5500.0 + weight * (stiffness - 5500.0);
                        }
                    }
                }
            }
        }
    }

    // Convert stiffness to density and speed of sound
    let mut density_map = Array3::<f64>::from_elem((grid.nx, grid.ny, grid.nz), 1050.0); // kg/m³
    let mut sound_speed_map = Array3::<f64>::from_elem((grid.nx, grid.ny, grid.nz), 1550.0); // m/s

    // Stiffer tissue has slightly higher density
    for k in 0..grid.nz {
        for j in 0..grid.ny {
            for i in 0..grid.nx {
                let stiffness = stiffness_map[[i, j, k]];
                density_map[[i, j, k]] = 1050.0 + (stiffness - 5500.0) * 0.01; // Small density increase
                sound_speed_map[[i, j, k]] = 1450.0 + (stiffness / 1000.0).sqrt() * 10.0; // Speed increases with sqrt(stiffness)
            }
        }
    }

    HeterogeneousMedium::new(
        stiffness_map,
        density_map,
        sound_speed_map,
        grid,
    ).expect("Failed to create heterogeneous liver medium")
}

/// Perform 3D elasticity reconstruction from wave front tracking data
fn perform_3d_elasticity_reconstruction(
    tracker: &WaveFrontTracker,
    push_pattern: &MultiDirectionalPush,
    elasticity_map: &mut ElasticityMap3D,
    grid: &Grid,
) -> Result<(), Box<dyn std::error::Error>> {
    // Simplified time-of-flight inversion for demonstration
    // In practice, this would use more sophisticated algorithms

    for k in 1..grid.nz - 1 {
        for j in 1..grid.ny - 1 {
            for i in 1..grid.nx - 1 {
                let arrival_time = tracker.arrival_times[[i, j, k]];

                if !arrival_time.is_infinite() && tracker.amplitudes[[i, j, k]] > 0.0 {
                    // Estimate distance from nearest push location
                    let mut min_distance = f64::INFINITY;
                    for push in &push_pattern.pushes {
                        let dx = i as f64 * grid.dx - push.location[0];
                        let dy = j as f64 * grid.dy - push.location[1];
                        let dz = k as f64 * grid.dz - push.location[2];
                        let distance = (dx * dx + dy * dy + dz * dz).sqrt();
                        min_distance = min_distance.min(distance);
                    }

                    if min_distance > 0.0 {
                        // Estimate shear wave speed: distance = speed × time
                        let estimated_speed = min_distance / arrival_time;

                        // Convert speed to Young's modulus: E = 3ρcₛ² (for Poisson's ratio ν=0.5)
                        let density = 1050.0; // kg/m³ (approximate liver density)
                        let young_modulus = 3.0 * density * estimated_speed * estimated_speed;

                        // Quality assessment
                        let quality = tracker.tracking_quality[[i, j, k]];
                        let confidence = if quality > 0.8 { 0.9 } else if quality > 0.6 { 0.7 } else { 0.5 };

                        // Store results
                        elasticity_map.young_modulus[[i, j, k]] = young_modulus;
                        elasticity_map.shear_speed[[i, j, k]] = estimated_speed;
                        elasticity_map.confidence[[i, j, k]] = confidence;
                        elasticity_map.quality[[i, j, k]] = quality;
                        elasticity_map.reliability_mask[[i, j, k]] = true;
                    }
                }
            }
        }
    }

    Ok(())
}

/// Calculate mean stiffness in ROI
fn calculate_mean_stiffness(elasticity_map: &ElasticityMap3D, roi: &VolumetricROI) -> f64 {
    let ([min_x, min_y, min_z], [max_x, max_y, max_z]) = roi.grid_bounds(&elasticity_map.grid);

    let mut sum_stiffness = 0.0;
    let mut count = 0;

    for k in min_z..=max_z {
        for j in min_y..=max_y {
            for i in min_x..=max_x {
                if elasticity_map.reliability_mask[[i, j, k]] &&
                   elasticity_map.confidence[[i, j, k]] >= roi.quality_threshold {
                    sum_stiffness += elasticity_map.young_modulus[[i, j, k]];
                    count += 1;
                }
            }
        }
    }

    if count > 0 { sum_stiffness / count as f64 } else { 0.0 }
}

/// Calculate percentage of reliable voxels in ROI
fn calculate_reliability_percentage(elasticity_map: &ElasticityMap3D, roi: &VolumetricROI) -> f64 {
    let ([min_x, min_y, min_z], [max_x, max_y, max_z]) = roi.grid_bounds(&elasticity_map.grid);

    let mut reliable_count = 0;
    let total_voxels = (max_x - min_x + 1) * (max_y - min_y + 1) * (max_z - min_z + 1);

    for k in min_z..=max_z {
        for j in min_y..=max_y {
            for i in min_x..=max_x {
                if elasticity_map.reliability_mask[[i, j, k]] &&
                   elasticity_map.confidence[[i, j, k]] >= roi.quality_threshold {
                    reliable_count += 1;
                }
            }
        }
    }

    reliable_count as f64 / total_voxels as f64
}

/// Display statistics for multi-planar slices
fn display_slice_statistics(mpr: &MultiPlanarReconstruction) {
    if !mpr.axial_slices.is_empty() {
        let axial_stats = calculate_slice_stats(&mpr.axial_slices);
        println!("   Axial slices: {} total, mean stiffness {:.1} kPa",
                mpr.axial_slices.len(), axial_stats.mean_stiffness / 1000.0);
    }

    if !mpr.sagittal_slices.is_empty() {
        let sagittal_stats = calculate_slice_stats(&mpr.sagittal_slices);
        println!("   Sagittal slices: {} total, mean stiffness {:.1} kPa",
                mpr.sagittal_slices.len(), sagittal_stats.mean_stiffness / 1000.0);
    }

    if !mpr.coronal_slices.is_empty() {
        let coronal_stats = calculate_slice_stats(&mpr.coronal_slices);
        println!("   Coronal slices: {} total, mean stiffness {:.1} kPa",
                mpr.coronal_slices.len(), coronal_stats.mean_stiffness / 1000.0);
    }
}

/// Calculate statistics for a set of slices
fn calculate_slice_stats(slices: &[kwavers::clinical::swe_3d_workflows::ElasticityMap2D]) -> SliceStats {
    let mut total_stiffness = 0.0;
    let mut total_voxels = 0;

    for slice in slices {
        for &stiffness in slice.young_modulus.iter() {
            if stiffness > 0.0 {
                total_stiffness += stiffness;
                total_voxels += 1;
            }
        }
    }

    let mean_stiffness = if total_voxels > 0 {
        total_stiffness / total_voxels as f64
    } else {
        0.0
    };

    SliceStats { mean_stiffness }
}

struct SliceStats {
    mean_stiffness: f64,
}
