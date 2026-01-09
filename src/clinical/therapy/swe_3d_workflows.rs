//! 3D Shear Wave Elastography Clinical Workflows
//!
//! Implements clinical workflows for 3D SWE including volumetric ROI analysis,
//! multi-planar reconstruction, and clinical decision support.
//!
//! ## Clinical Applications
//!
//! - Liver fibrosis staging (METAVIR F0-F4)
//! - Breast lesion characterization (BI-RADS)
//! - Prostate cancer detection and staging
//! - Thyroid nodule assessment
//! - Musculoskeletal tissue evaluation
//!
//! ## References
//!
//! - Nightingale, K. R., et al. (2015). "Shear wave elastography." *Physics in Medicine
//!   and Biology*, 60(2), R1-R41.
//! - Ferraioli, G., et al. (2018). "Guidelines and good clinical practice recommendations
//!   for contrast enhanced ultrasound (CEUS) in the liver." *Ultrasound in Medicine & Biology*.
//! - Barr, R. G., et al. (2019). "Elastography assessment of liver fibrosis." *Abdominal Radiology*.

use crate::domain::grid::Grid;
use ndarray::{Array3, Axis};
use std::collections::HashMap;

/// 3D Region of Interest for volumetric SWE analysis
#[derive(Debug, Clone)]
pub struct VolumetricROI {
    /// ROI center coordinates [x, y, z] in meters
    pub center: [f64; 3],
    /// ROI dimensions [width, height, depth] in meters
    pub size: [f64; 3],
    /// ROI orientation angles [yaw, pitch, roll] in radians
    pub orientation: [f64; 3],
    /// Quality threshold for inclusion (0-1)
    pub quality_threshold: f64,
    /// Minimum depth for analysis (m)
    pub min_depth: f64,
    /// Maximum depth for analysis (m)
    pub max_depth: f64,
}

impl Default for VolumetricROI {
    fn default() -> Self {
        Self {
            center: [0.0, 0.0, 0.04],     // 4cm depth (typical liver)
            size: [0.06, 0.06, 0.04],     // 6x6x4cm volume
            orientation: [0.0, 0.0, 0.0], // Axial orientation
            quality_threshold: 0.7,
            min_depth: 0.02, // 2cm minimum
            max_depth: 0.08, // 8cm maximum
        }
    }
}

impl VolumetricROI {
    /// Create ROI for liver fibrosis assessment
    pub fn liver_roi(center: [f64; 3]) -> Self {
        Self {
            center,
            size: [0.08, 0.08, 0.06], // 8x8x6cm for liver
            quality_threshold: 0.8,
            min_depth: 0.02,
            max_depth: 0.10,
            ..Default::default()
        }
    }

    /// Create ROI for breast lesion assessment
    pub fn breast_roi(center: [f64; 3]) -> Self {
        Self {
            center,
            size: [0.04, 0.04, 0.03], // 4x4x3cm for breast
            quality_threshold: 0.85,
            min_depth: 0.01,
            max_depth: 0.05,
            ..Default::default()
        }
    }

    /// Create ROI for prostate assessment
    pub fn prostate_roi(center: [f64; 3]) -> Self {
        Self {
            center,
            size: [0.05, 0.05, 0.04], // 5x5x4cm for prostate
            quality_threshold: 0.75,
            min_depth: 0.03,
            max_depth: 0.08,
            ..Default::default()
        }
    }

    /// Check if a point is within the ROI
    pub fn contains_point(&self, point: [f64; 3]) -> bool {
        // Simple axis-aligned bounding box check
        // In practice, this should account for orientation
        let dx = (point[0] - self.center[0]).abs();
        let dy = (point[1] - self.center[1]).abs();
        let dz = (point[2] - self.center[2]).abs();

        dx <= self.size[0] / 2.0
            && dy <= self.size[1] / 2.0
            && dz <= self.size[2] / 2.0
            && point[2] >= self.min_depth
            && point[2] <= self.max_depth
    }

    /// Get ROI bounds as grid indices
    pub fn grid_bounds(&self, grid: &Grid) -> ([usize; 3], [usize; 3]) {
        let min_x = ((self.center[0] - self.size[0] / 2.0) / grid.dx).max(0.0) as usize;
        let max_x =
            ((self.center[0] + self.size[0] / 2.0) / grid.dx).min(grid.nx as f64 - 1.0) as usize;
        let min_y = ((self.center[1] - self.size[1] / 2.0) / grid.dy).max(0.0) as usize;
        let max_y =
            ((self.center[1] + self.size[1] / 2.0) / grid.dy).min(grid.ny as f64 - 1.0) as usize;
        let min_z = ((self.center[2] - self.size[2] / 2.0) / grid.dz).max(0.0) as usize;
        let max_z =
            ((self.center[2] + self.size[2] / 2.0) / grid.dz).min(grid.nz as f64 - 1.0) as usize;

        ([min_x, min_y, min_z], [max_x, max_y, max_z])
    }
}

/// 3D elasticity map with confidence and quality metrics
#[derive(Debug, Clone)]
pub struct ElasticityMap3D {
    /// Young's modulus values (Pa) - shape: [nx, ny, nz]
    pub young_modulus: Array3<f64>,
    /// Shear wave speed (m/s) - shape: [nx, ny, nz]
    pub shear_speed: Array3<f64>,
    /// Confidence map (0-1) - shape: [nx, ny, nz]
    pub confidence: Array3<f64>,
    /// Quality map (0-1) - shape: [nx, ny, nz]
    pub quality: Array3<f64>,
    /// Reliability mask (true = reliable) - shape: [nx, ny, nz]
    pub reliability_mask: Array3<bool>,
    /// Computation grid
    pub grid: Grid,
}

impl ElasticityMap3D {
    /// Create new 3D elasticity map
    pub fn new(grid: &Grid) -> Self {
        let (nx, ny, nz) = grid.dimensions();
        Self {
            young_modulus: Array3::zeros((nx, ny, nz)),
            shear_speed: Array3::zeros((nx, ny, nz)),
            confidence: Array3::zeros((nx, ny, nz)),
            quality: Array3::zeros((nx, ny, nz)),
            reliability_mask: Array3::from_elem((nx, ny, nz), false),
            grid: grid.clone(),
        }
    }

    /// Extract 2D slice at given z-index
    pub fn axial_slice(&self, z_index: usize) -> ElasticityMap2D {
        ElasticityMap2D {
            young_modulus: self.young_modulus.index_axis(Axis(2), z_index).to_owned(),
            shear_speed: self.shear_speed.index_axis(Axis(2), z_index).to_owned(),
            confidence: self.confidence.index_axis(Axis(2), z_index).to_owned(),
            quality: self.quality.index_axis(Axis(2), z_index).to_owned(),
            reliability_mask: self
                .reliability_mask
                .index_axis(Axis(2), z_index)
                .to_owned(),
        }
    }

    /// Extract sagittal slice (YZ plane) at given x-index
    pub fn sagittal_slice(&self, x_index: usize) -> ElasticityMap2D {
        ElasticityMap2D {
            young_modulus: self.young_modulus.index_axis(Axis(0), x_index).to_owned(),
            shear_speed: self.shear_speed.index_axis(Axis(0), x_index).to_owned(),
            confidence: self.confidence.index_axis(Axis(0), x_index).to_owned(),
            quality: self.quality.index_axis(Axis(0), x_index).to_owned(),
            reliability_mask: self
                .reliability_mask
                .index_axis(Axis(0), x_index)
                .to_owned(),
        }
    }

    /// Extract coronal slice (XZ plane) at given y-index
    pub fn coronal_slice(&self, y_index: usize) -> ElasticityMap2D {
        ElasticityMap2D {
            young_modulus: self.young_modulus.index_axis(Axis(1), y_index).to_owned(),
            shear_speed: self.shear_speed.index_axis(Axis(1), y_index).to_owned(),
            confidence: self.confidence.index_axis(Axis(1), y_index).to_owned(),
            quality: self.quality.index_axis(Axis(1), y_index).to_owned(),
            reliability_mask: self
                .reliability_mask
                .index_axis(Axis(1), y_index)
                .to_owned(),
        }
    }

    /// Compute volumetric statistics within ROI
    pub fn volumetric_statistics(&self, roi: &VolumetricROI) -> VolumetricStatistics {
        let ([min_x, min_y, min_z], [max_x, max_y, max_z]) = roi.grid_bounds(&self.grid);

        let mut valid_points = 0;
        let mut sum_modulus = 0.0;
        let mut sum_speed = 0.0;
        let mut min_modulus = f64::INFINITY;
        let mut max_modulus = f64::NEG_INFINITY;
        let mut mean_confidence = 0.0;
        let mut mean_quality = 0.0;

        for k in min_z..=max_z {
            for j in min_y..=max_y {
                for i in min_x..=max_x {
                    if self.reliability_mask[[i, j, k]]
                        && self.confidence[[i, j, k]] >= roi.quality_threshold
                    {
                        let modulus = self.young_modulus[[i, j, k]];
                        let speed = self.shear_speed[[i, j, k]];

                        valid_points += 1;
                        sum_modulus += modulus;
                        sum_speed += speed;
                        min_modulus = min_modulus.min(modulus);
                        max_modulus = max_modulus.max(modulus);
                        mean_confidence += self.confidence[[i, j, k]];
                        mean_quality += self.quality[[i, j, k]];
                    }
                }
            }
        }

        if valid_points == 0 {
            return VolumetricStatistics {
                valid_voxels: 0,
                mean_modulus: 0.0,
                std_modulus: 0.0,
                median_modulus: 0.0,
                min_modulus: 0.0,
                max_modulus: 0.0,
                mean_speed: 0.0,
                mean_confidence: 0.0,
                mean_quality: 0.0,
                volume_coverage: 0.0,
            };
        }

        let mean_modulus = sum_modulus / valid_points as f64;
        let mean_speed = sum_speed / valid_points as f64;
        mean_confidence /= valid_points as f64;
        mean_quality /= valid_points as f64;

        // Compute standard deviation
        let mut sum_squared_diff = 0.0;
        for k in min_z..=max_z {
            for j in min_y..=max_y {
                for i in min_x..=max_x {
                    if self.reliability_mask[[i, j, k]]
                        && self.confidence[[i, j, k]] >= roi.quality_threshold
                    {
                        let diff = self.young_modulus[[i, j, k]] - mean_modulus;
                        sum_squared_diff += diff * diff;
                    }
                }
            }
        }
        let std_modulus = (sum_squared_diff / valid_points as f64).sqrt();

        // Simple median approximation (could be improved)
        let median_modulus = mean_modulus; // Placeholder

        let total_roi_voxels = (max_x - min_x + 1) * (max_y - min_y + 1) * (max_z - min_z + 1);
        let volume_coverage = valid_points as f64 / total_roi_voxels as f64;

        VolumetricStatistics {
            valid_voxels: valid_points,
            mean_modulus,
            std_modulus,
            median_modulus,
            min_modulus,
            max_modulus,
            mean_speed,
            mean_confidence,
            mean_quality,
            volume_coverage,
        }
    }
}

/// 2D elasticity map for slice visualization
#[derive(Debug, Clone)]
pub struct ElasticityMap2D {
    /// Young's modulus values (Pa)
    pub young_modulus: Array2<f64>,
    /// Shear wave speed (m/s)
    pub shear_speed: Array2<f64>,
    /// Confidence map (0-1)
    pub confidence: Array2<f64>,
    /// Quality map (0-1)
    pub quality: Array2<f64>,
    /// Reliability mask
    pub reliability_mask: Array2<bool>,
}

type Array2<T> = ndarray::Array2<T>;

/// Volumetric statistics for ROI analysis
#[derive(Debug, Clone)]
pub struct VolumetricStatistics {
    /// Number of valid voxels in analysis
    pub valid_voxels: usize,
    /// Mean Young's modulus (Pa)
    pub mean_modulus: f64,
    /// Standard deviation of Young's modulus (Pa)
    pub std_modulus: f64,
    /// Median Young's modulus (Pa)
    pub median_modulus: f64,
    /// Minimum Young's modulus (Pa)
    pub min_modulus: f64,
    /// Maximum Young's modulus (Pa)
    pub max_modulus: f64,
    /// Mean shear wave speed (m/s)
    pub mean_speed: f64,
    /// Mean confidence score (0-1)
    pub mean_confidence: f64,
    /// Mean quality score (0-1)
    pub mean_quality: f64,
    /// Volume coverage fraction (0-1)
    pub volume_coverage: f64,
}

/// Clinical decision support for 3D SWE
#[derive(Debug)]
pub struct ClinicalDecisionSupport {
    /// Reference ranges for different tissues
    _reference_ranges: HashMap<String, TissueReference>,
    /// Classification thresholds
    classification_thresholds: HashMap<String, Vec<f64>>,
}

impl Default for ClinicalDecisionSupport {
    fn default() -> Self {
        let mut reference_ranges = HashMap::new();
        let mut classification_thresholds = HashMap::new();

        // Liver reference ranges (kPa)
        reference_ranges.insert(
            "liver_normal".to_string(),
            TissueReference {
                mean_modulus: 5.5, // kPa
                std_modulus: 1.2,
                min_modulus: 3.0,
                max_modulus: 8.0,
            },
        );

        reference_ranges.insert(
            "liver_fibrosis_f4".to_string(),
            TissueReference {
                mean_modulus: 15.0, // kPa
                std_modulus: 3.0,
                min_modulus: 10.0,
                max_modulus: 25.0,
            },
        );

        // Liver fibrosis classification thresholds (kPa)
        classification_thresholds.insert(
            "liver_metavir".to_string(),
            vec![
                6.5,  // F0/F1 vs F2/F3/F4
                8.5,  // F0/F1/F2 vs F3/F4
                11.0, // F0/F1/F2/F3 vs F4
            ],
        );

        // Breast reference ranges (kPa)
        reference_ranges.insert(
            "breast_normal".to_string(),
            TissueReference {
                mean_modulus: 18.0,
                std_modulus: 5.0,
                min_modulus: 10.0,
                max_modulus: 30.0,
            },
        );

        reference_ranges.insert(
            "breast_malignant".to_string(),
            TissueReference {
                mean_modulus: 120.0,
                std_modulus: 40.0,
                min_modulus: 50.0,
                max_modulus: 200.0,
            },
        );

        Self {
            _reference_ranges: reference_ranges,
            classification_thresholds,
        }
    }
}

impl ClinicalDecisionSupport {
    /// Classify liver fibrosis stage using METAVIR scoring
    pub fn classify_liver_fibrosis(&self, stats: &VolumetricStatistics) -> LiverFibrosisStage {
        let thresholds = self.classification_thresholds.get("liver_metavir").unwrap();
        let mean_kpa = stats.mean_modulus / 1000.0; // Convert to kPa

        let stage = if mean_kpa < thresholds[0] {
            FibrosisStage::F0F1
        } else if mean_kpa < thresholds[1] {
            FibrosisStage::F2
        } else if mean_kpa < thresholds[2] {
            FibrosisStage::F3
        } else {
            FibrosisStage::F4
        };

        // Confidence based on standard deviation and quality
        let confidence = if stats.std_modulus / stats.mean_modulus < 0.3 && stats.mean_quality > 0.7
        {
            ClassificationConfidence::High
        } else if stats.std_modulus / stats.mean_modulus < 0.5 && stats.mean_quality > 0.5 {
            ClassificationConfidence::Medium
        } else {
            ClassificationConfidence::Low
        };

        LiverFibrosisStage {
            stage,
            mean_stiffness_kpa: mean_kpa,
            confidence,
            quality_score: stats.mean_quality,
        }
    }

    /// Classify breast lesion using BI-RADS criteria
    pub fn classify_breast_lesion(
        &self,
        stats: &VolumetricStatistics,
    ) -> BreastLesionClassification {
        let mean_kpa = stats.mean_modulus / 1000.0; // Convert to kPa

        // Simplified BI-RADS classification based on stiffness
        let (birads_category, malignancy_probability) = if mean_kpa < 20.0 {
            (2, 0.0) // Benign
        } else if mean_kpa < 50.0 {
            (3, 0.02) // Probably benign
        } else if mean_kpa < 80.0 {
            (4, 0.3) // Suspicious
        } else {
            (5, 0.95) // Highly suggestive of malignancy
        };

        let confidence = if stats.std_modulus / stats.mean_modulus < 0.4 && stats.mean_quality > 0.8
        {
            ClassificationConfidence::High
        } else if stats.std_modulus / stats.mean_modulus < 0.6 && stats.mean_quality > 0.6 {
            ClassificationConfidence::Medium
        } else {
            ClassificationConfidence::Low
        };

        BreastLesionClassification {
            birads_category,
            estimated_malignancy_probability: malignancy_probability,
            mean_stiffness_kpa: mean_kpa,
            confidence,
            quality_score: stats.mean_quality,
        }
    }

    /// Generate clinical report
    pub fn generate_report(&self, organ: &str, stats: &VolumetricStatistics) -> String {
        let mut report = format!("3D SWE Clinical Report - {}\n", organ.to_uppercase());
        report.push_str(&format!("={}\n\n", "=".repeat(40)));

        report.push_str("Volumetric Analysis:\n");
        report.push_str(&format!("- Valid voxels: {}\n", stats.valid_voxels));
        report.push_str(&format!(
            "- Volume coverage: {:.1}%\n",
            stats.volume_coverage * 100.0
        ));
        report.push_str(&format!("- Mean quality: {:.2}\n", stats.mean_quality));
        report.push_str(&format!(
            "- Mean confidence: {:.2}\n\n",
            stats.mean_confidence
        ));

        report.push_str("Elasticity Results:\n");
        report.push_str(&format!(
            "- Mean Young's modulus: {:.1} kPa\n",
            stats.mean_modulus / 1000.0
        ));
        report.push_str(&format!(
            "- Standard deviation: {:.1} kPa\n",
            stats.std_modulus / 1000.0
        ));
        report.push_str(&format!(
            "- Range: {:.1} - {:.1} kPa\n",
            stats.min_modulus / 1000.0,
            stats.max_modulus / 1000.0
        ));
        report.push_str(&format!(
            "- Mean shear speed: {:.1} m/s\n\n",
            stats.mean_speed
        ));

        // Organ-specific classification
        match organ.to_lowercase().as_str() {
            "liver" => {
                let classification = self.classify_liver_fibrosis(stats);
                report.push_str("Liver Fibrosis Assessment (METAVIR):\n");
                report.push_str(&format!("- Stage: {:?}\n", classification.stage));
                report.push_str(&format!("- Confidence: {:?}\n", classification.confidence));
                report.push_str(&format!(
                    "- Quality score: {:.2}\n",
                    classification.quality_score
                ));
            }
            "breast" => {
                let classification = self.classify_breast_lesion(stats);
                report.push_str("Breast Lesion Assessment (BI-RADS):\n");
                report.push_str(&format!("- Category: {}\n", classification.birads_category));
                report.push_str(&format!(
                    "- Estimated malignancy: {:.1}%\n",
                    classification.estimated_malignancy_probability * 100.0
                ));
                report.push_str(&format!("- Confidence: {:?}\n", classification.confidence));
            }
            _ => {
                report.push_str("General tissue assessment completed.\n");
            }
        }

        report.push_str("\nNote: This is an automated analysis. Clinical correlation required.\n");
        report
    }
}

/// Tissue reference ranges for clinical comparison
#[derive(Debug, Clone)]
pub struct TissueReference {
    /// Mean Young's modulus (kPa)
    pub mean_modulus: f64,
    /// Standard deviation (kPa)
    pub std_modulus: f64,
    /// Minimum expected modulus (kPa)
    pub min_modulus: f64,
    /// Maximum expected modulus (kPa)
    pub max_modulus: f64,
}

/// Liver fibrosis classification result
#[derive(Debug, Clone)]
pub struct LiverFibrosisStage {
    /// METAVIR fibrosis stage
    pub stage: FibrosisStage,
    /// Mean stiffness in kPa
    pub mean_stiffness_kpa: f64,
    /// Classification confidence
    pub confidence: ClassificationConfidence,
    /// Quality score (0-1)
    pub quality_score: f64,
}

/// METAVIR fibrosis stages
#[derive(Debug, Clone, Copy)]
pub enum FibrosisStage {
    /// No fibrosis (F0) or mild fibrosis (F1)
    F0F1,
    /// Moderate fibrosis (F2)
    F2,
    /// Severe fibrosis (F3)
    F3,
    /// Cirrhosis (F4)
    F4,
}

/// Breast lesion classification result
#[derive(Debug, Clone)]
pub struct BreastLesionClassification {
    /// BI-RADS category (2-5)
    pub birads_category: u8,
    /// Estimated probability of malignancy (0-1)
    pub estimated_malignancy_probability: f64,
    /// Mean stiffness in kPa
    pub mean_stiffness_kpa: f64,
    /// Classification confidence
    pub confidence: ClassificationConfidence,
    /// Quality score (0-1)
    pub quality_score: f64,
}

/// Classification confidence levels
#[derive(Debug, Clone, Copy)]
pub enum ClassificationConfidence {
    /// High confidence (>80% accuracy expected)
    High,
    /// Medium confidence (60-80% accuracy expected)
    Medium,
    /// Low confidence (<60% accuracy expected)
    Low,
}

/// Multi-planar reconstruction for 3D visualization
#[derive(Debug)]
pub struct MultiPlanarReconstruction {
    /// Axial slices (XY planes)
    pub axial_slices: Vec<ElasticityMap2D>,
    /// Sagittal slices (YZ planes)
    pub sagittal_slices: Vec<ElasticityMap2D>,
    /// Coronal slices (XZ planes)
    pub coronal_slices: Vec<ElasticityMap2D>,
    /// Slice positions for each plane
    pub slice_positions: SlicePositions,
}

impl MultiPlanarReconstruction {
    /// Create MPR from 3D elasticity map
    pub fn from_elasticity_map(map: &ElasticityMap3D, slice_spacing: f64) -> Self {
        let mut axial_slices = Vec::new();
        let mut sagittal_slices = Vec::new();
        let mut coronal_slices = Vec::new();

        let (nx, ny, nz) = map.grid.dimensions();

        // Generate axial slices (every slice_spacing meters)
        let axial_positions: Vec<f64> = (0..nz)
            .step_by((slice_spacing / map.grid.dz) as usize)
            .map(|k| k as f64 * map.grid.dz)
            .collect();

        for &z_pos in &axial_positions {
            let z_index = (z_pos / map.grid.dz) as usize;
            if z_index < nz {
                axial_slices.push(map.axial_slice(z_index));
            }
        }

        // Generate sagittal slices
        let sagittal_positions: Vec<f64> = (0..nx)
            .step_by((slice_spacing / map.grid.dx) as usize)
            .map(|i| i as f64 * map.grid.dx)
            .collect();

        for &x_pos in &sagittal_positions {
            let x_index = (x_pos / map.grid.dx) as usize;
            if x_index < nx {
                sagittal_slices.push(map.sagittal_slice(x_index));
            }
        }

        // Generate coronal slices
        let coronal_positions: Vec<f64> = (0..ny)
            .step_by((slice_spacing / map.grid.dy) as usize)
            .map(|j| j as f64 * map.grid.dy)
            .collect();

        for &y_pos in &coronal_positions {
            let y_index = (y_pos / map.grid.dy) as usize;
            if y_index < ny {
                coronal_slices.push(map.coronal_slice(y_index));
            }
        }

        Self {
            axial_slices,
            sagittal_slices,
            coronal_slices,
            slice_positions: SlicePositions {
                axial: axial_positions,
                sagittal: sagittal_positions,
                coronal: coronal_positions,
            },
        }
    }

    /// Get slice at specific position and orientation
    pub fn get_slice(
        &self,
        position: f64,
        orientation: SliceOrientation,
    ) -> Option<&ElasticityMap2D> {
        match orientation {
            SliceOrientation::Axial => {
                let index = self
                    .slice_positions
                    .axial
                    .iter()
                    .position(|&p| (p - position).abs() < 1e-6)?;
                self.axial_slices.get(index)
            }
            SliceOrientation::Sagittal => {
                let index = self
                    .slice_positions
                    .sagittal
                    .iter()
                    .position(|&p| (p - position).abs() < 1e-6)?;
                self.sagittal_slices.get(index)
            }
            SliceOrientation::Coronal => {
                let index = self
                    .slice_positions
                    .coronal
                    .iter()
                    .position(|&p| (p - position).abs() < 1e-6)?;
                self.coronal_slices.get(index)
            }
        }
    }
}

/// Slice orientation for multi-planar reconstruction
#[derive(Debug, Clone, Copy)]
pub enum SliceOrientation {
    /// Axial (XY plane)
    Axial,
    /// Sagittal (YZ plane)
    Sagittal,
    /// Coronal (XZ plane)
    Coronal,
}

/// Slice positions for each orientation
#[derive(Debug, Clone)]
pub struct SlicePositions {
    /// Axial slice positions (z-coordinates)
    pub axial: Vec<f64>,
    /// Sagittal slice positions (x-coordinates)
    pub sagittal: Vec<f64>,
    /// Coronal slice positions (y-coordinates)
    pub coronal: Vec<f64>,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::domain::grid::Grid;

    #[test]
    fn test_volumetric_roi_creation() {
        let roi = VolumetricROI::liver_roi([0.0, 0.0, 0.04]);
        assert_eq!(roi.center, [0.0, 0.0, 0.04]);
        assert_eq!(roi.size, [0.08, 0.08, 0.06]);
        assert_eq!(roi.quality_threshold, 0.8);
    }

    #[test]
    fn test_roi_contains_point() {
        let roi = VolumetricROI {
            center: [0.0, 0.0, 0.04],
            size: [0.06, 0.06, 0.04],
            quality_threshold: 0.7,
            min_depth: 0.02,
            max_depth: 0.06,
            orientation: [0.0, 0.0, 0.0],
        };

        // Point inside ROI
        assert!(roi.contains_point([0.01, 0.01, 0.04]));
        // Point outside ROI (too deep)
        assert!(!roi.contains_point([0.01, 0.01, 0.07]));
        // Point outside ROI (too shallow)
        assert!(!roi.contains_point([0.01, 0.01, 0.01]));
    }

    #[test]
    fn test_elasticity_map_3d_creation() {
        let grid = Grid::new(20, 20, 20, 0.001, 0.001, 0.001).unwrap();
        let map = ElasticityMap3D::new(&grid);

        assert_eq!(map.young_modulus.dim(), (20, 20, 20));
        assert_eq!(map.shear_speed.dim(), (20, 20, 20));
        assert_eq!(map.confidence.dim(), (20, 20, 20));
    }

    #[test]
    fn test_slice_extraction() {
        let grid = Grid::new(10, 10, 10, 0.001, 0.001, 0.001).unwrap();
        let mut map = ElasticityMap3D::new(&grid);

        // Set some test values
        map.young_modulus[[5, 5, 5]] = 10000.0; // 10 kPa
        map.confidence[[5, 5, 5]] = 0.9;

        // Extract axial slice
        let axial_slice = map.axial_slice(5);
        assert_eq!(axial_slice.young_modulus[[5, 5]], 10000.0);
        assert_eq!(axial_slice.confidence[[5, 5]], 0.9);

        // Extract sagittal slice
        let sagittal_slice = map.sagittal_slice(5);
        assert_eq!(sagittal_slice.young_modulus[[5, 5]], 10000.0);

        // Extract coronal slice
        let coronal_slice = map.coronal_slice(5);
        assert_eq!(coronal_slice.young_modulus[[5, 5]], 10000.0);
    }

    #[test]
    fn test_volumetric_statistics() {
        let grid = Grid::new(10, 10, 10, 0.001, 0.001, 0.001).unwrap();
        let mut map = ElasticityMap3D::new(&grid);

        // Set up test data in ROI
        let roi = VolumetricROI {
            center: [0.0045, 0.0045, 0.0045], // Center of 10x10x10 grid
            size: [0.009, 0.009, 0.009],      // Most of the volume
            quality_threshold: 0.5,
            min_depth: 0.0,
            max_depth: 0.01,
            orientation: [0.0, 0.0, 0.0],
        };

        // Set reliable data
        for k in 2..8 {
            for j in 2..8 {
                for i in 2..8 {
                    map.young_modulus[[i, j, k]] = 5000.0; // 5 kPa
                    map.shear_speed[[i, j, k]] = 2.0; // 2 m/s
                    map.confidence[[i, j, k]] = 0.8;
                    map.quality[[i, j, k]] = 0.9;
                    map.reliability_mask[[i, j, k]] = true;
                }
            }
        }

        let stats = map.volumetric_statistics(&roi);

        assert!(stats.valid_voxels > 0);
        assert!((stats.mean_modulus - 5000.0).abs() < 1.0);
        assert!((stats.mean_speed - 2.0).abs() < 0.1);
        assert!(stats.mean_confidence > 0.7);
        assert!(stats.volume_coverage > 0.0);
    }

    #[test]
    fn test_liver_fibrosis_classification() {
        let cds = ClinicalDecisionSupport::default();

        // Test normal liver
        let normal_stats = VolumetricStatistics {
            valid_voxels: 1000,
            mean_modulus: 5500.0, // 5.5 kPa
            std_modulus: 1200.0,
            median_modulus: 5500.0,
            min_modulus: 3000.0,
            max_modulus: 8000.0,
            mean_speed: 1.5,
            mean_confidence: 0.9,
            mean_quality: 0.85,
            volume_coverage: 0.95,
        };

        let classification = cds.classify_liver_fibrosis(&normal_stats);
        match classification.stage {
            FibrosisStage::F0F1 => {} // Expected for normal
            _ => panic!("Expected F0F1 for normal liver"),
        }

        // Test cirrhosis
        let cirrhosis_stats = VolumetricStatistics {
            valid_voxels: 1000,
            mean_modulus: 15000.0, // 15 kPa
            std_modulus: 3000.0,
            median_modulus: 15000.0,
            min_modulus: 10000.0,
            max_modulus: 25000.0,
            mean_speed: 2.5,
            mean_confidence: 0.9,
            mean_quality: 0.85,
            volume_coverage: 0.95,
        };

        let classification = cds.classify_liver_fibrosis(&cirrhosis_stats);
        match classification.stage {
            FibrosisStage::F4 => {} // Expected for cirrhosis
            _ => panic!("Expected F4 for cirrhosis"),
        }
    }

    #[test]
    fn test_breast_lesion_classification() {
        let cds = ClinicalDecisionSupport::default();

        // Test benign lesion
        let benign_stats = VolumetricStatistics {
            valid_voxels: 500,
            mean_modulus: 15000.0, // 15 kPa
            std_modulus: 3000.0,
            median_modulus: 15000.0,
            min_modulus: 10000.0,
            max_modulus: 20000.0,
            mean_speed: 2.2,
            mean_confidence: 0.85,
            mean_quality: 0.8,
            volume_coverage: 0.9,
        };

        let classification = cds.classify_breast_lesion(&benign_stats);
        assert_eq!(classification.birads_category, 2); // Benign

        // Test malignant lesion
        let malignant_stats = VolumetricStatistics {
            valid_voxels: 500,
            mean_modulus: 120000.0, // 120 kPa
            std_modulus: 20000.0,
            median_modulus: 120000.0,
            min_modulus: 80000.0,
            max_modulus: 160000.0,
            mean_speed: 8.0,
            mean_confidence: 0.85,
            mean_quality: 0.8,
            volume_coverage: 0.9,
        };

        let classification = cds.classify_breast_lesion(&malignant_stats);
        assert_eq!(classification.birads_category, 5); // Highly suggestive of malignancy
    }

    #[test]
    fn test_multi_planar_reconstruction() {
        let grid = Grid::new(10, 10, 10, 0.002, 0.002, 0.002).unwrap();
        let mut map = ElasticityMap3D::new(&grid);

        // Set test data
        map.young_modulus[[5, 5, 5]] = 10000.0;
        map.confidence[[5, 5, 5]] = 0.9;
        map.reliability_mask[[5, 5, 5]] = true;

        let mpr = MultiPlanarReconstruction::from_elasticity_map(&map, 0.004);

        // Should have slices
        assert!(!mpr.axial_slices.is_empty());
        assert!(!mpr.sagittal_slices.is_empty());
        assert!(!mpr.coronal_slices.is_empty());

        // Check slice positions
        assert!(!mpr.slice_positions.axial.is_empty());
        assert!(!mpr.slice_positions.sagittal.is_empty());
        assert!(!mpr.slice_positions.coronal.is_empty());

        // Test slice retrieval - use first available position
        let first_axial_pos = mpr.slice_positions.axial[0];
        let axial_slice = mpr.get_slice(first_axial_pos, SliceOrientation::Axial);
        assert!(axial_slice.is_some());
    }

    #[test]
    fn test_clinical_report_generation() {
        let cds = ClinicalDecisionSupport::default();

        let stats = VolumetricStatistics {
            valid_voxels: 1500,
            mean_modulus: 8000.0, // 8 kPa
            std_modulus: 1500.0,
            median_modulus: 8000.0,
            min_modulus: 5000.0,
            max_modulus: 12000.0,
            mean_speed: 1.8,
            mean_confidence: 0.85,
            mean_quality: 0.82,
            volume_coverage: 0.92,
        };

        let report = cds.generate_report("liver", &stats);

        // Check that report contains expected content
        assert!(report.contains("3D SWE Clinical Report"));
        assert!(report.contains("LIVER"));
        assert!(report.contains("Valid voxels: 1500"));
        assert!(report.contains("Mean Young's modulus: 8.0 kPa"));
        assert!(report.contains("Liver Fibrosis Assessment"));
    }
}
