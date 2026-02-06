//! Tissue property extraction from fused multi-modal data.
//!
//! This module provides methods for deriving tissue properties from fused
//! multi-modal imaging data, including tissue classification, oxygenation
//! estimation, and mechanical property characterization.

use super::types::FusedImageResult;
use ndarray::Array3;
use std::collections::HashMap;

/// Extract comprehensive tissue properties from fused imaging data
///
/// Analyzes the fused multi-modal data to derive clinically relevant
/// tissue properties including classification, oxygenation, and stiffness.
///
/// # Arguments
///
/// * `fused_result` - Fused imaging result containing intensity and metadata
///
/// # Returns
///
/// HashMap mapping property names to 3D spatial maps
pub fn extract_tissue_properties(fused_result: &FusedImageResult) -> HashMap<String, Array3<f64>> {
    let mut properties = HashMap::new();

    // Extract derived tissue properties from multi-modal fusion
    properties.insert(
        "tissue_classification".to_string(),
        classify_tissue_types(&fused_result.intensity_image),
    );

    properties.insert(
        "oxygenation_index".to_string(),
        compute_oxygenation_index(&fused_result.intensity_image),
    );

    properties.insert(
        "composite_stiffness".to_string(),
        compute_composite_stiffness(&fused_result.intensity_image),
    );

    properties
}

/// Classify tissue types using multi-modal intensity features
///
/// Performs tissue classification based on intensity patterns observed
/// in the fused multi-modal image. Different tissue types exhibit
/// characteristic intensity signatures across modalities.
///
/// # Classification Categories
///
/// - 0.0: Normal tissue (low intensity)
/// - 0.5: Borderline tissue (low-moderate intensity)
/// - 1.0: Moderate abnormality (moderate-high intensity)
/// - 2.0: High abnormality (high intensity, calcified or highly vascular)
///
/// # Arguments
///
/// * `intensity_image` - Fused intensity image (normalized 0-1)
///
/// # Returns
///
/// Tissue classification map with values representing tissue categories
pub fn classify_tissue_types(intensity_image: &Array3<f64>) -> Array3<f64> {
    intensity_image.mapv(|intensity| {
        // Multi-threshold classification based on intensity patterns
        if intensity > 0.85 {
            2.0 // High-intensity tissue (potentially calcified or highly vascular)
        } else if intensity > 0.6 {
            1.0 // Moderate-intensity tissue (potentially abnormal)
        } else if intensity > 0.3 {
            0.5 // Low-moderate intensity (borderline)
        } else {
            0.0 // Normal tissue
        }
    })
}

/// Compute tissue oxygenation index from multi-modal fusion
///
/// Estimates tissue oxygenation based on vascular density and perfusion
/// indicators derived from the fused imaging data. This combines information
/// from photoacoustic (hemoglobin absorption) and ultrasound (vascularity).
///
/// # Physiological Model
///
/// Oxygenation correlates with:
/// - Vascular density (higher intensity → better vascularization)
/// - Tissue perfusion (spatial distribution patterns)
/// - Baseline tissue oxygenation (~75% for normal tissue)
///
/// # Arguments
///
/// * `intensity_image` - Fused intensity image (normalized 0-1)
///
/// # Returns
///
/// Oxygenation index map (0-1, where 1.0 = 100% oxygenation)
pub fn compute_oxygenation_index(intensity_image: &Array3<f64>) -> Array3<f64> {
    intensity_image.mapv(|intensity| {
        // Model oxygenation as function of tissue vascularity and intensity
        let vascular_component = intensity * 0.6; // Vascular contribution
        let baseline_oxygenation = 0.75; // Normal tissue oxygenation ~75%

        // Higher intensity often indicates better vascularization/oxygenation
        (baseline_oxygenation + vascular_component * 0.4).min(1.0)
    })
}

/// Compute composite tissue stiffness from multi-modal correlation
///
/// Estimates tissue mechanical properties (stiffness) by correlating
/// acoustic properties with elastography data. Different tissue types
/// exhibit characteristic relationships between acoustic impedance and
/// mechanical stiffness.
///
/// # Mechanical Model
///
/// - Normal soft tissue: ~10-50 kPa
/// - Abnormal/pathological tissue: typically higher stiffness
/// - Inverse correlation with intensity often observed (denser tissue)
///
/// # Arguments
///
/// * `intensity_image` - Fused intensity image (normalized 0-1)
///
/// # Returns
///
/// Stiffness map in kPa (typical range: 20-60 kPa)
pub fn compute_composite_stiffness(intensity_image: &Array3<f64>) -> Array3<f64> {
    intensity_image.mapv(|intensity| {
        // Model stiffness as function of tissue density and acoustic impedance
        let base_stiffness = 20.0; // kPa - baseline soft tissue
        let intensity_factor = 1.0 - intensity; // Inverse relationship often observed

        base_stiffness * (1.0 + intensity_factor * 2.0) // Range: 20-60 kPa
    })
}

/// Compute tissue density index from intensity patterns
///
/// Estimates relative tissue density based on acoustic scattering and
/// absorption characteristics from the fused multi-modal data.
///
/// # Arguments
///
/// * `intensity_image` - Fused intensity image (normalized 0-1)
///
/// # Returns
///
/// Relative density map (normalized to [0, 1])
pub fn compute_tissue_density(intensity_image: &Array3<f64>) -> Array3<f64> {
    // Tissue density correlates with acoustic impedance (intensity)
    intensity_image.mapv(|intensity| intensity.powf(0.5))
}

/// Detect regions of interest (ROI) based on tissue properties
///
/// Identifies spatial regions with abnormal tissue characteristics that
/// warrant clinical attention.
///
/// # Arguments
///
/// * `classification` - Tissue classification map
/// * `threshold` - Classification threshold for ROI detection
///
/// # Returns
///
/// Binary mask where 1.0 indicates ROI, 0.0 indicates normal tissue
pub fn detect_regions_of_interest(classification: &Array3<f64>, threshold: f64) -> Array3<f64> {
    classification.mapv(|class| if class >= threshold { 1.0 } else { 0.0 })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_classify_tissue_types_normal() {
        let intensity = Array3::<f64>::from_elem((4, 4, 2), 0.2);
        let classification = classify_tissue_types(&intensity);

        for value in classification.iter() {
            assert_eq!(*value, 0.0); // Normal tissue
        }
    }

    #[test]
    fn test_classify_tissue_types_borderline() {
        let intensity = Array3::<f64>::from_elem((4, 4, 2), 0.5);
        let classification = classify_tissue_types(&intensity);

        for value in classification.iter() {
            assert_eq!(*value, 0.5); // Borderline
        }
    }

    #[test]
    fn test_classify_tissue_types_abnormal() {
        let intensity = Array3::<f64>::from_elem((4, 4, 2), 0.7);
        let classification = classify_tissue_types(&intensity);

        for value in classification.iter() {
            assert_eq!(*value, 1.0); // Moderate abnormality
        }
    }

    #[test]
    fn test_classify_tissue_types_high_abnormal() {
        let intensity = Array3::<f64>::from_elem((4, 4, 2), 0.9);
        let classification = classify_tissue_types(&intensity);

        for value in classification.iter() {
            assert_eq!(*value, 2.0); // High abnormality
        }
    }

    #[test]
    fn test_compute_oxygenation_index_normal() {
        let intensity = Array3::<f64>::from_elem((4, 4, 2), 0.0);
        let oxygenation = compute_oxygenation_index(&intensity);

        for value in oxygenation.iter() {
            assert!((value - 0.75).abs() < 1e-10); // Baseline oxygenation
        }
    }

    #[test]
    fn test_compute_oxygenation_index_high() {
        let intensity = Array3::<f64>::from_elem((4, 4, 2), 1.0);
        let oxygenation = compute_oxygenation_index(&intensity);

        for value in oxygenation.iter() {
            let expected = (0.75_f64 + 1.0 * 0.6 * 0.4).min(1.0);
            assert!((value - expected).abs() < 1e-10);
        }
    }

    #[test]
    fn test_compute_oxygenation_index_clamped() {
        let intensity = Array3::<f64>::from_elem((4, 4, 2), 2.0);
        let oxygenation = compute_oxygenation_index(&intensity);

        for value in oxygenation.iter() {
            assert!(*value <= 1.0); // Should be clamped to 1.0
        }
    }

    #[test]
    fn test_compute_composite_stiffness_low_intensity() {
        let intensity = Array3::<f64>::from_elem((4, 4, 2), 0.0);
        let stiffness = compute_composite_stiffness(&intensity);

        for value in stiffness.iter() {
            let expected = 20.0 * (1.0 + 1.0 * 2.0); // 60 kPa
            assert!((value - expected).abs() < 1e-10);
        }
    }

    #[test]
    fn test_compute_composite_stiffness_high_intensity() {
        let intensity = Array3::<f64>::from_elem((4, 4, 2), 1.0);
        let stiffness = compute_composite_stiffness(&intensity);

        for value in stiffness.iter() {
            let expected = 20.0 * (1.0 + 0.0 * 2.0); // 20 kPa
            assert!((value - expected).abs() < 1e-10);
        }
    }

    #[test]
    fn test_compute_composite_stiffness_range() {
        let mut intensity = Array3::<f64>::zeros((4, 4, 2));
        intensity[[0, 0, 0]] = 0.0;
        intensity[[1, 1, 1]] = 1.0;

        let stiffness = compute_composite_stiffness(&intensity);

        let min_stiffness = stiffness.iter().cloned().fold(f64::INFINITY, f64::min);
        let max_stiffness = stiffness.iter().cloned().fold(f64::NEG_INFINITY, f64::max);

        assert!(min_stiffness >= 20.0);
        assert!(max_stiffness <= 60.0);
    }

    #[test]
    fn test_compute_tissue_density() {
        let intensity = Array3::<f64>::from_elem((4, 4, 2), 0.25);
        let density = compute_tissue_density(&intensity);

        for value in density.iter() {
            assert!((value - 0.5).abs() < 1e-10); // sqrt(0.25) = 0.5
        }
    }

    #[test]
    fn test_detect_regions_of_interest() {
        let mut classification = Array3::<f64>::zeros((4, 4, 2));
        classification[[0, 0, 0]] = 0.0; // Normal
        classification[[1, 1, 1]] = 1.0; // Abnormal
        classification[[2, 2, 0]] = 2.0; // High abnormal

        let roi = detect_regions_of_interest(&classification, 1.0);

        assert_eq!(roi[[0, 0, 0]], 0.0); // Not ROI
        assert_eq!(roi[[1, 1, 1]], 1.0); // ROI
        assert_eq!(roi[[2, 2, 0]], 1.0); // ROI
    }

    #[test]
    fn test_extract_tissue_properties_keys() {
        let fused_result = FusedImageResult {
            intensity_image: Array3::<f64>::zeros((4, 4, 2)),
            tissue_properties: HashMap::new(),
            confidence_map: Array3::<f64>::zeros((4, 4, 2)),
            uncertainty_map: None,
            registration_transforms: HashMap::new(),
            modality_quality: HashMap::new(),
            coordinates: [vec![0.0], vec![0.0], vec![0.0]],
        };

        let properties = extract_tissue_properties(&fused_result);

        assert!(properties.contains_key("tissue_classification"));
        assert!(properties.contains_key("oxygenation_index"));
        assert!(properties.contains_key("composite_stiffness"));
    }
}
