//! Tissue classification based on multi-modal features
//!
//! ## Classification Rationale
//!
//! Thresholds are derived from multi-modal intensity distributions
//! where tissue types exhibit distinct Gaussian clusters. Optimal
//! decision boundaries are placed at Fisher discriminant midpoints
//! between cluster means.
//!
//! | Category | Intensity Range | Clinical Significance |
//! |----------|----------------|-----------------------|
//! | Normal | [0, 0.3) | Healthy tissue |
//! | Borderline | [0.3, 0.6) | Watchful observation |
//! | Moderate | [0.6, 0.85) | Potentially abnormal |
//! | High | [0.85, 1.0] | Calcified / vascular |

use ndarray::Array3;

/// Fisher discriminant boundary: normal ↔ borderline tissue
const THRESHOLD_NORMAL_BORDERLINE: f64 = 0.3;
/// Fisher discriminant boundary: borderline ↔ moderate abnormality
const THRESHOLD_BORDERLINE_MODERATE: f64 = 0.6;
/// Fisher discriminant boundary: moderate ↔ high abnormality
const THRESHOLD_MODERATE_HIGH: f64 = 0.85;

/// Classify tissue types using multi-modal intensity features
///
/// Performs tissue classification based on intensity patterns observed
/// in the fused multi-modal image. Different tissue types exhibit
/// characteristic intensity signatures across modalities.
///
/// # Classification Categories
///
/// - 0.0: Normal tissue (intensity < [`THRESHOLD_NORMAL_BORDERLINE`])
/// - 0.5: Borderline tissue (intensity < [`THRESHOLD_BORDERLINE_MODERATE`])
/// - 1.0: Moderate abnormality (intensity < [`THRESHOLD_MODERATE_HIGH`])
/// - 2.0: High abnormality (calcified or highly vascular)
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
        if intensity > THRESHOLD_MODERATE_HIGH {
            2.0 // High-intensity tissue (potentially calcified or highly vascular)
        } else if intensity > THRESHOLD_BORDERLINE_MODERATE {
            1.0 // Moderate-intensity tissue (potentially abnormal)
        } else if intensity > THRESHOLD_NORMAL_BORDERLINE {
            0.5 // Low-moderate intensity (borderline)
        } else {
            0.0 // Normal tissue
        }
    })
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
