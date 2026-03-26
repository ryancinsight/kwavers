use ndarray::ArrayView3;

/// Compute robust normalization bounds (1st and 99th percentiles)
pub(crate) fn compute_robust_bounds(data: ArrayView3<'_, f64>) -> (f64, f64) {
    let mut values: Vec<f64> = data.iter().cloned().filter(|v| v.is_finite()).collect();

    if values.is_empty() {
        return (0.0, 0.0);
    }

    values.sort_by(|a, b| a.partial_cmp(b).unwrap());

    let len = values.len();
    let lower_idx = (len as f64 * 0.01).floor() as usize;
    let upper_idx = (len as f64 * 0.99).ceil() as usize;

    let min_val = values[lower_idx.min(len - 1)];
    let max_val = values[upper_idx.min(len - 1)];

    if max_val <= min_val {
        (min_val, min_val + 1.0) // Avoid division by zero
    } else {
        (min_val, max_val)
    }
}

/// Classify tissue type based on multi-modal features
///
/// Returns (TissueType, Confidence)
/// Tissue Types:
/// 0: Fluid/Background (Low US, Low Stiffness)
/// 1: Soft Tissue (Mid US, Mid Stiffness)
/// 2: Vessel/Blood (High PA, Low Stiffness)
/// 3: Hard Tissue/Calcification (High US, High Stiffness)
pub(crate) fn classify_voxel_features(us: f64, pa: f64, stiffness: f64) -> (u8, f64) {
    // Heuristic classification logic

    // Check for Hard Tissue (High Stiffness is the strongest indicator)
    if stiffness > 0.7 {
        if us > 0.6 {
            return (3, stiffness * us); // High US + High Stiffness -> Calcification/Bone
        }
        return (3, stiffness * 0.8); // High Stiffness -> Hard Tissue
    }

    // Check for Vessel/Blood (High PA is key)
    if pa > 0.6 {
        if stiffness < 0.4 {
            return (2, pa * (1.0 - stiffness)); // High PA + Low Stiffness -> Vessel
        }
        return (2, pa * 0.7); // High PA -> Likely vessel
    }

    // Check for Fluid (Low signal across board, esp stiffness and US)
    if us < 0.3 && stiffness < 0.3 {
        return (0, (1.0 - us) * (1.0 - stiffness));
    }

    // Default to Soft Tissue
    (1, 0.5)
}
