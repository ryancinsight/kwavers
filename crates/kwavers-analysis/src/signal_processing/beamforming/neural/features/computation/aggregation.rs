use ndarray::Array3;

use super::local_ops::{
    compute_laplacian, compute_local_entropy, compute_local_std, compute_spatial_gradient,
};

/// Extract all feature types from an image as summary statistics.
///
/// Returns a 1D vector of 6 scalar features:
/// 1. Mean intensity
/// 2. Standard deviation (texture)
/// 3. Mean gradient magnitude (edges)
/// 4. Mean Laplacian (structural)
/// 5. Entropy (information content)
/// 6. Peak intensity
///
/// These summary statistics capture global image properties for neural network input.
///
/// # Arguments
///
/// * `image` - Input image as `Array3<f32>` (frames, angles, samples)
///
/// # Returns
///
/// `Array1<f32>` of 6 summary features.
///
/// # Example
///
/// ```rust,ignore
/// use ndarray::Array3;
/// let image = Array3::<f32>::zeros((1, 1, 256));
/// let features = extract_all_features(&image);
/// assert_eq!(features.len(), 6);
/// ```
pub fn extract_all_features(image: &Array3<f32>) -> ndarray::Array1<f32> {
    use ndarray::Array1;

    // 1. Mean intensity
    let mean_intensity = image.mean().unwrap_or(0.0);

    // 2. Standard deviation (global texture)
    let std_map = compute_local_std(image);
    let mean_std = std_map.mean().unwrap_or(0.0);

    // 3. Mean gradient magnitude (edge strength)
    let gradient_map = compute_spatial_gradient(image);
    let mean_gradient = gradient_map.mean().unwrap_or(0.0);

    // 4. Mean Laplacian (structural complexity)
    let laplacian_map = compute_laplacian(image);
    let mean_laplacian = laplacian_map.mean().unwrap_or(0.0);

    // 5. Entropy (information content)
    let entropy_map = compute_local_entropy(image);
    let mean_entropy = entropy_map.mean().unwrap_or(0.0);

    // 6. Peak intensity (dynamic range)
    let peak_intensity = image.iter().copied().fold(0.0f32, f32::max);

    Array1::from_vec(vec![
        mean_intensity,
        mean_std,
        mean_gradient,
        mean_laplacian,
        mean_entropy,
        peak_intensity,
    ])
}

/// Normalize features to [0, 1] range.
///
/// Applies min-max normalization to each feature map independently.
///
/// # Formula
///
/// ```text
/// I_norm = (I - I_min) / (I_max - I_min)
/// ```
///
/// # Arguments
///
/// * `features` - Vector of feature maps to normalize (modified in-place)
///
/// # Example
///
/// ```rust,ignore
/// let mut features = extract_all_features(&image);
/// normalize_features(&mut features);
/// // All features now in [0, 1] range
/// ```
pub fn normalize_features(features: &mut [Array3<f32>]) {
    for feature in features.iter_mut() {
        let min_val = feature.iter().copied().fold(f32::INFINITY, f32::min);
        let max_val = feature.iter().copied().fold(f32::NEG_INFINITY, f32::max);

        let range = max_val - min_val;
        if range > 1e-10 {
            feature.par_mapv_inplace(|v| (v - min_val) / range);
        }
    }
}

/// Concatenate features along a new dimension for network input.
///
/// Stacks multiple feature maps into a single 4D array suitable for
/// neural network processing.
///
/// # Arguments
///
/// * `features` - Vector of feature maps (each is frames × height × width)
///
/// # Returns
///
/// 4D array: (frames, features, height, width)
///
/// # Example
///
/// ```rust,ignore
/// let features = extract_all_features(&image); // 5 features
/// let stacked = concatenate_features(&features);
/// assert_eq!(stacked.shape(), &[1, 5, 256, 256]);
/// ```
#[must_use]
pub fn concatenate_features(features: &[Array3<f32>]) -> ndarray::Array4<f32> {
    if features.is_empty() {
        return ndarray::Array4::zeros((0, 0, 0, 0));
    }

    let (d0, d1, d2) = features[0].dim();
    let num_features = features.len();

    let mut stacked = ndarray::Array4::zeros((d0, num_features, d1, d2));

    for (feat_idx, feature) in features.iter().enumerate() {
        for k in 0..d0 {
            for i in 0..d1 {
                for j in 0..d2 {
                    stacked[[k, feat_idx, i, j]] = feature[[k, i, j]];
                }
            }
        }
    }

    stacked
}
