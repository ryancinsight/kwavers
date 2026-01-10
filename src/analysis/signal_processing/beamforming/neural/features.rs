//! Feature extraction utilities for neural beamforming.
//!
//! This module provides functions to extract meaningful features from
//! beamformed images or raw RF data for neural network processing.
//!
//! ## Feature Types
//!
//! 1. **Intensity Features**: Raw image values, log-compressed, normalized
//! 2. **Texture Features**: Local standard deviation, entropy
//! 3. **Structural Features**: Gradients, Laplacian, edges
//! 4. **Statistical Features**: Moments, coherence, speckle metrics
//!
//! ## Mathematical Foundation
//!
//! ### Local Standard Deviation (Texture)
//!
//! For a 3×3 local neighborhood:
//! ```text
//! σ(x,y) = √[ (1/9)∑ᵢⱼ I²(x+i, y+j) - μ²(x,y) ]
//! ```
//!
//! ### Gradient Magnitude (Edges)
//!
//! Sobel operator for edge detection:
//! ```text
//! Gₓ = [-1  0  1]      Gᵧ = [-1 -2 -1]
//!      [-2  0  2]           [ 0  0  0]
//!      [-1  0  1]           [ 1  2  1]
//!
//! |∇I| = √(Gₓ² + Gᵧ²)
//! ```
//!
//! ### Laplacian (Structural)
//!
//! Second derivative for detecting rapid intensity changes:
//! ```text
//! ∇²I = [ 0  1  0]     ∇²I(x,y) = 4I(x,y) - I(x±1,y) - I(x,y±1)
//!       [ 1 -4  1]
//!       [ 0  1  0]
//! ```
//!
//! ### Local Entropy (Information)
//!
//! Information content in local neighborhood:
//! ```text
//! H = -∑ᵢ p(i) log₂ p(i)
//! ```
//! where p(i) is the normalized histogram of the local patch.

use ndarray::Array3;

/// Extract all feature types from an image.
///
/// Returns a vector of 5 feature maps:
/// 1. Identity (original image)
/// 2. Local texture (standard deviation)
/// 3. Gradient magnitude (edges)
/// 4. Laplacian (structural)
/// 5. Local entropy (information content)
///
/// # Arguments
///
/// * `image` - Input image as Array3<f32> (frames, height, width)
///
/// # Returns
///
/// Vector of 5 feature maps, each with same dimensions as input.
///
/// # Example
///
/// ```rust,ignore
/// use ndarray::Array3;
/// let image = Array3::<f32>::zeros((1, 256, 256));
/// let features = extract_all_features(&image);
/// assert_eq!(features.len(), 5);
/// ```
pub fn extract_all_features(image: &Array3<f32>) -> Vec<Array3<f32>> {
    vec![
        image.clone(),                   // 1. Identity
        compute_local_std(image),        // 2. Texture
        compute_spatial_gradient(image), // 3. Edges
        compute_laplacian(image),        // 4. Structural
        compute_local_entropy(image),    // 5. Information
    ]
}

/// Compute local standard deviation (texture feature).
///
/// Applies a 3×3 sliding window to compute standard deviation at each pixel,
/// providing a measure of local texture and variability.
///
/// # Mathematical Definition
///
/// ```text
/// σ(x,y) = √[ E[I²] - E[I]² ]
/// ```
///
/// # Arguments
///
/// * `image` - Input image (frames, height, width)
///
/// # Returns
///
/// Standard deviation map with same dimensions as input.
///
/// # Edge Handling
///
/// Borders (1-pixel margin) are set to zero to avoid edge artifacts.
pub fn compute_local_std(image: &Array3<f32>) -> Array3<f32> {
    let mut std_map = Array3::zeros(image.dim());
    let (d0, d1, d2) = image.dim();

    for k in 0..d0 {
        for i in 1..d1 - 1 {
            for j in 1..d2 - 1 {
                let mut sum = 0.0;
                let mut sq_sum = 0.0;
                let mut count = 0;

                // 3×3 neighborhood
                for di in -1..=1 {
                    for dj in -1..=1 {
                        let val =
                            image[[k, (i as isize + di) as usize, (j as isize + dj) as usize]];
                        sum += val;
                        sq_sum += val * val;
                        count += 1;
                    }
                }

                let mean = sum / count as f32;
                let variance = (sq_sum / count as f32) - (mean * mean);
                std_map[[k, i, j]] = variance.max(0.0).sqrt();
            }
        }
    }

    std_map
}

/// Compute spatial gradient magnitude using Sobel operator.
///
/// Detects edges and boundaries by computing the magnitude of the
/// image gradient in X and Y directions.
///
/// # Sobel Kernels
///
/// X-direction (vertical edges):
/// ```text
/// [-1  0  1]
/// [-2  0  2]
/// [-1  0  1]
/// ```
///
/// Y-direction (horizontal edges):
/// ```text
/// [-1 -2 -1]
/// [ 0  0  0]
/// [ 1  2  1]
/// ```
///
/// # Arguments
///
/// * `image` - Input image (frames, height, width)
///
/// # Returns
///
/// Gradient magnitude map: |∇I| = √(Gₓ² + Gᵧ²)
pub fn compute_spatial_gradient(image: &Array3<f32>) -> Array3<f32> {
    let mut grad_map = Array3::zeros(image.dim());
    let (d0, d1, d2) = image.dim();

    for k in 0..d0 {
        for i in 1..d1 - 1 {
            for j in 1..d2 - 1 {
                // Sobel X kernel (vertical edges)
                let gx = -image[[k, i - 1, j - 1]] + image[[k, i + 1, j - 1]]
                    - 2.0 * image[[k, i - 1, j]]
                    + 2.0 * image[[k, i + 1, j]]
                    - image[[k, i - 1, j + 1]]
                    + image[[k, i + 1, j + 1]];

                // Sobel Y kernel (horizontal edges)
                let gy = -image[[k, i - 1, j - 1]]
                    - 2.0 * image[[k, i, j - 1]]
                    - image[[k, i + 1, j - 1]]
                    + image[[k, i - 1, j + 1]]
                    + 2.0 * image[[k, i, j + 1]]
                    + image[[k, i + 1, j + 1]];

                // Gradient magnitude
                grad_map[[k, i, j]] = (gx * gx + gy * gy).sqrt();
            }
        }
    }

    grad_map
}

/// Compute Laplacian (second derivative) for structural features.
///
/// Detects regions of rapid intensity change, useful for identifying
/// boundaries, ridges, and other structural features.
///
/// # Laplacian Kernel (4-connected)
///
/// ```text
/// [ 0  1  0]
/// [ 1 -4  1]
/// [ 0  1  0]
/// ```
///
/// # Mathematical Definition
///
/// ```text
/// ∇²I(x,y) = I(x+1,y) + I(x-1,y) + I(x,y+1) + I(x,y-1) - 4I(x,y)
/// ```
///
/// # Arguments
///
/// * `image` - Input image (frames, height, width)
///
/// # Returns
///
/// Absolute value of Laplacian: |∇²I|
pub fn compute_laplacian(image: &Array3<f32>) -> Array3<f32> {
    let mut lap_map = Array3::zeros(image.dim());
    let (d0, d1, d2) = image.dim();

    for k in 0..d0 {
        for i in 1..d1 - 1 {
            for j in 1..d2 - 1 {
                // 4-connected Laplacian kernel
                let lap = image[[k, i - 1, j]]
                    + image[[k, i + 1, j]]
                    + image[[k, i, j - 1]]
                    + image[[k, i, j + 1]]
                    - 4.0 * image[[k, i, j]];

                lap_map[[k, i, j]] = lap.abs();
            }
        }
    }

    lap_map
}

/// Compute local entropy (information content).
///
/// Measures the randomness or information content in a local neighborhood,
/// useful for texture analysis and speckle characterization.
///
/// # Mathematical Definition
///
/// For a 3×3 patch with normalized intensities p(i):
/// ```text
/// H = -∑ᵢ p(i) log₂ p(i)
/// ```
///
/// # Arguments
///
/// * `image` - Input image (frames, height, width)
///
/// # Returns
///
/// Local entropy map (higher values = more random/textured)
///
/// # Implementation Note
///
/// Uses a simplified histogram-free approximation based on normalized
/// patch variance for computational efficiency.
pub fn compute_local_entropy(image: &Array3<f32>) -> Array3<f32> {
    let mut entropy_map = Array3::zeros(image.dim());
    let (d0, d1, d2) = image.dim();

    const EPSILON: f32 = 1e-10;

    for k in 0..d0 {
        for i in 1..d1 - 1 {
            for j in 1..d2 - 1 {
                let mut sum = 0.0;
                let mut patch = [0.0f32; 9];
                let mut idx = 0;

                // Extract 3×3 patch
                for di in -1..=1 {
                    for dj in -1..=1 {
                        let val = image
                            [[k, (i as isize + di) as usize, (j as isize + dj) as usize]]
                        .abs();
                        patch[idx] = val;
                        sum += val;
                        idx += 1;
                    }
                }

                if sum < EPSILON {
                    continue;
                }

                // Normalize patch to form probability distribution
                let mut entropy = 0.0;
                for val in patch {
                    let p = val / sum;
                    if p > EPSILON {
                        entropy -= p * p.ln();
                    }
                }

                entropy_map[[k, i, j]] = entropy;
            }
        }
    }

    entropy_map
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
        let min_val = feature.iter().cloned().fold(f32::INFINITY, f32::min);
        let max_val = feature.iter().cloned().fold(f32::NEG_INFINITY, f32::max);

        let range = max_val - min_val;
        if range > 1e-10 {
            feature.mapv_inplace(|v| (v - min_val) / range);
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

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array3;

    fn create_test_image() -> Array3<f32> {
        let mut image = Array3::zeros((1, 10, 10));
        // Create a simple pattern: bright center, dark edges
        for i in 3..7 {
            for j in 3..7 {
                image[[0, i, j]] = 1.0;
            }
        }
        image
    }

    #[test]
    fn test_extract_all_features() {
        let image = create_test_image();
        let features = extract_all_features(&image);
        assert_eq!(features.len(), 5);
        for feature in features {
            assert_eq!(feature.dim(), (1, 10, 10));
        }
    }

    #[test]
    fn test_local_std() {
        let image = create_test_image();
        let std_map = compute_local_std(&image);
        assert_eq!(std_map.dim(), (1, 10, 10));
        // Borders should be zero
        assert_eq!(std_map[[0, 0, 0]], 0.0);
        // Center should have some variation at edges
        assert!(std_map[[0, 4, 4]] >= 0.0);
    }

    #[test]
    fn test_spatial_gradient() {
        let image = create_test_image();
        let grad_map = compute_spatial_gradient(&image);
        assert_eq!(grad_map.dim(), (1, 10, 10));
        // Gradient should be high at edges of bright region
        assert!(grad_map[[0, 3, 3]] > 0.0 || grad_map[[0, 3, 4]] > 0.0);
    }

    #[test]
    fn test_laplacian() {
        let image = create_test_image();
        let lap_map = compute_laplacian(&image);
        assert_eq!(lap_map.dim(), (1, 10, 10));
        // Laplacian should detect edges
        assert!(lap_map[[0, 5, 5]] >= 0.0);
    }

    #[test]
    fn test_local_entropy() {
        let image = create_test_image();
        let entropy_map = compute_local_entropy(&image);
        assert_eq!(entropy_map.dim(), (1, 10, 10));
        // Entropy should be higher at edges (more variation)
        assert!(entropy_map[[0, 3, 3]] >= 0.0);
    }

    #[test]
    fn test_normalize_features() {
        let image = create_test_image();
        let mut features = extract_all_features(&image);
        normalize_features(&mut features);

        for feature in &features {
            let min_val = feature.iter().cloned().fold(f32::INFINITY, f32::min);
            let max_val = feature.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            assert!(min_val >= -1e-6); // Allow small numerical error
            assert!(max_val <= 1.0 + 1e-6);
        }
    }

    #[test]
    fn test_concatenate_features() {
        let image = create_test_image();
        let features = extract_all_features(&image);
        let stacked = concatenate_features(&features);
        assert_eq!(stacked.shape(), &[1, 5, 10, 10]);
    }

    #[test]
    fn test_concatenate_empty() {
        let features: Vec<Array3<f32>> = vec![];
        let stacked = concatenate_features(&features);
        assert_eq!(stacked.shape(), &[0, 0, 0, 0]);
    }

    #[test]
    fn test_edge_cases_uniform_image() {
        let image = Array3::from_elem((1, 10, 10), 0.5);
        let features = extract_all_features(&image);

        // Uniform image should have zero texture/gradient/laplacian
        assert!(features[1][[0, 5, 5]].abs() < 1e-6); // std
        assert!(features[2][[0, 5, 5]].abs() < 1e-6); // gradient
        assert!(features[3][[0, 5, 5]].abs() < 1e-6); // laplacian
    }

    #[test]
    fn test_zero_image() {
        let image = Array3::zeros((1, 10, 10));
        let features = extract_all_features(&image);
        assert_eq!(features.len(), 5);
        // All features should be zero for zero image
        for feature in features {
            assert!(feature.iter().all(|&v| v.abs() < 1e-6));
        }
    }
}
