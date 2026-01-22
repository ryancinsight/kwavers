//! Feature Extraction for Neural Ultrasound Analysis
//!
//! This module implements multi-scale feature extraction for clinical ultrasound
//! analysis including morphological, spectral, and texture features.
//!
//! # Feature Categories
//!
//! - **Morphological**: Shape-based features (gradients, edges, Laplacian)
//! - **Spectral**: Frequency-domain features (local frequency estimation)
//! - **Texture**: Statistical texture features (speckle variance, homogeneity)
//!
//! # Algorithms
//!
//! - Gradient computation: Central differences with 3D Sobel operator
//! - Laplacian: 7-point stencil for blob detection
//! - Local frequency: Window-based variance estimation
//! - Speckle analysis: Local statistical moments
//! - Homogeneity: Gray-level co-occurrence matrix approximation
//!
//! # Literature References
//!
//! - Haralick et al. (1973): "Textural Features for Image Classification"
//! - Pratt (2007): "Digital Image Processing" (4th ed.)
//! - Gonzalez & Woods (2008): "Digital Image Processing"

use crate::core::error::KwaversResult;
use super::config::FeatureConfig;
use crate::clinical::imaging::workflows::neural::types::FeatureMap;
use ndarray::{Array3, ArrayView3};
use std::collections::HashMap;

/// Feature Extractor for Ultrasound Analysis
///
/// Extracts multi-scale morphological, spectral, and texture features from
/// 3D ultrasound volumes for AI analysis and clinical decision support.
///
/// # Example
///
/// ```ignore
/// use kwavers::domain::sensor::beamforming::neural::config::FeatureConfig;
/// use kwavers::domain::sensor::beamforming::neural::features::FeatureExtractor;
/// use ndarray::Array3;
///
/// let config = FeatureConfig::default();
/// let extractor = FeatureExtractor::new(config);
///
/// let volume = Array3::<f32>::zeros((64, 64, 100));
/// let features = extractor.extract_features(volume.view())?;
/// ```
#[derive(Debug, Clone)]
pub struct FeatureExtractor {
    config: FeatureConfig,
}

impl FeatureExtractor {
    /// Create new feature extractor with configuration
    ///
    /// # Arguments
    ///
    /// * `config` - Feature extraction configuration
    pub fn new(config: FeatureConfig) -> Self {
        Self { config }
    }

    /// Extract features from ultrasound volume
    ///
    /// Computes morphological, spectral, and texture features based on
    /// configuration settings. Each feature is a 3D array aligned with
    /// the input volume.
    ///
    /// # Arguments
    ///
    /// * `volume` - Input ultrasound volume [x, y, z]
    ///
    /// # Returns
    ///
    /// Feature map containing all extracted features organized by category
    ///
    /// # Performance
    ///
    /// Computational complexity: O(N·M) where N is volume size and M is window size
    /// Target: <20ms for 64x64x100 volume with default configuration
    pub fn extract_features(&self, volume: ArrayView3<f32>) -> KwaversResult<FeatureMap> {
        let mut morphological = HashMap::new();
        let mut spectral = HashMap::new();
        let mut texture = HashMap::new();

        if self.config.morphological_features {
            morphological.insert(
                "gradient_magnitude".to_string(),
                self.compute_gradient_magnitude(volume),
            );
            morphological.insert("laplacian".to_string(), self.compute_laplacian(volume));
        }

        if self.config.spectral_features {
            spectral.insert(
                "local_frequency".to_string(),
                self.compute_local_frequency(volume),
            );
        }

        if self.config.texture_features {
            texture.insert(
                "speckle_variance".to_string(),
                self.compute_speckle_variance(volume),
            );
            texture.insert("homogeneity".to_string(), self.compute_homogeneity(volume));
        }

        Ok(FeatureMap {
            morphological,
            spectral,
            texture,
        })
    }

    /// Compute gradient magnitude for edge detection
    ///
    /// Uses central differences to approximate first-order derivatives in 3D.
    /// Gradient magnitude highlights edges and boundaries in ultrasound images.
    ///
    /// # Mathematical Definition
    ///
    /// ∇f = (∂f/∂x, ∂f/∂y, ∂f/∂z)
    /// |∇f| = √(∂f/∂x)² + (∂f/∂y)² + (∂f/∂z)²
    ///
    /// # Literature Reference
    ///
    /// - Canny (1986): "A computational approach to edge detection"
    fn compute_gradient_magnitude(&self, volume: ArrayView3<f32>) -> Array3<f32> {
        let (nx, ny, nz) = volume.dim();
        let mut result = Array3::<f32>::zeros((nx, ny, nz));

        for z in 1..nz - 1 {
            for y in 1..ny - 1 {
                for x in 1..nx - 1 {
                    // Central differences for numerical derivatives
                    let dx = (volume[[x + 1, y, z]] - volume[[x - 1, y, z]]) / 2.0;
                    let dy = (volume[[x, y + 1, z]] - volume[[x, y - 1, z]]) / 2.0;
                    let dz = (volume[[x, y, z + 1]] - volume[[x, y, z - 1]]) / 2.0;

                    // Gradient magnitude
                    result[[x, y, z]] = (dx * dx + dy * dy + dz * dz).sqrt();
                }
            }
        }

        result
    }

    /// Compute Laplacian for blob detection
    ///
    /// The Laplacian is a second-order derivative operator that responds to
    /// blobs and ridges in the image. Useful for detecting lesions and
    /// structural features.
    ///
    /// # Mathematical Definition
    ///
    /// ∇²f = ∂²f/∂x² + ∂²f/∂y² + ∂²f/∂z²
    ///
    /// Approximated using 7-point stencil (6-connected neighbors):
    /// ∇²f ≈ f(x+1) + f(x-1) + f(y+1) + f(y-1) + f(z+1) + f(z-1) - 6·f(x,y,z)
    ///
    /// # Literature Reference
    ///
    /// - Lindeberg (1998): "Feature detection with automatic scale selection"
    fn compute_laplacian(&self, volume: ArrayView3<f32>) -> Array3<f32> {
        let (nx, ny, nz) = volume.dim();
        let mut result = Array3::<f32>::zeros((nx, ny, nz));

        for z in 1..nz - 1 {
            for y in 1..ny - 1 {
                for x in 1..nx - 1 {
                    let center = volume[[x, y, z]];

                    // 7-point stencil approximation
                    let laplacian = volume[[x + 1, y, z]]
                        + volume[[x - 1, y, z]]
                        + volume[[x, y + 1, z]]
                        + volume[[x, y - 1, z]]
                        + volume[[x, y, z + 1]]
                        + volume[[x, y, z - 1]]
                        - 6.0 * center;

                    result[[x, y, z]] = laplacian;
                }
            }
        }

        result
    }

    /// Compute local frequency content
    ///
    /// Estimates dominant frequency at each voxel using local variance
    /// as a proxy for frequency content. High variance indicates high-frequency
    /// components; low variance indicates low-frequency content.
    ///
    /// # Algorithm
    ///
    /// 1. Extract 3×3×3 local window around each voxel
    /// 2. Compute window variance: σ² = E[(x - μ)²]
    /// 3. Use standard deviation (σ) as frequency proxy
    ///
    /// # Literature Reference
    ///
    /// - Mallat (1989): "A theory for multiresolution signal decomposition"
    fn compute_local_frequency(&self, volume: ArrayView3<f32>) -> Array3<f32> {
        let (nx, ny, nz) = volume.dim();
        let mut result = Array3::<f32>::zeros((nx, ny, nz));

        for z in 1..nz - 1 {
            for y in 1..ny - 1 {
                for x in 1..nx - 1 {
                    // Extract 3×3×3 window
                    let mut window = Vec::with_capacity(27);
                    for dz in -1..=1 {
                        for dy in -1..=1 {
                            for dx in -1..=1 {
                                let xi = (x as isize + dx) as usize;
                                let yi = (y as isize + dy) as usize;
                                let zi = (z as isize + dz) as usize;
                                if xi < nx && yi < ny && zi < nz {
                                    window.push(volume[[xi, yi, zi]]);
                                }
                            }
                        }
                    }

                    // Compute local variance
                    let mean = window.iter().sum::<f32>() / window.len() as f32;
                    let variance = window.iter().map(|&v| (v - mean).powi(2)).sum::<f32>()
                        / window.len() as f32;

                    // Use standard deviation as frequency proxy
                    result[[x, y, z]] = variance.sqrt();
                }
            }
        }

        result
    }

    /// Compute speckle variance for tissue characterization
    ///
    /// Speckle is a granular pattern in ultrasound images caused by coherent
    /// interference. Speckle variance is a texture feature that characterizes
    /// tissue microstructure and can distinguish tissue types.
    ///
    /// # Algorithm
    ///
    /// 1. For each voxel, extract window of size `config.window_size`
    /// 2. Compute local variance: σ² = Σ(x - μ)² / N
    /// 3. High variance indicates heterogeneous tissue; low variance indicates homogeneous
    ///
    /// # Literature Reference
    ///
    /// - Wagner et al. (1983): "Statistics of speckle in ultrasound B-scans"
    /// - Dutt & Greenleaf (1994): "Adaptive speckle reduction filter"
    fn compute_speckle_variance(&self, volume: ArrayView3<f32>) -> Array3<f32> {
        let (nx, ny, nz) = volume.dim();
        let mut result = Array3::<f32>::zeros((nx, ny, nz));

        let window_size = self.config.window_size;
        let half_window = window_size / 2;

        for z in half_window..nz.saturating_sub(half_window) {
            for y in half_window..ny.saturating_sub(half_window) {
                for x in half_window..nx.saturating_sub(half_window) {
                    let mut window_values = Vec::new();

                    // Extract local window
                    for wz in z.saturating_sub(half_window)..=(z + half_window).min(nz - 1) {
                        for wy in y.saturating_sub(half_window)..=(y + half_window).min(ny - 1) {
                            for wx in x.saturating_sub(half_window)..=(x + half_window).min(nx - 1)
                            {
                                window_values.push(volume[[wx, wy, wz]]);
                            }
                        }
                    }

                    if !window_values.is_empty() {
                        let mean = window_values.iter().sum::<f32>() / window_values.len() as f32;
                        let variance = window_values
                            .iter()
                            .map(|&v| (v - mean).powi(2))
                            .sum::<f32>()
                            / window_values.len() as f32;

                        result[[x, y, z]] = variance;
                    }
                }
            }
        }

        result
    }

    /// Compute homogeneity measure (Gray-Level Co-occurrence Matrix approximation)
    ///
    /// Homogeneity measures the closeness of intensity distribution in a local region.
    /// High homogeneity indicates uniform tissue; low homogeneity indicates texture variation.
    ///
    /// # Mathematical Definition
    ///
    /// Homogeneity = Σ [1 / (1 + |I_center - I_neighbor|)]
    ///
    /// This is an approximation of GLCM homogeneity using immediate neighbors.
    ///
    /// # Literature Reference
    ///
    /// - Haralick et al. (1973): "Textural Features for Image Classification"
    fn compute_homogeneity(&self, volume: ArrayView3<f32>) -> Array3<f32> {
        let (nx, ny, nz) = volume.dim();
        let mut result = Array3::<f32>::zeros((nx, ny, nz));

        for z in 1..nz - 1 {
            for y in 1..ny - 1 {
                for x in 1..nx - 1 {
                    let center = volume[[x, y, z]];

                    // 8-connected neighbors in xy-plane (simplified 3D GLCM)
                    let neighbors = [
                        volume[[x - 1, y - 1, z]],
                        volume[[x, y - 1, z]],
                        volume[[x + 1, y - 1, z]],
                        volume[[x - 1, y, z]],
                        volume[[x + 1, y, z]],
                        volume[[x - 1, y + 1, z]],
                        volume[[x, y + 1, z]],
                        volume[[x + 1, y + 1, z]],
                    ];

                    // Compute homogeneity as inverse of intensity differences
                    let homogeneity = neighbors
                        .iter()
                        .map(|&n| 1.0 / (1.0 + (center - n).abs()))
                        .sum::<f32>()
                        / neighbors.len() as f32;

                    result[[x, y, z]] = homogeneity;
                }
            }
        }

        result
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_feature_extractor_creation() {
        let config = FeatureConfig::default();
        let extractor = FeatureExtractor::new(config);
        assert!(extractor.config.morphological_features);
    }

    #[test]
    fn test_extract_all_features() {
        let config = FeatureConfig::default();
        let extractor = FeatureExtractor::new(config);

        let volume = Array3::<f32>::from_elem((10, 10, 10), 1.0);
        let features = extractor.extract_features(volume.view()).unwrap();

        assert!(features.morphological.contains_key("gradient_magnitude"));
        assert!(features.morphological.contains_key("laplacian"));
        assert!(features.spectral.contains_key("local_frequency"));
        assert!(features.texture.contains_key("speckle_variance"));
        assert!(features.texture.contains_key("homogeneity"));
    }

    #[test]
    fn test_selective_feature_extraction() {
        let config = FeatureConfig {
            morphological_features: true,
            spectral_features: false,
            texture_features: false,
            ..Default::default()
        };

        let extractor = FeatureExtractor::new(config);
        let volume = Array3::<f32>::from_elem((10, 10, 10), 1.0);
        let features = extractor.extract_features(volume.view()).unwrap();

        assert!(features.morphological.contains_key("gradient_magnitude"));
        assert!(features.spectral.is_empty());
        assert!(features.texture.is_empty());
    }

    #[test]
    fn test_gradient_magnitude_constant_volume() {
        let config = FeatureConfig::default();
        let extractor = FeatureExtractor::new(config);

        // Constant volume should have zero gradient everywhere (except boundaries)
        let volume = Array3::<f32>::from_elem((10, 10, 10), 5.0);
        let gradient = extractor.compute_gradient_magnitude(volume.view());

        // Check interior points (boundaries may have artifacts)
        for z in 2..8 {
            for y in 2..8 {
                for x in 2..8 {
                    assert_relative_eq!(gradient[[x, y, z]], 0.0, epsilon = 1e-6);
                }
            }
        }
    }

    #[test]
    fn test_gradient_magnitude_step_edge() {
        let config = FeatureConfig::default();
        let extractor = FeatureExtractor::new(config);

        // Create volume with step edge in x-direction
        let mut volume = Array3::<f32>::zeros((10, 10, 10));
        for z in 0..10 {
            for y in 0..10 {
                for x in 5..10 {
                    volume[[x, y, z]] = 1.0;
                }
            }
        }

        let gradient = extractor.compute_gradient_magnitude(volume.view());

        // Gradient should be strong at x=5 (edge location)
        // Central difference at x=5: (1.0 - 0.0) / 2.0 = 0.5
        assert!(gradient[[5, 5, 5]] > 0.4);
        assert!(gradient[[5, 5, 5]] < 0.6);
    }

    #[test]
    fn test_laplacian_constant_volume() {
        let config = FeatureConfig::default();
        let extractor = FeatureExtractor::new(config);

        // Laplacian of constant function is zero
        let volume = Array3::<f32>::from_elem((10, 10, 10), 3.0);
        let laplacian = extractor.compute_laplacian(volume.view());

        for z in 2..8 {
            for y in 2..8 {
                for x in 2..8 {
                    assert_relative_eq!(laplacian[[x, y, z]], 0.0, epsilon = 1e-6);
                }
            }
        }
    }

    #[test]
    fn test_laplacian_spherical_blob() {
        let config = FeatureConfig::default();
        let extractor = FeatureExtractor::new(config);

        // Create spherical blob
        let mut volume = Array3::<f32>::zeros((20, 20, 20));
        let center = (10.0, 10.0, 10.0);
        let radius = 5.0;

        for z in 0..20 {
            for y in 0..20 {
                for x in 0..20 {
                    let dist = ((x as f32 - center.0).powi(2)
                        + (y as f32 - center.1).powi(2)
                        + (z as f32 - center.2).powi(2))
                    .sqrt();
                    if dist < radius {
                        volume[[x, y, z]] = 1.0;
                    }
                }
            }
        }

        let laplacian = extractor.compute_laplacian(volume.view());

        assert_relative_eq!(laplacian[[10, 10, 10]], 0.0, epsilon = 1e-6);
        assert!(laplacian[[14, 10, 10]] < 0.0);
        assert!(laplacian[[15, 10, 10]] > 0.0);
    }

    #[test]
    fn test_speckle_variance_uniform_region() {
        let config = FeatureConfig::default();
        let extractor = FeatureExtractor::new(config);

        // Uniform region should have low variance
        let volume = Array3::<f32>::from_elem((40, 40, 40), 1.0);
        let variance = extractor.compute_speckle_variance(volume.view());

        // Check central region
        let center_variance = variance[[20, 20, 20]];
        assert_relative_eq!(center_variance, 0.0, epsilon = 1e-6);
    }

    #[test]
    fn test_homogeneity_uniform_region() {
        let config = FeatureConfig::default();
        let extractor = FeatureExtractor::new(config);

        // Uniform region should have maximum homogeneity (1.0)
        let volume = Array3::<f32>::from_elem((10, 10, 10), 2.0);
        let homogeneity = extractor.compute_homogeneity(volume.view());

        // Homogeneity = 1 / (1 + 0) = 1.0 for uniform region
        assert_relative_eq!(homogeneity[[5, 5, 5]], 1.0, epsilon = 1e-6);
    }

    #[test]
    fn test_local_frequency_constant_region() {
        let config = FeatureConfig::default();
        let extractor = FeatureExtractor::new(config);

        // Constant region has zero variance (low frequency)
        let volume = Array3::<f32>::from_elem((10, 10, 10), 1.0);
        let frequency = extractor.compute_local_frequency(volume.view());

        assert_relative_eq!(frequency[[5, 5, 5]], 0.0, epsilon = 1e-6);
    }

    #[test]
    fn test_feature_extraction_preserves_dimensions() {
        let config = FeatureConfig::default();
        let extractor = FeatureExtractor::new(config);

        let volume = Array3::<f32>::zeros((20, 30, 40));
        let features = extractor.extract_features(volume.view()).unwrap();

        // All features should have same dimensions as input
        for (_, feature_array) in features.morphological.iter() {
            assert_eq!(feature_array.dim(), (20, 30, 40));
        }
        for (_, feature_array) in features.spectral.iter() {
            assert_eq!(feature_array.dim(), (20, 30, 40));
        }
        for (_, feature_array) in features.texture.iter() {
            assert_eq!(feature_array.dim(), (20, 30, 40));
        }
    }
}
