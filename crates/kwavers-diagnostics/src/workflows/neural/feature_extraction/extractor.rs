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
//! ## Not yet implemented
//!
//! - **Full GLCM**: All 14 Haralick statistical texture features from the Gray-Level
//!   Co-occurrence Matrix (Haralick et al. 1973).
//! - **Deep learning features**: CNN-based feature extraction for lesion characterization.
//! - **Radiomics standardization**: IBSI (Image Biomarker Standardization Initiative)
//!   guideline compliance for reproducible feature computation.
//! - **Multi-scale texture analysis**: Wavelet-based decomposition for scale-invariant features.
//! - **Shape features**: Morphological descriptors for lesion boundary characterization.
//!
//! # Literature References
//!
//! - Haralick et al. (1973): "Textural Features for Image Classification"
//! - Pratt (2007): "Digital Image Processing" (4th ed.)
//! - Gonzalez & Woods (2008): "Digital Image Processing"

use super::super::types::FeatureMap;
use kwavers_analysis::signal_processing::beamforming::neural::config::FeatureConfig;
use kwavers_core::error::KwaversResult;
use leto::{
    Array3,
    ArrayView3,
};
use std::collections::HashMap;

/// Feature Extractor for Ultrasound Analysis
///
/// Extracts multi-scale morphological, spectral, and texture features from
/// 3D ultrasound volumes for AI analysis and clinical decision support.
///
/// # Example
///
/// ```ignore
/// use kwavers_transducer::beamforming::neural::config::FeatureConfig;
/// use kwavers_transducer::beamforming::neural::features::FeatureExtractor;
/// use leto::Array3;
///
/// let config = FeatureConfig::default();
/// let extractor = FeatureExtractor::new(config);
///
/// let volume = Array3::<f32>::zeros((64, 64, 100));
/// let features = extractor.extract_features(volume.view())?;
/// ```
#[derive(Debug, Clone)]
pub struct FeatureExtractor {
    pub(super) config: FeatureConfig,
}

impl FeatureExtractor {
    /// Create new feature extractor with configuration
    ///
    /// # Arguments
    ///
    /// * `config` - Feature extraction configuration
    #[must_use]
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
    /// # Errors
    /// - Returns [`Err`] if an internal constraint is violated.
    ///
    pub fn extract_features(&self, volume: ArrayView3<f32>) -> KwaversResult<FeatureMap> {
        let mut morphological = HashMap::new();
        let mut spectral = HashMap::new();
        let mut texture = HashMap::new();

        if self.config.morphological_features {
            morphological.insert(
                "gradient_magnitude".to_owned(),
                self.compute_gradient_magnitude(volume),
            );
            morphological.insert("laplacian".to_owned(), self.compute_laplacian(volume));
        }

        if self.config.spectral_features {
            spectral.insert(
                "local_frequency".to_owned(),
                self.compute_local_frequency(volume),
            );
        }

        if self.config.texture_features {
            texture.insert(
                "speckle_variance".to_owned(),
                self.compute_speckle_variance(volume),
            );
            texture.insert("homogeneity".to_owned(), self.compute_homogeneity(volume));
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
    pub(super) fn compute_gradient_magnitude(&self, volume: ArrayView3<f32>) -> Array3<f32> {
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
    pub(super) fn compute_laplacian(&self, volume: ArrayView3<f32>) -> Array3<f32> {
        let (nx, ny, nz) = volume.dim();
        let mut result = Array3::<f32>::zeros((nx, ny, nz));

        for z in 1..nz - 1 {
            for y in 1..ny - 1 {
                for x in 1..nx - 1 {
                    let center = volume[[x, y, z]];

                    // 7-point stencil approximation
                    let laplacian = 6.0f32.mul_add(
                        -center,
                        volume[[x + 1, y, z]]
                            + volume[[x - 1, y, z]]
                            + volume[[x, y + 1, z]]
                            + volume[[x, y - 1, z]]
                            + volume[[x, y, z + 1]]
                            + volume[[x, y, z - 1]],
                    );

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
    pub(super) fn compute_local_frequency(&self, volume: ArrayView3<f32>) -> Array3<f32> {
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
    pub(super) fn compute_speckle_variance(&self, volume: ArrayView3<f32>) -> Array3<f32> {
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
    pub(super) fn compute_homogeneity(&self, volume: ArrayView3<f32>) -> Array3<f32> {
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
