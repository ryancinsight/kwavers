/// Feature Extraction Configuration
///
/// Controls which features are extracted from ultrasound volumes for
/// AI analysis and clinical decision support.
///
/// # Feature Types
///
/// - **Morphological**: Shape, boundaries, gradients (edge detection)
/// - **Spectral**: Frequency content, bandwidth, local frequency
/// - **Texture**: Speckle statistics, homogeneity, entropy
#[derive(Debug, Clone)]
pub struct FeatureConfig {
    /// Extract morphological features (gradient magnitude, Laplacian)
    pub morphological_features: bool,

    /// Extract spectral features (local frequency, bandwidth)
    pub spectral_features: bool,

    /// Extract texture features (speckle variance, homogeneity)
    pub texture_features: bool,

    /// Feature computation window size (voxels). Must be odd and >= 3.
    pub window_size: usize,

    /// Overlap between feature windows (0.0 to 1.0).
    pub overlap: f64,
}

impl Default for FeatureConfig {
    fn default() -> Self {
        Self {
            morphological_features: true,
            spectral_features: true,
            texture_features: true,
            window_size: 31,
            overlap: 0.5,
        }
    }
}

impl FeatureConfig {
    /// Validate feature configuration
    ///
    /// # Invariants
    ///
    /// - At least one feature type must be enabled
    /// - Window size must be odd and >= 3
    /// - Overlap must be in range [0.0, 1.0)
    pub fn validate(&self) -> Result<(), String> {
        if !self.morphological_features && !self.spectral_features && !self.texture_features {
            return Err("At least one feature type must be enabled".to_string());
        }

        if self.window_size < 3 {
            return Err("Window size must be >= 3".to_string());
        }

        if self.window_size.is_multiple_of(2) {
            return Err("Window size must be odd".to_string());
        }

        if self.overlap < 0.0 || self.overlap >= 1.0 {
            return Err("Overlap must be in range [0.0, 1.0)".to_string());
        }

        Ok(())
    }
}
