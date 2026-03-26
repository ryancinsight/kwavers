//! Fusion algorithms for combining multi-modal imaging data.
//!
//! This module implements various fusion strategies for combining data from
//! multiple imaging modalities, including weighted averaging, probabilistic
//! fusion, feature-based methods, and machine learning approaches.
//!
//! # Implementation Status
//!
//! Currently implemented:
//! - Weighted average fusion with registration and resampling
//! - Probabilistic fusion with uncertainty modeling (basic Bayesian fusion)
//! - Feature-based fusion using complementary tissue properties
//! - Maximum likelihood estimation fusion with EM algorithm
//!
//! Planned but not yet implemented:
//! - Deep learning fusion with U-Net architectures and attention mechanisms
//! - Advanced feature extraction with tissue classification and correlation analysis
//! - Real-time fusion for streaming multi-modal data
//! - Comprehensive quality assessment and fusion confidence metrics
//!
//! # Theoretical Foundation
//!
//! Information fusion: H(fused) ≤ min(H_i) where H is entropy, for complementary modalities
//! Bayesian fusion: P(fused|data) ∝ ∏ P(data_i|fused) P(fused) for independent measurements
//!
//! # References
//!
//! - Blum & Liu (2006) Multi-Sensor Image Fusion
//! - Ma et al. (2019) Medical Image Fusion

mod deep_learning;
mod feature_based;
mod maximum_likelihood;
mod probabilistic;
pub(crate) mod utils;
mod weighted_average;

use super::config::{FusionConfig, FusionMethod};
use super::quality;
use super::types::{FusedImageResult, RegisteredModality};
use crate::core::error::{KwaversError, KwaversResult};
use ndarray::Array3;
use std::collections::HashMap;

/// Multi-modal imaging fusion processor
#[derive(Debug)]
pub struct MultiModalFusion {
    /// Fusion configuration
    pub(crate) config: FusionConfig,
    /// Registered modality data
    pub(crate) registered_data: HashMap<String, RegisteredModality>,
}

impl MultiModalFusion {
    /// Create a new multi-modal fusion processor
    pub fn new(config: FusionConfig) -> Self {
        Self {
            config,
            registered_data: HashMap::new(),
        }
    }

    /// Register ultrasound data for multi-modal image fusion
    pub fn register_ultrasound(&mut self, ultrasound_data: &Array3<f64>) -> KwaversResult<()> {
        // Compute quality score from signal-to-noise ratio estimate
        let mean = ultrasound_data.mean().unwrap_or(0.0);
        let variance = ultrasound_data
            .mapv(|v| (v - mean).powi(2))
            .mean()
            .unwrap_or(1.0);
        let snr = if variance > 1e-15 {
            mean.abs() / variance.sqrt()
        } else {
            0.0
        };
        // Map SNR to 0..1 quality score via sigmoid
        let quality_score = 1.0 / (1.0 + (-0.5 * (snr - 5.0)).exp());

        let registered_data = RegisteredModality {
            data: ultrasound_data.clone(),
            quality_score,
        };

        self.registered_data
            .insert("ultrasound".to_string(), registered_data);
        Ok(())
    }

    /// Register photoacoustic data for fusion
    pub fn register_photoacoustic(
        &mut self,
        reconstructed_image: &Array3<f64>,
    ) -> KwaversResult<()> {
        let registered_data = RegisteredModality {
            data: reconstructed_image.clone(),
            quality_score: quality::compute_pa_quality(reconstructed_image),
        };

        self.registered_data
            .insert("photoacoustic".to_string(), registered_data);
        Ok(())
    }

    /// Register elastography data for fusion
    pub fn register_elastography(
        &mut self,
        elasticity_map: &crate::domain::imaging::ultrasound::elastography::ElasticityMap,
    ) -> KwaversResult<()> {
        let registered_data = RegisteredModality {
            data: elasticity_map.shear_modulus.clone(),
            quality_score: quality::compute_elastography_quality(elasticity_map),
        };

        self.registered_data
            .insert("elastography".to_string(), registered_data);
        Ok(())
    }

    /// Register optical/sonoluminescence data for fusion
    pub fn register_optical(
        &mut self,
        optical_intensity: &Array3<f64>,
        wavelength: f64,
    ) -> KwaversResult<()> {
        // Validate optical data represents intensity/emission
        if optical_intensity.iter().any(|&x| x < 0.0) {
            return Err(KwaversError::Validation(
                crate::core::error::ValidationError::ConstraintViolation {
                    message: "Optical intensity values must be non-negative".to_string(),
                },
            ));
        }

        let registered_data = RegisteredModality {
            data: optical_intensity.clone(),
            quality_score: quality::compute_optical_quality(optical_intensity, wavelength),
        };

        self.registered_data.insert(
            format!("optical_{}nm", (wavelength * 1e9) as usize),
            registered_data,
        );
        Ok(())
    }

    /// Get the number of registered modalities
    #[must_use]
    pub fn num_registered_modalities(&self) -> usize {
        self.registered_data.len()
    }

    /// Check if a modality is registered
    #[must_use]
    pub fn is_modality_registered(&self, modality_name: &str) -> bool {
        self.registered_data.contains_key(modality_name)
    }

    /// Perform multi-modal fusion
    ///
    /// Combines all registered modalities according to the configured fusion method.
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - Fewer than two modalities are registered
    /// - Registration or resampling fails
    /// - Configuration parameters are invalid
    pub fn fuse(&self) -> KwaversResult<FusedImageResult> {
        if self.registered_data.len() < 2 {
            return Err(KwaversError::Validation(
                crate::core::error::ValidationError::ConstraintViolation {
                    message: "At least two modalities required for fusion".to_string(),
                },
            ));
        }

        // Apply fusion method
        let fused_result = match self.config.fusion_method {
            FusionMethod::WeightedAverage => weighted_average::fuse_weighted_average(self),
            FusionMethod::FeatureBased => feature_based::fuse_feature_based(self),
            FusionMethod::Probabilistic => probabilistic::fuse_probabilistic(self),
            FusionMethod::DeepFusion => deep_learning::fuse_deep_learning(self),
            FusionMethod::MaximumLikelihood => maximum_likelihood::fuse_maximum_likelihood(self),
            FusionMethod::MaximumIntensity => {
                return Err(KwaversError::NotImplemented(
                    "Maximum Intensity Projection (MIP) fusion not yet implemented".into(),
                ))
            }
            FusionMethod::MinimumIntensity => {
                return Err(KwaversError::NotImplemented(
                    "Minimum Intensity Projection (MinIP) fusion not yet implemented".into(),
                ))
            }
            FusionMethod::PCA => {
                return Err(KwaversError::NotImplemented(
                    "PCA-based image fusion not yet implemented".into(),
                ))
            }
        }?;

        Ok(fused_result)
    }

    /// Extract tissue properties from fused imaging data
    ///
    /// Convenience method that delegates to the `properties` module's
    /// `extract_tissue_properties` function.
    ///
    /// # Arguments
    ///
    /// * `fused_result` - Fused imaging result
    ///
    /// # Returns
    ///
    /// HashMap mapping property names to 3D spatial maps
    pub fn extract_tissue_properties(
        &self,
        fused_result: &FusedImageResult,
    ) -> HashMap<String, Array3<f64>> {
        super::properties::extract_tissue_properties(fused_result)
    }
}

#[cfg(test)]
mod tests;
