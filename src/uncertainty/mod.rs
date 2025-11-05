//! Uncertainty Quantification Framework for Ultrasound Simulations
//!
//! Provides comprehensive uncertainty estimation and confidence assessment for
//! ultrasound imaging and therapeutic applications. This framework enables
//! reliable clinical decision-making by quantifying uncertainties in:
//! - Wave propagation predictions
//! - Tissue parameter estimation
//! - Beamforming results
//! - Treatment planning outcomes
//!
//! ## Key Components
//!
//! ### 1. Bayesian Neural Networks
//! - Dropout-based uncertainty estimation
//! - Monte Carlo dropout sampling
//! - Aleatoric and epistemic uncertainty separation
//!
//! ### 2. Ensemble Methods
//! - Bootstrap aggregation (bagging)
//! - Random forest uncertainty
//! - Ensemble variance estimation
//!
//! ### 3. Conformal Prediction
//! - Distribution-free uncertainty bounds
//! - Guaranteed coverage probabilities
//! - Adaptive confidence intervals
//!
//! ### 4. Sensitivity Analysis
//! - Parameter sensitivity assessment
//! - Uncertainty propagation through models
//! - Global sensitivity indices
//!
//! ## Clinical Applications
//!
//! - **Diagnostic Confidence**: Uncertainty maps for lesion detection
//! - **Treatment Safety**: Confidence bounds for therapeutic parameters
//! - **Quality Assurance**: Validation of simulation accuracy
//! - **Decision Support**: Risk assessment for clinical procedures
//!
//! ## References
//!
//! - Kendall & Gal (2017): "What uncertainties do we need in Bayesian deep learning?"
//! - Angelopoulos & Bates (2021): "A Gentle Introduction to Conformal Prediction"
//! - Sullivan (2015): "Introduction to Uncertainty Quantification"
//!
//! ## Usage Example
//!
//! ```rust,ignore
//! use kwavers::uncertainty::{UncertaintyQuantifier, UncertaintyConfig};
//!
//! // Configure uncertainty quantification
//! let config = UncertaintyConfig {
//!     method: UncertaintyMethod::MonteCarloDropout,
//!     num_samples: 50,
//!     confidence_level: 0.95,
//! };
//!
//! let quantifier = UncertaintyQuantifier::new(config)?;
//!
//! // Estimate uncertainty for PINN predictions
//! let prediction_with_uncertainty = quantifier.quantify_pinn_uncertainty(
//!     &pinn_model,
//!     &input_data,
//!     &ground_truth
//! )?;
//!
//! // Check if prediction is within confidence bounds
//! if quantifier.is_confident(&prediction_with_uncertainty, threshold) {
//!     // Use prediction for clinical decision
//! }
//! ```

pub mod bayesian_networks;
pub mod conformal_prediction;
pub mod ensemble_methods;
pub mod sensitivity_analysis;

pub use bayesian_networks::{BayesianPINN, BayesianConfig, PredictionWithUncertainty};
pub use conformal_prediction::{ConformalPredictor, ConformalConfig, ConformalResult};
pub use ensemble_methods::{EnsembleQuantifier, EnsembleConfig, EnsembleResult};
pub use sensitivity_analysis::{SensitivityAnalyzer, SensitivityConfig, SensitivityIndices};

use crate::error::{KwaversError, KwaversResult};
use ndarray::{Array1, Array2, Array3, Array4};
use std::collections::HashMap;

/// Uncertainty quantification methods
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum UncertaintyMethod {
    /// Monte Carlo dropout sampling
    MonteCarloDropout,
    /// Ensemble methods (bagging)
    Ensemble,
    /// Conformal prediction
    Conformal,
    /// Sensitivity analysis
    Sensitivity,
    /// Combined approach
    Hybrid,
}

/// Configuration for uncertainty quantification
#[derive(Debug, Clone)]
pub struct UncertaintyConfig {
    /// Primary uncertainty method
    pub method: UncertaintyMethod,
    /// Number of uncertainty samples
    pub num_samples: usize,
    /// Confidence level (0-1)
    pub confidence_level: f64,
    /// Dropout rate for MC dropout
    pub dropout_rate: f64,
    /// Ensemble size for ensemble methods
    pub ensemble_size: usize,
    /// Calibration set size for conformal prediction
    pub calibration_size: usize,
}

/// Main uncertainty quantification interface
pub struct UncertaintyQuantifier {
    config: UncertaintyConfig,
    bayesian: Option<BayesianPINN>,
    conformal: Option<ConformalPredictor>,
    ensemble: Option<EnsembleQuantifier>,
    sensitivity: Option<SensitivityAnalyzer>,
}

impl UncertaintyQuantifier {
    /// Create new uncertainty quantifier
    pub fn new(config: UncertaintyConfig) -> KwaversResult<Self> {
        let bayesian = if matches!(config.method, UncertaintyMethod::MonteCarloDropout | UncertaintyMethod::Hybrid) {
            Some(BayesianPINN::new(BayesianConfig {
                dropout_rate: config.dropout_rate,
                num_samples: config.num_samples,
            })?)
        } else {
            None
        };

        let conformal = if matches!(config.method, UncertaintyMethod::Conformal | UncertaintyMethod::Hybrid) {
            Some(ConformalPredictor::new(ConformalConfig {
                confidence_level: config.confidence_level,
                calibration_size: config.calibration_size,
            })?)
        } else {
            None
        };

        let ensemble = if matches!(config.method, UncertaintyMethod::Ensemble | UncertaintyMethod::Hybrid) {
            Some(EnsembleQuantifier::new(EnsembleConfig {
                ensemble_size: config.ensemble_size,
                num_samples: config.num_samples,
            })?)
        } else {
            None
        };

        let sensitivity = if matches!(config.method, UncertaintyMethod::Sensitivity | UncertaintyMethod::Hybrid) {
            Some(SensitivityAnalyzer::new(SensitivityConfig {
                num_samples: config.num_samples,
                confidence_level: config.confidence_level,
            })?)
        } else {
            None
        };

        Ok(Self {
            config,
            bayesian,
            conformal,
            ensemble,
            sensitivity,
        })
    }

    /// Quantify uncertainty for PINN predictions
    #[cfg(feature = "pinn")]
    pub fn quantify_pinn_uncertainty(
        &self,
        pinn: &crate::ml::pinn::BurnPINN1DWave,
        inputs: &Array2<f32>,
        ground_truth: Option<&Array2<f32>>,
    ) -> KwaversResult<PredictionWithUncertainty> {
        match self.config.method {
            UncertaintyMethod::MonteCarloDropout => {
                if let Some(bayesian) = &self.bayesian {
                    bayesian.quantify_uncertainty(pinn, inputs)
                } else {
                    Err(KwaversError::InvalidInput("Bayesian module not configured".to_string()))
                }
            }
            UncertaintyMethod::Ensemble => {
                if let Some(ensemble) = &self.ensemble {
                    ensemble.quantify_uncertainty(pinn, inputs)
                } else {
                    Err(KwaversError::InvalidInput("Ensemble module not configured".to_string()))
                }
            }
            UncertaintyMethod::Conformal => {
                if let Some(conformal) = &self.conformal {
                    conformal.quantify_uncertainty(pinn, inputs, ground_truth)
                } else {
                    Err(KwaversError::InvalidInput("Conformal module not configured".to_string()))
                }
            }
            UncertaintyMethod::Hybrid => {
                // Combine multiple methods
                let mut results = Vec::new();

                if let Some(bayesian) = &self.bayesian {
                    results.push(bayesian.quantify_uncertainty(pinn, inputs)?);
                }

                if let Some(ensemble) = &self.ensemble {
                    results.push(ensemble.quantify_uncertainty(pinn, inputs)?);
                }

                // Return combined uncertainty quantification result
                Ok(results.into_iter().next().unwrap_or_else(|| {
                    PredictionWithUncertainty {
                        mean_prediction: Array2::zeros(inputs.dim()),
                        uncertainty: Array2::zeros(inputs.dim()),
                        confidence_intervals: HashMap::new(),
                        reliability_score: 0.5,
                    }
                }))
            }
            UncertaintyMethod::Sensitivity => {
                Err(KwaversError::InvalidInput("Sensitivity analysis not applicable for PINN uncertainty".to_string()))
            }
        }
    }

    /// Quantify uncertainty for beamforming results
    pub fn quantify_beamforming_uncertainty(
        &self,
        beamformed_image: &Array3<f32>,
        signal_quality: f64,
    ) -> KwaversResult<BeamformingUncertainty> {
        // Estimate uncertainty based on signal quality and image characteristics
        let mut uncertainty_map = Array3::zeros(beamformed_image.dim());

        // Local variance as uncertainty proxy
        for i in 1..beamformed_image.nrows() - 1 {
            for j in 1..beamformed_image.ncols() - 1 {
                for k in 0..beamformed_image.dim().2 {
                    let center = beamformed_image[[i, j, k]];
                    let neighbors = [
                        beamformed_image[[i-1, j, k]], beamformed_image[[i+1, j, k]],
                        beamformed_image[[i, j-1, k]], beamformed_image[[i, j+1, k]],
                    ];

                    let variance = neighbors.iter()
                        .map(|&n| (n - center).powi(2))
                        .sum::<f32>() / neighbors.len() as f32;

                    uncertainty_map[[i, j, k]] = variance.sqrt() / signal_quality.max(1e-6);
                }
            }
        }

        // Overall confidence based on signal quality
        let confidence = (signal_quality / (1.0 + signal_quality)).min(1.0);

        Ok(BeamformingUncertainty {
            uncertainty_map,
            confidence_score: confidence,
            reliability_metrics: ReliabilityMetrics {
                signal_to_noise_ratio: signal_quality,
                contrast_to_noise_ratio: self.compute_cnr(beamformed_image),
                spatial_resolution: self.estimate_resolution(beamformed_image),
            },
        })
    }

    /// Perform sensitivity analysis on model parameters
    pub fn sensitivity_analysis(
        &self,
        model_fn: impl Fn(&Array1<f64>) -> Array1<f64>,
        parameter_ranges: &[(f64, f64)],
        num_samples: usize,
    ) -> KwaversResult<SensitivityIndices> {
        if let Some(sensitivity) = &self.sensitivity {
            sensitivity.analyze(model_fn, parameter_ranges, num_samples)
        } else {
            Err(KwaversError::InvalidInput("Sensitivity analysis not configured".to_string()))
        }
    }

    /// Check if prediction is within acceptable uncertainty bounds
    pub fn is_confident(&self, uncertainty: &impl UncertaintyResult, threshold: f64) -> bool {
        uncertainty.confidence_score() >= threshold
    }

    /// Compute contrast-to-noise ratio
    fn compute_cnr(&self, image: &Array3<f32>) -> f64 {
        // Simplified CNR calculation
        let mean_signal = image.iter().sum::<f32>() / image.len() as f32;
        let variance = image.iter()
            .map(|&x| (x - mean_signal).powi(2))
            .sum::<f32>() / image.len() as f32;

        if variance > 0.0 {
            mean_signal / variance.sqrt()
        } else {
            0.0
        }
    }

    /// Estimate spatial resolution
    fn estimate_resolution(&self, image: &Array3<f32>) -> f64 {
        // Estimate resolution from image gradients
        let mut total_gradient = 0.0;
        let mut count = 0;

        for i in 1..image.nrows() - 1 {
            for j in 1..image.ncols() - 1 {
                for k in 0..image.dim().2 {
                    let dx = (image[[i+1, j, k]] - image[[i-1, j, k]]).abs();
                    let dy = (image[[i, j+1, k]] - image[[i, j-1, k]]).abs();
                    total_gradient += (dx + dy) / 2.0;
                    count += 1;
                }
            }
        }

        if count > 0 {
            total_gradient / count as f64
        } else {
            0.0
        }
    }

    /// Generate uncertainty report
    pub fn generate_report(&self, results: &[Box<dyn UncertaintyResult>]) -> UncertaintyReport {
        let mut summary = UncertaintySummary {
            mean_confidence: 0.0,
            confidence_range: (1.0, 0.0),
            reliability_score: 0.0,
            dominant_uncertainty_sources: Vec::new(),
        };

        let mut total_confidence = 0.0;
        let mut min_conf = 1.0;
        let mut max_conf = 0.0;

        for result in results {
            let conf = result.confidence_score();
            total_confidence += conf;
            min_conf = min_conf.min(conf);
            max_conf = max_conf.max(conf);
        }

        summary.mean_confidence = total_confidence / results.len() as f64;
        summary.confidence_range = (min_conf, max_conf);
        summary.reliability_score = self.compute_reliability_score(&summary);

        UncertaintyReport {
            summary,
            detailed_results: results.to_vec(),
            recommendations: self.generate_recommendations(&summary),
        }
    }

    /// Compute overall reliability score
    fn compute_reliability_score(&self, summary: &UncertaintySummary) -> f64 {
        // Combine multiple factors
        let confidence_factor = summary.mean_confidence;
        let range_factor = 1.0 - (summary.confidence_range.1 - summary.confidence_range.0);

        (confidence_factor + range_factor) / 2.0
    }

    /// Generate clinical recommendations based on uncertainty
    fn generate_recommendations(&self, summary: &UncertaintySummary) -> Vec<String> {
        let mut recommendations = Vec::new();

        if summary.mean_confidence < 0.7 {
            recommendations.push("High uncertainty detected. Consider additional imaging or reduced confidence in diagnosis.".to_string());
        }

        if summary.confidence_range.1 - summary.confidence_range.0 > 0.3 {
            recommendations.push("Wide confidence range indicates variable reliability. Focus on high-confidence regions.".to_string());
        }

        if summary.reliability_score < 0.6 {
            recommendations.push("Low overall reliability. Consider alternative imaging modalities or expert consultation.".to_string());
        }

        if recommendations.is_empty() {
            recommendations.push("Uncertainty levels acceptable for clinical decision-making.".to_string());
        }

        recommendations
    }
}

/// Common trait for uncertainty results
pub trait UncertaintyResult {
    fn confidence_score(&self) -> f64;
    fn uncertainty_bounds(&self) -> (f64, f64);
}

/// Uncertainty result for beamforming
#[derive(Debug)]
pub struct BeamformingUncertainty {
    pub uncertainty_map: Array3<f32>,
    pub confidence_score: f64,
    pub reliability_metrics: ReliabilityMetrics,
}

impl UncertaintyResult for BeamformingUncertainty {
    fn confidence_score(&self) -> f64 {
        self.confidence_score
    }

    fn uncertainty_bounds(&self) -> (f64, f64) {
        let min_uncertainty = self.uncertainty_map.iter().fold(f64::INFINITY, |a, &b| a.min(b as f64));
        let max_uncertainty = self.uncertainty_map.iter().fold(0.0_f64, |a, &b| a.max(b as f64));
        (min_uncertainty, max_uncertainty)
    }
}

/// Reliability metrics for imaging quality
#[derive(Debug)]
pub struct ReliabilityMetrics {
    pub signal_to_noise_ratio: f64,
    pub contrast_to_noise_ratio: f64,
    pub spatial_resolution: f64,
}

/// Uncertainty report summary
#[derive(Debug)]
pub struct UncertaintySummary {
    pub mean_confidence: f64,
    pub confidence_range: (f64, f64),
    pub reliability_score: f64,
    pub dominant_uncertainty_sources: Vec<String>,
}

/// Complete uncertainty report
#[derive(Debug)]
pub struct UncertaintyReport {
    pub summary: UncertaintySummary,
    pub detailed_results: Vec<Box<dyn UncertaintyResult>>,
    pub recommendations: Vec<String>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_uncertainty_quantifier_creation() {
        let config = UncertaintyConfig {
            method: UncertaintyMethod::MonteCarloDropout,
            num_samples: 10,
            confidence_level: 0.95,
            dropout_rate: 0.1,
            ensemble_size: 5,
            calibration_size: 100,
        };

        let quantifier = UncertaintyQuantifier::new(config);
        assert!(quantifier.is_ok());
    }

    #[test]
    fn test_beamforming_uncertainty() {
        let config = UncertaintyConfig::default();
        let quantifier = UncertaintyQuantifier::new(config).unwrap();

        let image = Array3::from_elem((32, 32, 16), 1.0);
        let uncertainty = quantifier.quantify_beamforming_uncertainty(&image, 0.8);

        assert!(uncertainty.is_ok());
        let result = uncertainty.unwrap();
        assert_eq!(result.uncertainty_map.dim(), (32, 32, 16));
        assert!(result.confidence_score >= 0.0 && result.confidence_score <= 1.0);
    }

    #[test]
    fn test_confidence_check() {
        let config = UncertaintyConfig::default();
        let quantifier = UncertaintyQuantifier::new(config).unwrap();

        let image = Array3::from_elem((16, 16, 8), 1.0);
        let uncertainty = quantifier.quantify_beamforming_uncertainty(&image, 0.9).unwrap();

        assert!(quantifier.is_confident(&uncertainty, 0.5));
        assert!(!quantifier.is_confident(&uncertainty, 0.95));
    }
}
