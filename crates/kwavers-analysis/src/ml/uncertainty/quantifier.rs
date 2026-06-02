//! Main uncertainty quantification interface.

#[cfg(feature = "pinn")]
use super::bayesian_networks::MlPredictionWithUncertainty;
use super::bayesian_networks::{BayesianConfig, MlBayesianPINN};
use super::conformal_prediction::{ConformalConfig, MlConformalPredictor};
use super::ensemble_methods::{EnsembleConfig, EnsembleQuantifier};
use super::sensitivity_analysis::{SensitivityAnalyzer, SensitivityConfig, SensitivityIndices};
use super::types::{
    BeamformingUncertainty, MlUncertaintyConfig, MlUncertaintyMethod, ReliabilityMetrics,
    UncertaintyReport, UncertaintyResult, UncertaintySummary,
};
use kwavers_core::error::{KwaversError, KwaversResult};
#[cfg(feature = "pinn")]
use burn::tensor::backend::Backend;
#[cfg(feature = "pinn")]
use ndarray::{Array1, Array2, Array3};
#[cfg(not(feature = "pinn"))]
use ndarray::{Array1, Array3};
#[cfg(feature = "pinn")]
use std::collections::HashMap;

/// Main uncertainty quantification interface.
#[derive(Debug)]
pub struct UncertaintyQuantifier {
    pub(super) _config: MlUncertaintyConfig,
    pub(super) _bayesian: Option<MlBayesianPINN>,
    pub(super) _conformal: Option<MlConformalPredictor>,
    pub(super) _ensemble: Option<EnsembleQuantifier>,
    pub(super) _sensitivity: Option<SensitivityAnalyzer>,
}

impl UncertaintyQuantifier {
    /// Create new uncertainty quantifier.
    /// # Errors
    /// - Propagates any [`KwaversError`] returned by called functions.
    ///
    pub fn new(config: MlUncertaintyConfig) -> KwaversResult<Self> {
        let bayesian = if matches!(
            config.method,
            MlUncertaintyMethod::MonteCarloDropout | MlUncertaintyMethod::Hybrid
        ) {
            Some(MlBayesianPINN::new(BayesianConfig {
                dropout_rate: config.dropout_rate,
                num_samples: config.num_samples,
            })?)
        } else {
            None
        };

        let conformal = if matches!(
            config.method,
            MlUncertaintyMethod::Conformal | MlUncertaintyMethod::Hybrid
        ) {
            Some(MlConformalPredictor::new(ConformalConfig {
                confidence_level: config.confidence_level,
                calibration_size: config.calibration_size,
            })?)
        } else {
            None
        };

        let ensemble = if matches!(
            config.method,
            MlUncertaintyMethod::Ensemble | MlUncertaintyMethod::Hybrid
        ) {
            Some(EnsembleQuantifier::new(EnsembleConfig {
                ensemble_size: config.ensemble_size,
                num_samples: config.num_samples,
            })?)
        } else {
            None
        };

        let sensitivity = if matches!(
            config.method,
            MlUncertaintyMethod::Sensitivity | MlUncertaintyMethod::Hybrid
        ) {
            Some(SensitivityAnalyzer::new(SensitivityConfig {
                num_samples: config.num_samples,
                confidence_level: config.confidence_level,
            })?)
        } else {
            None
        };

        Ok(Self {
            _config: config,
            _bayesian: bayesian,
            _conformal: conformal,
            _ensemble: ensemble,
            _sensitivity: sensitivity,
        })
    }

    /// Quantify uncertainty for PINN predictions.
    /// # Errors
    /// - Returns [`KwaversError::InvalidInput`] if the precondition for invalid or out-of-range input parameters is violated.
    /// - Propagates any [`KwaversError`] returned by called functions.
    ///
    #[cfg(feature = "pinn")]
    pub fn quantify_pinn_uncertainty<B: Backend>(
        &self,
        pinn: &kwavers_solver::inverse::pinn::ml::BurnPINN1DWave<B>,
        inputs: &Array2<f32>,
        ground_truth: Option<&Array2<f32>>,
    ) -> KwaversResult<MlPredictionWithUncertainty> {
        match self._config.method {
            MlUncertaintyMethod::MonteCarloDropout => {
                if let Some(bayesian) = &self._bayesian {
                    bayesian.quantify_uncertainty(pinn, inputs)
                } else {
                    Err(KwaversError::InvalidInput(
                        "Bayesian module not configured".to_string(),
                    ))
                }
            }
            MlUncertaintyMethod::Ensemble => {
                if let Some(ensemble) = &self._ensemble {
                    ensemble.quantify_uncertainty(pinn, inputs)
                } else {
                    Err(KwaversError::InvalidInput(
                        "Ensemble module not configured".to_string(),
                    ))
                }
            }
            MlUncertaintyMethod::Conformal => {
                if let Some(conformal) = &self._conformal {
                    conformal.quantify_uncertainty(pinn, inputs, ground_truth)
                } else {
                    Err(KwaversError::InvalidInput(
                        "Conformal module not configured".to_string(),
                    ))
                }
            }
            MlUncertaintyMethod::Hybrid => {
                let mut results = Vec::new();

                if let Some(bayesian) = &self._bayesian {
                    results.push(bayesian.quantify_uncertainty(pinn, inputs)?);
                }

                if let Some(ensemble) = &self._ensemble {
                    results.push(ensemble.quantify_uncertainty(pinn, inputs)?);
                }

                Ok(results
                    .into_iter()
                    .next()
                    .unwrap_or_else(|| MlPredictionWithUncertainty {
                        mean_prediction: Array2::zeros(inputs.dim()),
                        uncertainty: Array2::zeros(inputs.dim()),
                        confidence_intervals: HashMap::new(),
                        reliability_score: 0.5,
                    }))
            }
            MlUncertaintyMethod::Sensitivity => Err(KwaversError::InvalidInput(
                "Sensitivity analysis not applicable for PINN uncertainty".to_string(),
            )),
        }
    }

    /// Quantify uncertainty for beamforming results.
    /// # Errors
    /// - Returns [`Err`] if an internal constraint is violated.
    ///
    pub fn quantify_beamforming_uncertainty(
        &self,
        beamformed_image: &Array3<f32>,
        signal_quality: f64,
    ) -> KwaversResult<BeamformingUncertainty> {
        let mut uncertainty_map = Array3::zeros(beamformed_image.dim());

        for i in 1..beamformed_image.dim().0 - 1 {
            for j in 1..beamformed_image.dim().1 - 1 {
                for k in 0..beamformed_image.dim().2 {
                    let center = beamformed_image[[i, j, k]];
                    let neighbors = [
                        beamformed_image[[i - 1, j, k]],
                        beamformed_image[[i + 1, j, k]],
                        beamformed_image[[i, j - 1, k]],
                        beamformed_image[[i, j + 1, k]],
                    ];

                    let variance = neighbors.iter().map(|&n| (n - center).powi(2)).sum::<f32>()
                        / neighbors.len() as f32;

                    uncertainty_map[[i, j, k]] =
                        (variance.sqrt() as f64 / signal_quality.max(1e-6)) as f32;
                }
            }
        }

        let confidence = signal_quality.clamp(0.0, 1.0);

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

    /// Perform sensitivity analysis on model parameters.
    /// # Errors
    /// - Returns [`KwaversError::InvalidInput`] if the precondition for invalid or out-of-range input parameters is violated.
    ///
    pub fn sensitivity_analysis(
        &self,
        model_fn: impl Fn(&Array1<f64>) -> Array1<f64>,
        parameter_ranges: &[(f64, f64)],
        num_samples: usize,
    ) -> KwaversResult<SensitivityIndices> {
        if let Some(sensitivity) = &self._sensitivity {
            sensitivity.analyze(model_fn, parameter_ranges, num_samples)
        } else {
            Err(KwaversError::InvalidInput(
                "Sensitivity analysis not configured".to_owned(),
            ))
        }
    }

    /// Check if prediction is within acceptable uncertainty bounds.
    #[must_use]
    pub fn is_confident(&self, uncertainty: &impl UncertaintyResult, threshold: f64) -> bool {
        uncertainty.confidence_score() >= threshold
    }

    fn compute_cnr(&self, image: &Array3<f32>) -> f64 {
        let mean_signal = image.iter().sum::<f32>() / image.len() as f32;
        let variance = image
            .iter()
            .map(|&x| (x - mean_signal).powi(2))
            .sum::<f32>()
            / image.len() as f32;

        if variance > 0.0 {
            (mean_signal / variance.sqrt()) as f64
        } else {
            0.0
        }
    }

    fn estimate_resolution(&self, image: &Array3<f32>) -> f64 {
        let mut total_gradient = 0.0;
        let mut count = 0;

        for i in 1..image.dim().0 - 1 {
            for j in 1..image.dim().1 - 1 {
                for k in 0..image.dim().2 {
                    let dx = (image[[i + 1, j, k]] - image[[i - 1, j, k]]).abs();
                    let dy = (image[[i, j + 1, k]] - image[[i, j - 1, k]]).abs();
                    total_gradient += (dx + dy) / 2.0;
                    count += 1;
                }
            }
        }

        if count > 0 {
            total_gradient as f64 / count as f64
        } else {
            0.0
        }
    }

    /// Generate uncertainty report.
    #[must_use]
    pub fn generate_report<'a>(
        &self,
        results: &'a [Box<dyn UncertaintyResult>],
    ) -> UncertaintyReport<'a> {
        let mut summary = UncertaintySummary {
            mean_confidence: 0.0,
            confidence_range: (1.0, 0.0),
            reliability_score: 0.0,
            dominant_uncertainty_sources: Vec::new(),
        };

        let mut total_confidence = 0.0;
        let mut min_conf: f64 = 1.0;
        let mut max_conf: f64 = 0.0;

        for result in results {
            let conf = result.confidence_score();
            total_confidence += conf;
            min_conf = min_conf.min(conf);
            max_conf = max_conf.max(conf);
        }

        if results.is_empty() {
            return UncertaintyReport {
                summary,
                detailed_results: Vec::new(),
                recommendations: Vec::new(),
            };
        }
        summary.mean_confidence = total_confidence / results.len() as f64;
        summary.confidence_range = (min_conf, max_conf);
        summary.reliability_score = self.compute_reliability_score(&summary);

        let recommendations = self.generate_recommendations(&summary);

        UncertaintyReport {
            summary,
            detailed_results: results.iter().map(|r| r.as_ref()).collect(),
            recommendations,
        }
    }

    fn compute_reliability_score(&self, summary: &UncertaintySummary) -> f64 {
        let confidence_factor = summary.mean_confidence;
        let range_factor = 1.0 - (summary.confidence_range.1 - summary.confidence_range.0);
        (confidence_factor + range_factor) / 2.0
    }

    fn generate_recommendations(&self, summary: &UncertaintySummary) -> Vec<String> {
        let mut recommendations = Vec::new();

        if summary.mean_confidence < 0.7 {
            recommendations.push("High uncertainty detected. Consider additional imaging or reduced confidence in diagnosis.".to_owned());
        }

        if summary.confidence_range.1 - summary.confidence_range.0 > 0.3 {
            recommendations.push("Wide confidence range indicates variable reliability. Focus on high-confidence regions.".to_owned());
        }

        if summary.reliability_score < 0.6 {
            recommendations.push("Low overall reliability. Consider alternative imaging modalities or expert consultation.".to_owned());
        }

        if recommendations.is_empty() {
            recommendations
                .push("Uncertainty levels acceptable for clinical decision-making.".to_owned());
        }

        recommendations
    }
}
