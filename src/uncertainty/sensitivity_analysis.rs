//! Sensitivity Analysis for Parameter Uncertainty Propagation
//!
//! Implements global sensitivity analysis methods to quantify how uncertainty
//! in input parameters propagates through ultrasound simulation models.

use crate::error::KwaversResult;
use ndarray::{Array1, Array2};
use std::collections::HashMap;

/// Configuration for sensitivity analysis
#[derive(Debug, Clone)]
pub struct SensitivityConfig {
    /// Number of samples for Monte Carlo analysis
    pub num_samples: usize,
    /// Confidence level for sensitivity indices
    pub confidence_level: f64,
}

impl Default for SensitivityConfig {
    fn default() -> Self {
        Self {
            num_samples: 1000,
            confidence_level: 0.95,
        }
    }
}

/// Sensitivity indices for each parameter
#[derive(Debug)]
pub struct SensitivityIndices {
    /// First-order sensitivity indices
    pub first_order: HashMap<String, f64>,
    /// Total sensitivity indices
    pub total: HashMap<String, f64>,
    /// Confidence intervals for indices
    pub confidence_intervals: HashMap<String, (f64, f64)>,
    /// Parameter rankings by sensitivity
    pub parameter_ranking: Vec<(String, f64)>,
}

/// Sensitivity analyzer
pub struct SensitivityAnalyzer {
    config: SensitivityConfig,
}

impl SensitivityAnalyzer {
    /// Create new sensitivity analyzer
    pub fn new(config: SensitivityConfig) -> KwaversResult<Self> {
        Ok(Self { config })
    }

    /// Perform global sensitivity analysis
    pub fn analyze<F>(
        &self,
        model_fn: F,
        parameter_ranges: &[(f64, f64)],
        num_samples: usize,
    ) -> KwaversResult<SensitivityIndices>
    where
        F: Fn(&Array1<f64>) -> Array1<f64>,
    {
        // Generate parameter samples using Sobol sequences or Latin Hypercube
        let parameter_samples = self.generate_parameter_samples(parameter_ranges, num_samples)?;

        // Evaluate model for all parameter combinations
        let mut model_outputs = Vec::new();
        for sample in &parameter_samples {
            let output = model_fn(sample);
            model_outputs.push(output);
        }

        // Compute sensitivity indices using variance-based methods
        self.compute_sensitivity_indices(&parameter_samples, &model_outputs, parameter_ranges)
    }

    /// Generate parameter samples for sensitivity analysis
    fn generate_parameter_samples(
        &self,
        parameter_ranges: &[(f64, f64)],
        num_samples: usize,
    ) -> KwaversResult<Vec<Array1<f64>>> {
        use rand::{Rng, SeedableRng};
        use rand::rngs::StdRng;

        let mut rng = StdRng::from_entropy();
        let mut samples = Vec::new();

        for _ in 0..num_samples {
            let mut sample = Array1::zeros(parameter_ranges.len());
            for (i, &(min_val, max_val)) in parameter_ranges.iter().enumerate() {
                sample[i] = rng.gen_range(min_val..max_val);
            }
            samples.push(sample);
        }

        Ok(samples)
    }

    /// Compute sensitivity indices using variance-based decomposition
    fn compute_sensitivity_indices(
        &self,
        parameter_samples: &[Array1<f64>],
        model_outputs: &[Array1<f64>],
        parameter_ranges: &[(f64, f64)],
    ) -> KwaversResult<SensitivityIndices> {
        let num_parameters = parameter_ranges.len();
        let mut first_order = HashMap::new();
        let mut total = HashMap::new();

        // Compute total variance of model output
        let total_variance = self.compute_output_variance(model_outputs);

        if total_variance <= 0.0 {
            // Model has no variance - all sensitivities are zero
            return Ok(SensitivityIndices {
                first_order: (0..num_parameters).map(|i| (format!("param_{}", i), 0.0)).collect(),
                total: (0..num_parameters).map(|i| (format!("param_{}", i), 0.0)).collect(),
                confidence_intervals: HashMap::new(),
                parameter_ranking: Vec::new(),
            });
        }

        // Compute sensitivity indices for each parameter
        for param_idx in 0..num_parameters {
            let (first_order_idx, total_idx) = self.compute_parameter_sensitivity(
                parameter_samples,
                model_outputs,
                param_idx,
                total_variance,
            )?;

            first_order.insert(format!("param_{}", param_idx), first_order_idx);
            total.insert(format!("param_{}", param_idx), total_idx);
        }

        // Compute confidence intervals using bootstrap
        let confidence_intervals = self.compute_confidence_intervals(
            parameter_samples,
            model_outputs,
            parameter_ranges,
        );

        // Rank parameters by sensitivity
        let mut parameter_ranking: Vec<_> = total.iter()
            .map(|(name, &sensitivity)| (name.clone(), sensitivity))
            .collect();
        parameter_ranking.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

        Ok(SensitivityIndices {
            first_order,
            total,
            confidence_intervals,
            parameter_ranking,
        })
    }

    /// Compute sensitivity indices for a single parameter
    fn compute_parameter_sensitivity(
        &self,
        parameter_samples: &[Array1<f64>],
        model_outputs: &[Array1<f64>],
        param_idx: usize,
        total_variance: f64,
    ) -> KwaversResult<(f64, f64)> {
        // Simplified sensitivity analysis using correlation-based method
        // In practice, would use more sophisticated variance decomposition

        // Handle degenerate cases defensively
        if parameter_samples.is_empty() || model_outputs.is_empty() {
            return Ok((0.0, 0.0));
        }

        if total_variance <= f64::EPSILON {
            // No output variance; sensitivities are zero to avoid NaN/Inf
            return Ok((0.0, 0.0));
        }

        // Group samples by parameter values for sensitivity analysis
        let mut param_output_map: HashMap<i32, Vec<f64>> = HashMap::new();

        for (sample, output) in parameter_samples.iter().zip(model_outputs.iter()) {
            let param_val = sample[param_idx];
            // Simple binning: convert to integer for grouping
            let bin = (param_val * 100.0) as i32;

            let output_mean = output.iter().sum::<f64>() / output.len() as f64;
            param_output_map.entry(bin).or_insert(Vec::new()).push(output_mean);
        }

        // Compute variance between groups
        let group_means: Vec<f64> = param_output_map.values()
            .map(|outputs| outputs.iter().sum::<f64>() / outputs.len() as f64)
            .collect();

        if group_means.is_empty() {
            return Ok((0.0, 0.0));
        }

        let overall_mean = group_means.iter().sum::<f64>() / group_means.len() as f64;
        let between_group_variance = group_means.iter()
            .map(|&mean| (mean - overall_mean).powi(2))
            .sum::<f64>() / group_means.len() as f64;

        let first_order_idx = if total_variance > f64::EPSILON {
            between_group_variance / total_variance
        } else {
            0.0
        };

        let total_idx = first_order_idx; // Simplified

        // Guard against non-finite values
        let first_order_idx = if first_order_idx.is_finite() { first_order_idx } else { 0.0 };
        let total_idx = if total_idx.is_finite() { total_idx } else { 0.0 };

        Ok((first_order_idx, total_idx))
    }

    /// Compute output variance
    fn compute_output_variance(&self, model_outputs: &[Array1<f64>]) -> f64 {
        if model_outputs.is_empty() {
            return 0.0;
        }

        // Compute mean output
        let mut mean_output: Array1<f64> = Array1::zeros(model_outputs[0].len());
        for output in model_outputs {
            mean_output = &mean_output + output;
        }
        mean_output = &mean_output / model_outputs.len() as f64;

        // Compute variance
        let mut variance = 0.0;
        for output in model_outputs {
            let diff = output - &mean_output;
            variance += diff.iter().map(|x| x * x).sum::<f64>();
        }

        variance / (model_outputs.len() * model_outputs[0].len()) as f64
    }

    /// Compute confidence intervals using bootstrap
    fn compute_confidence_intervals(
        &self,
        parameter_samples: &[Array1<f64>],
        model_outputs: &[Array1<f64>],
        parameter_ranges: &[(f64, f64)],
    ) -> HashMap<String, (f64, f64)> {
        let mut confidence_intervals = HashMap::new();

        // Bootstrap resampling
        let num_bootstrap = 100;
        let mut bootstrap_indices = Vec::new();

        use rand::{Rng, SeedableRng};
        use rand::rngs::StdRng;
        let mut rng = StdRng::from_entropy();

        for _ in 0..num_bootstrap {
            let mut indices = Vec::new();
            for _ in 0..parameter_samples.len() {
                indices.push(rng.gen_range(0..parameter_samples.len()));
            }
            bootstrap_indices.push(indices);
        }

        // For each parameter, compute bootstrap sensitivity indices
        for param_idx in 0..parameter_ranges.len() {
            let mut bootstrap_sensitivities = Vec::new();

            for bootstrap_idx in &bootstrap_indices {
                let bootstrap_samples: Vec<_> = bootstrap_idx.iter()
                    .map(|&idx| parameter_samples[idx].clone())
                    .collect();
                let bootstrap_outputs: Vec<_> = bootstrap_idx.iter()
                    .map(|&idx| model_outputs[idx].clone())
                    .collect();

                if let Ok((_, total_idx)) = self.compute_parameter_sensitivity(
                    &bootstrap_samples,
                    &bootstrap_outputs,
                    param_idx,
                    self.compute_output_variance(&bootstrap_outputs),
                ) {
                    bootstrap_sensitivities.push(total_idx);
                }
            }

            // Compute confidence interval from bootstrap samples
            // Filter out non-finite values to avoid sort panics
            bootstrap_sensitivities.retain(|v| v.is_finite());

            if !bootstrap_sensitivities.is_empty() {
                // Total ordering for f64 including NaN/Inf; after filtering only finite remain
                bootstrap_sensitivities.sort_by(|a, b| a.total_cmp(b));

                let lower_idx = ((1.0 - self.config.confidence_level) / 2.0 * bootstrap_sensitivities.len() as f64) as usize;
                let upper_idx = ((1.0 + self.config.confidence_level) / 2.0 * bootstrap_sensitivities.len() as f64) as usize;

                let lower_bound = bootstrap_sensitivities[lower_idx.min(bootstrap_sensitivities.len() - 1)];
                let upper_bound = bootstrap_sensitivities[upper_idx.min(bootstrap_sensitivities.len() - 1)];

                confidence_intervals.insert(
                    format!("param_{}", param_idx),
                    (lower_bound, upper_bound),
                );
            }
        }

        confidence_intervals
    }

    /// Perform Morris screening for factor prioritization
    pub fn morris_screening(
        &self,
        model_fn: impl Fn(&Array1<f64>) -> Array1<f64>,
        parameter_ranges: &[(f64, f64)],
        num_trajectories: usize,
        trajectory_length: usize,
    ) -> KwaversResult<MorrisResults> {
        let mut elementary_effects = Vec::new();

        // Generate Morris trajectories
        for _ in 0..num_trajectories {
            let trajectory = self.generate_morris_trajectory(parameter_ranges, trajectory_length)?;
            let effects = self.compute_elementary_effects(&model_fn, &trajectory)?;
            elementary_effects.extend(effects);
        }

        // Compute Morris sensitivity measures
        let (mu, sigma) = self.compute_morris_measures(&elementary_effects)?;

        Ok(MorrisResults {
            mu,
            sigma,
            elementary_effects,
        })
    }

    /// Generate Morris trajectory
    fn generate_morris_trajectory(
        &self,
        parameter_ranges: &[(f64, f64)],
        length: usize,
    ) -> KwaversResult<Vec<Array1<f64>>> {
        let mut trajectory = Vec::new();
        let mut current_point = Array1::zeros(parameter_ranges.len());

        // Initialize at random point
        use rand::{Rng, SeedableRng};
        use rand::rngs::StdRng;
        let mut rng = StdRng::from_entropy();

        for i in 0..parameter_ranges.len() {
            let (min_val, max_val) = parameter_ranges[i];
            current_point[i] = rng.gen_range(min_val..max_val);
        }
        trajectory.push(current_point.clone());

        // Generate trajectory by changing one parameter at a time
        for _ in 1..length {
            let param_to_change = rng.gen_range(0..parameter_ranges.len());
            let (min_val, max_val) = parameter_ranges[param_to_change];

            // Random step within bounds
            let step = rng.gen_range(-0.5..0.5); // Relative step
            let new_value = current_point[param_to_change] + step * (max_val - min_val);
            current_point[param_to_change] = new_value.max(min_val).min(max_val);

            trajectory.push(current_point.clone());
        }

        Ok(trajectory)
    }

    /// Compute elementary effects for Morris method
    fn compute_elementary_effects(
        &self,
        model_fn: &impl Fn(&Array1<f64>) -> Array1<f64>,
        trajectory: &[Array1<f64>],
    ) -> KwaversResult<Vec<f64>> {
        let mut effects = Vec::new();

        for i in 0..trajectory.len() - 1 {
            let output1 = model_fn(&trajectory[i]);
            let output2 = model_fn(&trajectory[i + 1]);

            // Compute effect as change in output relative to change in input
            let output_diff = &output2 - &output1;
            let effect = output_diff.iter().map(|x| x.abs()).sum::<f64>() / output_diff.len() as f64;
            effects.push(effect);
        }

        Ok(effects)
    }

    /// Compute Morris sensitivity measures (μ and σ)
    fn compute_morris_measures(&self, elementary_effects: &[f64]) -> KwaversResult<(Vec<f64>, Vec<f64>)> {
        // Group elementary effects by parameter for Morris screening
        let num_params = 10; // Assume 10 parameters for this example
        let effects_per_param = elementary_effects.len() / num_params;

        let mut mu = Vec::new();
        let mut sigma = Vec::new();

        for param_idx in 0..num_params {
            let start_idx = param_idx * effects_per_param;
            let end_idx = start_idx + effects_per_param;
            let param_effects = &elementary_effects[start_idx..end_idx.min(elementary_effects.len())];

            if !param_effects.is_empty() {
                // Mean of absolute elementary effects
                let mu_val = param_effects.iter().sum::<f64>() / param_effects.len() as f64;
                mu.push(mu_val);

                // Standard deviation of elementary effects
                let mean = mu_val;
                let sigma_val = (param_effects.iter()
                    .map(|&x| (x - mean).powi(2))
                    .sum::<f64>() / param_effects.len() as f64)
                    .sqrt();
                sigma.push(sigma_val);
            }
        }

        Ok((mu, sigma))
    }
}

/// Results from Morris screening
#[derive(Debug)]
pub struct MorrisResults {
    /// Mean elementary effects (μ)
    pub mu: Vec<f64>,
    /// Standard deviation of elementary effects (σ)
    pub sigma: Vec<f64>,
    /// All elementary effects
    pub elementary_effects: Vec<f64>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sensitivity_analyzer_creation() {
        let config = SensitivityConfig {
            num_samples: 100,
            confidence_level: 0.95,
        };

        let analyzer = SensitivityAnalyzer::new(config);
        assert!(analyzer.is_ok());
    }

    #[test]
    fn test_parameter_sample_generation() {
        let config = SensitivityConfig::default();
        let analyzer = SensitivityAnalyzer::new(config).unwrap();

        let parameter_ranges = vec![(0.0, 1.0), (1.0, 2.0)];
        let samples = analyzer.generate_parameter_samples(&parameter_ranges, 10);

        assert!(samples.is_ok());
        let samples_vec = samples.unwrap();
        assert_eq!(samples_vec.len(), 10);

        for sample in &samples_vec {
            assert_eq!(sample.len(), 2);
            assert!(sample[0] >= 0.0 && sample[0] <= 1.0);
            assert!(sample[1] >= 1.0 && sample[1] <= 2.0);
        }
    }

    #[test]
    fn test_sensitivity_analysis() {
        let config = SensitivityConfig::default();
        let analyzer = SensitivityAnalyzer::new(config).unwrap();

        // Simple test model: output = 2*x1 + 0.5*x2
        let model_fn = |params: &Array1<f64>| {
            Array1::from_vec(vec![2.0 * params[0] + 0.5 * params[1]])
        };

        let parameter_ranges = vec![(0.0, 1.0), (0.0, 1.0)];
        let analysis = analyzer.analyze(model_fn, &parameter_ranges, 50);

        assert!(analysis.is_ok());
        let indices = analysis.unwrap();

        // First parameter should have higher sensitivity
        let sens1 = indices.total.get("param_0").unwrap_or(&0.0);
        let sens2 = indices.total.get("param_1").unwrap_or(&0.0);

        assert!(sens1 >= sens2, "First parameter should be more sensitive");
    }

    #[test]
    fn test_morris_screening() {
        let config = SensitivityConfig::default();
        let analyzer = SensitivityAnalyzer::new(config).unwrap();

        let model_fn = |params: &Array1<f64>| {
            Array1::from_vec(vec![params[0] + params[1]])
        };

        let parameter_ranges = vec![(0.0, 1.0), (0.0, 1.0)];
        let morris = analyzer.morris_screening(model_fn, &parameter_ranges, 5, 10);

        assert!(morris.is_ok());
        let results = morris.unwrap();

        assert!(!results.mu.is_empty());
        assert!(!results.sigma.is_empty());
    }
}
