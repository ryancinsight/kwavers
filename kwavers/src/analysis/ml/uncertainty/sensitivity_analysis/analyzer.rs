//! `SensitivityAnalyzer` — variance-based global sensitivity analysis.

use super::config::SensitivityConfig;
use super::types::{MorrisResults, SensitivityIndices};
use crate::core::error::KwaversResult;
use ndarray::Array1;
use std::collections::HashMap;

/// Sensitivity analyzer
#[derive(Debug)]
pub struct SensitivityAnalyzer {
    pub(super) config: SensitivityConfig,
}

impl SensitivityAnalyzer {
    /// Create new sensitivity analyzer
    /// # Errors
    /// - Returns [`Err`] if an internal constraint is violated.
    ///
    pub fn new(config: SensitivityConfig) -> KwaversResult<Self> {
        Ok(Self { config })
    }

    /// Perform global sensitivity analysis
    /// # Errors
    /// - Propagates any [`KwaversError`] returned by called functions.
    ///
    pub fn analyze<F>(
        &self,
        model_fn: F,
        parameter_ranges: &[(f64, f64)],
        num_samples: usize,
    ) -> KwaversResult<SensitivityIndices>
    where
        F: Fn(&Array1<f64>) -> Array1<f64>,
    {
        let parameter_samples = self.generate_parameter_samples(parameter_ranges, num_samples)?;
        let mut model_outputs = Vec::new();
        for sample in &parameter_samples {
            let output = model_fn(sample);
            model_outputs.push(output);
        }
        self.compute_sensitivity_indices(&parameter_samples, &model_outputs, parameter_ranges)
    }

    /// Generate parameter samples for sensitivity analysis
    /// # Errors
    /// - Returns [`Err`] if an internal constraint is violated.
    ///
    pub(super) fn generate_parameter_samples(
        &self,
        parameter_ranges: &[(f64, f64)],
        num_samples: usize,
    ) -> KwaversResult<Vec<Array1<f64>>> {
        use rand::rngs::StdRng;
        use rand::{Rng, SeedableRng};

        let mut rng = StdRng::seed_from_u64(0);
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

    fn compute_sensitivity_indices(
        &self,
        parameter_samples: &[Array1<f64>],
        model_outputs: &[Array1<f64>],
        parameter_ranges: &[(f64, f64)],
    ) -> KwaversResult<SensitivityIndices> {
        let num_parameters = parameter_ranges.len();
        let mut first_order = HashMap::new();
        let mut total = HashMap::new();

        let total_variance = self.compute_output_variance(model_outputs);

        if total_variance <= 0.0 {
            return Ok(SensitivityIndices {
                first_order: (0..num_parameters)
                    .map(|i| (format!("param_{}", i), 0.0))
                    .collect(),
                total: (0..num_parameters)
                    .map(|i| (format!("param_{}", i), 0.0))
                    .collect(),
                confidence_intervals: HashMap::new(),
                parameter_ranking: Vec::new(),
            });
        }

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

        let confidence_intervals =
            self.compute_confidence_intervals(parameter_samples, model_outputs, parameter_ranges);

        let mut parameter_ranking: Vec<_> = total
            .iter()
            .map(|(name, &sensitivity)| (name.clone(), sensitivity))
            .collect();
        parameter_ranking.sort_by(|a, b| b.1.total_cmp(&a.1));

        Ok(SensitivityIndices {
            first_order,
            total,
            confidence_intervals,
            parameter_ranking,
        })
    }

    fn compute_parameter_sensitivity(
        &self,
        parameter_samples: &[Array1<f64>],
        model_outputs: &[Array1<f64>],
        param_idx: usize,
        total_variance: f64,
    ) -> KwaversResult<(f64, f64)> {
        if parameter_samples.is_empty() || model_outputs.is_empty() {
            return Ok((0.0, 0.0));
        }
        if total_variance <= f64::EPSILON {
            return Ok((0.0, 0.0));
        }
        let n = parameter_samples.len().min(model_outputs.len());
        let mut sum_x = 0.0;
        let mut sum_y = 0.0;
        for i in 0..n {
            sum_x += parameter_samples[i][param_idx];
            let output_len = model_outputs[i].len();
            let y = if output_len == 0 {
                0.0
            } else {
                model_outputs[i].iter().sum::<f64>() / output_len as f64
            };
            sum_y += y;
        }
        let mean_x = sum_x / n as f64;
        let mean_y = sum_y / n as f64;
        let mut var_x = 0.0;
        let mut cov_xy = 0.0;
        for i in 0..n {
            let x = parameter_samples[i][param_idx] - mean_x;
            let output_len_i = model_outputs[i].len();
            let y = (if output_len_i == 0 {
                0.0
            } else {
                model_outputs[i].iter().sum::<f64>() / output_len_i as f64
            }) - mean_y;
            var_x += x * x;
            cov_xy += x * y;
        }
        if var_x <= f64::EPSILON {
            return Ok((0.0, 0.0));
        }
        let first_order_idx = (cov_xy * cov_xy) / (var_x * (n as f64) * total_variance);
        let first_order_idx = if first_order_idx.is_finite() {
            first_order_idx.clamp(0.0, 1.0)
        } else {
            0.0
        };
        Ok((first_order_idx, first_order_idx))
    }

    fn compute_output_variance(&self, model_outputs: &[Array1<f64>]) -> f64 {
        if model_outputs.is_empty() {
            return 0.0;
        }
        let mut mean_output: Array1<f64> = Array1::zeros(model_outputs[0].len());
        for output in model_outputs {
            mean_output = &mean_output + output;
        }
        mean_output = &mean_output / model_outputs.len() as f64;
        let mut variance = 0.0;
        for output in model_outputs {
            let diff = output - &mean_output;
            variance += diff.iter().map(|x| x * x).sum::<f64>();
        }
        let denominator = model_outputs.len() * model_outputs[0].len();
        if denominator == 0 {
            return 0.0;
        }
        variance / denominator as f64
    }

    fn compute_confidence_intervals(
        &self,
        parameter_samples: &[Array1<f64>],
        model_outputs: &[Array1<f64>],
        parameter_ranges: &[(f64, f64)],
    ) -> HashMap<String, (f64, f64)> {
        let mut confidence_intervals = HashMap::new();
        let num_bootstrap = 100;
        let mut bootstrap_indices = Vec::new();

        use rand::rngs::StdRng;
        use rand::{Rng, SeedableRng};
        let mut rng = StdRng::from_entropy();

        for _ in 0..num_bootstrap {
            let mut indices = Vec::new();
            for _ in 0..parameter_samples.len() {
                indices.push(rng.gen_range(0..parameter_samples.len()));
            }
            bootstrap_indices.push(indices);
        }

        for param_idx in 0..parameter_ranges.len() {
            let mut bootstrap_sensitivities = Vec::new();
            for bootstrap_idx in &bootstrap_indices {
                let bootstrap_samples: Vec<_> = bootstrap_idx
                    .iter()
                    .map(|&idx| parameter_samples[idx].clone())
                    .collect();
                let bootstrap_outputs: Vec<_> = bootstrap_idx
                    .iter()
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
            bootstrap_sensitivities.retain(|v| v.is_finite());
            if !bootstrap_sensitivities.is_empty() {
                bootstrap_sensitivities.sort_by(|a, b| a.total_cmp(b));
                let lower_idx = ((1.0 - self.config.confidence_level) / 2.0
                    * bootstrap_sensitivities.len() as f64)
                    as usize;
                let upper_idx = ((1.0 + self.config.confidence_level) / 2.0
                    * bootstrap_sensitivities.len() as f64)
                    as usize;
                let lower_bound =
                    bootstrap_sensitivities[lower_idx.min(bootstrap_sensitivities.len() - 1)];
                let upper_bound =
                    bootstrap_sensitivities[upper_idx.min(bootstrap_sensitivities.len() - 1)];
                confidence_intervals
                    .insert(format!("param_{}", param_idx), (lower_bound, upper_bound));
            }
        }
        confidence_intervals
    }

    /// Perform Morris screening for factor prioritization
    /// # Errors
    /// - Propagates any [`KwaversError`] returned by called functions.
    ///
    pub fn morris_screening(
        &self,
        model_fn: impl Fn(&Array1<f64>) -> Array1<f64>,
        parameter_ranges: &[(f64, f64)],
        num_trajectories: usize,
        trajectory_length: usize,
    ) -> KwaversResult<MorrisResults> {
        let mut elementary_effects = Vec::new();
        for _ in 0..num_trajectories {
            let trajectory =
                self.generate_morris_trajectory(parameter_ranges, trajectory_length)?;
            let effects = self.compute_elementary_effects(&model_fn, &trajectory)?;
            elementary_effects.extend(effects);
        }
        let (mu, sigma) = self.compute_morris_measures(&elementary_effects)?;
        Ok(MorrisResults {
            mu,
            sigma,
            elementary_effects,
        })
    }

    fn generate_morris_trajectory(
        &self,
        parameter_ranges: &[(f64, f64)],
        length: usize,
    ) -> KwaversResult<Vec<Array1<f64>>> {
        let mut trajectory = Vec::new();
        let mut current_point = Array1::zeros(parameter_ranges.len());

        use rand::rngs::StdRng;
        use rand::{Rng, SeedableRng};
        let mut rng = StdRng::from_entropy();

        for i in 0..parameter_ranges.len() {
            let (min_val, max_val) = parameter_ranges[i];
            current_point[i] = rng.gen_range(min_val..max_val);
        }
        trajectory.push(current_point.clone());

        for _ in 1..length {
            let param_to_change = rng.gen_range(0..parameter_ranges.len());
            let (min_val, max_val) = parameter_ranges[param_to_change];
            let step: f64 = rng.gen_range(-0.5..0.5);
            let new_value = step.mul_add(max_val - min_val, current_point[param_to_change]);
            current_point[param_to_change] = new_value.max(min_val).min(max_val);
            trajectory.push(current_point.clone());
        }
        Ok(trajectory)
    }

    fn compute_elementary_effects(
        &self,
        model_fn: &impl Fn(&Array1<f64>) -> Array1<f64>,
        trajectory: &[Array1<f64>],
    ) -> KwaversResult<Vec<f64>> {
        let mut effects = Vec::new();
        for i in 0..trajectory.len() - 1 {
            let output1 = model_fn(&trajectory[i]);
            let output2 = model_fn(&trajectory[i + 1]);
            let output_diff = &output2 - &output1;
            let diff_len = output_diff.len();
            let effect = if diff_len == 0 {
                0.0
            } else {
                output_diff.iter().map(|x| x.abs()).sum::<f64>() / diff_len as f64
            };
            effects.push(effect);
        }
        Ok(effects)
    }

    fn compute_morris_measures(
        &self,
        elementary_effects: &[f64],
    ) -> KwaversResult<(Vec<f64>, Vec<f64>)> {
        let num_params = 10;
        let effects_per_param = elementary_effects.len() / num_params;
        let mut mu = Vec::new();
        let mut sigma = Vec::new();
        for param_idx in 0..num_params {
            let start_idx = param_idx * effects_per_param;
            let end_idx = start_idx + effects_per_param;
            let param_effects =
                &elementary_effects[start_idx..end_idx.min(elementary_effects.len())];
            if !param_effects.is_empty() {
                let mu_val = param_effects.iter().sum::<f64>() / param_effects.len() as f64;
                mu.push(mu_val);
                let mean = mu_val;
                let sigma_val = (param_effects
                    .iter()
                    .map(|&x| (x - mean).powi(2))
                    .sum::<f64>()
                    / param_effects.len() as f64)
                    .sqrt();
                sigma.push(sigma_val);
            }
        }
        Ok((mu, sigma))
    }
}
