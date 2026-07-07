use super::BatchIterator;
use kwavers_core::error::{KwaversError, KwaversResult};
use rand::prelude::*;

/// Adaptive sampling strategy: how collocation points are selected.
#[derive(Debug, Clone, PartialEq)]
pub enum ElasticAdaptiveSamplingStrategy {
    Uniform,
    /// `p_i ∝ r_i^α`, `keep_ratio` fraction of old points retained.
    ResidualWeighted {
        alpha: f64,
        keep_ratio: f64,
    },
    ImportanceThreshold {
        threshold: f64,
        top_k_ratio: f64,
    },
    Hybrid {
        uniform_ratio: f64,
        alpha: f64,
    },
}

impl Default for ElasticAdaptiveSamplingStrategy {
    fn default() -> Self {
        ElasticAdaptiveSamplingStrategy::ResidualWeighted {
            alpha: 1.0,
            keep_ratio: 0.0,
        }
    }
}

/// Adaptive collocation point sampler with mini-batching.
#[derive(Debug)]
pub struct AdaptiveSampler {
    pub strategy: ElasticAdaptiveSamplingStrategy,
    pub n_points: usize,
    pub batch_size: usize,
    rng: StdRng,
    current_indices: Vec<usize>,
}

impl AdaptiveSampler {
    pub fn new(
        strategy: ElasticAdaptiveSamplingStrategy,
        n_points: usize,
        batch_size: usize,
    ) -> Self {
        Self::with_seed(strategy, n_points, batch_size, 42)
    }

    pub fn with_seed(
        strategy: ElasticAdaptiveSamplingStrategy,
        n_points: usize,
        batch_size: usize,
        seed: u64,
    ) -> Self {
        Self {
            strategy,
            n_points,
            batch_size,
            rng: StdRng::seed_from_u64(seed),
            current_indices: (0..n_points).collect(),
        }
    }

    /// Resample collocation points based on PDE residuals.
    /// # Errors
    /// - Returns [`KwaversError::InvalidInput`] if the precondition for invalid or out-of-range input parameters is violated.
    /// - Propagates any [`KwaversError`] returned by called functions.
    ///
    /// # Panics
    /// - Panics if an internal invariant assumed to hold at this call site is violated.
    ///
    pub fn resample(&mut self, residuals: &[f64]) -> KwaversResult<Vec<usize>> {
        let n_candidates = residuals.len();
        if n_candidates == 0 {
            return Err(KwaversError::InvalidInput(
                "No candidate points for resampling".into(),
            ));
        }
        if self.n_points > n_candidates {
            return Err(KwaversError::InvalidInput(format!(
                "Requested {} points but only {} candidates available",
                self.n_points, n_candidates
            )));
        }

        match &self.strategy.clone() {
            ElasticAdaptiveSamplingStrategy::Uniform => {
                let mut indices: Vec<usize> = (0..n_candidates).collect();
                indices.shuffle(&mut self.rng);
                indices.truncate(self.n_points);
                Ok(indices)
            }
            ElasticAdaptiveSamplingStrategy::ResidualWeighted { alpha, keep_ratio } => {
                let (alpha, keep_ratio) = (*alpha, *keep_ratio);
                let weights: Vec<f64> = residuals.iter().map(|&r| r.abs().powf(alpha)).collect();
                let total_weight: f64 = weights.iter().sum();
                if total_weight == 0.0 {
                    return self.resample(&vec![1.0; n_candidates]);
                }
                let probs: Vec<f64> = weights.iter().map(|w| w / total_weight).collect();
                let n_new = ((1.0 - keep_ratio) * self.n_points as f64) as usize;
                let n_keep = self.n_points - n_new;
                let mut selected = Vec::with_capacity(self.n_points);
                if n_keep > 0 && !self.current_indices.is_empty() {
                    let mut old = self.current_indices.clone();
                    old.shuffle(&mut self.rng);
                    selected.extend_from_slice(&old[..n_keep.min(old.len())]);
                }
                selected.extend(self.weighted_sample(&probs, n_new)?);
                self.current_indices = selected.clone();
                Ok(selected)
            }
            ElasticAdaptiveSamplingStrategy::ImportanceThreshold {
                threshold,
                top_k_ratio,
            } => {
                let (threshold, top_k_ratio) = (*threshold, *top_k_ratio);
                let mut candidates: Vec<(usize, f64)> = residuals
                    .iter()
                    .enumerate()
                    .filter(|(_, &r)| r.abs() >= threshold)
                    .map(|(i, &r)| (i, r.abs()))
                    .collect();
                if candidates.is_empty() {
                    return self.resample(&vec![1.0; n_candidates]);
                }
                candidates.sort_by(|a, b| b.1.total_cmp(&a.1));
                let k = ((top_k_ratio * candidates.len() as f64) as usize)
                    .max(self.n_points)
                    .min(candidates.len());
                candidates.truncate(k);
                let mut indices: Vec<usize> = candidates.into_iter().map(|(i, _)| i).collect();
                indices.shuffle(&mut self.rng);
                indices.truncate(self.n_points);
                self.current_indices = indices.clone();
                Ok(indices)
            }
            ElasticAdaptiveSamplingStrategy::Hybrid {
                uniform_ratio,
                alpha,
            } => {
                let (uniform_ratio, alpha) = (*uniform_ratio, *alpha);
                let n_uniform = (uniform_ratio * self.n_points as f64) as usize;
                let n_weighted = self.n_points - n_uniform;
                let mut selected = Vec::with_capacity(self.n_points);
                if n_uniform > 0 {
                    let mut all: Vec<usize> = (0..n_candidates).collect();
                    all.shuffle(&mut self.rng);
                    selected.extend_from_slice(&all[..n_uniform.min(n_candidates)]);
                }
                if n_weighted > 0 {
                    let weights: Vec<f64> =
                        residuals.iter().map(|&r| r.abs().powf(alpha)).collect();
                    let total_weight: f64 = weights.iter().sum();
                    if total_weight > 0.0 {
                        let probs: Vec<f64> = weights.iter().map(|w| w / total_weight).collect();
                        selected.extend(self.weighted_sample(&probs, n_weighted)?);
                    }
                }
                self.current_indices = selected.clone();
                Ok(selected)
            }
        }
    }

    fn weighted_sample(&mut self, probs: &[f64], n_samples: usize) -> KwaversResult<Vec<usize>> {
        if n_samples == 0 {
            return Ok(Vec::new());
        }
        let n = probs.len();
        if n_samples > n {
            return Err(KwaversError::InvalidInput(format!(
                "Cannot sample {} points from {} candidates",
                n_samples, n
            )));
        }
        let mut keys: Vec<(usize, f64)> = probs
            .iter()
            .enumerate()
            .filter(|(_, &p)| p > 0.0)
            .map(|(i, &p)| {
                let u: f64 = self.rng.gen();
                let key = if u > 0.0 { u.powf(1.0 / p) } else { 0.0 };
                (i, key)
            })
            .collect();
        keys.sort_by(|a, b| b.1.total_cmp(&a.1));
        keys.truncate(n_samples);
        Ok(keys.into_iter().map(|(i, _)| i).collect())
    }

    pub fn iter_batches(&mut self) -> BatchIterator {
        let mut indices = self.current_indices.clone();
        indices.shuffle(&mut self.rng);
        let batch_size = if self.batch_size == 0 {
            indices.len()
        } else {
            self.batch_size
        };
        BatchIterator::new(indices, batch_size)
    }

    pub fn current_indices(&self) -> &[usize] {
        &self.current_indices
    }

    pub fn n_batches(&self) -> usize {
        if self.batch_size == 0 {
            1
        } else {
            self.n_points.div_ceil(self.batch_size)
        }
    }
}
