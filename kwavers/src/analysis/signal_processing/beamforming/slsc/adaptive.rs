use super::{beamformer::compute_lag_coherence, SlscBeamformer, SlscConfig};
use crate::core::error::KwaversResult;
use ndarray::Array1;
use num_complex::Complex64;

/// Adaptive SLSC with automatic parameter selection
#[derive(Debug, Clone)]
pub struct AdaptiveSlsc {
    base_config: SlscConfig,
    adaptation_rate: f64,
}

impl AdaptiveSlsc {
    /// Create new adaptive SLSC processor
    #[must_use]
    pub fn new() -> Self {
        Self {
            base_config: SlscConfig::default(),
            adaptation_rate: 0.1,
        }
    }

    /// Process with automatic max_lag selection based on signal characteristics
    ///
    /// # Arguments
    /// * `data` - Input RF data
    ///
    /// # Returns
    /// * Optimized coherence values
    /// # Errors
    /// - Returns [`Err`] if an internal constraint is violated.
    ///
    pub fn process_adaptive(
        &self,
        data: &ndarray::Array2<Complex64>,
    ) -> KwaversResult<Array1<f64>> {
        let n_elements = data.nrows();

        let optimal_lag = self.estimate_optimal_lag(data);
        let max_lag = ((optimal_lag as f64 * self.adaptation_rate) as usize)
            .max(2)
            .min(n_elements - 1);

        let config = SlscConfig {
            max_lag,
            ..self.base_config.clone()
        };

        let slsc = SlscBeamformer::with_config(config);
        slsc.process(data)
    }

    fn estimate_optimal_lag(&self, data: &ndarray::Array2<Complex64>) -> usize {
        let n_elements = data.nrows();
        let sample_idx = data.ncols() / 2;

        let sample_data: Vec<Complex64> = (0..n_elements).map(|i| data[[i, sample_idx]]).collect();

        let threshold = 0.5;
        for lag in 1..n_elements {
            let coherence = compute_lag_coherence(&sample_data, lag);
            if coherence < threshold {
                return lag;
            }
        }

        n_elements / 4
    }
}

impl Default for AdaptiveSlsc {
    fn default() -> Self {
        Self::new()
    }
}
