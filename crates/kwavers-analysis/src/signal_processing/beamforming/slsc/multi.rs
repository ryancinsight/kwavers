use super::{SlscBeamformer, SlscConfig};
use kwavers_core::error::KwaversResult;
use leto::Array1;
use eunomia::Complex64;

/// Multi-lag SLSC for enhanced imaging
#[derive(Debug, Clone)]
pub struct MultiLagSlsc {
    configs: Vec<SlscConfig>,
    combination_weights: Vec<f64>,
}

impl MultiLagSlsc {
    /// Create multi-lag SLSC with multiple configurations
    /// # Errors
    /// - Returns [`Err`] if an internal constraint is violated.
    ///
    #[must_use]
    pub fn with_configs(configs: Vec<SlscConfig>) -> Self {
        let n = configs.len();
        let weights = vec![1.0 / n as f64; n];
        Self {
            configs,
            combination_weights: weights,
        }
    }

    /// Process with multiple lag configurations and combine results
    ///
    /// # Arguments
    /// * `data` - Input RF data
    ///
    /// # Returns
    /// * Combined coherence values
    /// # Errors
    /// - Propagates any [`KwaversError`] returned by called functions.
    ///
    pub fn process_multi(&self, data: &leto::Array2<Complex64>) -> KwaversResult<Array1<f64>> {
        let mut combined = Array1::zeros(data.shape()[1]);

        for (config, weight) in self.configs.iter().zip(&self.combination_weights) {
            let slsc = SlscBeamformer::with_config(config.clone());
            let coherence = slsc.process(data)?;

            let scaled = coherence.mapv(|v| v * weight);
            for (c, s) in combined.iter_mut().zip(scaled.iter()) {
                *c += *s;
            }
        }

        Ok(combined)
    }
}

