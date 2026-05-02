use super::{SlscBeamformer, SlscConfig};
use crate::core::error::KwaversResult;
use ndarray::Array1;
use num_complex::Complex64;

/// Multi-lag SLSC for enhanced imaging
#[derive(Debug, Clone)]
pub struct MultiLagSlsc {
    configs: Vec<SlscConfig>,
    combination_weights: Vec<f64>,
}

impl MultiLagSlsc {
    /// Create multi-lag SLSC with multiple configurations
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
    pub fn process_multi(&self, data: &ndarray::Array2<Complex64>) -> KwaversResult<Array1<f64>> {
        let mut combined = Array1::zeros(data.ncols());

        for (config, weight) in self.configs.iter().zip(&self.combination_weights) {
            let slsc = SlscBeamformer::with_config(config.clone());
            let coherence = slsc.process(data)?;

            combined += &coherence.mapv(|v| v * weight);
        }

        Ok(combined)
    }
}
