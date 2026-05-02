use super::SlscConfig;
use crate::core::error::{KwaversError, KwaversResult};
use ndarray::{Array1, Array2, Array3, Axis};
use num_complex::Complex64;
use rayon::prelude::*;

/// Short-Lag Spatial Coherence beamformer
#[derive(Debug, Clone)]
pub struct SlscBeamformer {
    config: SlscConfig,
}

impl SlscBeamformer {
    /// Create a new SLSC beamformer with default config
    #[must_use]
    pub fn new() -> Self {
        Self {
            config: SlscConfig::default(),
        }
    }

    /// Create a new SLSC beamformer with custom config
    #[must_use]
    pub fn with_config(config: SlscConfig) -> Self {
        Self { config }
    }

    /// Get the current configuration
    #[must_use]
    pub fn config(&self) -> &SlscConfig {
        &self.config
    }

    /// Process a single beamformed line using SLSC
    ///
    /// # Arguments
    /// * `data` - Complex RF data with shape (n_elements, n_samples)
    ///
    /// # Returns
    /// * Coherence values with shape (n_samples,)
    ///
    /// # Errors
    /// Returns error if input data has fewer than 2 elements
    pub fn process(&self, data: &Array2<Complex64>) -> KwaversResult<Array1<f64>> {
        let (n_elements, n_samples) = (data.nrows(), data.ncols());

        if n_elements < 2 {
            return Err(KwaversError::Validation(
                crate::core::error::ValidationError::InvalidParameter {
                    parameter: "n_elements".to_string(),
                    reason: "SLSC requires at least 2 array elements".to_string(),
                },
            ));
        }

        let max_lag = self.config.max_lag.min(n_elements - 1);
        let mut coherence = Array1::zeros(n_samples);

        for sample_idx in 0..n_samples {
            let sample_data = data.column(sample_idx);
            let slice: Vec<Complex64> = sample_data.iter().copied().collect();
            coherence[sample_idx] = self.compute_short_lag_coherence(&slice, max_lag);
        }

        Ok(coherence)
    }

    /// Process data in parallel for multiple beams
    ///
    /// # Arguments
    /// * `data` - Complex RF data with shape (n_elements, n_samples)
    ///
    /// # Returns
    /// * Coherence values with shape (n_samples,)
    pub fn process_parallel(&self, data: &Array2<Complex64>) -> KwaversResult<Array1<f64>> {
        let (n_elements, n_samples) = (data.nrows(), data.ncols());

        if n_elements < 2 {
            return Err(KwaversError::Validation(
                crate::core::error::ValidationError::InvalidParameter {
                    parameter: "n_elements".to_string(),
                    reason: "SLSC requires at least 2 array elements".to_string(),
                },
            ));
        }

        let max_lag = self.config.max_lag.min(n_elements - 1);

        let coherence_vec: Vec<f64> = (0..n_samples)
            .into_par_iter()
            .map(|sample_idx| {
                let sample_data = data.column(sample_idx);
                let slice: Vec<Complex64> = sample_data.iter().copied().collect();
                self.compute_short_lag_coherence(&slice, max_lag)
            })
            .collect();

        Ok(Array1::from(coherence_vec))
    }

    /// Compute short-lag spatial coherence for a single sample
    pub(super) fn compute_short_lag_coherence(&self, signals: &[Complex64], max_lag: usize) -> f64 {
        let n = signals.len();
        let mut total_coherence = 0.0;
        let mut total_weight = 0.0;

        for lag in 1..=max_lag {
            if lag >= n {
                break;
            }

            let weight = self.config.weighting.weight(lag, max_lag);
            let coherence = compute_lag_coherence(signals, lag);

            total_coherence += weight * coherence;
            total_weight += weight;
        }

        if total_weight > 0.0 {
            total_coherence / total_weight
        } else {
            0.0
        }
    }

    /// Process 3D volume data
    ///
    /// # Arguments
    /// * `data` - Complex RF data with shape (n_elements, n_beams, n_samples)
    ///
    /// # Returns
    /// * Coherence volume with shape (n_beams, n_samples)
    pub fn process_volume(&self, data: &Array3<Complex64>) -> KwaversResult<Array2<f64>> {
        let (n_elements, n_beams, n_samples) = (data.dim().0, data.dim().1, data.dim().2);

        if n_elements < 2 {
            return Err(KwaversError::Validation(
                crate::core::error::ValidationError::InvalidParameter {
                    parameter: "n_elements".to_string(),
                    reason: "SLSC requires at least 2 array elements".to_string(),
                },
            ));
        }

        let max_lag = self.config.max_lag.min(n_elements - 1);
        let mut coherence = Array2::zeros((n_beams, n_samples));

        coherence
            .axis_iter_mut(Axis(0))
            .into_par_iter()
            .enumerate()
            .for_each(|(beam_idx, mut beam_coherence)| {
                for sample_idx in 0..n_samples {
                    let sample_data: Vec<Complex64> = (0..n_elements)
                        .map(|elem| data[[elem, beam_idx, sample_idx]])
                        .collect();

                    beam_coherence[sample_idx] =
                        self.compute_short_lag_coherence(&sample_data, max_lag);
                }
            });

        Ok(coherence)
    }

    /// Process data for a grid of pixels (for imaging)
    ///
    /// # Arguments
    /// * `data` - Complex RF data with shape (n_elements, n_pixels)
    /// * `grid_shape` - Output grid shape (height, width)
    ///
    /// # Returns
    /// * Coherence image with shape (height, width)
    pub fn process_grid(
        &self,
        data: &Array2<Complex64>,
        grid_shape: (usize, usize),
    ) -> KwaversResult<Array2<f64>> {
        let coherence = self.process(data)?;
        let (height, width) = grid_shape;

        if coherence.len() != height * width {
            return Err(KwaversError::Validation(
                crate::core::error::ValidationError::DimensionMismatch {
                    expected: format!("{}x{}={}", height, width, height * width),
                    actual: format!("{}", coherence.len()),
                },
            ));
        }

        let mut image = Array2::zeros((height, width));
        for (idx, &value) in coherence.iter().enumerate() {
            let row = idx / width;
            let col = idx % width;
            image[[row, col]] = value;
        }

        Ok(image)
    }

    /// Create a coherence map for visualization
    ///
    /// # Arguments
    /// * `data` - Complex RF data
    /// * `dynamic_range_db` - Dynamic range in dB for scaling (default: 40)
    ///
    /// # Returns
    /// * Scaled coherence values in [0, 1] range
    pub fn create_coherence_map(
        &self,
        data: &Array2<Complex64>,
        dynamic_range_db: Option<f64>,
    ) -> KwaversResult<Array2<f64>> {
        let coherence = self.process(data)?;
        let dyn_range = dynamic_range_db.unwrap_or(40.0);

        let mut map = Array2::zeros((1, coherence.len()));
        for (i, &value) in coherence.iter().enumerate() {
            let db_value = 20.0 * value.max(1e-10).log10();
            map[[0, i]] = (db_value + dyn_range) / dyn_range;
        }

        map.mapv_inplace(|v| v.clamp(0.0, 1.0));

        Ok(map)
    }
}

impl Default for SlscBeamformer {
    fn default() -> Self {
        Self::new()
    }
}

/// Compute coherence for a specific lag (module-level helper)
pub(super) fn compute_lag_coherence(signals: &[Complex64], lag: usize) -> f64 {
    let n = signals.len();
    if lag >= n {
        return 0.0;
    }

    let mut numerator = 0.0;
    let mut energy1 = 0.0;
    let mut energy2 = 0.0;

    for i in 0..(n - lag) {
        let s_i = signals[i];
        let s_j = signals[i + lag];

        numerator += (s_i * s_j.conj()).re;
        energy1 += s_i.norm_sqr();
        energy2 += s_j.norm_sqr();
    }

    let denominator = (energy1 * energy2).sqrt();

    if denominator > 1e-10 {
        (numerator / denominator).abs()
    } else {
        0.0
    }
}
