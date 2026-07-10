use super::SlscConfig;
use kwavers_core::error::{KwaversError, KwaversResult};
use moirai_parallel::{map_collect_index_with, Adaptive};
use leto::{
    Array1,
    Array2,
    Array3,
};
use eunomia::Complex64;

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
    /// # Errors
    /// - Returns [`Err`] if an internal constraint is violated.
    ///
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
        let [n_elements, n_samples] = data.shape();

        if n_elements < 2 {
            return Err(KwaversError::Validation(
                kwavers_core::error::ValidationError::InvalidParameter {
                    parameter: "n_elements".to_owned(),
                    reason: "SLSC requires at least 2 array elements".to_owned(),
                },
            ));
        }

        let max_lag = self.config.max_lag.min(n_elements - 1);
        let mut coherence = Array1::zeros(n_samples);

        for sample_idx in 0..n_samples {
            let sample_data = data
                .index_axis::<1>(1, sample_idx)
                .expect("invariant: sample_idx < n_samples");
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
    /// # Errors
    /// - Returns [`KwaversError::Validation`] if the precondition for a Validation-class constraint is violated.
    ///
    pub fn process_parallel(&self, data: &Array2<Complex64>) -> KwaversResult<Array1<f64>> {
        let [n_elements, n_samples] = data.shape();

        if n_elements < 2 {
            return Err(KwaversError::Validation(
                kwavers_core::error::ValidationError::InvalidParameter {
                    parameter: "n_elements".to_owned(),
                    reason: "SLSC requires at least 2 array elements".to_owned(),
                },
            ));
        }

        let max_lag = self.config.max_lag.min(n_elements - 1);

        let coherence_vec: Vec<f64> =
            map_collect_index_with::<Adaptive, _, _>(n_samples, |sample_idx| {
                let sample_data = data
                    .index_axis::<1>(1, sample_idx)
                    .expect("invariant: sample_idx < n_samples");
                let slice: Vec<Complex64> = sample_data.iter().copied().collect();
                self.compute_short_lag_coherence(&slice, max_lag)
            });

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
    /// # Errors
    /// - Returns [`KwaversError::Validation`] if the precondition for a Validation-class constraint is violated.
    ///
    pub fn process_volume(&self, data: &Array3<Complex64>) -> KwaversResult<Array2<f64>> {
        let [n_elements, n_beams, n_samples] = data.shape();

        if n_elements < 2 {
            return Err(KwaversError::Validation(
                kwavers_core::error::ValidationError::InvalidParameter {
                    parameter: "n_elements".to_owned(),
                    reason: "SLSC requires at least 2 array elements".to_owned(),
                },
            ));
        }

        let max_lag = self.config.max_lag.min(n_elements - 1);
        let values = map_collect_index_with::<Adaptive, _, _>(n_beams * n_samples, |flat_index| {
            let beam_idx = flat_index / n_samples;
            let sample_idx = flat_index % n_samples;
            let sample_data: Vec<Complex64> = (0..n_elements)
                .map(|elem| data[[elem, beam_idx, sample_idx]])
                .collect();

            self.compute_short_lag_coherence(&sample_data, max_lag)
        });

        Array2::from_shape_vec((n_beams, n_samples), values).map_err(|err| {
            KwaversError::InvalidInput(format!(
                "SLSC volume output shape invariant violated: {err}"
            ))
        })
    }

    /// Process data for a grid of pixels (for imaging)
    ///
    /// # Arguments
    /// * `data` - Complex RF data with shape (n_elements, n_pixels)
    /// * `grid_shape` - Output grid shape (height, width)
    ///
    /// # Returns
    /// * Coherence image with shape (height, width)
    /// # Errors
    /// - Returns [`KwaversError::Validation`] if the precondition for a Validation-class constraint is violated.
    /// - Propagates any [`KwaversError`] returned by called functions.
    ///
    pub fn process_grid(
        &self,
        data: &Array2<Complex64>,
        grid_shape: (usize, usize),
    ) -> KwaversResult<Array2<f64>> {
        let coherence = self.process(data)?;
        let (height, width) = grid_shape;

        if coherence.len() != height * width {
            return Err(KwaversError::Validation(
                kwavers_core::error::ValidationError::DimensionMismatch {
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
    /// # Errors
    /// - Propagates any [`KwaversError`] returned by called functions.
    ///
    pub fn create_coherence_map(
        &self,
        data: &Array2<Complex64>,
        dynamic_range_db: Option<f64>,
    ) -> KwaversResult<Array2<f64>> {
        let coherence = self.process(data)?;
        let dyn_range = dynamic_range_db.unwrap_or(40.0);

        let mut map = Array2::<f64>::zeros((1, coherence.len()));
        for (i, &value) in coherence.iter().enumerate() {
            let db_value = 20.0 * value.max(1e-10).log10();
            map[[0, i]] = (db_value + dyn_range) / dyn_range;
        }

        for value in map.iter_mut() {
            *value = value.clamp(0.0, 1.0);
        }

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

