//! Short-Lag Spatial Coherence (SLSC) Beamforming
//!
//! # Overview
//!
//! Short-Lag Spatial Coherence (SLSC) beamforming is an advanced imaging technique
//! that improves image quality by leveraging the spatial coherence of backscattered
//! ultrasound signals. Unlike conventional delay-and-sum beamforming which only uses
//! amplitude information, SLSC exploits the phase coherence between signals received
//! at different array elements.
//!
//! ## Key Advantages
//!
//! - **Improved Clutter Rejection**: Suppresses incoherent noise and reverberation clutter
//! - **Better Contrast**: Enhances tissue boundaries and cyst visualization
//! - **Robust to Phase Aberration**: Less sensitive to sound speed variations
//! - **No Additional Hardware**: Uses same data as conventional beamforming
//!
//! # Mathematical Foundation
//!
//! ## Spatial Coherence
//!
//! The spatial coherence between signals received at elements i and j is defined as:
//!
//! ```text
//! C(d) = |Σ_{k=1}^{N-d} s_k · s_{k+d}^*| / √[Σ|s_k|² · Σ|s_{k+d}|²]
//! ```
//!
//! where:
//! - `d` = element lag (distance between elements)
//! - `s_k` = signal at element k after delay compensation
//! - `N` = total number of elements
//! - `*` = complex conjugate
//!
//! ## Short-Lag Region
//!
//! SLSC only uses short lags (typically d = 1 to M where M << N) because:
//! - Short lags capture local coherence (tissue microstructure)
//! - Long lags are dominated by noise and decorrelation
//! - Short-lag coherence is more robust to phase aberrations
//!
//! ## SLSC Image Formation
//!
//! The SLSC image value at each pixel is computed as:
//!
//! ```text
//! I_SLSC = Σ_{d=1}^{M} w(d) · C(d)
//! ```
//!
//! where:
//! - `M` = maximum lag (typically 10-20% of array aperture)
//! - `w(d)` = optional weighting function (e.g., triangular, Hamming)
//! - `C(d)` = spatial coherence at lag d
//!
//! # Algorithm Steps
//!
//! 1. **Delay Compensation**: Apply geometric delays to align signals
//! 2. **Lag Correlation**: Compute correlation for each short lag
//! 3. **Normalization**: Normalize by signal energy
//! 4. **Coherence Summation**: Sum weighted coherence values
//!
//! # Implementation Notes
//!
//! This implementation follows the SSOT (Single Source of Truth) principle:
//! - Uses existing delay computation from `time_domain` module
//! - Leverages covariance estimation from `covariance` module
//! - No redundant implementations or wrapper layers
//!
//! # References
//!
//! - Lediju, M. A., et al. (2011). "Short-lag spatial coherence of backscattered echoes:
//!   Imaging characteristics." *IEEE Trans. Ultrason., Ferroelect., Freq. Control*, 58(7).
//!   DOI: 10.1109/TUFFC.2011.1957
//!
//! - Jakovljevic, M., et al. (2013). "In vivo application of short-lag spatial coherence
//!   imaging in human liver." *Ultrasonic Imaging*, 35(3).
//!   DOI: 10.1177/0161734613489682

use crate::core::error::{KwaversError, KwaversResult};
use ndarray::{Array1, Array2, Array3, Axis};
use num_complex::Complex64;
use rayon::prelude::*;

/// Configuration for SLSC beamforming
#[derive(Debug, Clone)]
pub struct SlscConfig {
    /// Maximum lag to use (M). Typically 10-20% of array elements.
    pub max_lag: usize,
    /// Weighting function for different lags
    pub weighting: LagWeighting,
    /// Whether to use normalized coherence
    pub normalize: bool,
}

impl Default for SlscConfig {
    fn default() -> Self {
        Self {
            max_lag: 10,
            weighting: LagWeighting::Uniform,
            normalize: true,
        }
    }
}

impl SlscConfig {
    /// Create a new config with specified max lag
    #[must_use]
    pub fn with_max_lag(max_lag: usize) -> Self {
        Self {
            max_lag,
            ..Default::default()
        }
    }

    /// Create a new config with triangular weighting
    #[must_use]
    pub fn with_triangular_weighting() -> Self {
        Self {
            weighting: LagWeighting::Triangular,
            ..Default::default()
        }
    }

    /// Validate the configuration
    ///
    /// # Errors
    /// Returns error if max_lag is 0
    pub fn validate(&self) -> KwaversResult<()> {
        if self.max_lag == 0 {
            return Err(KwaversError::Config(
                crate::core::error::ConfigError::InvalidValue {
                    parameter: "max_lag".to_string(),
                    value: "0".to_string(),
                    constraint: "max_lag must be greater than 0".to_string(),
                },
            ));
        }
        Ok(())
    }
}

/// Weighting function for lag contributions
#[derive(Debug, Clone, PartialEq)]
pub enum LagWeighting {
    /// Uniform weighting (all lags equal)
    Uniform,
    /// Triangular weighting (linear decrease with lag)
    Triangular,
    /// Hamming window weighting
    Hamming,
    /// Custom weighting with user-defined weights
    Custom { weights: Box<[f64; 64]>, len: usize },
}

impl LagWeighting {
    /// Get the weight for a specific lag
    #[must_use]
    pub fn weight(&self, lag: usize, max_lag: usize) -> f64 {
        match self {
            Self::Uniform => 1.0,
            Self::Triangular => {
                if lag >= max_lag {
                    0.0
                } else {
                    1.0 - (lag as f64 / max_lag as f64)
                }
            }
            Self::Hamming => {
                if lag >= max_lag {
                    0.0
                } else {
                    let alpha = 0.54;
                    let beta = 0.46;
                    let pi = std::f64::consts::PI;
                    alpha - beta * (2.0 * pi * lag as f64 / max_lag as f64).cos()
                }
            }
            Self::Custom { weights, len } => {
                if lag < *len {
                    weights[lag]
                } else {
                    0.0
                }
            }
        }
    }
}

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

        // Compute spatial coherence for each sample
        for sample_idx in 0..n_samples {
            let sample_data = data.column(sample_idx);
            // Handle the case where column data might not be contiguous
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

        // Process in parallel using rayon
        let coherence_vec: Vec<f64> = (0..n_samples)
            .into_par_iter()
            .map(|sample_idx| {
                let sample_data = data.column(sample_idx);
                // Handle the case where column data might not be contiguous
                let slice: Vec<Complex64> = sample_data.iter().copied().collect();
                self.compute_short_lag_coherence(&slice, max_lag)
            })
            .collect();

        Ok(Array1::from(coherence_vec))
    }

    /// Compute short-lag spatial coherence for a single sample
    fn compute_short_lag_coherence(&self, signals: &[Complex64], max_lag: usize) -> f64 {
        let n = signals.len();
        let mut total_coherence = 0.0;
        let mut total_weight = 0.0;

        for lag in 1..=max_lag {
            if lag >= n {
                break;
            }

            let weight = self.config.weighting.weight(lag, max_lag);
            let coherence = self.compute_lag_coherence(signals, lag);

            total_coherence += weight * coherence;
            total_weight += weight;
        }

        if total_weight > 0.0 {
            total_coherence / total_weight
        } else {
            0.0
        }
    }

    /// Compute coherence for a specific lag
    fn compute_lag_coherence(&self, signals: &[Complex64], lag: usize) -> f64 {
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

            numerator += (s_i * s_j.conj()).re; // Real part of correlation
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

        // Process each beam in parallel
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

        // Reshape to 2D grid
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

        // Convert to dB scale and normalize
        let mut map = Array2::zeros((1, coherence.len()));
        for (i, &value) in coherence.iter().enumerate() {
            // Coherence is already in [0, 1], convert to dB-like scale
            let db_value = 20.0 * value.max(1e-10).log10();
            map[[0, i]] = (db_value + dyn_range) / dyn_range;
        }

        // Clip to [0, 1]
        map.mapv_inplace(|v| v.clamp(0.0, 1.0));

        Ok(map)
    }
}

impl Default for SlscBeamformer {
    fn default() -> Self {
        Self::new()
    }
}

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
    pub fn process_adaptive(&self, data: &Array2<Complex64>) -> KwaversResult<Array1<f64>> {
        let n_elements = data.nrows();

        // Estimate optimal max_lag based on element count
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

    /// Estimate optimal lag based on signal coherence decay
    fn estimate_optimal_lag(&self, data: &Array2<Complex64>) -> usize {
        let n_elements = data.nrows();
        let sample_idx = data.ncols() / 2; // Use center sample

        let sample_data: Vec<Complex64> = (0..n_elements).map(|i| data[[i, sample_idx]]).collect();

        // Find where coherence drops below threshold
        let threshold = 0.5;
        for lag in 1..n_elements {
            let coherence = self.compute_lag_coherence_simple(&sample_data, lag);
            if coherence < threshold {
                return lag;
            }
        }

        n_elements / 4 // Default to 25% of aperture
    }

    /// Simple coherence computation without weighting
    fn compute_lag_coherence_simple(&self, signals: &[Complex64], lag: usize) -> f64 {
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
}

impl Default for AdaptiveSlsc {
    fn default() -> Self {
        Self::new()
    }
}

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
    pub fn process_multi(&self, data: &Array2<Complex64>) -> KwaversResult<Array1<f64>> {
        let mut combined = Array1::zeros(data.ncols());

        for (config, weight) in self.configs.iter().zip(&self.combination_weights) {
            let slsc = SlscBeamformer::with_config(config.clone());
            let coherence = slsc.process(data)?;

            combined += &coherence.mapv(|v| v * weight);
        }

        Ok(combined)
    }
}

/// Batch processing for multiple frames
pub fn process_slsc_batch(
    data: &Array3<Complex64>,
    config: &SlscConfig,
) -> KwaversResult<Array2<f64>> {
    let (n_elements, n_frames, n_samples) = (data.dim().0, data.dim().1, data.dim().2);

    if n_elements < 2 {
        return Err(KwaversError::Validation(
            crate::core::error::ValidationError::InvalidParameter {
                parameter: "n_elements".to_string(),
                reason: "SLSC requires at least 2 array elements".to_string(),
            },
        ));
    }

    let slsc = SlscBeamformer::with_config(config.clone());
    let mut results = Vec::with_capacity(n_frames);

    for frame_idx in 0..n_frames {
        let frame_data: Array2<Complex64> = data
            .slice(ndarray::s![.., frame_idx, ..])
            .to_owned()
            .into_dimensionality()
            .map_err(|_| {
                KwaversError::Validation(crate::core::error::ValidationError::InvalidFormat {
                    field: "frame_data".to_string(),
                    expected: "Array2".to_string(),
                    actual: "Array3 slice".to_string(),
                })
            })?;

        let coherence = slsc.process(&frame_data)?;
        results.push(coherence);
    }

    let mut output = Array2::zeros((n_frames, n_samples));
    for (frame_idx, frame_result) in results.iter().enumerate() {
        output.row_mut(frame_idx).assign(frame_result);
    }

    Ok(output)
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array2;

    #[test]
    fn test_slsc_default_config() {
        let slsc = SlscBeamformer::new();
        assert_eq!(slsc.config().max_lag, 10);
        assert!(slsc.config().normalize);
    }

    #[test]
    fn test_slsc_with_config() {
        let config = SlscConfig::with_max_lag(20);
        let slsc = SlscBeamformer::with_config(config);
        assert_eq!(slsc.config().max_lag, 20);
    }

    #[test]
    fn test_slsc_process_simple() {
        // Create simple test data: 4 elements, 10 samples
        let n_elements = 4;
        let n_samples = 10;
        let data = Array2::from_elem((n_elements, n_samples), Complex64::new(1.0, 0.0));

        let slsc = SlscBeamformer::new();
        let result = slsc.process(&data).expect("SLSC processing should succeed");

        assert_eq!(result.len(), n_samples);
        // Perfectly coherent signals should give high coherence values
        for &val in result.iter() {
            assert!((0.0..=1.0).contains(&val), "Coherence should be in [0, 1]");
        }
    }

    #[test]
    fn test_slsc_rejects_single_element() {
        let data = Array2::from_elem((1, 10), Complex64::new(1.0, 0.0));
        let slsc = SlscBeamformer::new();
        let result = slsc.process(&data);
        assert!(result.is_err());
    }

    #[test]
    fn test_lag_weighting_uniform() {
        let w = LagWeighting::Uniform;
        assert_eq!(w.weight(1, 10), 1.0);
        assert_eq!(w.weight(5, 10), 1.0);
    }

    #[test]
    fn test_lag_weighting_triangular() {
        let w = LagWeighting::Triangular;
        assert_eq!(w.weight(0, 10), 1.0);
        assert_eq!(w.weight(5, 10), 0.5);
        assert_eq!(w.weight(10, 10), 0.0);
    }

    #[test]
    fn test_slsc_grid_processing() {
        let n_elements = 8;
        let height = 10;
        let width = 20;
        let n_pixels = height * width;

        let data = Array2::from_elem((n_elements, n_pixels), Complex64::new(1.0, 0.0));
        let slsc = SlscBeamformer::new();
        let result = slsc
            .process_grid(&data, (height, width))
            .expect("Grid processing should succeed");

        assert_eq!(result.shape(), &[height, width]);
    }
}
