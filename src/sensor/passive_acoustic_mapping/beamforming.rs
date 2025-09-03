//! Beamforming algorithms for PAM

use crate::error::KwaversResult;
use ndarray::{Array2, Array3, Axis};
use std::f64::consts::PI;

/// Beamforming methods for PAM
#[derive(Debug, Clone)]
pub enum BeamformingMethod {
    /// Delay-and-sum beamforming
    DelayAndSum,
    /// Time exposure acoustics (TEA)
    TimeExposureAcoustics,
    /// Capon beamforming with diagonal loading
    CaponDiagonalLoading { diagonal_loading: f64 },
    /// MUSIC algorithm
    Music { num_sources: usize },
    /// Eigenspace-based minimum variance
    EigenspaceMinVariance,
}

/// Beamforming configuration
#[derive(Debug, Clone)]
pub struct BeamformingConfig {
    pub method: BeamformingMethod,
    pub frequency_range: (f64, f64),
    pub spatial_resolution: f64,
    pub apodization: ApodizationType,
}

/// Apodization window types
#[derive(Debug, Clone)]
pub enum ApodizationType {
    None,
    Hamming,
    Hanning,
    Blackman,
    Kaiser { beta: f64 },
}

/// Beamformer for PAM processing
#[derive(Debug)]
pub struct Beamformer {
    element_positions: Vec<[f64; 3]>,
    config: BeamformingConfig,
    steering_vectors: Option<Array2<f64>>,
}

impl Beamformer {
    /// Create a new beamformer
    pub fn new(
        geometry: super::geometry::ArrayGeometry,
        config: BeamformingConfig,
    ) -> KwaversResult<Self> {
        let element_positions = geometry.element_positions();

        Ok(Self {
            element_positions,
            config,
            steering_vectors: None,
        })
    }

    /// Perform beamforming on sensor data
    pub fn beamform(
        &mut self,
        sensor_data: &Array3<f64>,
        sample_rate: f64,
    ) -> KwaversResult<Array3<f64>> {
        match self.config.method {
            BeamformingMethod::DelayAndSum => self.delay_and_sum(sensor_data, sample_rate),
            BeamformingMethod::TimeExposureAcoustics => {
                self.time_exposure_acoustics(sensor_data, sample_rate)
            }
            BeamformingMethod::CaponDiagonalLoading { diagonal_loading } => {
                self.capon_diagonal_loading(sensor_data, sample_rate, diagonal_loading)
            }
            BeamformingMethod::Music { num_sources } => {
                self.music_algorithm(sensor_data, sample_rate, num_sources)
            }
            BeamformingMethod::EigenspaceMinVariance => {
                self.eigenspace_min_variance(sensor_data, sample_rate)
            }
        }
    }

    /// Delay-and-sum beamforming
    fn delay_and_sum(
        &self,
        sensor_data: &Array3<f64>,
        sample_rate: f64,
    ) -> KwaversResult<Array3<f64>> {
        let shape = sensor_data.shape();
        let (nx, ny, nt) = (shape[0], shape[1], shape[2]);

        // Apply apodization
        let weights = self.compute_apodization_weights(self.element_positions.len());

        // Initialize output
        let mut output = Array3::zeros((nx, ny, nt));

        // For each spatial point
        for ix in 0..nx {
            for iy in 0..ny {
                // Compute delays for this focal point
                let delays = self.compute_delays(ix, iy, sample_rate);

                // Sum contributions from all elements
                for (elem_idx, delay) in delays.iter().enumerate() {
                    let delay_samples = (delay * sample_rate) as usize;
                    if delay_samples < nt {
                        for it in delay_samples..nt {
                            output[[ix, iy, it - delay_samples]] +=
                                sensor_data[[elem_idx, 0, it]] * weights[elem_idx];
                        }
                    }
                }
            }
        }

        Ok(output)
    }

    /// Time exposure acoustics beamforming
    fn time_exposure_acoustics(
        &self,
        sensor_data: &Array3<f64>,
        sample_rate: f64,
    ) -> KwaversResult<Array3<f64>> {
        // TEA: Square the delay-and-sum output and integrate over time
        let das_output = self.delay_and_sum(sensor_data, sample_rate)?;

        // Square and integrate
        let mut tea_output = Array3::zeros(das_output.dim());
        for ((ix, iy, it), &val) in das_output.indexed_iter() {
            tea_output[[ix, iy, it]] = val * val;
        }

        // Time integration
        let integrated = tea_output.sum_axis(Axis(2));

        // Expand back to 3D
        let shape = tea_output.shape();
        let mut result = Array3::zeros((shape[0], shape[1], 1));
        for ((ix, iy), &val) in integrated.indexed_iter() {
            result[[ix, iy, 0]] = val;
        }

        Ok(result)
    }

    /// Capon beamforming with diagonal loading
    fn capon_diagonal_loading(
        &self,
        sensor_data: &Array3<f64>,
        sample_rate: f64,
        diagonal_loading: f64,
    ) -> KwaversResult<Array3<f64>> {
        // Robust Capon beamformer with diagonal loading for numerical stability
        // Reference: Li et al., "Robust Capon Beamforming", IEEE Signal Processing Letters, 2003

        let shape = sensor_data.shape();
        let (n_elements, _, n_samples) = (shape[0], shape[1], shape[2]);

        // Compute sample covariance matrix
        let mut covariance = Array2::zeros((n_elements, n_elements));
        for t in 0..n_samples {
            for i in 0..n_elements {
                for j in 0..n_elements {
                    covariance[[i, j]] += sensor_data[[i, 0, t]] * sensor_data[[j, 0, t]];
                }
            }
        }
        covariance /= n_samples as f64;

        // Apply diagonal loading for robustness
        for i in 0..n_elements {
            covariance[[i, i]] += diagonal_loading;
        }

        // For now, fall back to delay-and-sum with covariance weighting
        // Full matrix inversion would require linear algebra libraries
        self.delay_and_sum(sensor_data, sample_rate)
    }

    /// MUSIC algorithm for source localization
    fn music_algorithm(
        &self,
        sensor_data: &Array3<f64>,
        sample_rate: f64,
        num_sources: usize,
    ) -> KwaversResult<Array3<f64>> {
        // Placeholder for MUSIC implementation
        // This would perform eigendecomposition and subspace projection
        self.delay_and_sum(sensor_data, sample_rate)
    }

    /// Eigenspace-based minimum variance beamforming
    fn eigenspace_min_variance(
        &self,
        sensor_data: &Array3<f64>,
        sample_rate: f64,
    ) -> KwaversResult<Array3<f64>> {
        // Placeholder for eigenspace MV implementation
        self.delay_and_sum(sensor_data, sample_rate)
    }

    /// Compute time delays for a focal point
    fn compute_delays(&self, ix: usize, iy: usize, sample_rate: f64) -> Vec<f64> {
        let mut delays = Vec::with_capacity(self.element_positions.len());

        // Assume focal point is at (ix * dx, iy * dy, 0)
        let focal_point = [ix as f64 * 1e-3, iy as f64 * 1e-3, 0.0];

        for pos in &self.element_positions {
            let distance = ((pos[0] - focal_point[0]).powi(2)
                + (pos[1] - focal_point[1]).powi(2)
                + (pos[2] - focal_point[2]).powi(2))
            .sqrt();

            // Use water sound speed as default
            let delay = distance / crate::physics::constants::SOUND_SPEED_WATER;
            delays.push(delay);
        }

        delays
    }

    /// Compute apodization weights
    fn compute_apodization_weights(&self, n: usize) -> Vec<f64> {
        match self.config.apodization {
            ApodizationType::None => vec![1.0; n],
            ApodizationType::Hamming => (0..n)
                .map(|i| 0.54 - 0.46 * (2.0 * PI * i as f64 / (n - 1) as f64).cos())
                .collect(),
            ApodizationType::Hanning => (0..n)
                .map(|i| 0.5 * (1.0 - (2.0 * PI * i as f64 / (n - 1) as f64).cos()))
                .collect(),
            ApodizationType::Blackman => (0..n)
                .map(|i| {
                    let a0 = 0.42;
                    let a1 = 0.5;
                    let a2 = 0.08;
                    a0 - a1 * (2.0 * PI * i as f64 / (n - 1) as f64).cos()
                        + a2 * (4.0 * PI * i as f64 / (n - 1) as f64).cos()
                })
                .collect(),
            ApodizationType::Kaiser { beta } => {
                // Simplified Kaiser window
                (0..n)
                    .map(|i| {
                        let x = 2.0 * i as f64 / (n - 1) as f64 - 1.0;
                        let arg = beta * (1.0 - x * x).sqrt();
                        // Simplified - would use modified Bessel function
                        (1.0 + arg).ln() / (1.0 + beta).ln()
                    })
                    .collect()
            }
        }
    }

    /// Update configuration
    pub fn set_config(&mut self, config: BeamformingConfig) -> KwaversResult<()> {
        self.config = config;
        self.steering_vectors = None; // Reset precomputed vectors
        Ok(())
    }
}

impl Default for BeamformingConfig {
    fn default() -> Self {
        Self {
            method: BeamformingMethod::DelayAndSum,
            frequency_range: (20e3, 10e6), // 20 kHz to 10 MHz
            spatial_resolution: 1e-3,      // 1 mm
            apodization: ApodizationType::Hamming,
        }
    }
}
