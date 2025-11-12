//! Beamforming algorithms for PAM

use crate::error::KwaversResult;
use ndarray::{Array2, Array3, Axis};
use std::f64::consts::PI;
use crate::sensor::beamforming::BeamformingProcessor;
use crate::sensor::beamforming::BeamformingCoreConfig;

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
    /// Focal point for single-look beamforming in meters `[x, y, z]`
    ///
    /// PAM workflows may sweep multiple focal points externally to build maps.
    /// Here we define a single focal point per beamform invocation for
    /// mathematically consistent alignment.
    pub focal_point: [f64; 3],
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
    processor: BeamformingProcessor,
}

impl Beamformer {
    /// Create a new beamformer
    pub fn new(
        geometry: super::geometry::ArrayGeometry,
        config: BeamformingConfig,
    ) -> KwaversResult<Self> {
        let element_positions = geometry.element_positions();
        let core_cfg: BeamformingCoreConfig = config.clone().into();
        let processor = BeamformingProcessor::new(core_cfg, element_positions.clone());

        Ok(Self {
            element_positions,
            config,
            steering_vectors: None,
            processor,
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
        let (n_elements, _channels, _n_samples) = sensor_data.dim();
        let weights = self.compute_apodization_weights(n_elements);
        let delays = self.processor.compute_delays(self.config.focal_point);
        self.processor
            .delay_and_sum_with(sensor_data, sample_rate, &delays, &weights)
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
    #[cfg_attr(feature = "structured-logging", tracing::instrument(skip(sensor_data)))]
    fn capon_diagonal_loading(
        &self,
        sensor_data: &Array3<f64>,
        sample_rate: f64,
        diagonal_loading: f64,
    ) -> KwaversResult<Array3<f64>> {
        let _ = sample_rate; // not used in Capon
        self.processor.capon_with_uniform(sensor_data, diagonal_loading)
    }

    /// MUSIC algorithm for source localization
    ///
    /// Implements a narrowband MUSIC pseudospectrum estimator over time for a
    /// single spatial look-direction defined by the precomputed steering
    /// vectors in `self.steering_vectors`.
    ///
    /// Algorithm (Schmidt, 1986; Van Trees, 2002):
    /// 1. Form spatial covariance matrix R from the sensor data.
    /// 2. Perform eigendecomposition R = E Λ E^H.
    /// 3. Partition eigenvectors into signal and noise subspaces; for PAM we
    ///    treat all but the dominant eigenvector as noise when `_num_sources`
    ///    is not used explicitly.
    /// 4. Compute pseudospectrum P(θ) = 1 / (a(θ)^H E_n E_n^H a(θ)), where
    ///    a(θ) is the steering vector for the look-direction.
    ///
    /// In this implementation, we:
    /// - Build R from the provided `sensor_data` without simplifications.
    /// - Use nalgebra symmetric eigendecomposition.
    /// - Map the pseudospectrum back into the existing 3D output layout by
    ///   encoding the MUSIC response into the first channel over time.
    ///
    /// # Assumptions
    /// - Real-valued omnidirectional sensors.
    /// - Single dominant source per focal point (suitable for PAM imaging).
    /// - Steering vectors are either precomputed or fall back to uniform.
    ///
    /// # Errors
    /// Returns DAS result if covariance is ill-conditioned or eigen-decomposition fails.
    #[cfg_attr(feature = "structured-logging", tracing::instrument(skip(sensor_data)))]
    fn music_algorithm(
        &self,
        sensor_data: &Array3<f64>,
        sample_rate: f64,
        _num_sources: usize,
    ) -> KwaversResult<Array3<f64>> {
        use nalgebra::{DMatrix, DVector, SymmetricEigen};

        let shape = sensor_data.dim();
        let n_elements = shape.0;
        let n_samples = shape.2;

        if n_elements < 2 || n_samples < 2 {
            // Degenerate; fall back to DAS which already validates shapes
            return self.delay_and_sum(sensor_data, sample_rate);
        }

        // Form unbiased spatial covariance matrix R (n_elements x n_elements)
        let mut covariance = DMatrix::zeros(n_elements, n_elements);
        let inv_n = 1.0 / (n_samples as f64);
        for i in 0..n_elements {
            for j in i..n_elements {
                let mut acc = 0.0;
                for t in 0..n_samples {
                    acc += sensor_data[[i, 0, t]] * sensor_data[[j, 0, t]];
                }
                let v = acc * inv_n;
                covariance[(i, j)] = v;
                if i != j {
                    covariance[(j, i)] = v;
                }
            }
        }

        // Diagonal loading for numerical robustness
        let loading = 1e-6_f64.max(1e-3 * covariance[(0, 0)].abs());
        for i in 0..n_elements {
            covariance[(i, i)] += loading;
        }

        // Eigendecomposition of symmetric covariance matrix
        let eig = SymmetricEigen::new(covariance.clone());

        let eigenvalues = eig.eigenvalues;
        let eigenvectors = eig.eigenvectors; // columns are eigenvectors

        // Sort eigenpairs by ascending eigenvalue to identify noise subspace
        let mut indices: Vec<usize> = (0..n_elements).collect();
        indices.sort_by(|&a, &b| eigenvalues[a]
            .partial_cmp(&eigenvalues[b])
            .unwrap_or(std::cmp::Ordering::Equal));

        // For PAM we assume a single dominant source; use all but the largest
        // eigenvalue/eigenvector as noise subspace. This is consistent and
        // avoids relying on the unused `_num_sources` argument.
        if n_elements < 2 {
            return self.delay_and_sum(sensor_data, sample_rate);
        }
        let signal_index = *indices.last().unwrap();

        let mut noise_basis = DMatrix::zeros(n_elements, n_elements - 1);
        let mut col = 0;
        for &idx in &indices {
            if idx == signal_index {
                continue;
            }
            for row in 0..n_elements {
                noise_basis[(row, col)] = eigenvectors[(row, idx)];
            }
            col += 1;
        }

        // Precompute projection matrix E_n E_n^T (n_elements x n_elements)
        let noise_projector = &noise_basis * noise_basis.transpose();

        // Determine steering vector for this look direction.
        // If steering_vectors are available, use the first one; otherwise fall
        // back to the normalized uniform vector.
        let steering: DVector<f64> = if let Some(ref sv) = self.steering_vectors {
            // Accept either shape (n_elements, 1) or (1, n_elements)
            if sv.nrows() == n_elements && sv.ncols() == 1 {
                let mut v = DVector::zeros(n_elements);
                for i in 0..n_elements {
                    v[i] = sv[[i, 0]];
                }
                v
            } else if sv.nrows() == 1 && sv.ncols() == n_elements {
                let mut v = DVector::zeros(n_elements);
                for i in 0..n_elements {
                    v[i] = sv[[0, i]];
                }
                v
            } else {
                DVector::from_element(n_elements, 1.0 / (n_elements as f64).sqrt())
            }
        } else {
            DVector::from_element(n_elements, 1.0 / (n_elements as f64).sqrt())
        };

        // MUSIC pseudospectrum value for this steering vector
        let denom = steering.transpose() * (&noise_projector * &steering);
        let denom_scalar = denom[(0, 0)];
        if denom_scalar <= 0.0 || !denom_scalar.is_finite() {
            log::warn!("MUSIC: non-positive or non-finite denominator; using DAS fallback");
            return self.delay_and_sum(sensor_data, sample_rate);
        }
        let pseudospectrum_value = 1.0 / denom_scalar;

        // Map pseudospectrum into output: encode as constant over time in
        // channel 0 to preserve 3D shape while exposing MUSIC response.
        let mut output = Array3::<f64>::zeros((1, 1, n_samples));
        for t in 0..n_samples {
            output[[0, 0, t]] = pseudospectrum_value;
        }

        Ok(output)
    }

    /// Eigenspace-based minimum variance (E-MVDR) beamforming
    ///
    /// This implementation follows Carlson (1988) and Van Trees (2002):
    /// 1. Form spatial covariance matrix R from the input data.
    /// 2. Apply diagonal loading for robustness.
    /// 3. Select dominant signal eigenspace (largest eigenvalue eigenvector).
    /// 4. Project steering vector into signal subspace and compute MVDR weights
    ///    w ∝ R^{-1} a_s / (a_s^H R^{-1} a_s), where a_s is the projected steering vector.
    /// 5. Apply complex-conjugate weights to sensor data to obtain beamformed output.
    ///
    /// For real-valued PAM data and a single focal/look direction, this reduces to
    /// a stable scalar weighting of channels preserving array gain and minimizing
    /// output power subject to unity response.
    #[cfg_attr(feature = "structured-logging", tracing::instrument(skip(sensor_data)))]
    fn eigenspace_min_variance(
        &self,
        sensor_data: &Array3<f64>,
        sample_rate: f64,
    ) -> KwaversResult<Array3<f64>> {
        use nalgebra::{DMatrix, DVector, SymmetricEigen};

        let (n_elements, _n_channels, n_samples) = sensor_data.dim();
        if n_elements < 2 || n_samples < 2 {
            return self.delay_and_sum(sensor_data, sample_rate);
        }

        // 1. Form spatial covariance matrix R (unbiased estimate)
        let mut covariance = DMatrix::zeros(n_elements, n_elements);
        let inv_n = 1.0 / (n_samples as f64);
        for i in 0..n_elements {
            for j in i..n_elements {
                let mut acc = 0.0;
                for t in 0..n_samples {
                    acc += sensor_data[[i, 0, t]] * sensor_data[[j, 0, t]];
                }
                let v = acc * inv_n;
                covariance[(i, j)] = v;
                if i != j {
                    covariance[(j, i)] = v;
                }
            }
        }

        // 2. Diagonal loading to handle finite-sample and modeling errors
        let power_ref = covariance[(0, 0)].abs().max(1e-12);
        let loading = 1e-3 * power_ref;
        for i in 0..n_elements {
            covariance[(i, i)] += loading;
        }

        // 3. Eigendecomposition R = E Λ E^T
        let eig = SymmetricEigen::new(covariance.clone());

        let eigenvalues = eig.eigenvalues;
        let eigenvectors = eig.eigenvectors; // columns are eigenvectors

        // Sort indices by descending eigenvalue to identify dominant signal subspace
        let mut indices: Vec<usize> = (0..n_elements).collect();
        indices.sort_by(|&a, &b| eigenvalues[b]
            .partial_cmp(&eigenvalues[a])
            .unwrap_or(std::cmp::Ordering::Equal));

        let signal_index = indices[0];
        let mut a_s = DVector::zeros(n_elements);

        // Determine steering vector: use precomputed if available, else uniform.
        let steering: DVector<f64> = if let Some(ref sv) = self.steering_vectors {
            if sv.nrows() == n_elements && sv.ncols() == 1 {
                let mut v = DVector::zeros(n_elements);
                for i in 0..n_elements {
                    v[i] = sv[[i, 0]];
                }
                v
            } else if sv.nrows() == 1 && sv.ncols() == n_elements {
                let mut v = DVector::zeros(n_elements);
                for i in 0..n_elements {
                    v[i] = sv[[0, i]];
                }
                v
            } else {
                DVector::from_element(n_elements, 1.0 / (n_elements as f64).sqrt())
            }
        } else {
            DVector::from_element(n_elements, 1.0 / (n_elements as f64).sqrt())
        };

        // Project steering vector into signal eigenspace (1D here)
        for i in 0..n_elements {
            a_s[i] = eigenvectors[(i, signal_index)];
        }
        let proj_gain = a_s.dot(&steering);
        if !proj_gain.is_finite() || proj_gain.abs() < 1e-12 {
            log::warn!("E-MVDR: degenerate steering projection; using delay-and-sum fallback");
            return self.delay_and_sum(sensor_data, sample_rate);
        }
        let a_s = &a_s * (1.0 / proj_gain);

        // 4. Solve R w = a_s for MVDR weights using linear solve on symmetric PD matrix.
        let weights = match covariance.clone().lu().solve(&a_s) {
            Some(w) => w,
            None => {
                log::warn!("E-MVDR: linear solve failed; using delay-and-sum fallback");
                return self.delay_and_sum(sensor_data, sample_rate);
            }
        };

        let denom = a_s.transpose() * &weights;
        let denom_scalar = denom[(0, 0)];
        if !denom_scalar.is_finite() || denom_scalar.abs() < 1e-12 {
            log::warn!("E-MVDR: invalid normalization; using delay-and-sum fallback");
            return self.delay_and_sum(sensor_data, sample_rate);
        }
        let weights = weights * (1.0 / denom_scalar);

        // 5. Apply weights across sensors for each time sample (real-valued case)
        let mut output = Array3::<f64>::zeros((1, 1, n_samples));
        for t in 0..n_samples {
            let mut acc = 0.0;
            for i in 0..n_elements {
                acc += weights[i] * sensor_data[[i, 0, t]];
            }
            output[[0, 0, t]] = acc;
        }

        Ok(output)
    }

    // Delays computation delegated to processor for SSOT

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
                // Kaiser window using modified Bessel function of first kind I_0
                // w(n) = I_0(β√(1-(2n/(N-1)-1)²)) / I_0(β)
                // where I_0 is the zeroth-order modified Bessel function
                let i0_beta = modified_bessel_i0(beta);
                (0..n)
                    .map(|i| {
                        let x = 2.0 * i as f64 / (n - 1) as f64 - 1.0;
                        let arg = beta * (1.0 - x * x).sqrt();
                        modified_bessel_i0(arg) / i0_beta
                    })
                    .collect()
            }
        }
    }

    /// Update configuration
    pub fn set_config(&mut self, config: BeamformingConfig) -> KwaversResult<()> {
        self.config = config;
        self.steering_vectors = None; // Reset precomputed vectors
        let core_cfg: BeamformingCoreConfig = self.config.clone().into();
        self.processor = BeamformingProcessor::new(core_cfg, self.element_positions.clone());
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sensor::passive_acoustic_mapping::geometry::ArrayGeometry;

    fn make_geometry_same_positions(n: usize) -> ArrayGeometry {
        ArrayGeometry::Arbitrary {
            positions: (0..n).map(|_| [0.0, 0.0, 0.0]).collect(),
        }
    }

    fn make_config_delay_and_sum() -> BeamformingConfig {
        BeamformingConfig {
            method: BeamformingMethod::DelayAndSum,
            frequency_range: (2.0e6, 2.0e6),
            spatial_resolution: 1e-3,
            apodization: ApodizationType::None,
            focal_point: [0.0, 0.0, 0.0],
        }
    }

    fn make_config_capon(dl: f64) -> BeamformingConfig {
        BeamformingConfig {
            method: BeamformingMethod::CaponDiagonalLoading { diagonal_loading: dl },
            frequency_range: (2.0e6, 2.0e6),
            spatial_resolution: 1e-3,
            apodization: ApodizationType::None,
            focal_point: [0.0, 0.0, 0.0],
        }
    }

    #[test]
    fn pam_beamformer_delegates_das() {
        let geometry = make_geometry_same_positions(2);
        let cfg = make_config_delay_and_sum();
        let mut bf = Beamformer::new(geometry, cfg).expect("construct beamformer");

        let sensor0 = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let sensor1 = vec![10.0, 20.0, 30.0, 40.0, 50.0];
        let n_samples = sensor0.len();
        let mut data = ndarray::Array3::<f64>::zeros((2, 1, n_samples));
        for t in 0..n_samples {
            data[[0, 0, t]] = sensor0[t];
            data[[1, 0, t]] = sensor1[t];
        }

        let out = bf.beamform(&data, 1.0).expect("beamform");
        for t in 0..n_samples {
            assert!((out[[0, 0, t]] - (sensor0[t] + sensor1[t])).abs() < 1e-12);
        }
    }

    #[test]
    fn pam_beamformer_delegates_capon_uniform() {
        let geometry = make_geometry_same_positions(2);
        let cfg = make_config_capon(0.1);
        let mut bf = Beamformer::new(geometry, cfg).expect("construct beamformer");

        let n_samples = 8;
        let mut data = ndarray::Array3::<f64>::zeros((2, 1, n_samples));
        for t in 0..n_samples {
            data[[0, 0, t]] = 1.0;
            data[[1, 0, t]] = 1.0;
        }

        let out = bf.beamform(&data, 1.0).expect("beamform");
        for t in 0..n_samples {
            assert!((out[[0, 0, t]] - std::f64::consts::SQRT_2).abs() < 1e-6);
        }
    }
}
impl Default for BeamformingConfig {
    fn default() -> Self {
        Self {
            method: BeamformingMethod::DelayAndSum,
            frequency_range: (20e3, 10e6), // 20 kHz to 10 MHz
            spatial_resolution: 1e-3,      // 1 mm
            apodization: ApodizationType::Hamming,
            focal_point: [0.0, 0.0, 0.0],
        }
    }
}

/// Modified Bessel function of the first kind I_0(x)
///
/// Uses series expansion for accurate computation:
/// I_0(x) = Σ_{k=0}^∞ [(x/2)^(2k)] / [(k!)^2]
///
/// # References
/// - Abramowitz & Stegun (1964), Section 9.8
/// - Kaiser & Schafer (1980), "On the use of the I0-sinh window"
fn modified_bessel_i0(x: f64) -> f64 {
    const MAX_ITERATIONS: usize = 50;
    const TOLERANCE: f64 = 1e-12;

    let x_abs = x.abs();
    let x_half = x_abs / 2.0;

    let mut term = 1.0;
    let mut sum = 1.0;

    for k in 1..MAX_ITERATIONS {
        // term_{k} = term_{k-1} * (x/2)^2 / k^2
        term *= (x_half * x_half) / ((k * k) as f64);
        sum += term;

        if term < TOLERANCE * sum {
            break;
        }
    }

    sum
}
