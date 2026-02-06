//! MUSIC (Multiple Signal Classification) Algorithm
//!
//! Implements super-resolution direction-of-arrival (DoA) estimation using subspace methods.
//!
//! # Theory
//!
//! MUSIC exploits the eigenstructure of the spatial covariance matrix R to achieve
//! super-resolution source localization beyond the Rayleigh limit.
//!
//! Given M sensors and K sources (K < M):
//! - Signal subspace: Span of K steering vectors
//! - Noise subspace: Orthogonal complement (M-K dimensional)
//!
//! The MUSIC pseudospectrum is defined as:
//!
//! P_MUSIC(θ) = 1 / (a(θ)^H E_n E_n^H a(θ))
//!
//! where:
//! - a(θ) is the steering vector for location θ
//! - E_n is the noise subspace eigenvector matrix
//! - ^H denotes conjugate transpose
//!
//! Source locations correspond to peaks (nulls in denominator) where a(θ) ⊥ E_n.
//!
//! # References
//!
//! - Schmidt, R. O. (1986). "Multiple emitter location and signal parameter estimation"
//!   IEEE Trans. Antennas Propag., 34(3), 276-280.
//! - Stoica, P., & Nehorai, A. (1989). "MUSIC, maximum likelihood, and Cramér–Rao bound"
//!   IEEE Trans. Acoust., Speech, Signal Process., 37(5), 720-741.
//! - Van Trees, H. L. (2002). "Optimum Array Processing" - Part IV of Detection, Estimation,
//!   and Modulation Theory. Wiley-Interscience.
//! - Wax, M., & Kailath, T. (1985). "Detection of signals by information theoretic criteria"
//!   IEEE Trans. Acoust., Speech, Signal Process., 33(2), 387-392.

use super::config::LocalizationConfig;
use super::model_order::{ModelOrderConfig, ModelOrderCriterion, ModelOrderEstimator};
use crate::core::error::{KwaversError, KwaversResult, NumericalError};
use crate::domain::signal_processing::localization::{LocalizationProcessor, SourceLocation};
use crate::math::linear_algebra::EigenDecomposition;
use ndarray::{Array1, Array2};
use num_complex::Complex;

/// MUSIC configuration
#[derive(Debug, Clone)]
pub struct MUSICConfig {
    /// Base localization config
    pub config: LocalizationConfig,

    /// Number of sources to detect (None = automatic via AIC/MDL)
    pub num_sources: Option<usize>,

    /// Model order selection criterion (used if num_sources is None)
    pub model_order_criterion: ModelOrderCriterion,

    /// Search grid resolution (number of points per dimension)
    pub grid_resolution: usize,

    /// Search region bounds [xmin, xmax, ymin, ymax, zmin, zmax] in meters
    pub search_bounds: [f64; 6],

    /// Minimum separation between detected sources [m]
    pub min_source_separation: f64,

    /// Number of temporal snapshots for covariance estimation
    pub num_snapshots: usize,

    /// Diagonal loading factor for covariance regularization
    ///
    /// Added to diagonal: R_reg = R + δI where δ = loading_factor × trace(R)/M
    /// Prevents ill-conditioning. Typical value: 1e-6 to 1e-3.
    pub diagonal_loading: f64,

    /// Center frequency for steering vector calculation [Hz]
    pub center_frequency: f64,
}

impl MUSICConfig {
    /// Create new MUSIC configuration
    ///
    /// # Arguments
    ///
    /// * `config` - Base localization config with sensor positions
    /// * `num_sources` - Number of sources (None = automatic estimation)
    pub fn new(config: LocalizationConfig, num_sources: Option<usize>) -> Self {
        let center_frequency = config.sampling_frequency / 4.0; // Default to Nyquist/2
        Self {
            config,
            num_sources,
            model_order_criterion: ModelOrderCriterion::MDL,
            grid_resolution: 50,
            search_bounds: [-0.1, 0.1, -0.1, 0.1, -0.1, 0.1],
            min_source_separation: 0.01,
            num_snapshots: 100,
            diagonal_loading: 1e-6,
            center_frequency,
        }
    }

    /// Set model order selection criterion
    pub fn with_criterion(mut self, criterion: ModelOrderCriterion) -> Self {
        self.model_order_criterion = criterion;
        self
    }

    /// Set grid resolution
    pub fn with_grid_resolution(mut self, resolution: usize) -> Self {
        self.grid_resolution = resolution;
        self
    }

    /// Set search region bounds
    pub fn with_search_bounds(mut self, bounds: [f64; 6]) -> Self {
        self.search_bounds = bounds;
        self
    }

    /// Set minimum source separation
    pub fn with_min_separation(mut self, separation: f64) -> Self {
        self.min_source_separation = separation;
        self
    }

    /// Set number of snapshots
    pub fn with_num_snapshots(mut self, snapshots: usize) -> Self {
        self.num_snapshots = snapshots;
        self
    }

    /// Set diagonal loading factor
    pub fn with_diagonal_loading(mut self, loading: f64) -> Self {
        self.diagonal_loading = loading;
        self
    }

    /// Set center frequency
    pub fn with_center_frequency(mut self, frequency: f64) -> Self {
        self.center_frequency = frequency;
        self
    }
}

impl Default for MUSICConfig {
    fn default() -> Self {
        Self::new(LocalizationConfig::default(), Some(1))
    }
}

/// MUSIC result with multiple sources
#[derive(Debug, Clone)]
pub struct MUSICResult {
    /// Detected source locations
    pub sources: Vec<SourceLocation>,

    /// MUSIC pseudospectrum (flattened grid)
    pub pseudospectrum: Vec<f64>,

    /// Grid dimensions [nx, ny, nz]
    pub grid_dims: [usize; 3],

    /// Search bounds used [xmin, xmax, ymin, ymax, zmin, zmax]
    pub search_bounds: [f64; 6],

    /// Number of sources detected
    pub num_sources: usize,

    /// Noise subspace dimension
    pub noise_subspace_dim: usize,
}

/// MUSIC processor for direction-of-arrival estimation
#[derive(Debug)]
pub struct MUSICProcessor {
    config: MUSICConfig,
}

impl MUSICProcessor {
    /// Create new MUSIC processor
    pub fn new(config: &MUSICConfig) -> KwaversResult<Self> {
        config.config.validate()?;

        let num_sensors = config.config.sensor_positions.len();

        if num_sensors < 2 {
            return Err(KwaversError::InvalidInput(
                "MUSIC requires at least 2 sensors".to_string(),
            ));
        }

        if let Some(k) = config.num_sources {
            if k == 0 {
                return Err(KwaversError::InvalidInput(
                    "Number of sources must be > 0".to_string(),
                ));
            }

            if k >= num_sensors {
                return Err(KwaversError::InvalidInput(format!(
                    "Number of sources ({}) must be < number of sensors ({})",
                    k, num_sensors
                )));
            }
        }

        if config.num_snapshots < num_sensors {
            return Err(KwaversError::InvalidInput(format!(
                "Number of snapshots ({}) must be ≥ number of sensors ({})",
                config.num_snapshots, num_sensors
            )));
        }

        if config.grid_resolution == 0 {
            return Err(KwaversError::InvalidInput(
                "Grid resolution must be > 0".to_string(),
            ));
        }

        Ok(Self {
            config: config.clone(),
        })
    }

    /// Estimate spatial covariance matrix from complex sensor snapshots
    ///
    /// # Arguments
    ///
    /// * `snapshots` - Complex sensor data (M sensors × N snapshots)
    ///
    /// # Returns
    ///
    /// Spatial covariance matrix R = (1/N) ∑ₙ x(n) x(n)^H ∈ ℂ^(M×M)
    ///
    /// # Mathematical Properties
    ///
    /// - R is Hermitian: R^H = R
    /// - R is positive semi-definite: x^H R x ≥ 0 for all x
    /// - Eigenvalues are real and non-negative
    pub fn estimate_covariance(
        &self,
        snapshots: &Array2<Complex<f64>>,
    ) -> KwaversResult<Array2<Complex<f64>>> {
        let (num_sensors, num_snapshots) = snapshots.dim();

        if num_snapshots == 0 {
            return Err(KwaversError::InvalidInput(
                "Cannot compute covariance from zero snapshots".to_string(),
            ));
        }

        // R = (1/N) X X^H where X is M×N snapshot matrix
        let mut covariance = Array2::zeros((num_sensors, num_sensors));

        for i in 0..num_sensors {
            for j in 0..num_sensors {
                let mut sum = Complex::new(0.0, 0.0);
                for k in 0..num_snapshots {
                    sum += snapshots[[i, k]] * snapshots[[j, k]].conj();
                }
                covariance[[i, j]] = sum / num_snapshots as f64;
            }
        }

        // Apply diagonal loading for regularization: R_reg = R + δI
        if self.config.diagonal_loading > 0.0 {
            let trace: Complex<f64> = (0..num_sensors).map(|i| covariance[[i, i]]).sum();
            let loading = self.config.diagonal_loading * (trace / num_sensors as f64).re;

            for i in 0..num_sensors {
                covariance[[i, i]] += Complex::new(loading, 0.0);
            }
        }

        Ok(covariance)
    }

    /// Compute steering vector for given source location
    ///
    /// # Arguments
    ///
    /// * `source_position` - Source location [x, y, z]
    /// * `sensor_positions` - Array element positions [[x, y, z], ...]
    /// * `frequency` - Signal frequency [Hz]
    /// * `speed_of_sound` - Propagation speed [m/s]
    ///
    /// # Returns
    ///
    /// Steering vector a(θ) ∈ ℂ^M where a_m = exp(-j 2π f τ_m)
    /// and τ_m is the time delay to sensor m
    ///
    /// # Theory
    ///
    /// For narrowband signals, the steering vector encodes the phase delays
    /// from the source to each sensor:
    ///
    /// a_m(θ) = exp(-j 2π f ||θ - r_m|| / c)
    ///
    /// where θ is source location, r_m is sensor position, c is speed of sound.
    fn steering_vector(
        source_position: [f64; 3],
        sensor_positions: &[[f64; 3]],
        frequency: f64,
        speed_of_sound: f64,
    ) -> Array1<Complex<f64>> {
        let num_sensors = sensor_positions.len();
        let mut steering = Array1::zeros(num_sensors);

        let k = 2.0 * std::f64::consts::PI * frequency / speed_of_sound;

        for (m, sensor_pos) in sensor_positions.iter().enumerate() {
            // Distance from source to sensor
            let dx = source_position[0] - sensor_pos[0];
            let dy = source_position[1] - sensor_pos[1];
            let dz = source_position[2] - sensor_pos[2];
            let distance = (dx * dx + dy * dy + dz * dz).sqrt();

            // Phase delay: exp(-j k r)
            let phase = -k * distance;
            steering[m] = Complex::new(phase.cos(), phase.sin());
        }

        steering
    }

    /// Compute MUSIC pseudospectrum over 3D search grid
    ///
    /// # Arguments
    ///
    /// * `noise_eigenvectors` - Noise subspace eigenvector matrix E_n (M × (M-K))
    ///
    /// # Returns
    ///
    /// Tuple of (pseudospectrum, grid_dims) where pseudospectrum is flattened 3D array
    ///
    /// # Algorithm
    ///
    /// For each point θ in the search grid:
    /// 1. Compute steering vector a(θ)
    /// 2. Project onto noise subspace: proj = E_n E_n^H a(θ)
    /// 3. Compute pseudospectrum: P(θ) = 1 / ||proj||²
    ///
    /// Sources are located at peaks of P(θ) (nulls of denominator).
    fn compute_pseudospectrum(
        &self,
        noise_eigenvectors: &Array2<Complex<f64>>,
    ) -> KwaversResult<(Vec<f64>, [usize; 3])> {
        let sensor_positions = &self.config.config.sensor_positions;
        let frequency = self.config.center_frequency;
        let speed_of_sound = self.config.config.sound_speed;
        let res = self.config.grid_resolution;

        let [xmin, xmax, ymin, ymax, zmin, zmax] = self.config.search_bounds;

        let nx = res;
        let ny = res;
        let nz = res;

        let dx = if nx > 1 {
            (xmax - xmin) / (nx - 1) as f64
        } else {
            0.0
        };
        let dy = if ny > 1 {
            (ymax - ymin) / (ny - 1) as f64
        } else {
            0.0
        };
        let dz = if nz > 1 {
            (zmax - zmin) / (nz - 1) as f64
        } else {
            0.0
        };

        let mut pseudospectrum = vec![0.0; nx * ny * nz];

        // Precompute E_n E_n^H for efficiency
        let num_sensors = sensor_positions.len();
        let noise_subspace_dim = noise_eigenvectors.ncols();
        let mut noise_projector = Array2::zeros((num_sensors, num_sensors));

        for i in 0..num_sensors {
            for j in 0..num_sensors {
                let mut sum = Complex::new(0.0, 0.0);
                for k in 0..noise_subspace_dim {
                    sum += noise_eigenvectors[[i, k]] * noise_eigenvectors[[j, k]].conj();
                }
                noise_projector[[i, j]] = sum;
            }
        }

        // Compute MUSIC spectrum over grid
        for ix in 0..nx {
            for iy in 0..ny {
                for iz in 0..nz {
                    let x = xmin + ix as f64 * dx;
                    let y = ymin + iy as f64 * dy;
                    let z = zmin + iz as f64 * dz;

                    let source_pos = [x, y, z];

                    // Compute steering vector a(θ)
                    let steering = Self::steering_vector(
                        source_pos,
                        sensor_positions,
                        frequency,
                        speed_of_sound,
                    );

                    // Compute a^H (E_n E_n^H) a = ||E_n^H a||²
                    let mut denominator = 0.0;
                    for i in 0..num_sensors {
                        for j in 0..num_sensors {
                            let term = steering[i].conj() * noise_projector[[i, j]] * steering[j];
                            denominator += term.re;
                        }
                    }

                    // MUSIC pseudospectrum: P(θ) = 1 / denominator
                    // Clamp to prevent division by zero
                    let p_music = if denominator > 1e-12 {
                        1.0 / denominator
                    } else {
                        1e12 // Very large value (sharp peak)
                    };

                    let idx = ix * ny * nz + iy * nz + iz;
                    pseudospectrum[idx] = p_music;
                }
            }
        }

        Ok((pseudospectrum, [nx, ny, nz]))
    }

    /// Find peaks in MUSIC pseudospectrum
    ///
    /// # Arguments
    ///
    /// * `pseudospectrum` - Flattened 3D MUSIC spectrum
    /// * `grid_dims` - Grid dimensions [nx, ny, nz]
    /// * `num_peaks` - Number of peaks to detect
    ///
    /// # Returns
    ///
    /// Vector of detected source locations sorted by spectrum magnitude
    fn find_peaks(
        &self,
        pseudospectrum: &[f64],
        grid_dims: [usize; 3],
        num_peaks: usize,
    ) -> Vec<SourceLocation> {
        let [nx, ny, nz] = grid_dims;
        let [xmin, xmax, ymin, ymax, zmin, zmax] = self.config.search_bounds;

        // Find local maxima in 3D grid
        let mut candidates = Vec::new();

        for ix in 1..nx - 1 {
            for iy in 1..ny - 1 {
                for iz in 1..nz - 1 {
                    let idx = ix * ny * nz + iy * nz + iz;
                    let value = pseudospectrum[idx];

                    // Check if this is a local maximum (26-connectivity in 3D)
                    let mut is_local_max = true;
                    'neighbor_loop: for di in -1..=1 {
                        for dj in -1..=1 {
                            for dk in -1..=1 {
                                if di == 0 && dj == 0 && dk == 0 {
                                    continue;
                                }

                                let ni = (ix as i32 + di) as usize;
                                let nj = (iy as i32 + dj) as usize;
                                let nk = (iz as i32 + dk) as usize;

                                let neighbor_idx = ni * ny * nz + nj * nz + nk;
                                if pseudospectrum[neighbor_idx] > value {
                                    is_local_max = false;
                                    break 'neighbor_loop;
                                }
                            }
                        }
                    }

                    if is_local_max {
                        // Convert grid indices to physical coordinates
                        let x = xmin + ix as f64 * (xmax - xmin) / (nx - 1) as f64;
                        let y = ymin + iy as f64 * (ymax - ymin) / (ny - 1) as f64;
                        let z = zmin + iz as f64 * (zmax - zmin) / (nz - 1) as f64;

                        candidates.push((value, [x, y, z]));
                    }
                }
            }
        }

        // Sort by spectrum magnitude (descending)
        candidates.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));

        // Filter by minimum separation and take top N
        let mut sources: Vec<SourceLocation> = Vec::new();
        for (magnitude, position) in candidates {
            // Check separation from already detected sources
            let mut too_close = false;
            for existing in &sources {
                let dx = position[0] - existing.position[0];
                let dy = position[1] - existing.position[1];
                let dz = position[2] - existing.position[2];
                let distance = (dx * dx + dy * dy + dz * dz).sqrt();

                if distance < self.config.min_source_separation {
                    too_close = true;
                    break;
                }
            }

            if !too_close {
                sources.push(SourceLocation {
                    position,
                    confidence: magnitude.log10().clamp(0.0, 1.0), // Normalize
                    uncertainty: 1.0 / magnitude.sqrt().max(1.0),
                });

                if sources.len() >= num_peaks {
                    break;
                }
            }
        }

        sources
    }

    /// Run complete MUSIC algorithm
    ///
    /// # Arguments
    ///
    /// * `snapshots` - Complex sensor snapshots (M sensors × N time samples)
    ///
    /// # Returns
    ///
    /// `MUSICResult` containing detected sources and pseudospectrum
    ///
    /// # Algorithm
    ///
    /// 1. Estimate spatial covariance matrix R from snapshots
    /// 2. Compute eigendecomposition R = V Λ V^H
    /// 3. Determine number of sources (if not specified) via AIC/MDL
    /// 4. Partition into signal and noise subspaces
    /// 5. Compute MUSIC pseudospectrum over search grid
    /// 6. Detect peaks corresponding to source locations
    pub fn run(&self, snapshots: &Array2<Complex<f64>>) -> KwaversResult<MUSICResult> {
        let num_sensors = snapshots.nrows();
        let num_snapshots = snapshots.ncols();

        // Step 1: Estimate spatial covariance matrix
        let covariance = self.estimate_covariance(snapshots)?;

        // Step 2: Eigendecomposition
        let (eigenvalues, eigenvectors) =
            EigenDecomposition::hermitian_eigendecomposition_complex(&covariance)?;

        // Step 3: Determine number of sources
        let num_sources = if let Some(k) = self.config.num_sources {
            k
        } else {
            // Automatic source estimation via AIC/MDL
            let model_config = ModelOrderConfig::new(num_sensors, num_snapshots)?
                .with_criterion(self.config.model_order_criterion);
            let estimator = ModelOrderEstimator::new(model_config)?;

            // Convert to real eigenvalues (they should already be real for Hermitian matrix)
            let real_eigenvalues: Vec<f64> = eigenvalues.to_vec();

            let result = estimator.estimate(&real_eigenvalues)?;
            result.num_sources
        };

        if num_sources == 0 {
            return Ok(MUSICResult {
                sources: Vec::new(),
                pseudospectrum: Vec::new(),
                grid_dims: [0, 0, 0],
                search_bounds: self.config.search_bounds,
                num_sources: 0,
                noise_subspace_dim: num_sensors,
            });
        }

        // Step 4: Extract noise subspace (last M-K eigenvectors)
        let noise_subspace_dim = num_sensors - num_sources;
        let mut noise_eigenvectors = Array2::zeros((num_sensors, noise_subspace_dim));

        for i in 0..num_sensors {
            for j in 0..noise_subspace_dim {
                // Noise eigenvectors are the last M-K columns (smallest eigenvalues)
                noise_eigenvectors[[i, j]] = eigenvectors[[i, num_sources + j]];
            }
        }

        // Step 5: Compute MUSIC pseudospectrum
        let (pseudospectrum, grid_dims) = self.compute_pseudospectrum(&noise_eigenvectors)?;

        // Step 6: Find peaks
        let sources = self.find_peaks(&pseudospectrum, grid_dims, num_sources);

        Ok(MUSICResult {
            sources,
            pseudospectrum,
            grid_dims,
            search_bounds: self.config.search_bounds,
            num_sources,
            noise_subspace_dim,
        })
    }
}

impl LocalizationProcessor for MUSICProcessor {
    fn localize(
        &self,
        _time_delays: &[f64],
        _sensor_positions: &[[f64; 3]],
    ) -> KwaversResult<SourceLocation> {
        // This interface expects single source from time delays, but MUSIC works
        // with complex snapshots. Return error indicating API mismatch.
        Err(KwaversError::Numerical(NumericalError::NotImplemented {
            feature: "MUSIC via time delays (use MUSICProcessor::run with snapshots instead)"
                .to_string(),
        }))
    }

    fn name(&self) -> &str {
        "MUSIC"
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_music_processor_creation() {
        let config = MUSICConfig::default();
        let result = MUSICProcessor::new(&config);
        assert!(result.is_ok());
    }

    #[test]
    fn test_music_invalid_num_sources_zero() {
        let mut config = MUSICConfig::default();
        config.num_sources = Some(0);
        let result = MUSICProcessor::new(&config);
        assert!(result.is_err());
    }

    #[test]
    fn test_music_invalid_num_sources_too_many() {
        let mut config = MUSICConfig::default();
        config.num_sources = Some(10); // More than sensors
        let result = MUSICProcessor::new(&config);
        assert!(result.is_err());
    }

    #[test]
    fn test_music_config_builder() {
        let config = MUSICConfig::default()
            .with_grid_resolution(100)
            .with_min_separation(0.05)
            .with_criterion(ModelOrderCriterion::AIC);

        assert_eq!(config.grid_resolution, 100);
        assert_eq!(config.min_source_separation, 0.05);
        assert_eq!(config.model_order_criterion, ModelOrderCriterion::AIC);
    }

    #[test]
    fn test_covariance_estimation() {
        let config = MUSICConfig::default();
        let processor = MUSICProcessor::new(&config).unwrap();

        // Create simple test snapshots (2 sensors, 10 snapshots)
        let snapshots = Array2::from_shape_fn((2, 10), |(i, j)| Complex::new((i + j) as f64, 0.0));

        let cov = processor.estimate_covariance(&snapshots).unwrap();

        // Check dimensions
        assert_eq!(cov.dim(), (2, 2));

        // Check Hermitian property: R[i,j] = conj(R[j,i])
        for i in 0..2 {
            for j in 0..2 {
                assert!((cov[[i, j]] - cov[[j, i]].conj()).norm() < 1e-10);
            }
        }
    }

    #[test]
    fn test_steering_vector() {
        let source_pos = [0.0, 0.0, 0.0];
        let sensor_positions = vec![[0.0, 0.0, 0.0], [0.1, 0.0, 0.0], [0.0, 0.1, 0.0]];
        let frequency = 1000.0; // 1 kHz
        let speed_of_sound = 1500.0; // m/s

        let steering = MUSICProcessor::steering_vector(
            source_pos,
            &sensor_positions,
            frequency,
            speed_of_sound,
        );

        assert_eq!(steering.len(), 3);

        // All sensors at source location should have phase 0 (magnitude 1, angle 0)
        for val in steering.iter() {
            assert!((val.norm() - 1.0).abs() < 1e-10);
        }
    }

    #[test]
    fn test_music_run_single_source() {
        let mut config = MUSICConfig::default();
        config.num_sources = Some(1);
        config.grid_resolution = 10;
        config.num_snapshots = 50;

        let processor = MUSICProcessor::new(&config).unwrap();

        // Create synthetic snapshots with one source
        // This is a simplified test - real data would have phase shifts
        let num_sensors = config.config.sensor_positions.len();
        let snapshots = Array2::from_shape_fn((num_sensors, 50), |(i, j)| {
            Complex::new((i + j) as f64 / 10.0, (i * j) as f64 / 20.0)
        });

        let result = processor.run(&snapshots);
        assert!(result.is_ok());

        let music_result = result.unwrap();
        assert_eq!(music_result.num_sources, 1);
        assert!(music_result.sources.len() <= 1);
    }

    #[test]
    fn test_music_automatic_source_detection() {
        let mut config = MUSICConfig::default();
        config.num_sources = None; // Automatic detection
        config.num_snapshots = 100;
        config.model_order_criterion = ModelOrderCriterion::MDL;

        let processor = MUSICProcessor::new(&config).unwrap();

        // Create snapshots with 2 clear signal eigenvalues
        let num_sensors = 4;
        let snapshots = Array2::from_shape_fn((num_sensors, 100), |(i, j)| {
            if i < 2 {
                Complex::new(10.0 * (i + j) as f64, 0.0)
            } else {
                Complex::new((i + j) as f64 / 10.0, 0.0)
            }
        });

        let result = processor.run(&snapshots);
        assert!(result.is_ok());
    }
}
