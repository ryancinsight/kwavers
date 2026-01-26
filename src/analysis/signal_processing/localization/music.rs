//! MUSIC (MUltiple SIgnal Classification) Source Localization
//!
//! This module implements the MUSIC algorithm for super-resolution acoustic source
//! localization. MUSIC exploits the eigenstructure of the sensor covariance matrix
//! to achieve spatial resolution beyond the classical diffraction limit.
//!
//! # Mathematical Foundation
//!
//! ## Signal Model
//!
//! For K narrowband sources at locations **r_k** and N sensors at **p_n**:
//!
//! ```text
//! x(t) = A(θ)·s(t) + n(t)
//! ```
//!
//! where:
//! - x(t) ∈ ℂ^N = sensor array output
//! - A(θ) = [a(θ₁), ..., a(θ_K)] = array manifold (N×K)
//! - s(t) ∈ ℂ^K = source signals
//! - n(t) ∈ ℂ^N = additive noise
//!
//! The steering vector a(θ) describes phase delays from source to sensors:
//!
//! ```text
//! a(θ)_n = exp(-jω τ_n(θ))
//! ```
//!
//! where τ_n(θ) = ||r(θ) - p_n|| / c is the propagation delay.
//!
//! ## Covariance Matrix Eigendecomposition
//!
//! The sensor covariance matrix R = E[x·x^H] has eigendecomposition:
//!
//! ```text
//! R = [U_s  U_n] [Λ_s   0 ] [U_s^H]
//!                 [0   Λ_n] [U_n^H]
//! ```
//!
//! where:
//! - U_s = signal subspace eigenvectors (span{a(θ₁), ..., a(θ_K)})
//! - U_n = noise subspace eigenvectors (orthogonal to signal subspace)
//! - Λ_s > σ² = signal eigenvalues
//! - Λ_n = σ² I = noise eigenvalues
//!
//! ## MUSIC Spatial Spectrum
//!
//! The MUSIC pseudospectrum is defined as:
//!
//! ```text
//! P_MUSIC(θ) = 1 / ||U_n^H·a(θ)||²
//! ```
//!
//! Key properties:
//! - P_MUSIC(θ) → ∞ when a(θ) ⊥ U_n (i.e., θ is a true source direction)
//! - Sharp peaks at source locations
//! - Super-resolution: can resolve sources closer than λ/2
//!
//! ## Subspace Orthogonality Principle
//!
//! For a true source at θ_k, the steering vector lies in the signal subspace:
//!
//! ```text
//! a(θ_k) ∈ span{U_s}  ⟹  a(θ_k) ⊥ U_n
//! ```
//!
//! MUSIC exploits this by searching for directions where a(θ) is orthogonal
//! to the noise subspace.
//!
//! ## Grid Search Procedure
//!
//! 1. Estimate covariance: R̂ = (1/T) Σ_t x(t)·x(t)^H
//! 2. Eigendecompose: [U, Λ] = eig(R̂)
//! 3. Partition: U_n = eigenvectors with K smallest eigenvalues
//! 4. Grid search: Evaluate P_MUSIC(θ) on spatial grid
//! 5. Peak detection: Find local maxima → source locations
//!
//! # Resolution and Performance
//!
//! ## Spatial Resolution
//!
//! MUSIC can resolve sources separated by:
//!
//! ```text
//! Δr ≈ λ / (2√(2·SNR·T))
//! ```
//!
//! where:
//! - λ = wavelength
//! - SNR = signal-to-noise ratio
//! - T = number of snapshots
//!
//! This is **sub-wavelength** (Δr < λ/2) for high SNR!
//!
//! ## Advantages
//!
//! - Super-resolution (beyond diffraction limit)
//! - No bias from source correlation
//! - Statistically consistent
//! - Works with arbitrary array geometry
//!
//! ## Limitations
//!
//! - Requires K < N (more sensors than sources)
//! - Assumes narrowband signals
//! - Sensitive to array calibration errors
//! - Requires eigendecomposition (O(N³))
//!
//! # References
//!
//! - Schmidt, R. O. (1986). "Multiple Emitter Location and Signal Parameter Estimation"
//!   *IEEE Transactions on Antennas and Propagation*, 34(3), 276-280
//!   DOI: 10.1109/TAP.1986.1143830
//!
//! - Bienvenu, G., & Kopp, L. (1983). "Optimality of High Resolution Array Processing
//!   Using the Eigensystem Approach" *IEEE Trans. ASSP*, 31(5), 1235-1248
//!
//! - Stoica, P., & Nehorai, A. (1989). "MUSIC, Maximum Likelihood, and Cramér-Rao Bound"
//!   *IEEE Trans. ASSP*, 37(5), 720-741
//!   DOI: 10.1109/29.17564

use crate::analysis::signal_processing::localization::LocalizationResult;
use crate::core::error::{KwaversError, KwaversResult};
use ndarray::{Array1, Array2};
use num_complex::Complex;
use serde::{Deserialize, Serialize};

/// Configuration for MUSIC localization
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MusicConfig {
    /// Sound speed in medium (m/s)
    pub sound_speed: f64,

    /// Center frequency of narrowband signals (Hz)
    pub frequency: f64,

    /// Number of sources to locate
    pub num_sources: usize,

    /// Grid search bounds in x-direction (m)
    pub x_bounds: [f64; 2],

    /// Grid search bounds in y-direction (m)
    pub y_bounds: [f64; 2],

    /// Grid search bounds in z-direction (m)
    pub z_bounds: [f64; 2],

    /// Grid resolution in each dimension (m)
    pub grid_resolution: f64,

    /// Minimum peak separation for multi-source detection (m)
    pub peak_separation: f64,

    /// Threshold for peak detection (relative to maximum)
    pub peak_threshold: f64,
}

impl Default for MusicConfig {
    fn default() -> Self {
        Self {
            sound_speed: 1540.0,
            frequency: 1e6, // 1 MHz
            num_sources: 1,
            x_bounds: [-0.02, 0.02],
            y_bounds: [-0.02, 0.02],
            z_bounds: [-0.02, 0.02],
            grid_resolution: 0.0002, // 0.2 mm
            peak_separation: 0.001,  // 1 mm
            peak_threshold: 0.5,
        }
    }
}

/// MUSIC source localizer using eigendecomposition
///
/// Provides super-resolution source localization via subspace methods.
/// Resolves multiple sources below the classical diffraction limit.
#[derive(Debug)]
pub struct MusicLocalizer {
    config: MusicConfig,
    sensor_positions: Vec<[f64; 3]>,
    num_sensors: usize,
}

/// MUSIC localization result with spatial spectrum
#[derive(Debug, Clone)]
pub struct MusicResult {
    /// Detected source positions
    pub sources: Vec<[f64; 3]>,

    /// Peak values at each source (pseudospectrum magnitude)
    pub peak_values: Vec<f64>,

    /// Full spatial spectrum (flattened 3D grid)
    pub spectrum: Vec<f64>,

    /// Grid dimensions [nx, ny, nz]
    pub grid_shape: [usize; 3],

    /// Signal subspace dimension used
    pub signal_subspace_dim: usize,

    /// Eigenvalues of covariance matrix
    pub eigenvalues: Vec<f64>,
}

impl MusicLocalizer {
    /// Create a new MUSIC localizer
    ///
    /// # Arguments
    ///
    /// * `sensor_positions` - Sensor array positions [[x, y, z], ...] (m)
    /// * `config` - MUSIC configuration
    ///
    /// # Returns
    ///
    /// Configured MUSIC localizer
    pub fn new(sensor_positions: Vec<[f64; 3]>, config: MusicConfig) -> KwaversResult<Self> {
        let num_sensors = sensor_positions.len();

        if num_sensors < config.num_sources + 1 {
            return Err(KwaversError::InvalidInput(format!(
                "MUSIC requires at least {} sensors for {} sources",
                config.num_sources + 1,
                config.num_sources
            )));
        }

        if config.sound_speed <= 0.0 {
            return Err(KwaversError::InvalidInput(
                "Sound speed must be positive".to_string(),
            ));
        }

        if config.frequency <= 0.0 {
            return Err(KwaversError::InvalidInput(
                "Frequency must be positive".to_string(),
            ));
        }

        if config.grid_resolution <= 0.0 {
            return Err(KwaversError::InvalidInput(
                "Grid resolution must be positive".to_string(),
            ));
        }

        Ok(Self {
            config,
            sensor_positions,
            num_sensors,
        })
    }

    /// Localize sources from sensor covariance matrix
    ///
    /// # Arguments
    ///
    /// * `covariance` - Estimated sensor covariance matrix (N×N complex)
    ///
    /// # Returns
    ///
    /// MUSIC result with detected source positions and spatial spectrum
    pub fn localize(&self, covariance: &Array2<Complex<f64>>) -> KwaversResult<MusicResult> {
        if covariance.nrows() != self.num_sensors || covariance.ncols() != self.num_sensors {
            return Err(KwaversError::InvalidInput(format!(
                "Covariance must be {}×{}, got {}×{}",
                self.num_sensors,
                self.num_sensors,
                covariance.nrows(),
                covariance.ncols()
            )));
        }

        // 1. Eigendecomposition of covariance matrix
        let (eigenvalues, eigenvectors) = self.hermitian_eigendecomposition(covariance)?;

        // 2. Sort eigenvalues/eigenvectors in descending order
        let mut eigen_pairs: Vec<(f64, Array1<Complex<f64>>)> = eigenvalues
            .iter()
            .enumerate()
            .map(|(i, &val)| (val, eigenvectors.column(i).to_owned()))
            .collect();
        eigen_pairs.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap());

        let sorted_eigenvalues: Vec<f64> = eigen_pairs.iter().map(|(val, _)| *val).collect();

        // 3. Extract noise subspace (smallest N - K eigenvectors)
        let noise_subspace_dim = self.num_sensors - self.config.num_sources;
        let mut noise_subspace: Array2<Complex<f64>> = Array2::from_elem(
            (self.num_sensors, noise_subspace_dim),
            Complex::new(0.0, 0.0),
        );

        for (i, (_, eigvec)) in eigen_pairs.iter().enumerate().skip(self.config.num_sources) {
            for j in 0..self.num_sensors {
                noise_subspace[[j, i - self.config.num_sources]] = eigvec[j];
            }
        }

        // 4. Grid search over spatial region
        let (spectrum, grid_shape) = self.compute_spatial_spectrum(&noise_subspace)?;

        // 5. Detect peaks (source locations)
        let (sources, peak_values) = self.detect_peaks(&spectrum, &grid_shape)?;

        Ok(MusicResult {
            sources,
            peak_values,
            spectrum,
            grid_shape,
            signal_subspace_dim: self.config.num_sources,
            eigenvalues: sorted_eigenvalues,
        })
    }

    /// Compute covariance matrix from snapshot data
    ///
    /// # Arguments
    ///
    /// * `snapshots` - Sensor data snapshots [N_sensors × N_snapshots]
    ///
    /// # Returns
    ///
    /// Sample covariance matrix R = (1/T) Σ x·x^H
    pub fn estimate_covariance(
        &self,
        snapshots: &Array2<Complex<f64>>,
    ) -> KwaversResult<Array2<Complex<f64>>> {
        let num_snapshots = snapshots.ncols();

        if snapshots.nrows() != self.num_sensors {
            return Err(KwaversError::InvalidInput(format!(
                "Snapshots must have {} rows (sensors), got {}",
                self.num_sensors,
                snapshots.nrows()
            )));
        }

        let mut covariance: Array2<Complex<f64>> =
            Array2::from_elem((self.num_sensors, self.num_sensors), Complex::new(0.0, 0.0));

        // R = (1/T) Σ_t x(t)·x(t)^H
        for t in 0..num_snapshots {
            let snapshot = snapshots.column(t);
            for i in 0..self.num_sensors {
                for j in 0..self.num_sensors {
                    covariance[[i, j]] += snapshot[i] * snapshot[j].conj();
                }
            }
        }

        // Normalize by number of snapshots
        let scale = 1.0 / (num_snapshots as f64);
        for elem in covariance.iter_mut() {
            *elem *= scale;
        }

        Ok(covariance)
    }

    /// Compute MUSIC spatial spectrum on 3D grid
    fn compute_spatial_spectrum(
        &self,
        noise_subspace: &Array2<Complex<f64>>,
    ) -> KwaversResult<(Vec<f64>, [usize; 3])> {
        let nx = ((self.config.x_bounds[1] - self.config.x_bounds[0]) / self.config.grid_resolution)
            .ceil() as usize;
        let ny = ((self.config.y_bounds[1] - self.config.y_bounds[0]) / self.config.grid_resolution)
            .ceil() as usize;
        let nz = ((self.config.z_bounds[1] - self.config.z_bounds[0]) / self.config.grid_resolution)
            .ceil() as usize;

        let mut spectrum = Vec::with_capacity(nx * ny * nz);
        let wavelength = self.config.sound_speed / self.config.frequency;
        let k = 2.0 * std::f64::consts::PI / wavelength; // wavenumber

        for iz in 0..nz {
            let z = self.config.z_bounds[0] + (iz as f64) * self.config.grid_resolution;
            for iy in 0..ny {
                let y = self.config.y_bounds[0] + (iy as f64) * self.config.grid_resolution;
                for ix in 0..nx {
                    let x = self.config.x_bounds[0] + (ix as f64) * self.config.grid_resolution;
                    let position = [x, y, z];

                    // Compute steering vector a(θ)
                    let steering_vector = self.compute_steering_vector(&position, k);

                    // Compute MUSIC pseudospectrum: P = 1 / ||U_n^H · a(θ)||²
                    let mut projection_norm_sq = 0.0;
                    for j in 0..noise_subspace.ncols() {
                        let mut dot = Complex::new(0.0, 0.0);
                        for i in 0..self.num_sensors {
                            dot += noise_subspace[[i, j]].conj() * steering_vector[i];
                        }
                        projection_norm_sq += dot.norm_sqr();
                    }

                    let pseudospectrum = if projection_norm_sq > 1e-20 {
                        1.0 / projection_norm_sq
                    } else {
                        1e20 // Avoid division by zero
                    };

                    spectrum.push(pseudospectrum);
                }
            }
        }

        Ok((spectrum, [nx, ny, nz]))
    }

    /// Compute steering vector for a given source position
    ///
    /// a(θ)_n = exp(-j k ||r(θ) - p_n||)
    ///
    /// # Arguments
    ///
    /// * `position` - Source position [x, y, z] (m)
    /// * `k` - Wavenumber 2π/λ (rad/m)
    ///
    /// # Returns
    ///
    /// Complex steering vector for all sensors
    pub fn compute_steering_vector(&self, position: &[f64; 3], k: f64) -> Vec<Complex<f64>> {
        self.sensor_positions
            .iter()
            .map(|sensor_pos| {
                let dx = position[0] - sensor_pos[0];
                let dy = position[1] - sensor_pos[1];
                let dz = position[2] - sensor_pos[2];
                let distance = (dx * dx + dy * dy + dz * dz).sqrt();
                let phase = -k * distance;
                Complex::new(phase.cos(), phase.sin())
            })
            .collect()
    }

    /// Detect peaks in spatial spectrum
    fn detect_peaks(
        &self,
        spectrum: &[f64],
        grid_shape: &[usize; 3],
    ) -> KwaversResult<(Vec<[f64; 3]>, Vec<f64>)> {
        let [nx, ny, nz] = *grid_shape;

        // Find global maximum
        let max_value = spectrum
            .iter()
            .copied()
            .max_by(|a, b| a.partial_cmp(b).unwrap())
            .unwrap_or(0.0);

        if max_value <= 0.0 {
            return Err(KwaversError::InvalidInput(
                "No peaks found in MUSIC spectrum".to_string(),
            ));
        }

        let threshold = max_value * self.config.peak_threshold;

        // Find all local maxima above threshold
        let mut peaks = Vec::new();
        let mut peak_values = Vec::new();

        for iz in 1..(nz - 1) {
            for iy in 1..(ny - 1) {
                for ix in 1..(nx - 1) {
                    let idx = iz * (nx * ny) + iy * nx + ix;
                    let value = spectrum[idx];

                    if value < threshold {
                        continue;
                    }

                    // Check if local maximum (3×3×3 neighborhood)
                    let mut is_local_max = true;
                    'outer: for dz in -1..=1 {
                        for dy in -1..=1 {
                            for dx in -1..=1 {
                                if dx == 0 && dy == 0 && dz == 0 {
                                    continue;
                                }
                                let neighbor_idx = (iz as i32 + dz) as usize * (nx * ny)
                                    + (iy as i32 + dy) as usize * nx
                                    + (ix as i32 + dx) as usize;
                                if spectrum[neighbor_idx] > value {
                                    is_local_max = false;
                                    break 'outer;
                                }
                            }
                        }
                    }

                    if is_local_max {
                        let x = self.config.x_bounds[0] + (ix as f64) * self.config.grid_resolution;
                        let y = self.config.y_bounds[0] + (iy as f64) * self.config.grid_resolution;
                        let z = self.config.z_bounds[0] + (iz as f64) * self.config.grid_resolution;
                        peaks.push([x, y, z]);
                        peak_values.push(value);
                    }
                }
            }
        }

        // Sort peaks by value (descending)
        let mut peak_pairs: Vec<([f64; 3], f64)> =
            peaks.into_iter().zip(peak_values.into_iter()).collect();
        peak_pairs.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

        // Apply peak separation constraint and limit to num_sources
        let mut filtered_peaks: Vec<[f64; 3]> = Vec::new();
        let mut filtered_values: Vec<f64> = Vec::new();

        for (pos, val) in peak_pairs {
            let mut too_close = false;
            for existing_pos in &filtered_peaks {
                let dx = pos[0] - existing_pos[0];
                let dy = pos[1] - existing_pos[1];
                let dz = pos[2] - existing_pos[2];
                let dist = (dx * dx + dy * dy + dz * dz).sqrt();
                if dist < self.config.peak_separation {
                    too_close = true;
                    break;
                }
            }

            if !too_close {
                filtered_peaks.push(pos);
                filtered_values.push(val);
                if filtered_peaks.len() >= self.config.num_sources {
                    break;
                }
            }
        }

        Ok((filtered_peaks, filtered_values))
    }

    /// Hermitian eigendecomposition (uses simplified algorithm)
    fn hermitian_eigendecomposition(
        &self,
        matrix: &Array2<Complex<f64>>,
    ) -> KwaversResult<(Array1<f64>, Array2<Complex<f64>>)> {
        use crate::math::linear_algebra::eigen::EigenDecomposition;

        // Delegate to existing eigendecomposition utility
        EigenDecomposition::hermitian_eigendecomposition_complex(matrix)
            .map_err(|e| KwaversError::InvalidInput(format!("Eigendecomposition failed: {}", e)))
    }

    /// Convert MUSIC result to standard LocalizationResult (single source)
    pub fn to_localization_result(music_result: &MusicResult) -> KwaversResult<LocalizationResult> {
        if music_result.sources.is_empty() {
            return Err(KwaversError::InvalidInput(
                "No sources detected in MUSIC result".to_string(),
            ));
        }

        // Use first (strongest) source
        let position = music_result.sources[0];
        let peak_value = music_result.peak_values[0];

        // Estimate uncertainty from peak sharpness
        let uncertainty = 1.0 / peak_value.sqrt().max(1.0);

        Ok(LocalizationResult {
            position,
            uncertainty,
            residual: 0.0, // Not applicable for MUSIC
            iterations: 0,
            converged: true,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;
    use ndarray::Array2;
    use num_complex::Complex;

    #[test]
    fn test_music_creation() {
        let sensors = vec![
            [0.0, 0.0, 0.0],
            [0.005, 0.0, 0.0],
            [0.0, 0.005, 0.0],
            [0.0, 0.0, 0.005],
        ];

        let config = MusicConfig {
            num_sources: 1,
            grid_resolution: 0.001,
            ..Default::default()
        };

        let music = MusicLocalizer::new(sensors, config);
        assert!(music.is_ok());
    }

    #[test]
    fn test_insufficient_sensors() {
        let sensors = vec![[0.0, 0.0, 0.0], [0.005, 0.0, 0.0]];

        let config = MusicConfig {
            num_sources: 2,
            ..Default::default()
        };

        assert!(MusicLocalizer::new(sensors, config).is_err());
    }

    #[test]
    fn test_steering_vector() {
        let sensors = vec![[0.0, 0.0, 0.0], [0.005, 0.0, 0.0], [0.0, 0.005, 0.0]];

        let config = MusicConfig {
            frequency: 1e6,
            sound_speed: 1500.0,
            ..Default::default()
        };

        let music = MusicLocalizer::new(sensors, config).unwrap();

        let position = [0.01, 0.0, 0.0];
        let wavelength = 1500.0 / 1e6;
        let k = 2.0 * std::f64::consts::PI / wavelength;

        let steering_vec = music.compute_steering_vector(&position, k);

        // Check magnitude (should be unity)
        for elem in &steering_vec {
            assert_relative_eq!(elem.norm(), 1.0, epsilon = 1e-10);
        }
    }

    #[test]
    fn test_covariance_estimation() {
        let sensors = vec![
            [0.0, 0.0, 0.0],
            [0.005, 0.0, 0.0],
            [0.0, 0.005, 0.0],
            [0.0, 0.0, 0.005],
        ];

        let config = MusicConfig::default();
        let music = MusicLocalizer::new(sensors, config).unwrap();

        // Create synthetic snapshots
        let num_snapshots = 100;
        let mut snapshots: Array2<Complex<f64>> =
            Array2::from_elem((4, num_snapshots), Complex::new(0.0, 0.0));

        for t in 0..num_snapshots {
            for i in 0..4 {
                let phase = 2.0 * std::f64::consts::PI * (t as f64) / (num_snapshots as f64);
                snapshots[[i, t]] = Complex::new(phase.cos(), phase.sin());
            }
        }

        let covariance = music.estimate_covariance(&snapshots).unwrap();

        // Check Hermitian property
        for i in 0..4 {
            for j in 0..4 {
                assert_relative_eq!(
                    covariance[[i, j]].re,
                    covariance[[j, i]].re,
                    epsilon = 1e-10
                );
                assert_relative_eq!(
                    covariance[[i, j]].im,
                    -covariance[[j, i]].im,
                    epsilon = 1e-10
                );
            }
        }
    }

    #[test]
    fn test_single_source_localization() {
        let c = 1500.0;
        let freq = 1e6;
        let wavelength = c / freq;

        // Small array (4 sensors in tetrahedral configuration)
        let sensors = vec![
            [0.0, 0.0, 0.0],
            [wavelength, 0.0, 0.0],
            [0.0, wavelength, 0.0],
            [0.0, 0.0, wavelength],
        ];

        let source_pos = [wavelength / 2.0, wavelength / 2.0, wavelength / 2.0];

        let config = MusicConfig {
            frequency: freq,
            sound_speed: c,
            num_sources: 1,
            x_bounds: [0.0, wavelength],
            y_bounds: [0.0, wavelength],
            z_bounds: [0.0, wavelength],
            grid_resolution: wavelength / 10.0,
            peak_separation: wavelength / 20.0,
            ..Default::default()
        };

        let music = MusicLocalizer::new(sensors.clone(), config).unwrap();

        // Generate synthetic covariance from single source
        let k = 2.0 * std::f64::consts::PI / wavelength;
        let steering_vec = music.compute_steering_vector(&source_pos, k);

        // R = a(θ)·a(θ)^H + σ²I (signal + noise)
        let mut covariance: Array2<Complex<f64>> =
            Array2::from_elem((4, 4), Complex::new(0.0, 0.0));
        let signal_power = 10.0;
        let noise_power = 0.1;

        for i in 0..4 {
            for j in 0..4 {
                covariance[[i, j]] = steering_vec[i] * steering_vec[j].conj() * signal_power;
                if i == j {
                    covariance[[i, j]] += Complex::new(noise_power, 0.0);
                }
            }
        }

        let result = music.localize(&covariance).unwrap();

        // Should detect one source
        assert_eq!(result.sources.len(), 1);

        // Should be close to true position (within grid resolution)
        let detected = result.sources[0];
        assert!((detected[0] - source_pos[0]).abs() < wavelength / 5.0);
        assert!((detected[1] - source_pos[1]).abs() < wavelength / 5.0);
        assert!((detected[2] - source_pos[2]).abs() < wavelength / 5.0);
    }

    #[test]
    fn test_to_localization_result() {
        let music_result = MusicResult {
            sources: vec![[0.01, 0.02, 0.03]],
            peak_values: vec![100.0],
            spectrum: vec![],
            grid_shape: [10, 10, 10],
            signal_subspace_dim: 1,
            eigenvalues: vec![10.0, 1.0, 1.0, 1.0],
        };

        let loc_result = MusicLocalizer::to_localization_result(&music_result).unwrap();

        assert_eq!(loc_result.position, [0.01, 0.02, 0.03]);
        assert!(loc_result.converged);
    }
}
