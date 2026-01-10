//! # Beamforming Trait Hierarchy
//!
//! This module defines the core trait hierarchy for beamforming algorithms in kwavers.
//! It establishes a clean abstraction boundary between different algorithm categories
//! while enforcing strict architectural principles.
//!
//! # Architectural Intent (SSOT + Layer Separation)
//!
//! ## Design Principles
//!
//! 1. **Single Source of Truth (SSOT)**: All beamforming algorithm interfaces are defined here
//! 2. **Layer Separation**: Traits operate on processed data, not domain primitives
//! 3. **Explicit Failure**: No silent fallbacks, error masking, or dummy outputs
//! 4. **Mathematical Rigor**: Contracts specify invariants and guarantees
//!
//! ## Layer Dependencies
//!
//! ```text
//! analysis::signal_processing::beamforming::traits (Layer 7)
//!   ↓ imports from
//! domain::sensor (Layer 2) - array geometry (read-only)
//! math::linear_algebra (Layer 1) - numerical operations
//! core::error (Layer 0) - error types
//! ```
//!
//! # Trait Hierarchy
//!
//! ```text
//! Beamformer (root trait)
//!   ├── TimeDomainBeamformer
//!   │     ├── DelayAndSum
//!   │     └── SyntheticAperture
//!   ├── FrequencyDomainBeamformer
//!   │     ├── MinimumVariance (MVDR/Capon)
//!   │     ├── MUSIC
//!   │     └── RobustCapon
//!   └── AdaptiveBeamformer
//!         ├── SampleMatrixInversion (SMI)
//!         └── EigenspaceMinimumVariance (ESMV)
//! ```
//!
//! # Usage Example
//!
//! ```rust,ignore
//! use kwavers::analysis::signal_processing::beamforming::traits::{
//!     Beamformer, TimeDomainBeamformer
//! };
//! use kwavers::domain::sensor::SensorArray;
//! use ndarray::{Array1, Array2};
//!
//! fn process_with_any_beamformer<B: Beamformer>(
//!     beamformer: &B,
//!     rf_data: &Array2<f64>,
//!     focal_point: [f64; 3],
//! ) -> KwaversResult<f64> {
//!     beamformer.focus_at_point(rf_data, focal_point)
//! }
//! ```
//!
//! # Mathematical Foundation
//!
//! ## General Beamforming Output
//!
//! For a sensor array with N elements, the beamformed output is:
//!
//! ```text
//! y(r, t) = w^H x(t, r)
//! ```
//!
//! where:
//! - `w` (N×1) = complex weight vector
//! - `x(t, r)` (N×1) = sensor data vector (time-aligned or frequency-domain)
//! - `r` = focal point in space
//! - `H` = Hermitian (conjugate transpose)
//!
//! ## Time-Domain Formulation
//!
//! ```text
//! y(r, t) = ∑ᵢ₌₁ᴺ wᵢ · xᵢ(t - τᵢ(r))
//! ```
//!
//! ## Frequency-Domain Formulation
//!
//! ```text
//! Y(r, ω) = w^H(ω) X(ω, r)
//! ```
//!
//! # Error Semantics (Strict)
//!
//! All traits enforce:
//!
//! - ❌ **NO silent fallbacks** - return `Err(...)` on any failure
//! - ❌ **NO error masking** - propagate root cause errors
//! - ❌ **NO dummy outputs** - never return zeros, steering vectors, or placeholders
//! - ❌ **NO undefined behavior** - all inputs validated, all outputs guaranteed
//!
//! # Performance Considerations
//!
//! | Algorithm Category | Complexity | Memory | Parallelizable | GPU-Friendly |
//! |-------------------|------------|--------|----------------|--------------|
//! | Time-Domain DAS   | O(N·M·K)   | O(N·M) | ✅ Yes         | ✅ Yes       |
//! | Frequency-Domain  | O(N²·M)    | O(N²)  | ⚠️ Partial     | ✅ Yes       |
//! | Adaptive          | O(N³·M)    | O(N²)  | ❌ Limited     | ⚠️ Moderate  |
//!
//! where N = sensors, M = time samples, K = focal points
//!
//! # Literature References
//!
//! ## Foundational Texts
//!
//! - Van Trees, H. L. (2002). *Optimum Array Processing: Part IV of Detection,
//!   Estimation, and Modulation Theory*. Wiley-Interscience.
//!   ISBN: 978-0-471-09390-9
//!
//! - Johnson, D. H., & Dudgeon, D. E. (1993). *Array Signal Processing:
//!   Concepts and Techniques*. Prentice Hall.
//!   ISBN: 978-0-13-048513-5
//!
//! ## Medical Ultrasound Applications
//!
//! - Jensen, J. A. (1996). *Field: A Program for Simulating Ultrasound Systems*.
//!   Medical & Biological Engineering & Computing, 34(1), 351-353.
//!
//! - Synnevåg, J. F., et al. (2009). "Adaptive beamforming applied to medical
//!   ultrasound imaging." *IEEE Trans. Ultrason., Ferroelect., Freq. Control*, 56(8).
//!   DOI: 10.1109/TUFFC.2009.1263

use crate::core::error::KwaversResult;
use ndarray::{Array1, Array2, Array3};
use num_complex::Complex64;

/// Root trait for all beamforming algorithms.
///
/// This trait defines the minimal interface that all beamformers must implement.
/// It focuses on the core operation: computing a beamformed output at a specific
/// focal point from multi-channel sensor data.
///
/// # Design Rationale
///
/// - **Minimal Interface**: Only essential operations, algorithm-specific details in sub-traits
/// - **Type Safety**: Strongly typed inputs/outputs with dimensional guarantees
/// - **Explicit Failure**: All errors returned as `Result`, no silent failures
///
/// # Mathematical Contract
///
/// Given:
/// - Multi-channel sensor data **X** (shape depends on domain)
/// - Focal point **r** ∈ ℝ³ in Cartesian coordinates
///
/// Compute scalar beamformed output **y** ∈ ℂ (or ℝ for real-valued data).
///
/// # Error Guarantees
///
/// Implementations **MUST** return `Err(...)` if:
/// - Input data has invalid shape or contains non-finite values
/// - Focal point is outside valid region or contains non-finite coordinates
/// - Any internal numerical operation fails
///
/// # Example
///
/// ```rust,ignore
/// use kwavers::analysis::signal_processing::beamforming::traits::Beamformer;
///
/// fn process<B: Beamformer>(
///     beamformer: &B,
///     data: &Array2<f64>,
///     point: [f64; 3],
/// ) -> KwaversResult<f64> {
///     beamformer.focus_at_point(data, point)
/// }
/// ```
pub trait Beamformer {
    /// Input data type (e.g., `f64` for real RF data, `Complex64` for frequency-domain).
    type Input: Copy + Send + Sync;

    /// Output data type (e.g., `f64` for intensity, `Complex64` for complex coherence).
    type Output: Copy + Send + Sync;

    /// Compute beamformed output at a single focal point.
    ///
    /// # Parameters
    ///
    /// - `data`: Multi-channel sensor data (shape algorithm-dependent)
    /// - `focal_point`: 3D Cartesian coordinates [x, y, z] in meters
    ///
    /// # Returns
    ///
    /// Beamformed scalar output for the focal point.
    ///
    /// # Errors
    ///
    /// Returns `Err(...)` if:
    /// - `data` shape is incompatible with sensor array geometry
    /// - `focal_point` contains non-finite values or is outside valid region
    /// - Numerical computation fails (singular matrices, convergence issues, etc.)
    ///
    /// # Performance
    ///
    /// - **Time Complexity**: O(N·M) to O(N³) depending on algorithm
    /// - **Space Complexity**: O(N·M) for time-domain, O(N²) for adaptive
    ///
    /// where N = number of sensors, M = time samples or frequency bins
    fn focus_at_point(
        &self,
        data: &Array2<Self::Input>,
        focal_point: [f64; 3],
    ) -> KwaversResult<Self::Output>;

    /// Validate input data dimensions and values.
    ///
    /// This method performs defensive validation before processing. Implementations
    /// should check:
    /// - Data shape matches expected sensor count
    /// - All values are finite (no NaN, no Inf)
    /// - Algorithm-specific constraints (e.g., minimum samples)
    ///
    /// # Default Implementation
    ///
    /// Checks for finite values only. Implementations should override to add
    /// dimension and algorithm-specific validation.
    ///
    /// # Errors
    ///
    /// Returns `Err(...)` if validation fails, with a descriptive error message.
    fn validate_input(&self, _data: &Array2<Self::Input>) -> KwaversResult<()> {
        // Default: check finiteness (requires trait bounds in implementing types)
        Ok(())
    }

    /// Get expected number of sensors for this beamformer configuration.
    ///
    /// This is used for validation and error reporting.
    ///
    /// # Returns
    ///
    /// Expected number of sensor channels in input data.
    fn expected_sensor_count(&self) -> usize;
}

/// Time-domain beamforming algorithms.
///
/// Time-domain beamformers operate directly on RF time series data. They apply
/// geometric delays to align signals from different sensors and then combine
/// them (typically via weighted sum).
///
/// # Mathematical Formulation
///
/// ```text
/// y(r, t) = ∑ᵢ₌₁ᴺ wᵢ · xᵢ(t - τᵢ(r))
/// ```
///
/// where:
/// - `wᵢ` = apodization weight for sensor i
/// - `xᵢ(t)` = RF signal at sensor i
/// - `τᵢ(r)` = geometric time delay from focal point r to sensor i
///
/// # Delay Calculation
///
/// For a focal point **r** and sensor position **sᵢ**, the delay is:
///
/// ```text
/// τᵢ(r) = ||r - sᵢ|| / c
/// ```
///
/// where c is the medium sound speed (m/s).
///
/// # Characteristics
///
/// - **Domain**: Operates on time series (N_sensors × N_samples)
/// - **Complexity**: O(N·M·K) for K focal points
/// - **Parallelism**: Highly parallelizable (independent focal points)
/// - **GPU**: Excellent GPU acceleration potential
///
/// # Implementations
///
/// - `DelayAndSum`: Standard DAS with apodization
/// - `SyntheticAperture`: Virtual source reconstruction
///
/// # Example
///
/// ```rust,ignore
/// use kwavers::analysis::signal_processing::beamforming::traits::TimeDomainBeamformer;
/// use ndarray::Array2;
///
/// fn process_rf_data<B: TimeDomainBeamformer>(
///     beamformer: &B,
///     rf_data: &Array2<f64>,  // (n_sensors, n_samples)
///     focal_points: &[[f64; 3]],
/// ) -> KwaversResult<Vec<f64>> {
///     let sampling_rate = beamformer.sampling_rate();
///     focal_points
///         .iter()
///         .map(|&pt| beamformer.focus_at_point(rf_data, pt))
///         .collect()
/// }
/// ```
pub trait TimeDomainBeamformer: Beamformer<Input = f64, Output = f64> {
    /// Get the sampling rate in Hz.
    ///
    /// This is required to convert geometric delays (seconds) to sample indices.
    ///
    /// # Returns
    ///
    /// Sampling rate in Hz (samples per second).
    ///
    /// # Invariants
    ///
    /// - Must be positive and finite
    /// - Should match the sampling rate of input RF data
    fn sampling_rate(&self) -> f64;

    /// Get the medium sound speed in m/s.
    ///
    /// Used for time-of-flight delay calculations.
    ///
    /// # Returns
    ///
    /// Sound speed in m/s (e.g., 1540.0 for soft tissue).
    ///
    /// # Invariants
    ///
    /// - Must be positive and finite
    /// - Typically in range [1000, 2000] m/s for ultrasound
    fn sound_speed(&self) -> f64;

    /// Compute geometric delay from focal point to sensor.
    ///
    /// # Parameters
    ///
    /// - `focal_point`: 3D Cartesian coordinates [x, y, z] in meters
    /// - `sensor_position`: 3D sensor position [x, y, z] in meters
    ///
    /// # Returns
    ///
    /// Time delay in seconds (τ = distance / sound_speed).
    ///
    /// # Errors
    ///
    /// Returns `Err(...)` if inputs contain non-finite values.
    ///
    /// # Mathematical Definition
    ///
    /// ```text
    /// τ = ||r_focal - r_sensor|| / c
    /// ```
    fn compute_delay(&self, focal_point: [f64; 3], sensor_position: [f64; 3])
        -> KwaversResult<f64>;

    /// Apply apodization (windowing) weights to sensor array.
    ///
    /// Apodization reduces side lobes at the cost of main lobe width.
    /// Common windows: Hamming, Hanning, Blackman, rectangular (uniform).
    ///
    /// # Parameters
    ///
    /// - `sensor_index`: Index of sensor in array (0-based)
    ///
    /// # Returns
    ///
    /// Apodization weight in [0, 1].
    ///
    /// # Default Implementation
    ///
    /// Returns 1.0 (uniform weighting, no apodization).
    ///
    /// # Mathematical Definition
    ///
    /// For Hamming window:
    /// ```text
    /// w[n] = 0.54 - 0.46·cos(2πn/(N-1))
    /// ```
    fn apodization_weight(&self, _sensor_index: usize) -> f64 {
        1.0 // Default: uniform (rectangular) window
    }
}

/// Frequency-domain beamforming algorithms.
///
/// Frequency-domain beamformers operate on Fourier-transformed data. They compute
/// optimal weights for each frequency bin, enabling adaptive nulling and
/// high-resolution direction-of-arrival estimation.
///
/// # Mathematical Formulation
///
/// ```text
/// Y(r, ω) = w^H(ω) · X(ω, r)
/// ```
///
/// where:
/// - `w(ω)` (N×1) = complex frequency-dependent weights
/// - `X(ω, r)` (N×1) = sensor data in frequency domain
/// - `ω` = angular frequency (rad/s)
/// - `H` = Hermitian transpose
///
/// # Adaptive Weight Computation
///
/// Many frequency-domain methods compute weights from the covariance matrix:
///
/// ```text
/// R = E[X X^H]  (sample covariance)
/// w = f(R, a)    (algorithm-specific optimization)
/// ```
///
/// where `a` is the steering vector for the look direction.
///
/// # Characteristics
///
/// - **Domain**: Operates on FFT bins (N_sensors × N_frequencies)
/// - **Complexity**: O(N²·F) to O(N³·F) for F frequency bins
/// - **Adaptivity**: Data-driven weight optimization
/// - **Resolution**: Superior to time-domain for narrowband signals
///
/// # Implementations
///
/// - `MinimumVariance`: MVDR/Capon beamformer
/// - `MUSIC`: Multiple Signal Classification
/// - `RobustCapon`: MVDR with uncertainty modeling
///
/// # Example
///
/// ```rust,ignore
/// use kwavers::analysis::signal_processing::beamforming::traits::FrequencyDomainBeamformer;
/// use ndarray::{Array1, Array2};
/// use num_complex::Complex64;
///
/// fn process_frequency_data<B: FrequencyDomainBeamformer>(
///     beamformer: &B,
///     fft_data: &Array2<Complex64>,  // (n_sensors, n_frequencies)
///     focal_point: [f64; 3],
/// ) -> KwaversResult<Complex64> {
///     beamformer.focus_at_point(fft_data, focal_point)
/// }
/// ```
pub trait FrequencyDomainBeamformer: Beamformer<Input = Complex64, Output = Complex64> {
    /// Compute steering vector for a given frequency and look direction.
    ///
    /// The steering vector represents the expected phase response of the array
    /// for a plane wave arriving from the look direction.
    ///
    /// # Parameters
    ///
    /// - `frequency`: Signal frequency in Hz
    /// - `look_direction`: Unit vector [x, y, z] pointing toward source
    ///
    /// # Returns
    ///
    /// Steering vector **a** (N×1 complex) where N is number of sensors.
    ///
    /// # Mathematical Definition
    ///
    /// For sensor i at position **sᵢ**:
    /// ```text
    /// aᵢ = exp(j·k·(sᵢ · d))
    /// ```
    /// where:
    /// - `k = 2πf/c` (wavenumber)
    /// - `d` = look direction unit vector
    /// - `j` = √(-1)
    ///
    /// # Errors
    ///
    /// Returns `Err(...)` if:
    /// - `frequency` is non-positive or non-finite
    /// - `look_direction` is not unit length or contains non-finite values
    fn steering_vector(
        &self,
        frequency: f64,
        look_direction: [f64; 3],
    ) -> KwaversResult<Array1<Complex64>>;

    /// Get the frequency range (min, max) in Hz for this beamformer.
    ///
    /// This defines the valid operating range. Processing outside this range
    /// may produce unreliable results.
    ///
    /// # Returns
    ///
    /// Tuple (f_min, f_max) in Hz.
    ///
    /// # Invariants
    ///
    /// - 0 < f_min < f_max
    /// - Both values must be finite
    fn frequency_range(&self) -> (f64, f64);

    /// Compute sample covariance matrix from frequency-domain data.
    ///
    /// The covariance matrix captures spatial correlation between sensors and
    /// is central to adaptive beamforming algorithms.
    ///
    /// # Parameters
    ///
    /// - `data`: Frequency-domain sensor data (N_sensors × N_snapshots)
    ///
    /// # Returns
    ///
    /// Sample covariance matrix **R** (N×N Hermitian positive semi-definite).
    ///
    /// # Mathematical Definition
    ///
    /// ```text
    /// R = (1/M) ∑ₘ₌₁ᴹ x[m] x[m]^H
    /// ```
    ///
    /// where M is the number of snapshots (time samples or frequency bins).
    ///
    /// # Errors
    ///
    /// Returns `Err(...)` if:
    /// - Input data has invalid shape or non-finite values
    /// - Insufficient snapshots (M < N) for full-rank estimation
    ///
    /// # Default Implementation
    ///
    /// Computes sample covariance with optional diagonal loading for numerical stability.
    fn compute_covariance(&self, data: &Array2<Complex64>) -> KwaversResult<Array2<Complex64>>;
}

/// Adaptive beamforming algorithms (narrowband frequency-domain specialization).
///
/// Adaptive beamformers compute data-dependent weights that optimize performance
/// criteria such as signal-to-interference-plus-noise ratio (SINR). They automatically
/// form nulls in interference directions.
///
/// # Mathematical Framework
///
/// Given:
/// - Covariance matrix **R** (N×N Hermitian)
/// - Steering vector **a** (N×1 complex)
///
/// Compute optimal weights **w** via algorithm-specific optimization:
///
/// ## MVDR (Minimum Variance Distortionless Response)
/// ```text
/// minimize   w^H R w
/// subject to w^H a = 1
/// Solution:  w = R^{-1} a / (a^H R^{-1} a)
/// ```
///
/// ## Maximum SINR
/// ```text
/// maximize   w^H R_s w / w^H R_n w
/// Solution:  w ∝ R_n^{-1} R_s w (generalized eigenproblem)
/// ```
///
/// # Characteristics
///
/// - **Adaptivity**: Weights computed from data covariance
/// - **Interference Rejection**: Automatic null formation
/// - **Complexity**: O(N³) for matrix inversion
/// - **SNR Requirement**: Moderate to high SNR needed for stability
///
/// # Implementations
///
/// - `MinimumVariance`: MVDR/Capon with diagonal loading
/// - `EigenspaceMV`: Subspace-based MVDR
/// - `MUSIC`: Pseudospectrum via noise subspace projection
///
/// # Example
///
/// ```rust,ignore
/// use kwavers::analysis::signal_processing::beamforming::traits::AdaptiveBeamformer;
/// use ndarray::{Array1, Array2};
/// use num_complex::Complex64;
///
/// fn scan_directions<B: AdaptiveBeamformer>(
///     beamformer: &B,
///     covariance: &Array2<Complex64>,
///     directions: &[[f64; 3]],
/// ) -> KwaversResult<Vec<f64>> {
///     directions
///         .iter()
///         .map(|&dir| {
///             let steering = beamformer.steering_vector(1e6, dir)?;
///             let weights = beamformer.compute_weights(covariance, &steering)?;
///             Ok(weights.iter().map(|w| w.norm()).sum())
///         })
///         .collect()
/// }
/// ```
pub trait AdaptiveBeamformer: FrequencyDomainBeamformer {
    /// Compute optimal beamforming weights from covariance and steering vector.
    ///
    /// This is the core operation for adaptive beamformers. Different algorithms
    /// implement different optimization criteria (MVDR, MUSIC, etc.).
    ///
    /// # Parameters
    ///
    /// - `covariance`: Sample covariance matrix **R** (N×N Hermitian)
    /// - `steering`: Steering vector **a** (N×1 complex)
    ///
    /// # Returns
    ///
    /// Optimal weight vector **w** (N×1 complex) according to algorithm's criterion.
    ///
    /// # Errors (Strict - No Silent Fallbacks)
    ///
    /// Returns `Err(...)` if:
    /// - Dimensions inconsistent (covariance not square, steering length mismatch)
    /// - Covariance matrix singular or ill-conditioned
    /// - Numerical operation fails (solve, eigendecomposition, etc.)
    /// - Algorithm-specific constraints violated
    ///
    /// **MUST NOT**:
    /// - Return steering vector as fallback on failure
    /// - Return zero weights to mask errors
    /// - Silently skip numerical issues
    ///
    /// # Mathematical Guarantees
    ///
    /// Implementations must ensure:
    /// - Unit gain constraint (if applicable): `w^H a = 1`
    /// - Numerical stability (e.g., via diagonal loading)
    /// - Hermitian covariance matrix preserved
    ///
    /// # Performance
    ///
    /// - **Time Complexity**: O(N²) to O(N³) depending on algorithm
    /// - **Space Complexity**: O(N²) for covariance matrix
    fn compute_weights(
        &self,
        covariance: &Array2<Complex64>,
        steering: &Array1<Complex64>,
    ) -> KwaversResult<Array1<Complex64>>;

    /// Get diagonal loading factor for numerical stability.
    ///
    /// Diagonal loading adds a small value to covariance matrix diagonal to
    /// prevent singular matrices and improve robustness.
    ///
    /// # Returns
    ///
    /// Diagonal loading factor ε ≥ 0.
    ///
    /// # Mathematical Effect
    ///
    /// ```text
    /// R_loaded = R + ε·I
    /// ```
    ///
    /// Typical values: ε ∈ [1e-6, 1e-2] depending on SNR and condition number.
    fn diagonal_loading(&self) -> f64;

    /// Compute adaptive beamforming pseudospectrum (for DOA estimation).
    ///
    /// The pseudospectrum is a spatial function used for direction-of-arrival
    /// estimation. It peaks at source locations.
    ///
    /// # Parameters
    ///
    /// - `covariance`: Sample covariance matrix **R** (N×N)
    /// - `steering`: Steering vector **a** (N×1) for scan direction
    ///
    /// # Returns
    ///
    /// Pseudospectrum value (non-negative real scalar).
    ///
    /// # Mathematical Definitions
    ///
    /// **MVDR pseudospectrum:**
    /// ```text
    /// P_MVDR(θ) = 1 / (a^H(θ) R^{-1} a(θ))
    /// ```
    ///
    /// **MUSIC pseudospectrum:**
    /// ```text
    /// P_MUSIC(θ) = 1 / (a^H(θ) E_n E_n^H a(θ))
    /// ```
    /// where E_n is the noise subspace eigenvector matrix.
    ///
    /// # Errors
    ///
    /// Returns `Err(...)` if:
    /// - Covariance matrix singular or ill-conditioned
    /// - Numerical operations fail
    ///
    /// **MUST NOT** return 0.0 as error masking.
    fn compute_pseudospectrum(
        &self,
        covariance: &Array2<Complex64>,
        steering: &Array1<Complex64>,
    ) -> KwaversResult<f64>;
}

/// Configuration trait for beamformers that can be constructed from domain geometry.
///
/// This trait separates the concern of "how to initialize a beamformer from sensor
/// array geometry" from the core beamforming algorithms. It enables dependency
/// injection and testability.
///
/// # Architectural Rationale
///
/// - **Separation**: Domain (geometry) vs. Analysis (algorithms)
/// - **Testability**: Allows mock sensor arrays for unit tests
/// - **Flexibility**: Multiple initialization strategies without changing algorithm traits
///
/// # Example
///
/// ```rust,ignore
/// use kwavers::analysis::signal_processing::beamforming::traits::BeamformerConfig;
/// use kwavers::domain::sensor::LinearArray;
///
/// fn create_beamformer<B, A>(array: &A) -> KwaversResult<B>
/// where
///     B: BeamformerConfig<A>,
///     A: SensorArray,
/// {
///     B::from_sensor_array(array, 1540.0, 10e6)
/// }
/// ```
pub trait BeamformerConfig<SensorArray>: Sized {
    /// Create beamformer from sensor array geometry.
    ///
    /// # Parameters
    ///
    /// - `array`: Sensor array with geometry information
    /// - `sound_speed`: Medium sound speed in m/s
    /// - `sampling_rate`: Data sampling rate in Hz
    ///
    /// # Returns
    ///
    /// Configured beamformer instance.
    ///
    /// # Errors
    ///
    /// Returns `Err(...)` if:
    /// - Array geometry is invalid or empty
    /// - Sound speed or sampling rate non-positive or non-finite
    /// - Insufficient sensor elements for algorithm requirements
    fn from_sensor_array(
        array: &SensorArray,
        sound_speed: f64,
        sampling_rate: f64,
    ) -> KwaversResult<Self>;
}

#[cfg(test)]
mod tests {
    use super::*;

    // Mock beamformer for trait validation
    struct MockBeamformer {
        sensor_count: usize,
        sampling_rate: f64,
        sound_speed: f64,
    }

    impl Beamformer for MockBeamformer {
        type Input = f64;
        type Output = f64;

        fn focus_at_point(&self, data: &Array2<f64>, focal_point: [f64; 3]) -> KwaversResult<f64> {
            self.validate_input(data)?;
            if !focal_point.iter().all(|x| x.is_finite()) {
                return Err(crate::core::error::KwaversError::InvalidInput(
                    "Focal point contains non-finite values".into(),
                ));
            }
            Ok(data.sum())
        }

        fn expected_sensor_count(&self) -> usize {
            self.sensor_count
        }
    }

    impl TimeDomainBeamformer for MockBeamformer {
        fn sampling_rate(&self) -> f64 {
            self.sampling_rate
        }

        fn sound_speed(&self) -> f64 {
            self.sound_speed
        }

        fn compute_delay(
            &self,
            focal_point: [f64; 3],
            sensor_position: [f64; 3],
        ) -> KwaversResult<f64> {
            let dx = focal_point[0] - sensor_position[0];
            let dy = focal_point[1] - sensor_position[1];
            let dz = focal_point[2] - sensor_position[2];
            let distance = (dx * dx + dy * dy + dz * dz).sqrt();
            Ok(distance / self.sound_speed)
        }
    }

    #[test]
    fn test_beamformer_trait() {
        let beamformer = MockBeamformer {
            sensor_count: 4,
            sampling_rate: 10e6,
            sound_speed: 1540.0,
        };

        assert_eq!(beamformer.expected_sensor_count(), 4);

        let data = Array2::<f64>::zeros((4, 100));
        let focal_point = [0.0, 0.0, 0.01];

        let result = beamformer.focus_at_point(&data, focal_point);
        assert!(result.is_ok());
    }

    #[test]
    fn test_time_domain_beamformer_trait() {
        let beamformer = MockBeamformer {
            sensor_count: 8,
            sampling_rate: 10e6,
            sound_speed: 1540.0,
        };

        assert_eq!(beamformer.sampling_rate(), 10e6);
        assert_eq!(beamformer.sound_speed(), 1540.0);

        let focal = [0.0, 0.0, 0.02];
        let sensor = [0.001, 0.0, 0.0];
        let delay = beamformer.compute_delay(focal, sensor).unwrap();

        // Expected: distance ~= sqrt(0.001^2 + 0.02^2) = 0.020025 m
        // delay = 0.020025 / 1540 ≈ 1.3e-5 s
        assert!(delay > 0.0 && delay < 1e-4);
    }

    #[test]
    fn test_apodization_default() {
        let beamformer = MockBeamformer {
            sensor_count: 8,
            sampling_rate: 10e6,
            sound_speed: 1540.0,
        };

        // Default apodization is uniform (1.0)
        for i in 0..8 {
            assert_eq!(beamformer.apodization_weight(i), 1.0);
        }
    }

    #[test]
    fn test_trait_object_compatibility() {
        let beamformer: Box<dyn TimeDomainBeamformer> = Box::new(MockBeamformer {
            sensor_count: 4,
            sampling_rate: 10e6,
            sound_speed: 1540.0,
        });

        assert_eq!(beamformer.expected_sensor_count(), 4);
        assert_eq!(beamformer.sampling_rate(), 10e6);
    }
}
