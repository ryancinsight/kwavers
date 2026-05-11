//! Frequency-domain beamforming trait — FFT-bin steering vectors and sample
//! covariance for narrowband and adaptive subspace methods.

use ndarray::{Array1, Array2};
use num_complex::Complex64;

use crate::core::error::KwaversResult;

use super::Beamformer;

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
    /// R = (1/M) ∑ₘ₌₁ᴹ x(m) x(m)^H
    /// ```
    ///
    /// where M is the number of snapshots (time samples or frequency bins).
    ///
    /// # Errors
    ///
    /// Returns `Err(...)` if:
    /// - Input data has invalid shape or non-finite values
    /// - Insufficient snapshots (M < N) for full-rank estimation
    fn compute_covariance(&self, data: &Array2<Complex64>) -> KwaversResult<Array2<Complex64>>;
}
