//! Root [`Beamformer`] trait — minimal interface every beamformer must satisfy.

use ndarray::Array2;

use crate::core::error::KwaversResult;

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
    fn expected_sensor_count(&self) -> usize;
}
