//! Core boundary condition traits: `BoundaryCondition`, `AbsorbingBoundary`,
//! `ReflectiveBoundary`, and `PeriodicBoundary`.

use crate::core::constants::fundamental::SOUND_SPEED_WATER_SIM;
use crate::core::error::KwaversResult;
use crate::domain::grid::GridTopology;
use ndarray::{Array3, ArrayViewMut3};
use std::fmt::Debug;

use super::types::{BoundaryDirections, BoundaryFieldType};

/// Core boundary condition trait.
///
/// Defines the interface that all boundary conditions must implement.
///
/// # Invariants
///
/// All implementations must preserve:
/// 1. **Energy Conservation**: Total energy cannot increase at boundaries.
/// 2. **Stability**: Boundary updates must be numerically stable.
/// 3. **Causality**: No dependence on future field values.
pub trait BoundaryCondition: Debug + Send + Sync {
    /// Return the boundary type name for logging/debugging.
    fn name(&self) -> &str;

    /// Return the directions where this boundary is active.
    /// # Errors
    /// - Propagates any [`KwaversError`] returned by called functions.
    ///
    fn active_directions(&self) -> BoundaryDirections;

    /// Apply the boundary to a scalar field in the spatial domain.
    ///
    /// For absorbing boundaries this typically applies:
    /// ```text
    /// u(x,t) → u(x,t) * exp(-σ(x) * Δt)
    /// ```
    /// # Errors
    /// - Propagates any [`KwaversError`] returned by called functions.
    ///
    fn apply_scalar_spatial(
        &mut self,
        field: ArrayViewMut3<f64>,
        grid: &dyn GridTopology,
        time_step: usize,
        dt: f64,
    ) -> KwaversResult<()>;

    /// Apply the boundary to a scalar field in the frequency domain.
    ///
    /// Used for spectral / k-space pseudospectral solvers.
    /// # Errors
    /// - Propagates any [`KwaversError`] returned by called functions.
    ///
    fn apply_scalar_frequency(
        &mut self,
        field: &mut Array3<num_complex::Complex<f64>>,
        grid: &dyn GridTopology,
        time_step: usize,
        dt: f64,
    ) -> KwaversResult<()>;

    /// Apply the boundary to a 3-component vector field in the spatial domain.
    ///
    /// Default: applies `apply_scalar_spatial` independently to each component.
    /// # Errors
    /// - Propagates any [`KwaversError`] returned by called functions.
    ///
    fn apply_vector_spatial(
        &mut self,
        field_x: ArrayViewMut3<f64>,
        field_y: ArrayViewMut3<f64>,
        field_z: ArrayViewMut3<f64>,
        grid: &dyn GridTopology,
        time_step: usize,
        dt: f64,
    ) -> KwaversResult<()> {
        self.apply_scalar_spatial(field_x, grid, time_step, dt)?;
        self.apply_scalar_spatial(field_y, grid, time_step, dt)?;
        self.apply_scalar_spatial(field_z, grid, time_step, dt)?;
        Ok(())
    }

    /// Return `true` if the boundary supports the given field type.
    ///
    /// Default: supports `Pressure`, `Velocity`, and `Displacement`.
    fn supports_field_type(&self, field_type: BoundaryFieldType) -> bool {
        matches!(
            field_type,
            BoundaryFieldType::Pressure
                | BoundaryFieldType::Velocity
                | BoundaryFieldType::Displacement
        )
    }

    /// Estimate the reflection coefficient magnitude `|r| ∈ [0, 1]` at a given angle.
    ///
    /// Default: perfect reflection (1.0).
    fn reflection_coefficient(
        &self,
        _angle_degrees: f64,
        _frequency: f64,
        _sound_speed: f64,
    ) -> f64 {
        1.0
    }

    /// Reset internal state (e.g., PML memory variables).
    fn reset(&mut self);

    /// Return `true` if this boundary maintains internal state across steps.
    fn is_stateful(&self) -> bool {
        false
    }

    /// Return the approximate memory usage of this boundary in bytes.
    fn memory_usage(&self) -> usize {
        std::mem::size_of_val(self)
    }
}

/// Trait for absorbing boundary conditions (PML, CPML, sponge layers).
///
/// Absorbing boundaries attenuate outgoing waves to minimise reflections.
/// The absorption profile σ(x) grows from zero inside the domain to maximum
/// at the boundary.
pub trait AbsorbingBoundary: BoundaryCondition {
    /// Return the thickness of the absorbing layer in grid points.
    /// # Errors
    /// - Returns [`Err`] if an internal constraint is violated.
    ///
    fn thickness(&self) -> usize;

    /// Return the absorption coefficient σ at the given grid indices.
    ///
    /// The field decays as `exp(-σ * Δt)`.
    /// # Errors
    /// - Returns [`Err`] if an internal constraint is violated.
    ///
    fn absorption_profile(&self, indices: [usize; 3], grid: &dyn GridTopology) -> f64;

    /// Return the design target reflection coefficient R₀.
    /// # Errors
    /// - Returns [`Err`] if an internal constraint is violated.
    ///
    fn target_reflection(&self) -> f64;

    /// Validate that the layer is thick enough for the given maximum frequency.
    ///
    /// Rule of thumb: at least λ_min / 4 or 10 grid points.
    /// # Errors
    /// - Returns [`Err`] if an internal constraint is violated.
    ///
    fn validate_thickness(&self, grid: &dyn GridTopology, max_frequency: f64) -> KwaversResult<()> {
        let min_spacing = grid.spacing().iter().copied().fold(f64::INFINITY, f64::min);
        let wavelength = SOUND_SPEED_WATER_SIM / max_frequency;
        let min_thickness = (wavelength / (4.0 * min_spacing)).ceil() as usize;

        if self.thickness() < min_thickness.max(10) {
            return Err(crate::core::error::KwaversError::Config(
                crate::core::error::ConfigError::InvalidValue {
                    parameter: "boundary thickness".to_owned(),
                    value: self.thickness().to_string(),
                    constraint: format!("Must be at least {} points", min_thickness.max(10)),
                },
            ));
        }

        Ok(())
    }
}

/// Trait for reflective boundary conditions (rigid, soft, impedance-matched).
pub trait ReflectiveBoundary: BoundaryCondition {
    /// Return the complex reflection coefficient at the given frequency.
    ///
    /// - Rigid boundary: r = +1.0
    /// - Soft boundary:  r = −1.0
    /// - Impedance-matched: r = (Z − Z₀) / (Z + Z₀)
    fn reflection_coefficient_complex(&self, frequency: f64) -> num_complex::Complex<f64>;

    /// Return `true` if the boundary is perfectly rigid (no normal velocity).
    fn is_rigid(&self) -> bool {
        false
    }

    /// Return `true` if the boundary is perfectly soft (zero pressure).
    /// # Errors
    /// - Returns [`Err`] if an internal constraint is violated.
    ///
    fn is_soft(&self) -> bool {
        false
    }
}

/// Trait for periodic boundary conditions.
pub trait PeriodicBoundary: BoundaryCondition {
    /// Enforce periodicity: `field(x_max) = field(x_min)` for periodic dimensions.
    /// # Errors
    /// - Returns [`Err`] if an internal constraint is violated.
    ///
    fn wrap_periodic(
        &mut self,
        field: ArrayViewMut3<f64>,
        grid: &dyn GridTopology,
    ) -> KwaversResult<()>;

    /// Return the Bloch phase shift `k·L` per dimension (default: no shift).
    fn phase_shift(&self) -> [f64; 3] {
        [0.0, 0.0, 0.0]
    }
}
