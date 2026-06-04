//! Boundary field updater for solver integration.
//!
//! Provides a clean abstraction for applying boundary conditions during solver
//! field updates, separating boundary logic from solver logic.
//!
//! # Architecture
//!
//! ```text
//! Solver
//!   ↓
//! FieldUpdater (this module)
//!   ↓
//! BoundaryCondition (trait)
//!   ↓
//! CPMLBoundary / DomainPMLBoundary / etc.
//! ```

mod gradient;

#[cfg(test)]
mod tests;

pub use gradient::GradientFieldUpdater;

use kwavers_core::error::KwaversResult;
use crate::traits::{BoundaryCondition, BoundaryFieldType};
use kwavers_grid::GridTopology;
use ndarray::Array3;

/// Field updater that applies boundary conditions during solver steps.
///
/// Encapsulates boundary application logic, providing a clean interface for
/// solvers to integrate boundaries without knowing implementation details.
///
/// # Example
///
/// ```ignore
/// use kwavers_boundary::{CPMLBoundary, FieldUpdater};
///
/// let boundary = CPMLBoundary::new(config, &grid, sound_speed)?;
/// let mut updater = FieldUpdater::new(boundary);
///
/// // During solver step:
/// updater.apply_to_scalar_field(&mut pressure, &grid, step, dt)?;
/// ```
#[derive(Debug)]
pub struct FieldUpdater<B: BoundaryCondition> {
    /// The boundary condition to apply.
    boundary: B,
    /// Field type for validation.
    field_type: BoundaryFieldType,
}

impl<B: BoundaryCondition> FieldUpdater<B> {
    /// Create a new field updater with default field type (Pressure).
    pub fn new(boundary: B) -> Self {
        Self {
            boundary,
            field_type: BoundaryFieldType::Pressure,
        }
    }

    /// Create with explicit field type.
    pub fn with_field_type(boundary: B, field_type: BoundaryFieldType) -> Self {
        Self {
            boundary,
            field_type,
        }
    }

    /// Set the field type.
    pub fn set_field_type(&mut self, field_type: BoundaryFieldType) {
        self.field_type = field_type;
    }

    /// Apply boundary to a scalar field in spatial domain.
    ///
    /// Primary method for pressure, temperature, and similar scalar fields.
    /// # Errors
    /// - Returns [`kwavers_core::error::KwaversError::Config`] if the boundary does not support this field type.
    /// - Propagates any [`KwaversError`] returned by called functions.
    ///
    pub fn apply_to_scalar_field(
        &mut self,
        field: &mut Array3<f64>,
        grid: &dyn GridTopology,
        time_step: usize,
        dt: f64,
    ) -> KwaversResult<()> {
        if !self.boundary.supports_field_type(self.field_type) {
            return Err(kwavers_core::error::KwaversError::Config(
                kwavers_core::error::ConfigError::InvalidValue {
                    parameter: "field_type".to_owned(),
                    value: format!("{:?}", self.field_type),
                    constraint: format!(
                        "Boundary '{}' does not support this field type",
                        self.boundary.name()
                    ),
                },
            ));
        }

        self.boundary
            .apply_scalar_spatial(field.view_mut(), grid, time_step, dt)
    }

    /// Apply boundary to vector field components (e.g., velocity, EM fields).
    /// # Errors
    /// - Returns [`Err`] if an internal constraint is violated.
    ///
    pub fn apply_to_vector_field(
        &mut self,
        field_x: &mut Array3<f64>,
        field_y: &mut Array3<f64>,
        field_z: &mut Array3<f64>,
        grid: &dyn GridTopology,
        time_step: usize,
        dt: f64,
    ) -> KwaversResult<()> {
        self.boundary.apply_vector_spatial(
            field_x.view_mut(),
            field_y.view_mut(),
            field_z.view_mut(),
            grid,
            time_step,
            dt,
        )
    }

    /// Apply boundary in frequency domain (for k-space and spectral solvers).
    /// # Errors
    /// - Returns [`Err`] if an internal constraint is violated.
    ///
    pub fn apply_to_frequency_field(
        &mut self,
        field: &mut Array3<kwavers_math::fft::Complex64>,
        grid: &dyn GridTopology,
        time_step: usize,
        dt: f64,
    ) -> KwaversResult<()> {
        self.boundary
            .apply_scalar_frequency(field, grid, time_step, dt)
    }

    /// Borrow the boundary condition.
    pub fn boundary(&self) -> &B {
        &self.boundary
    }

    /// Mutably borrow the boundary condition.
    pub fn boundary_mut(&mut self) -> &mut B {
        &mut self.boundary
    }

    /// Reset boundary state (e.g., PML memory variables).
    pub fn reset(&mut self) {
        self.boundary.reset();
    }

    /// Estimate reflection coefficient at given angle, frequency, and sound speed.
    pub fn reflection_coefficient(
        &self,
        angle_degrees: f64,
        frequency: f64,
        sound_speed: f64,
    ) -> f64 {
        self.boundary
            .reflection_coefficient(angle_degrees, frequency, sound_speed)
    }
}
