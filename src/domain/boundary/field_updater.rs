//! Boundary field updater for solver integration
//!
//! This module provides a clean abstraction for applying boundary conditions
//! during solver field updates. It separates boundary logic from solver logic,
//! enabling:
//! - Generic solvers that work with any boundary type
//! - Testable boundary integration
//! - Clear separation of concerns
//!
//! # Design Philosophy
//!
//! The field updater acts as a bridge between solvers and boundary conditions:
//! - Solvers focus on physics (wave propagation, etc.)
//! - Boundaries focus on absorption/reflection
//! - Updater coordinates the interaction
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
//! CPMLBoundary / PMLBoundary / etc.
//! ```

use crate::core::error::KwaversResult;
use crate::domain::boundary::traits::{BoundaryCondition, FieldType};
use crate::domain::grid::{Grid, GridTopology};
use ndarray::{Array3, ArrayView3, ArrayViewMut3};

/// Field updater that applies boundary conditions during solver steps
///
/// This struct encapsulates boundary application logic, providing a clean
/// interface for solvers to integrate boundaries without knowing implementation details.
///
/// # Example
///
/// ```ignore
/// use kwavers::domain::boundary::{CPMLBoundary, FieldUpdater};
///
/// let boundary = CPMLBoundary::new(config, &grid, sound_speed)?;
/// let mut updater = FieldUpdater::new(boundary);
///
/// // During solver step:
/// updater.apply_to_scalar_field(&mut pressure, &grid, step, dt)?;
/// ```
#[derive(Debug)]
pub struct FieldUpdater<B: BoundaryCondition> {
    /// The boundary condition to apply
    boundary: B,
    /// Field type for validation
    field_type: FieldType,
}

impl<B: BoundaryCondition> FieldUpdater<B> {
    /// Create a new field updater
    ///
    /// # Arguments
    ///
    /// * `boundary` - The boundary condition implementation
    pub fn new(boundary: B) -> Self {
        Self {
            boundary,
            field_type: FieldType::Pressure, // Default
        }
    }

    /// Create with explicit field type
    pub fn with_field_type(boundary: B, field_type: FieldType) -> Self {
        Self {
            boundary,
            field_type,
        }
    }

    /// Set the field type
    pub fn set_field_type(&mut self, field_type: FieldType) {
        self.field_type = field_type;
    }

    /// Apply boundary to a scalar field in spatial domain
    ///
    /// This is the primary method for most solvers.
    ///
    /// # Arguments
    ///
    /// * `field` - The field to modify (pressure, temperature, etc.)
    /// * `grid` - Grid topology
    /// * `time_step` - Current time step
    /// * `dt` - Time step size
    pub fn apply_to_scalar_field(
        &mut self,
        field: &mut Array3<f64>,
        grid: &dyn GridTopology,
        time_step: usize,
        dt: f64,
    ) -> KwaversResult<()> {
        if !self.boundary.supports_field_type(self.field_type) {
            return Err(crate::domain::core::error::KwaversError::Config(
                crate::domain::core::error::ConfigError::InvalidValue {
                    parameter: "field_type".to_string(),
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

    /// Apply boundary to vector field components
    ///
    /// Used for velocity fields, electric/magnetic fields, etc.
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

    /// Apply boundary in frequency domain
    ///
    /// For k-space and spectral solvers.
    pub fn apply_to_frequency_field(
        &mut self,
        field: &mut Array3<rustfft::num_complex::Complex<f64>>,
        grid: &dyn GridTopology,
        time_step: usize,
        dt: f64,
    ) -> KwaversResult<()> {
        self.boundary
            .apply_scalar_frequency(field, grid, time_step, dt)
    }

    /// Get a reference to the boundary condition
    pub fn boundary(&self) -> &B {
        &self.boundary
    }

    /// Get a mutable reference to the boundary condition
    pub fn boundary_mut(&mut self) -> &mut B {
        &mut self.boundary
    }

    /// Reset boundary state (e.g., PML memory variables)
    pub fn reset(&mut self) {
        self.boundary.reset();
    }

    /// Estimate reflection coefficient
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

/// Gradient-based field updater for FDTD solvers
///
/// This struct handles the gradient computation and boundary correction
/// pattern common in FDTD solvers with split-field PML.
///
/// # Mathematical Background
///
/// FDTD acoustic wave equation:
/// ```text
/// ∂v/∂t = -(1/ρ) ∇p
/// ∂p/∂t = -ρc² ∇·v
/// ```
///
/// With CPML, gradients are modified by memory variables:
/// ```text
/// ∇p → ∇p + ψ(memory correction)
/// ```
#[derive(Debug)]
pub struct GradientFieldUpdater {
    /// Temporary storage for gradients
    grad_x: Array3<f64>,
    grad_y: Array3<f64>,
    grad_z: Array3<f64>,
}

impl GradientFieldUpdater {
    /// Create a new gradient field updater
    ///
    /// # Arguments
    ///
    /// * `grid` - Grid to allocate gradient arrays
    pub fn new(grid: &dyn GridTopology) -> Self {
        let dims = grid.dimensions();
        Self {
            grad_x: Array3::zeros((dims[0], dims[1], dims[2])),
            grad_y: Array3::zeros((dims[0], dims[1], dims[2])),
            grad_z: Array3::zeros((dims[0], dims[1], dims[2])),
        }
    }

    /// Compute spatial gradients using central differences
    ///
    /// # Arguments
    ///
    /// * `field` - The field to differentiate
    /// * `grid` - Grid providing spacing information
    pub fn compute_gradients(&mut self, field: &Array3<f64>, grid: &dyn GridTopology) {
        let dims = grid.dimensions();
        let spacing = grid.spacing();
        let (nx, ny, nz) = (dims[0], dims[1], dims[2]);
        let (dx, dy, dz) = (spacing[0], spacing[1], spacing[2]);

        // X-gradient (central differences in interior)
        for i in 1..nx - 1 {
            for j in 0..ny {
                for k in 0..nz {
                    self.grad_x[[i, j, k]] =
                        (field[[i + 1, j, k]] - field[[i - 1, j, k]]) / (2.0 * dx);
                }
            }
        }

        // X boundaries (one-sided differences)
        for j in 0..ny {
            for k in 0..nz {
                self.grad_x[[0, j, k]] = (field[[1, j, k]] - field[[0, j, k]]) / dx;
                self.grad_x[[nx - 1, j, k]] = (field[[nx - 1, j, k]] - field[[nx - 2, j, k]]) / dx;
            }
        }

        // Y-gradient
        for i in 0..nx {
            for j in 1..ny - 1 {
                for k in 0..nz {
                    self.grad_y[[i, j, k]] =
                        (field[[i, j + 1, k]] - field[[i, j - 1, k]]) / (2.0 * dy);
                }
            }
        }

        // Y boundaries
        for i in 0..nx {
            for k in 0..nz {
                self.grad_y[[i, 0, k]] = (field[[i, 1, k]] - field[[i, 0, k]]) / dy;
                self.grad_y[[i, ny - 1, k]] = (field[[i, ny - 1, k]] - field[[i, ny - 2, k]]) / dy;
            }
        }

        // Z-gradient
        for i in 0..nx {
            for j in 0..ny {
                for k in 1..nz - 1 {
                    self.grad_z[[i, j, k]] =
                        (field[[i, j, k + 1]] - field[[i, j, k - 1]]) / (2.0 * dz);
                }
            }
        }

        // Z boundaries
        for i in 0..nx {
            for j in 0..ny {
                self.grad_z[[i, j, 0]] = (field[[i, j, 1]] - field[[i, j, 0]]) / dz;
                self.grad_z[[i, j, nz - 1]] = (field[[i, j, nz - 1]] - field[[i, j, nz - 2]]) / dz;
            }
        }
    }

    /// Get immutable views of gradient components
    pub fn gradients(
        &self,
    ) -> (
        ArrayView3<'_, f64>,
        ArrayView3<'_, f64>,
        ArrayView3<'_, f64>,
    ) {
        (self.grad_x.view(), self.grad_y.view(), self.grad_z.view())
    }

    /// Get mutable views of gradient components
    pub fn gradients_mut(
        &mut self,
    ) -> (
        ArrayViewMut3<'_, f64>,
        ArrayViewMut3<'_, f64>,
        ArrayViewMut3<'_, f64>,
    ) {
        (
            self.grad_x.view_mut(),
            self.grad_y.view_mut(),
            self.grad_z.view_mut(),
        )
    }

    /// Apply boundary correction to computed gradients
    ///
    /// This modifies the gradients in-place to include PML memory effects.
    /// After calling this, the gradients can be used directly in field updates.
    ///
    /// # Note
    ///
    /// This is typically called by CPML-specific update methods.
    pub fn apply_cpml_correction<F>(&mut self, mut correction_fn: F)
    where
        F: FnMut(&mut Array3<f64>, usize),
    {
        correction_fn(&mut self.grad_x, 0);
        correction_fn(&mut self.grad_y, 1);
        correction_fn(&mut self.grad_z, 2);
    }

    /// Compute divergence from vector field
    ///
    /// Returns ∇·v = ∂vx/∂x + ∂vy/∂y + ∂vz/∂z
    pub fn compute_divergence(
        vx: &Array3<f64>,
        vy: &Array3<f64>,
        vz: &Array3<f64>,
        grid: &dyn GridTopology,
    ) -> Array3<f64> {
        let dims = grid.dimensions();
        let spacing = grid.spacing();
        let (nx, ny, nz) = (dims[0], dims[1], dims[2]);
        let (dx, dy, dz) = (spacing[0], spacing[1], spacing[2]);

        let mut div = Array3::zeros((nx, ny, nz));

        for i in 1..nx - 1 {
            for j in 1..ny - 1 {
                for k in 1..nz - 1 {
                    let dvx_dx = (vx[[i + 1, j, k]] - vx[[i - 1, j, k]]) / (2.0 * dx);
                    let dvy_dy = (vy[[i, j + 1, k]] - vy[[i, j - 1, k]]) / (2.0 * dy);
                    let dvz_dz = (vz[[i, j, k + 1]] - vz[[i, j, k - 1]]) / (2.0 * dz);

                    div[[i, j, k]] = dvx_dx + dvy_dy + dvz_dz;
                }
            }
        }

        div
    }
}

/// Boundary-aware field updater for legacy Grid compatibility
///
/// This provides backward compatibility with code using the legacy `Grid` struct
/// while still benefiting from the new boundary trait system.
pub struct LegacyFieldUpdater<B: BoundaryCondition> {
    inner: FieldUpdater<B>,
}

impl<B: BoundaryCondition> std::fmt::Debug for LegacyFieldUpdater<B> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("LegacyFieldUpdater").finish_non_exhaustive()
    }
}

impl<B: BoundaryCondition> LegacyFieldUpdater<B> {
    /// Create from boundary condition
    pub fn new(boundary: B) -> Self {
        Self {
            inner: FieldUpdater::new(boundary),
        }
    }

    /// Apply to scalar field using legacy Grid
    pub fn apply_to_scalar_field_legacy(
        &mut self,
        field: &mut Array3<f64>,
        grid: &Grid,
        time_step: usize,
        dt: f64,
    ) -> KwaversResult<()> {
        use crate::domain::grid::GridTopologyExt;
        let adapter = grid.as_topology();
        self.inner
            .apply_to_scalar_field(field, &adapter, time_step, dt)
    }

    /// Get inner updater
    pub fn inner(&self) -> &FieldUpdater<B> {
        &self.inner
    }

    /// Get mutable inner updater
    pub fn inner_mut(&mut self) -> &mut FieldUpdater<B> {
        &mut self.inner
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::domain::boundary::cpml::{CPMLBoundary, CPMLConfig};
    use crate::domain::grid::{CartesianTopology, Grid};

    #[test]
    fn test_field_updater_creation() {
        let grid = Grid::new(32, 32, 32, 1e-3, 1e-3, 1e-3).unwrap();
        let config = CPMLConfig::default();
        let boundary = CPMLBoundary::new(config, &grid, 1500.0).unwrap();

        let updater = FieldUpdater::new(boundary);
        assert_eq!(updater.boundary().name(), "CPML (Convolutional PML)");
    }

    #[test]
    fn test_gradient_updater() {
        let topo =
            CartesianTopology::new([32, 32, 32], [1e-3, 1e-3, 1e-3], [0.0, 0.0, 0.0]).unwrap();
        let mut grad_updater = GradientFieldUpdater::new(&topo);

        // Create test field with linear gradient in x
        let mut field = Array3::zeros((32, 32, 32));
        for i in 0..32 {
            for j in 0..32 {
                for k in 0..32 {
                    field[[i, j, k]] = i as f64;
                }
            }
        }

        grad_updater.compute_gradients(&field, &topo);

        // Check gradient is approximately 1/dx in interior
        let expected_grad = 1.0 / 1e-3;
        for i in 1..31 {
            for j in 0..32 {
                for k in 0..32 {
                    let computed = grad_updater.grad_x[[i, j, k]];
                    assert!((computed - expected_grad).abs() < 1e-6);
                }
            }
        }
    }

    #[test]
    fn test_divergence_computation() {
        let topo = CartesianTopology::new([16, 16, 16], [0.1, 0.1, 0.1], [0.0, 0.0, 0.0]).unwrap();

        // Create uniform expansion field: v = (x, y, z)
        let mut vx = Array3::zeros((16, 16, 16));
        let mut vy = Array3::zeros((16, 16, 16));
        let mut vz = Array3::zeros((16, 16, 16));

        for i in 0..16 {
            for j in 0..16 {
                for k in 0..16 {
                    vx[[i, j, k]] = i as f64 * 0.1;
                    vy[[i, j, k]] = j as f64 * 0.1;
                    vz[[i, j, k]] = k as f64 * 0.1;
                }
            }
        }

        let div = GradientFieldUpdater::compute_divergence(&vx, &vy, &vz, &topo);

        // Divergence should be approximately 3.0 (1.0 from each component)
        for i in 1..15 {
            for j in 1..15 {
                for k in 1..15 {
                    assert!((div[[i, j, k]] - 3.0).abs() < 1e-6);
                }
            }
        }
    }

    #[test]
    fn test_legacy_compatibility() {
        let grid = Grid::new(64, 64, 64, 1e-3, 1e-3, 1e-3).unwrap();
        let config = CPMLConfig::with_thickness(10);
        let boundary = CPMLBoundary::new(config, &grid, 1500.0).unwrap();

        let mut updater = LegacyFieldUpdater::new(boundary);
        let mut field = Array3::ones((64, 64, 64));

        // Should not panic
        updater
            .apply_to_scalar_field_legacy(&mut field, &grid, 0, 1e-7)
            .unwrap();
    }
}
