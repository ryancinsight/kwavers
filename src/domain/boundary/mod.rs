//! # Boundary Conditions for Acoustic Simulations
//!
//! This module provides a comprehensive boundary condition system supporting
//! multiple numerical methods and physics domains.
//!
//! ## Boundary Condition Types
//!
//! ### Time-Domain Boundaries (FDTD, PSTD)
//! Applied during time stepping to prevent reflections:
//!
//! - **[`PMLBoundary`]**: Perfectly Matched Layer absorbing boundaries
//! - **[`CPMLBoundary`]**: Convolutional Perfectly Matched Layer (advanced)
//!
//! ### Variational Method Boundaries (FEM, BEM)
//! Applied during matrix assembly for variational solvers:
//!
//! - **[`FemBoundaryManager`]**: FEM boundary conditions (Dirichlet, Neumann, Robin, radiation)
//! - **[`BemBoundaryManager`]**: BEM boundary conditions for boundary element methods
//!
//! ## Usage Examples
//!
//! ### Time-Domain Solver
//! ```rust,ignore
//! use kwavers::domain::boundary::{PMLBoundary, PMLConfig};
//!
//! let pml_config = PMLConfig { thickness: 20, alpha: 2.0 };
//! let mut boundary = PMLBoundary::new(pml_config)?;
//!
//! // Apply during time stepping
//! boundary.apply_acoustic(field.view_mut(), &grid, time_step)?;
//! ```
//!
//! ### FEM Solver
//! ```rust,ignore
//! use kwavers::domain::boundary::FemBoundaryManager;
//!
//! let mut bc_manager = FemBoundaryManager::new();
//! bc_manager.add_dirichlet(vec![(node_id, pressure_value)]);
//! bc_manager.add_radiation(vec![boundary_nodes]);
//!
//! // Apply during matrix assembly
//! bc_manager.apply_all(&mut stiffness, &mut mass, &mut rhs, wavenumber)?;
//! ```
//!
//! ### BEM Solver
//! ```rust,ignore
//! use kwavers::domain::boundary::BemBoundaryManager;
//!
//! let mut bc_manager = BemBoundaryManager::new();
//! bc_manager.add_dirichlet(vec![(node_id, pressure_value)]);
//!
//! // Apply to BEM boundary integrals
//! bc_manager.apply_all(&mut h_matrix, &mut g_matrix, &mut boundary_values, wavenumber)?;
//! ```
//!
//! ## Architecture Notes
//!
//! Runtime boundary conditions live under `simulation::environment`.
//! This module must not depend on crate-root re-exports; use direct imports only.

use crate::core::error::KwaversResult;
use crate::domain::grid::Grid;
use ndarray::{Array3, ArrayViewMut3};
use std::fmt::Debug;

pub mod bem;
pub mod config;
pub mod coupling;
pub mod cpml;
pub mod fem;
pub mod field_updater;
pub mod pml;
pub mod smoothing;
pub mod traits;
pub mod types;

pub use bem::{BemBoundaryCondition, BemBoundaryManager};
pub use config::{BoundaryParameters, BoundaryType};
pub use coupling::{
    AdaptiveBoundary, ImpedanceBoundary, MaterialInterface, MultiPhysicsInterface, SchwarzBoundary,
};
pub use fem::{FemBoundaryCondition, FemBoundaryManager};
pub use field_updater::{FieldUpdater, GradientFieldUpdater, LegacyFieldUpdater};
pub use traits::{
    AbsorbingBoundary, BoundaryCondition, BoundaryDirections, BoundaryDomain, BoundaryLayer,
    BoundaryLayerManager, FieldType, PeriodicBoundary, ReflectiveBoundary,
};
pub use types::{
    AcousticBoundaryType, BoundaryComponent, BoundaryFace, BoundarySpec, ElasticBoundaryType,
    ElectromagneticBoundaryType,
};
// Note: BoundaryType from types module is canonical SSOT
// BoundaryType from config module is legacy - will be deprecated
pub use types::BoundaryType as CanonicalBoundaryType;

/// Trait for runtime boundary condition implementations.
///
/// ## Multi-physics note
/// Kwavers supports optics in addition to acoustics. The runtime boundary trait
/// therefore includes `apply_light(...)` for fluence/fluence-rate boundary handling.
pub trait Boundary: Debug + Send + Sync {
    /// Applies boundary conditions to the acoustic field in spatial domain.
    fn apply_acoustic(
        &mut self,
        field: ArrayViewMut3<f64>,
        grid: &Grid,
        time_step: usize,
    ) -> KwaversResult<()>;

    /// Applies boundary conditions to the acoustic field in frequency domain (k-space).
    fn apply_acoustic_freq(
        &mut self,
        field: &mut Array3<rustfft::num_complex::Complex<f64>>,
        grid: &Grid,
        time_step: usize,
    ) -> KwaversResult<()>;

    /// Applies boundary conditions to the acoustic field with a scaling factor.
    ///
    /// Default implementation applies regular boundary conditions.
    fn apply_acoustic_with_factor(
        &mut self,
        field: ArrayViewMut3<f64>,
        grid: &Grid,
        time_step: usize,
        _factor: f64,
    ) -> KwaversResult<()> {
        self.apply_acoustic(field, grid, time_step)
    }

    /// Applies boundary conditions to the light fluence rate field (spatial domain).
    fn apply_light(&mut self, field: ArrayViewMut3<f64>, grid: &Grid, time_step: usize);
}

// Time-domain boundary conditions (FDTD, PSTD)
pub use cpml::{CPMLBoundary, CPMLConfig};
pub use pml::{PMLBoundary, PMLConfig};
