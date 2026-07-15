//! `FemHelmholtzSolver` — P1 tetrahedral FEM Helmholtz solver.
//!
//! ## Mathematical Foundation
//!
//! Galerkin discretization of ∇²u + k²u = −f on a tetrahedral mesh with P1 basis
//! functions (Ihlenburg 1998, §2.1):
//! ```text
//! a(u,v) = ∫_Ω (∇u·∇v − k²uv) dΩ = ∫_Ω fv dΩ + ∫_Γ (∂u/∂n)v dΓ
//! ```
//!
//! **Element matrices** for tetrahedron {p₀,p₁,p₂,p₃}:
//! - J = [p₁−p₀ | p₂−p₀ | p₃−p₀], V = |det J|/6
//! - K_ij = V · (∇φᵢ · ∇φⱼ) — stiffness
//! - M_ij = V/(10+10·δᵢⱼ) — consistent mass (analytical P1 formula)
//! - System: A = K − k²M, solve via BiCGSTAB
//!
//! ## References
//! - Ihlenburg F (1998). *Finite Element Analysis of Acoustic Scattering*. Springer.

mod accessors;
mod assembly;
mod element;
mod interpolation;
mod solve;

use super::config::FemHelmholtzConfig;
use kwavers_boundary::FemBoundaryManager;
use kwavers_core::error::KwaversResult;
use kwavers_grid::Grid;
use kwavers_math::fft::Complex64;
use kwavers_math::linear_algebra::sparse::csr::CompressedSparseRowMatrix;
use kwavers_mesh::TetrahedralMesh;
use leto::Array1;

/// Finite Element Helmholtz solver for complex geometries.
#[derive(Debug)]
pub struct FemHelmholtzSolver {
    pub(super) config: FemHelmholtzConfig,
    pub(super) mesh: TetrahedralMesh,
    pub(super) boundary_manager: FemBoundaryManager,
    pub(super) system_matrix: CompressedSparseRowMatrix<Complex64>,
    pub(super) rhs: Array1<Complex64>,
    /// Nodal solution vector u_h ∈ ℂ^{n_dof}.
    pub solution: Array1<Complex64>,
}

impl FemHelmholtzSolver {
    /// Construct solver from configuration and mesh.
    #[must_use]
    pub fn new(config: FemHelmholtzConfig, mesh: TetrahedralMesh) -> Self {
        let num_dofs = mesh.nodes.len();
        Self {
            config,
            mesh,
            boundary_manager: FemBoundaryManager::new(),
            system_matrix: CompressedSparseRowMatrix::create(num_dofs, num_dofs),
            rhs: Array1::zeros(num_dofs),
            solution: Array1::zeros(num_dofs),
        }
    }

    /// Construct a FEM Helmholtz solver from a structured Cartesian grid.
    ///
    /// The grid is interpreted as a vertex lattice and tetrahedralized by
    /// `TetrahedralMesh::from_grid_vertices`, preserving exact domain volume
    /// through the six-tetrahedra-per-cell split.
    /// # Errors
    /// - Propagates any [`crate::KwaversError`] returned by called functions.
    ///
    pub fn from_grid(config: FemHelmholtzConfig, grid: &Grid) -> KwaversResult<Self> {
        let mesh = TetrahedralMesh::from_grid_vertices(grid)?;
        Ok(Self::new(config, mesh))
    }
}
