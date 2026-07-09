use super::FemHelmholtzSolver;
use kwavers_boundary::FemBoundaryManager;
use kwavers_mesh::TetrahedralMesh;
use leto::Array1;
use kwavers_math::fft::Complex64;

impl FemHelmholtzSolver {
    /// Borrow the assembled right-hand side vector.
    #[must_use]
    pub fn rhs(&self) -> &Array1<Complex64> {
        &self.rhs
    }

    /// Borrow the tetrahedral mesh owned by the solver.
    #[must_use]
    pub fn mesh(&self) -> &TetrahedralMesh {
        &self.mesh
    }

    /// Mutable reference to the boundary condition manager.
    #[must_use]
    pub fn boundary_manager(&mut self) -> &mut FemBoundaryManager {
        &mut self.boundary_manager
    }

    /// Immutable reference to the boundary condition manager.
    #[must_use]
    pub fn boundary_manager_ref(&self) -> &FemBoundaryManager {
        &self.boundary_manager
    }

    /// Nodal solution vector u_h.
    #[must_use]
    pub fn solution(&self) -> &Array1<Complex64> {
        &self.solution
    }
}
