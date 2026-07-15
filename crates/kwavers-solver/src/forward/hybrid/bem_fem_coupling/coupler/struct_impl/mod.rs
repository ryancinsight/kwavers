//! `BemFemCoupler` — struct definition, constructor, and query methods.
//!
//! Partitioned by responsibility:
//! - `coupled`   — `solve_coupled` coupling-iteration loop.
//! - `interface` — FEM/BEM interface data transfer methods.
//! - `solvers`   — BEM system solve, FEM matrix assembly, linear solver.

mod coupled;
mod interface;
mod solvers;

use crate::forward::bem::solver::{BemConfig, BemSolver};
use kwavers_core::error::KwaversResult;
use kwavers_math::numerics::operators::NumericsTrilinearInterpolator;
use kwavers_mesh::tetrahedral::TetrahedralMesh;

use super::super::{BemFemCouplingConfig, BemFemInterface};

/// BEM-FEM Coupling Solver.
#[derive(Debug)]
pub struct BemFemCoupler {
    pub(super) config: BemFemCouplingConfig,
    /// Interface accessible from `solver.rs` in the parent `bem_fem_coupling` module.
    pub(crate) interface: BemFemInterface,
    pub(super) _fem_interpolator: NumericsTrilinearInterpolator,
    pub(super) convergence_history: Vec<f64>,
    pub(super) iteration_count: usize,
    pub(super) bem_solver: BemSolver,
}

impl BemFemCoupler {
    /// Create a new BEM-FEM coupler.
    ///
    /// # Errors
    /// Propagates errors from `BemFemInterface::new` and `BemSolver::new`.
    pub fn new(
        config: BemFemCouplingConfig,
        fem_mesh: &TetrahedralMesh,
        bem_boundary: &[usize],
    ) -> KwaversResult<Self> {
        let interface = BemFemInterface::new(fem_mesh, bem_boundary)?;

        let bb = &fem_mesh.bounding_box;
        let lx = (bb.max[0] - bb.min[0]).max(1e-12);
        let ly = (bb.max[1] - bb.min[1]).max(1e-12);
        let lz = (bb.max[2] - bb.min[2]).max(1e-12);
        let l_max = lx.max(ly).max(lz);
        let n_e = (fem_mesh.elements.len()).max(1) as f64;
        let n1d = n_e.cbrt();
        let dx = lx / (n1d * lx / l_max).max(1.0);
        let dy = ly / (n1d * ly / l_max).max(1.0);
        let dz = lz / (n1d * lz / l_max).max(1.0);
        let fem_interpolator = NumericsTrilinearInterpolator::new(dx, dy, dz);

        let bem_config = BemConfig::default();
        let boundary_verts: Vec<[f64; 3]> = fem_mesh.nodes.iter().map(|n| n.coordinates).collect();
        let boundary_tris: Vec<[usize; 3]> = fem_mesh
            .boundary_faces
            .keys()
            .filter_map(|face| {
                if (face.len()) == 3 {
                    Some([face[0], face[1], face[2]])
                } else {
                    None
                }
            })
            .collect();
        let bem_solver = BemSolver::new(bem_config, boundary_verts, boundary_tris)?;

        Ok(Self {
            config,
            interface,
            _fem_interpolator: fem_interpolator,
            convergence_history: Vec::new(),
            iteration_count: 0,
            bem_solver,
        })
    }

    /// Get the convergence residual history.
    #[must_use]
    pub fn convergence_history(&self) -> &[f64] {
        &self.convergence_history
    }

    /// Return `true` if the last `solve_coupled` call converged below tolerance.
    ///
    /// # Panics
    /// Panics if called when `convergence_history` is non-empty but
    /// `last()` returns `None` (unreachable in practice).
    #[must_use]
    pub fn has_converged(&self) -> bool {
        if self.convergence_history.is_empty() {
            return false;
        }
        let last_residual = *self.convergence_history.last().unwrap();
        last_residual < self.config.convergence_tolerance
    }

    /// Get the number of coupling iterations performed.
    #[must_use]
    pub fn iterations(&self) -> usize {
        self.iteration_count
    }

    /// Reset convergence tracking for a fresh solve.
    pub fn reset(&mut self) {
        self.convergence_history.clear();
        self.iteration_count = 0;
    }
}
