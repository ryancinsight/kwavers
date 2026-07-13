//! `FrequencyDomainAcousticBackend` implementation for FEM Helmholtz.

use super::backend::FemHelmholtzBackend;
use crate::backends::acoustic::backend::FrequencyDomainAcousticBackend;
use kwavers_core::error::{KwaversError, KwaversResult};
use kwavers_math::fft::Complex64;
use kwavers_mesh::MeshBoundaryType;
use leto::{Array1, ArrayView2};

impl FrequencyDomainAcousticBackend for FemHelmholtzBackend<'_> {
    fn add_nodal_load(&mut self, node_idx: usize, value: Complex64) -> KwaversResult<()> {
        if node_idx >= self.solver.mesh().nodes.len() {
            return Err(KwaversError::InvalidInput(format!(
                "FEM nodal load index {node_idx} is outside mesh node count {}",
                self.solver.mesh().nodes.len()
            )));
        }
        if !value.re.is_finite() || !value.im.is_finite() {
            return Err(KwaversError::InvalidInput(format!(
                "FEM nodal load must be finite, got {value}"
            )));
        }

        self.pending_nodal_loads.push((node_idx, value));
        Ok(())
    }

    fn add_dirichlet_on_boundary_type(
        &mut self,
        boundary_type: MeshBoundaryType,
        value: Complex64,
    ) -> KwaversResult<usize> {
        self.solver
            .add_dirichlet_on_boundary_type(boundary_type, value)
    }

    fn solve(&mut self) -> KwaversResult<()> {
        self.solver.assemble_system(self.medium)?;
        for &(node_idx, value) in &self.pending_nodal_loads {
            self.solver.add_nodal_load(node_idx, value)?;
        }
        self.solver.solve_system()
    }

    fn pressure_solution(&self) -> &Array1<Complex64> {
        self.solver.solution()
    }

    fn interpolate_pressure(
        &self,
        query_points: ArrayView2<'_, f64>,
    ) -> KwaversResult<Array1<Complex64>> {
        self.solver.interpolate_solution(query_points)
    }

    fn wavenumber(&self) -> f64 {
        self.wavenumber
    }

    fn mesh_size(&self) -> (usize, usize) {
        (
            self.solver.mesh().nodes.len(),
            self.solver.mesh().elements.len(),
        )
    }
}
