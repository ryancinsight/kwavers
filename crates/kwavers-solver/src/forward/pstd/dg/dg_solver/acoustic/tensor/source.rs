use super::projection::axis_map_for_index;
use super::{validate_tensor_state, AcousticDgTensorSourceWeight, AcousticDgTensorWorkspace};
use crate::forward::pstd::dg::dg_solver::core::DGSolver;
use crate::forward::pstd::dg::dg_solver::rk_update::{
    update_euler, update_ssp_final, update_ssp_second,
};
use crate::forward::pstd::dg::dg_solver::topology::DgTopology;
use kwavers_core::error::KwaversResult;
use ndarray::Array3;

impl DGSolver {
    /// Advance a tensor-product acoustic state by one SSP-RK3 step with a
    /// time-dependent source term added to each stage RHS.
    ///
    /// The source callback receives the physical stage time and mutable RHS.
    ///
    /// # Errors
    /// Returns an error for shape mismatches or non-positive density.
    pub fn step_acoustic_tensor_ssp_rk3_with_source<F>(
        &self,
        state: &mut Array3<f64>,
        density: f64,
        dt: f64,
        workspace: &mut AcousticDgTensorWorkspace,
        t: f64,
        mut add_source_rhs: F,
    ) -> KwaversResult<()>
    where
        F: FnMut(f64, &mut Array3<f64>),
    {
        validate_tensor_state(self, state, density)?;
        workspace.ensure_dim(state.dim());
        workspace.original.assign(state);

        self.compute_acoustic_tensor_rhs_into(&workspace.original, density, &mut workspace.rhs)?;
        add_source_rhs(t, &mut workspace.rhs);
        update_euler(
            &mut workspace.stage,
            &workspace.original,
            &workspace.rhs,
            dt,
        );

        self.compute_acoustic_tensor_rhs_into(&workspace.stage, density, &mut workspace.rhs)?;
        add_source_rhs(t + dt, &mut workspace.rhs);
        update_ssp_second(
            &mut workspace.stage,
            &workspace.original,
            &workspace.rhs,
            dt,
        );

        self.compute_acoustic_tensor_rhs_into(&workspace.stage, density, &mut workspace.rhs)?;
        add_source_rhs(t + 0.5 * dt, &mut workspace.rhs);
        update_ssp_final(
            state,
            &workspace.original,
            &workspace.stage,
            &workspace.rhs,
            dt,
        );
        Ok(())
    }

    /// Return cell-integrated weak source weights for a uniform grid cell index.
    ///
    /// The returned weights distribute one uniform-cell source over the local
    /// GLL nodal mass. For each active axis the factor is
    /// `phi_i(x_s) * 2 / ((p + 1) w_i)`.
    ///
    /// # Errors
    /// Returns an error for out-of-bounds indices.
    pub fn acoustic_tensor_cell_source_weights(
        &self,
        index: [usize; 3],
    ) -> KwaversResult<Vec<AcousticDgTensorSourceWeight>> {
        let topology = DgTopology::from_grid(&self.grid, self.n_nodes)?;
        let maps = [
            axis_map_for_index(self, topology, 0, index[0])?,
            axis_map_for_index(self, topology, 1, index[1])?,
            axis_map_for_index(self, topology, 2, index[2])?,
        ];
        let elem = topology.element_index([maps[0].elem, maps[1].elem, maps[2].elem]);
        let mut out = Vec::with_capacity(topology.nodes_per_element);
        for node in 0..topology.nodes_per_element {
            let coords = topology.node_coords(node);
            let mut weight = 1.0;
            for axis in 0..3 {
                if topology.active_axes[axis] {
                    let node_axis = coords[axis];
                    weight *= maps[axis].basis[node_axis] * 2.0
                        / (self.n_nodes as f64 * self.weights[node_axis]);
                }
            }
            out.push(AcousticDgTensorSourceWeight { elem, node, weight });
        }
        Ok(out)
    }
}
