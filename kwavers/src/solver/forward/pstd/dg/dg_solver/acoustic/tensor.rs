//! Tensor-product first-order acoustic DG stepping.
//!
//! The state layout is `[p, u_x, u_y, u_z]` per tensor-product DG node. Active
//! dimensions are derived from the grid shape. Inactive velocity components are
//! carried for a stable public layout but receive no volume or surface update.
//!
//! ## Semi-discrete system
//! ```text
//! p_t + K div(u) = 0
//! u_t + (1/rho) grad(p) = 0
//! K = rho c^2
//! ```
//!
//! At each element face with normal axis `a`, the physical normal flux is
//! `F_a(q) = [K u_a, 0, 0, 0]` with `F_a(u_a)=p/rho`. The Rusanov flux is
//! applied only to the characteristic pair `(p, u_a)`, so tangential velocities
//! are not artificially diffused by unrelated face jumps.

use super::super::core::DGSolver;
use super::super::topology::DgTopology;
use crate::core::error::{KwaversError, KwaversResult};
use ndarray::Array3;

mod boundary;
mod projection;
mod source;

pub const ACOUSTIC_PRESSURE_VAR: usize = 0;
pub const ACOUSTIC_VELOCITY_X_VAR: usize = 1;
pub const ACOUSTIC_VELOCITY_Y_VAR: usize = 2;
pub const ACOUSTIC_VELOCITY_Z_VAR: usize = 3;
pub const ACOUSTIC_VARIABLES: usize = 4;

/// Tensor DG pressure-source contribution for one state node.
#[derive(Debug, Clone, Copy)]
pub struct AcousticDgTensorSourceWeight {
    pub elem: usize,
    pub node: usize,
    pub weight: f64,
}

#[derive(Debug, Clone)]
pub struct AcousticDgTensorWorkspace {
    original: Array3<f64>,
    stage: Array3<f64>,
    rhs: Array3<f64>,
}

impl AcousticDgTensorWorkspace {
    #[must_use]
    pub fn new(dim: (usize, usize, usize)) -> Self {
        Self {
            original: Array3::zeros(dim),
            stage: Array3::zeros(dim),
            rhs: Array3::zeros(dim),
        }
    }

    fn ensure_dim(&mut self, dim: (usize, usize, usize)) {
        if self.original.dim() != dim {
            *self = Self::new(dim);
        }
    }
}

impl DGSolver {
    /// Return the tensor acoustic state shape for this solver grid.
    ///
    /// # Errors
    /// Returns an error when any active grid axis is not divisible by `p + 1`.
    pub fn acoustic_tensor_state_shape(&self) -> KwaversResult<(usize, usize, usize)> {
        let topology = DgTopology::from_grid(&self.grid, self.n_nodes)?;
        Ok((
            topology.n_elements,
            topology.nodes_per_element,
            ACOUSTIC_VARIABLES,
        ))
    }

    /// Map a physical grid index to `(element, tensor_node)` in the acoustic DG state.
    ///
    /// # Errors
    /// Returns an error for out-of-bounds indices or inactive-axis indices that
    /// are not zero.
    pub fn acoustic_tensor_state_index(&self, index: [usize; 3]) -> KwaversResult<(usize, usize)> {
        let topology = DgTopology::from_grid(&self.grid, self.n_nodes)?;
        let dims = [self.grid.nx, self.grid.ny, self.grid.nz];
        let mut elem_coords = [0_usize; 3];
        let mut node_coords = [0_usize; 3];
        for axis in 0..3 {
            if index[axis] >= dims[axis] {
                return Err(KwaversError::InvalidInput(format!(
                    "DG acoustic grid index {:?} exceeds grid dimensions {:?}",
                    index, dims
                )));
            }
            if topology.active_axes[axis] {
                elem_coords[axis] = index[axis] / self.n_nodes;
                node_coords[axis] = index[axis] % self.n_nodes;
            } else if index[axis] != 0 {
                return Err(KwaversError::InvalidInput(format!(
                    "DG acoustic inactive axis {axis} requires index 0, got {}",
                    index[axis]
                )));
            }
        }
        Ok((
            topology.element_index(elem_coords),
            topology.node_index(node_coords),
        ))
    }

    /// Advance a tensor-product acoustic state by one SSP-RK3 time step.
    ///
    /// State layout is `(n_elements, nodes_per_element, 4)` with variables
    /// `[p, u_x, u_y, u_z]`.
    ///
    /// # Errors
    /// Returns an error for shape mismatches or non-positive density.
    pub fn step_acoustic_tensor_ssp_rk3(
        &self,
        state: &mut Array3<f64>,
        density: f64,
        dt: f64,
        workspace: &mut AcousticDgTensorWorkspace,
    ) -> KwaversResult<()> {
        self.step_acoustic_tensor_ssp_rk3_with_source(state, density, dt, workspace, 0.0, |_, _| {})
    }

    /// Compute the tensor-product first-order acoustic RHS.
    ///
    /// # Errors
    /// Returns an error for shape mismatches or non-positive density.
    pub fn compute_acoustic_tensor_rhs_into(
        &self,
        state: &Array3<f64>,
        density: f64,
        rhs: &mut Array3<f64>,
    ) -> KwaversResult<()> {
        let topology = validate_tensor_state(self, state, density)?;
        if rhs.dim() != state.dim() {
            return Err(KwaversError::InvalidInput(format!(
                "DG acoustic tensor RHS shape {:?} does not match state {:?}",
                rhs.dim(),
                state.dim()
            )));
        }

        rhs.fill(0.0);
        let axis_scales = self.acoustic_axis_scales();
        let bulk = density * self.config().sound_speed * self.config().sound_speed;
        let inv_density = density.recip();

        for elem in 0..topology.n_elements {
            for node in 0..topology.nodes_per_element {
                let node_coords = topology.node_coords(node);
                for axis in 0..3 {
                    if !topology.active_axes[axis] {
                        continue;
                    }
                    let velocity_var = velocity_var(axis);
                    let mut du = 0.0;
                    let mut dp = 0.0;
                    for j in 0..self.n_nodes {
                        let source_node = topology.node_with_axis(node, axis, j);
                        du += self.diff_matrix[[node_coords[axis], j]]
                            * state[(elem, source_node, velocity_var)];
                        dp += self.diff_matrix[[node_coords[axis], j]]
                            * state[(elem, source_node, ACOUSTIC_PRESSURE_VAR)];
                    }
                    rhs[(elem, node, ACOUSTIC_PRESSURE_VAR)] -= bulk * axis_scales[axis] * du;
                    rhs[(elem, node, velocity_var)] -= inv_density * axis_scales[axis] * dp;
                }
            }
        }

        for elem in 0..topology.n_elements {
            for axis in 0..3 {
                if topology.active_axes[axis] {
                    boundary::add_axis_surface_flux(
                        self,
                        topology,
                        state,
                        elem,
                        axis,
                        bulk,
                        inv_density,
                        rhs,
                    );
                }
            }
        }
        Ok(())
    }

    /// Project acoustic pressure from tensor DG state to a grid array.
    ///
    /// # Errors
    /// Returns an error for shape mismatches.
    pub fn project_acoustic_tensor_pressure_to_grid(
        &self,
        state: &Array3<f64>,
        pressure: &mut Array3<f64>,
    ) -> KwaversResult<()> {
        let topology = validate_tensor_state(self, state, 1.0)?;
        if pressure.dim() != (self.grid.nx, self.grid.ny, self.grid.nz) {
            return Err(KwaversError::InvalidInput(format!(
                "DG acoustic pressure grid shape {:?} does not match ({}, {}, {})",
                pressure.dim(),
                self.grid.nx,
                self.grid.ny,
                self.grid.nz
            )));
        }
        for elem in 0..topology.n_elements {
            for node in 0..topology.nodes_per_element {
                let [i, j, k] = topology.grid_index(elem, node);
                pressure[(i, j, k)] = state[(elem, node, ACOUSTIC_PRESSURE_VAR)];
            }
        }
        Ok(())
    }

    /// Project pressure and particle velocity components to grid arrays.
    ///
    /// # Errors
    /// Returns an error for state or output shape mismatches.
    pub fn project_acoustic_tensor_fields_to_grid(
        &self,
        state: &Array3<f64>,
        pressure: &mut Array3<f64>,
        ux: &mut Array3<f64>,
        uy: &mut Array3<f64>,
        uz: &mut Array3<f64>,
    ) -> KwaversResult<()> {
        let topology = validate_tensor_state(self, state, 1.0)?;
        let expected = (self.grid.nx, self.grid.ny, self.grid.nz);
        for (name, dim) in [
            ("pressure", pressure.dim()),
            ("ux", ux.dim()),
            ("uy", uy.dim()),
            ("uz", uz.dim()),
        ] {
            if dim != expected {
                return Err(KwaversError::InvalidInput(format!(
                    "DG acoustic {name} grid shape {dim:?} does not match {expected:?}"
                )));
            }
        }
        for elem in 0..topology.n_elements {
            for node in 0..topology.nodes_per_element {
                let index = topology.grid_index(elem, node);
                let [i, j, k] = index;
                pressure[(i, j, k)] = state[(elem, node, ACOUSTIC_PRESSURE_VAR)];
                ux[(i, j, k)] = state[(elem, node, ACOUSTIC_VELOCITY_X_VAR)];
                uy[(i, j, k)] = state[(elem, node, ACOUSTIC_VELOCITY_Y_VAR)];
                uz[(i, j, k)] = state[(elem, node, ACOUSTIC_VELOCITY_Z_VAR)];
            }
        }
        Ok(())
    }

    /// Copy grid pressure samples into the tensor acoustic pressure component.
    ///
    /// Velocity components are preserved so repeated `run` calls on the
    /// simulation adapter continue the acoustic state instead of resetting it.
    ///
    /// # Errors
    /// Returns an error for state or pressure shape mismatches.
    pub fn assign_acoustic_tensor_pressure_from_grid(
        &self,
        pressure: &Array3<f64>,
        state: &mut Array3<f64>,
    ) -> KwaversResult<()> {
        let topology = validate_tensor_state(self, state, 1.0)?;
        let expected = (self.grid.nx, self.grid.ny, self.grid.nz);
        if pressure.dim() != expected {
            return Err(KwaversError::InvalidInput(format!(
                "DG acoustic pressure grid shape {:?} does not match {expected:?}",
                pressure.dim()
            )));
        }
        for elem in 0..topology.n_elements {
            for node in 0..topology.nodes_per_element {
                let [i, j, k] = topology.grid_index(elem, node);
                state[(elem, node, ACOUSTIC_PRESSURE_VAR)] = pressure[(i, j, k)];
            }
        }
        Ok(())
    }

    pub(super) fn acoustic_axis_scales(&self) -> [f64; 3] {
        let element_spans = [
            self.n_nodes as f64 * self.grid.dx,
            self.n_nodes as f64 * self.grid.dy,
            self.n_nodes as f64 * self.grid.dz,
        ];
        [
            2.0 / element_spans[0],
            2.0 / element_spans[1],
            2.0 / element_spans[2],
        ]
    }
}

fn validate_tensor_state(
    solver: &DGSolver,
    state: &Array3<f64>,
    density: f64,
) -> KwaversResult<DgTopology> {
    if !density.is_finite() || density <= 0.0 {
        return Err(KwaversError::InvalidInput(format!(
            "DG acoustic density must be finite and positive, got {density}"
        )));
    }
    let sound_speed = solver.config().sound_speed;
    if !sound_speed.is_finite() || sound_speed <= 0.0 {
        return Err(KwaversError::InvalidInput(format!(
            "DG acoustic sound speed must be finite and positive, got {sound_speed}"
        )));
    }
    let topology = DgTopology::from_grid(&solver.grid, solver.n_nodes)?;
    let expected = (
        topology.n_elements,
        topology.nodes_per_element,
        ACOUSTIC_VARIABLES,
    );
    if state.dim() != expected {
        return Err(KwaversError::InvalidInput(format!(
            "DG acoustic tensor state shape {:?} does not match expected {:?}",
            state.dim(),
            expected
        )));
    }
    Ok(topology)
}

#[inline]
pub(super) fn velocity_var(axis: usize) -> usize {
    match axis {
        0 => ACOUSTIC_VELOCITY_X_VAR,
        1 => ACOUSTIC_VELOCITY_Y_VAR,
        2 => ACOUSTIC_VELOCITY_Z_VAR,
        _ => unreachable!("axis validated by caller"),
    }
}

#[cfg(test)]
mod tests;
