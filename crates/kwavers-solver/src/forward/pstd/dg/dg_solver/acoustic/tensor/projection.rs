//! Projection of tensor DG state onto the uniform solver grid.
//!
//! A uniform grid index lands on element `index / n_nodes` at local position
//! `index % n_nodes`, so the GLL basis at that index depends on the local
//! position alone: every element reuses the same `n_nodes` patterns, and every
//! active axis reuses them too, because the quadrature nodes and the local
//! coordinate mapping are the same on all three. The basis is therefore stored
//! once, in one contiguous buffer, and the element is derived arithmetically.
//!
//! The per-node interpolation weights depend on the grid point alone, so one
//! computation serves every variable projected there. Computing them per
//! variable repeated the same `nodes_per_element` products once per output
//! field.

use super::{validate_tensor_state, ACOUSTIC_PRESSURE_VAR};
use crate::forward::pstd::dg::dg_solver::core::DGSolver;
use crate::forward::pstd::dg::dg_solver::topology::DgTopology;
use kwavers_core::error::{KwaversError, KwaversResult};
use leto::Array3;

/// The local GLL basis evaluated at every uniform-grid position.
///
/// `values[local * n_nodes + node]` is the basis value of `node` at local
/// position `local`. One buffer serves all three axes and every element.
pub(super) struct AxisBasis {
    values: Vec<f64>,
    n_nodes: usize,
}

impl AxisBasis {
    /// Evaluate the basis at each of the `n_nodes` uniform local positions.
    pub(super) fn new(solver: &DGSolver) -> Self {
        let n_nodes = solver.n_nodes;
        let mut values = vec![0.0; n_nodes * n_nodes];
        for (local, row) in values.chunks_exact_mut(n_nodes).enumerate() {
            let xi = local_coordinate(n_nodes, local);
            lagrange_values_into(solver, xi, row);
        }
        Self { values, n_nodes }
    }

    /// The basis value of `axis_node` at a grid index along one axis.
    #[inline]
    pub(super) fn at(&self, index: usize, axis_node: usize) -> f64 {
        self.values[(index % self.n_nodes) * self.n_nodes + axis_node]
    }
}

impl DGSolver {
    /// Project acoustic pressure from tensor DG state to the solver grid by
    /// evaluating the local GLL polynomial at uniform grid coordinates.
    ///
    /// # Errors
    /// Returns an error for state or pressure shape mismatches.
    pub fn project_acoustic_tensor_pressure_to_uniform_grid(
        &self,
        state: &Array3<f64>,
        pressure: &mut Array3<f64>,
    ) -> KwaversResult<()> {
        let topology = validate_tensor_state(self, state, 1.0)?;
        validate_grid_shape(self, "pressure", pressure.shape())?;
        let basis = AxisBasis::new(self);
        let mut weights = Vec::new();
        for i in 0..self.grid.nx {
            for j in 0..self.grid.ny {
                for k in 0..self.grid.nz {
                    let elem = point_weights(topology, &basis, [i, j, k], &mut weights);
                    pressure[[i, j, k]] =
                        interpolate_var(state, elem, &weights, ACOUSTIC_PRESSURE_VAR);
                }
            }
        }
        Ok(())
    }

    /// Project pressure and particle velocity components to uniform grid arrays.
    ///
    /// # Errors
    /// Returns an error for state or output shape mismatches.
    pub fn project_acoustic_tensor_fields_to_uniform_grid(
        &self,
        state: &Array3<f64>,
        pressure: &mut Array3<f64>,
        ux: &mut Array3<f64>,
        uy: &mut Array3<f64>,
        uz: &mut Array3<f64>,
    ) -> KwaversResult<()> {
        let topology = validate_tensor_state(self, state, 1.0)?;
        for (name, dim) in [
            ("pressure", pressure.shape()),
            ("ux", ux.shape()),
            ("uy", uy.shape()),
            ("uz", uz.shape()),
        ] {
            validate_grid_shape(self, name, dim)?;
        }
        let basis = AxisBasis::new(self);
        let mut weights = Vec::new();
        for i in 0..self.grid.nx {
            for j in 0..self.grid.ny {
                for k in 0..self.grid.nz {
                    let index = [i, j, k];
                    // One weight set, four fields: the weights are a property
                    // of the point, not of the variable.
                    let elem = point_weights(topology, &basis, index, &mut weights);
                    pressure[[i, j, k]] =
                        interpolate_var(state, elem, &weights, ACOUSTIC_PRESSURE_VAR);
                    ux[[i, j, k]] = interpolate_var(state, elem, &weights, 1);
                    uy[[i, j, k]] = interpolate_var(state, elem, &weights, 2);
                    uz[[i, j, k]] = interpolate_var(state, elem, &weights, 3);
                }
            }
        }
        Ok(())
    }
}

/// The element a uniform grid point lands in.
#[inline]
pub(super) fn point_element_of(topology: DgTopology, index: [usize; 3]) -> usize {
    let n_nodes = topology.n_nodes;
    topology.element_index([index[0] / n_nodes, index[1] / n_nodes, index[2] / n_nodes])
}

/// Fill `weights` with the per-node interpolation weights for one grid point,
/// and return the element the point lands in.
///
/// `weights` is caller scratch: the projection reuses it across the whole grid,
/// so the inner loop allocates nothing.
fn point_weights(
    topology: DgTopology,
    basis: &AxisBasis,
    index: [usize; 3],
    weights: &mut Vec<f64>,
) -> usize {
    weights.clear();
    for node in 0..topology.nodes_per_element {
        let coords = topology.node_coords(node);
        let mut weight = 1.0;
        for axis in 0..3 {
            if topology.active_axes[axis] {
                weight *= basis.at(index[axis], coords[axis]);
            }
        }
        weights.push(weight);
    }
    point_element_of(topology, index)
}

/// The weighted sum of one variable over an element's nodes.
#[inline]
fn interpolate_var(state: &Array3<f64>, elem: usize, weights: &[f64], var: usize) -> f64 {
    weights
        .iter()
        .enumerate()
        .map(|(node, weight)| weight * state[[elem, node, var]])
        .sum()
}

/// The local coordinate of `local` in the reference element `[-1, 1]`.
#[inline]
fn local_coordinate(n_nodes: usize, local: usize) -> f64 {
    (local as f64 / n_nodes as f64).mul_add(2.0, -1.0)
}

/// Evaluate the Lagrange basis at `xi` into `out`.
fn lagrange_values_into(solver: &DGSolver, xi: f64, out: &mut [f64]) {
    for (node, slot) in out.iter_mut().enumerate() {
        let xi_node = solver.xi_nodes[node];
        let mut basis = 1.0;
        for (other, xi_other) in solver.xi_nodes.iter().enumerate() {
            if other != node {
                basis *= (xi - xi_other) / (xi_node - xi_other);
            }
        }
        *slot = basis;
    }
}

fn validate_grid_shape(solver: &DGSolver, name: &str, dim: [usize; 3]) -> KwaversResult<()> {
    let expected = [solver.grid.nx, solver.grid.ny, solver.grid.nz];
    if dim != expected {
        return Err(KwaversError::InvalidInput(format!(
            "DG acoustic {name} grid shape {dim:?} does not match {expected:?}"
        )));
    }
    Ok(())
}

/// Reject a uniform grid index outside the grid, or a nonzero index on an
/// inactive axis.
pub(super) fn validate_uniform_index(
    solver: &DGSolver,
    topology: DgTopology,
    index: [usize; 3],
) -> KwaversResult<()> {
    let dims = [solver.grid.nx, solver.grid.ny, solver.grid.nz];
    for axis in 0..3 {
        if index[axis] >= dims[axis] {
            return Err(KwaversError::InvalidInput(format!(
                "DG acoustic uniform index {} exceeds axis {axis} dimension {}",
                index[axis], dims[axis]
            )));
        }
        if !topology.active_axes[axis] && index[axis] != 0 {
            return Err(KwaversError::InvalidInput(format!(
                "DG acoustic inactive axis {axis} requires index 0, got {}",
                index[axis]
            )));
        }
    }
    Ok(())
}
