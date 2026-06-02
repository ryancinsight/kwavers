use super::{validate_tensor_state, ACOUSTIC_PRESSURE_VAR};
use kwavers_core::error::{KwaversError, KwaversResult};
use crate::forward::pstd::dg::dg_solver::core::DGSolver;
use crate::forward::pstd::dg::dg_solver::topology::DgTopology;
use ndarray::Array3;

#[derive(Debug, Clone)]
pub(super) struct AxisMap {
    pub(super) elem: usize,
    pub(super) basis: Vec<f64>,
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
        validate_grid_shape(self, "pressure", pressure.dim())?;
        let maps = axis_maps(self, topology);
        for i in 0..self.grid.nx {
            for j in 0..self.grid.ny {
                for k in 0..self.grid.nz {
                    pressure[(i, j, k)] =
                        interpolate_var(topology, state, &maps, [i, j, k], ACOUSTIC_PRESSURE_VAR);
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
            ("pressure", pressure.dim()),
            ("ux", ux.dim()),
            ("uy", uy.dim()),
            ("uz", uz.dim()),
        ] {
            validate_grid_shape(self, name, dim)?;
        }
        let maps = axis_maps(self, topology);
        for i in 0..self.grid.nx {
            for j in 0..self.grid.ny {
                for k in 0..self.grid.nz {
                    let index = [i, j, k];
                    pressure[(i, j, k)] =
                        interpolate_var(topology, state, &maps, index, ACOUSTIC_PRESSURE_VAR);
                    ux[(i, j, k)] = interpolate_var(topology, state, &maps, index, 1);
                    uy[(i, j, k)] = interpolate_var(topology, state, &maps, index, 2);
                    uz[(i, j, k)] = interpolate_var(topology, state, &maps, index, 3);
                }
            }
        }
        Ok(())
    }
}

pub(super) fn axis_map_for_index(
    solver: &DGSolver,
    topology: DgTopology,
    axis: usize,
    index: usize,
) -> KwaversResult<AxisMap> {
    let dims = [solver.grid.nx, solver.grid.ny, solver.grid.nz];
    if index >= dims[axis] {
        return Err(KwaversError::InvalidInput(format!(
            "DG acoustic uniform index {index} exceeds axis {axis} dimension {}",
            dims[axis]
        )));
    }
    if !topology.active_axes[axis] {
        if index != 0 {
            return Err(KwaversError::InvalidInput(format!(
                "DG acoustic inactive axis {axis} requires index 0, got {index}"
            )));
        }
        return Ok(AxisMap {
            elem: 0,
            basis: vec![1.0],
        });
    }
    let elem = index / solver.n_nodes;
    let local = (index % solver.n_nodes) as f64 / solver.n_nodes as f64;
    let xi = local.mul_add(2.0, -1.0);
    Ok(AxisMap {
        elem,
        basis: lagrange_values(solver, xi),
    })
}

#[allow(clippy::needless_range_loop)]
pub(super) fn lagrange_values(solver: &DGSolver, xi: f64) -> Vec<f64> {
    let mut values = vec![1.0; solver.n_nodes];
    for node in 0..solver.n_nodes {
        let xi_node = solver.xi_nodes[node];
        let mut basis = 1.0;
        for other in 0..solver.n_nodes {
            if other != node {
                basis *= (xi - solver.xi_nodes[other]) / (xi_node - solver.xi_nodes[other]);
            }
        }
        values[node] = basis;
    }
    values
}

fn axis_maps(solver: &DGSolver, topology: DgTopology) -> Vec<Vec<AxisMap>> {
    [solver.grid.nx, solver.grid.ny, solver.grid.nz]
        .into_iter()
        .enumerate()
        .map(|(axis, dim)| {
            (0..dim)
                .map(|index| {
                    axis_map_for_index(solver, topology, axis, index).expect("validated grid index")
                })
                .collect()
        })
        .collect()
}

fn interpolate_var(
    topology: DgTopology,
    state: &Array3<f64>,
    maps: &[Vec<AxisMap>],
    index: [usize; 3],
    var: usize,
) -> f64 {
    let axis_maps = [&maps[0][index[0]], &maps[1][index[1]], &maps[2][index[2]]];
    let elem = topology.element_index([axis_maps[0].elem, axis_maps[1].elem, axis_maps[2].elem]);
    let mut value = 0.0;
    for node in 0..topology.nodes_per_element {
        let coords = topology.node_coords(node);
        let mut weight = 1.0;
        for axis in 0..3 {
            if topology.active_axes[axis] {
                weight *= axis_maps[axis].basis[coords[axis]];
            }
        }
        value += weight * state[(elem, node, var)];
    }
    value
}

fn validate_grid_shape(
    solver: &DGSolver,
    name: &str,
    dim: (usize, usize, usize),
) -> KwaversResult<()> {
    let expected = (solver.grid.nx, solver.grid.ny, solver.grid.nz);
    if dim != expected {
        return Err(KwaversError::InvalidInput(format!(
            "DG acoustic {name} grid shape {dim:?} does not match {expected:?}"
        )));
    }
    Ok(())
}
