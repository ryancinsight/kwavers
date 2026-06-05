//! Tensor-product DG element topology for Cartesian grids.

use kwavers_core::error::{KwaversError, KwaversResult};
use kwavers_grid::Grid;
use ndarray::Array1;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum CoefficientLayout {
    Line1D,
    TensorProduct(DgTopology),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) struct DgTopology {
    pub(super) active_axes: [bool; 3],
    pub(super) element_counts: [usize; 3],
    pub(super) active_dim: usize,
    pub(super) n_nodes: usize,
    pub(super) nodes_per_element: usize,
    pub(super) n_elements: usize,
}

impl DgTopology {
    pub(super) fn from_grid(grid: &Grid, n_nodes: usize) -> KwaversResult<Self> {
        if n_nodes == 0 {
            return Err(KwaversError::InvalidInput(
                "DG topology requires at least one node per element".to_owned(),
            ));
        }

        let dims = [grid.nx, grid.ny, grid.nz];
        let active_axes = [dims[0] > 1, dims[1] > 1, dims[2] > 1];
        let active_dim = active_axes.iter().filter(|&&active| active).count();
        if active_dim == 0 {
            return Err(KwaversError::InvalidInput(
                "DG topology requires at least one active grid axis".to_owned(),
            ));
        }

        let mut element_counts = [1_usize; 3];
        for axis in 0..3 {
            if active_axes[axis] {
                if !dims[axis].is_multiple_of(n_nodes) {
                    return Err(KwaversError::InvalidInput(format!(
                        "DG axis {axis} has {} nodes, which is not divisible by nodes_per_element={n_nodes}",
                        dims[axis]
                    )));
                }
                element_counts[axis] = dims[axis] / n_nodes;
            }
        }

        let nodes_per_element = (0..active_dim).fold(1_usize, |acc, _| acc * n_nodes);
        let n_elements = element_counts.iter().product();

        Ok(Self {
            active_axes,
            element_counts,
            active_dim,
            n_nodes,
            nodes_per_element,
            n_elements,
        })
    }

    #[inline]
    pub(super) fn element_coords(self, elem: usize) -> [usize; 3] {
        let ex = elem % self.element_counts[0];
        let ey = (elem / self.element_counts[0]) % self.element_counts[1];
        let ez = elem / (self.element_counts[0] * self.element_counts[1]);
        [ex, ey, ez]
    }

    #[inline]
    pub(super) fn element_index(self, coords: [usize; 3]) -> usize {
        coords[0] + self.element_counts[0] * (coords[1] + self.element_counts[1] * coords[2])
    }

    #[inline]
    pub(super) fn node_coords(self, node: usize) -> [usize; 3] {
        let mut coords = [0_usize; 3];
        let mut remaining = node;
        for (axis, active) in self.active_axes.iter().copied().enumerate() {
            if active {
                coords[axis] = remaining % self.n_nodes;
                remaining /= self.n_nodes;
            }
        }
        coords
    }

    #[inline]
    pub(super) fn node_index(self, coords: [usize; 3]) -> usize {
        let mut index = 0_usize;
        let mut stride = 1_usize;
        for (axis, active) in self.active_axes.iter().copied().enumerate() {
            if active {
                index += coords[axis] * stride;
                stride *= self.n_nodes;
            }
        }
        index
    }

    #[inline]
    pub(super) fn node_with_axis(self, node: usize, axis: usize, axis_node: usize) -> usize {
        let mut coords = self.node_coords(node);
        coords[axis] = axis_node;
        self.node_index(coords)
    }

    #[inline]
    pub(super) fn grid_index(self, elem: usize, node: usize) -> [usize; 3] {
        let elem_coords = self.element_coords(elem);
        let node_coords = self.node_coords(node);
        let mut index = [0_usize; 3];
        for axis in 0..3 {
            index[axis] = if self.active_axes[axis] {
                elem_coords[axis] * self.n_nodes + node_coords[axis]
            } else {
                0
            };
        }
        index
    }

    #[inline]
    pub(super) fn neighbor(self, elem: usize, axis: usize, positive: bool) -> usize {
        let mut coords = self.element_coords(elem);
        let count = self.element_counts[axis];
        coords[axis] = if positive {
            (coords[axis] + 1) % count
        } else if coords[axis] == 0 {
            count - 1
        } else {
            coords[axis] - 1
        };
        self.element_index(coords)
    }

    #[inline]
    pub(super) fn node_weight(self, node: usize, weights: &Array1<f64>) -> f64 {
        let coords = self.node_coords(node);
        let mut weight = 1.0;
        for axis in 0..3 {
            if self.active_axes[axis] {
                weight *= weights[coords[axis]];
            }
        }
        weight
    }
}
