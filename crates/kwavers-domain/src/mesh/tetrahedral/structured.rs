//! Structured Cartesian-grid tetrahedralization.
//!
//! # Algorithm
//!
//! Given a Cartesian vertex lattice with dimensions `(nx, ny, nz)`, each
//! hexahedral cell is split into six tetrahedra sharing the body diagonal
//! `v000 -> v111`:
//!
//! ```text
//! [000,100,110,111], [000,100,101,111], [000,010,110,111],
//! [000,010,011,111], [000,001,101,111], [000,001,011,111]
//! ```
//!
//! # Theorem
//!
//! For an orthogonal cell with side lengths `(dx, dy, dz)`, each tetrahedron has
//! volume `dx*dy*dz/6`, because the determinant of its three edge vectors has
//! magnitude `dx*dy*dz`. The six tetrahedra are non-overlapping interiors whose
//! union is the original hexahedron, so total generated volume is
//! `(nx-1)(ny-1)(nz-1) dx dy dz`.

use super::mesh::TetrahedralMesh;
use super::types::MeshBoundaryType;
use kwavers_core::error::{KwaversError, KwaversResult};
use crate::grid::Grid;

impl TetrahedralMesh {
    /// Build a tetrahedral mesh from the grid's Cartesian vertex lattice.
    ///
    /// The grid points are interpreted as FEM vertices, not cell centers. At
    /// least two points in each axis are required to form a 3-D cell.
    /// # Errors
    /// - Returns [`KwaversError::InvalidInput`] if the precondition for invalid or out-of-range input parameters is violated.
    /// - Propagates any [`KwaversError`] returned by called functions.
    ///
    pub fn from_grid_vertices(grid: &Grid) -> KwaversResult<Self> {
        if grid.nx < 2 || grid.ny < 2 || grid.nz < 2 {
            return Err(KwaversError::InvalidInput(format!(
                "Tetrahedral mesh generation requires at least 2 vertices per axis, got {}x{}x{}",
                grid.nx, grid.ny, grid.nz
            )));
        }

        if !grid.dx.is_finite()
            || !grid.dy.is_finite()
            || !grid.dz.is_finite()
            || grid.dx <= 0.0
            || grid.dy <= 0.0
            || grid.dz <= 0.0
        {
            return Err(KwaversError::InvalidInput(format!(
                "Tetrahedral mesh generation requires finite positive spacing, got ({}, {}, {})",
                grid.dx, grid.dy, grid.dz
            )));
        }

        let mut mesh = Self::new();
        let node_count = grid.nx * grid.ny * grid.nz;
        mesh.nodes.reserve(node_count);
        mesh.elements
            .reserve((grid.nx - 1) * (grid.ny - 1) * (grid.nz - 1) * 6);

        for k in 0..grid.nz {
            for j in 0..grid.ny {
                for i in 0..grid.nx {
                    let coordinates = [
                        (i as f64).mul_add(grid.dx, grid.origin[0]),
                        (j as f64).mul_add(grid.dy, grid.origin[1]),
                        (k as f64).mul_add(grid.dz, grid.origin[2]),
                    ];
                    mesh.add_node(coordinates, MeshBoundaryType::Interior);
                }
            }
        }

        for k in 0..(grid.nz - 1) {
            for j in 0..(grid.ny - 1) {
                for i in 0..(grid.nx - 1) {
                    let v000 = grid_vertex_index(grid, i, j, k);
                    let v100 = grid_vertex_index(grid, i + 1, j, k);
                    let v010 = grid_vertex_index(grid, i, j + 1, k);
                    let v110 = grid_vertex_index(grid, i + 1, j + 1, k);
                    let v001 = grid_vertex_index(grid, i, j, k + 1);
                    let v101 = grid_vertex_index(grid, i + 1, j, k + 1);
                    let v011 = grid_vertex_index(grid, i, j + 1, k + 1);
                    let v111 = grid_vertex_index(grid, i + 1, j + 1, k + 1);

                    for tet in [
                        [v000, v100, v110, v111],
                        [v000, v100, v101, v111],
                        [v000, v010, v110, v111],
                        [v000, v010, v011, v111],
                        [v000, v001, v101, v111],
                        [v000, v001, v011, v111],
                    ] {
                        mesh.add_element(tet, 0)?;
                    }
                }
            }
        }

        Ok(mesh)
    }
}

fn grid_vertex_index(grid: &Grid, i: usize, j: usize, k: usize) -> usize {
    i + grid.nx * (j + grid.ny * k)
}
