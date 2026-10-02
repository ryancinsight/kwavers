//! Diffusion-solver reference fluence for the Monte Carlo validation.

use anyhow::Result;
use kwavers_grid::Grid3D;
use kwavers_medium::optical_map::OpticalPropertyMap;
use kwavers_medium::properties::OpticalPropertyData;
use kwavers_solver::forward::optical::diffusion::{DiffusionSolver, DiffusionSolverConfig};
use leto::Array3;

pub(crate) fn solve_diffusion_fluence(
    grid: &Grid3D,
    optical_map: &OpticalPropertyMap,
    source_position: [f64; 3],
) -> Result<Vec<f64>> {
    let config = DiffusionSolverConfig::default();
    let optical_properties = optical_property_map_to_array3(optical_map);
    let solver = DiffusionSolver::new(grid.clone(), optical_properties, config)?;

    let (nx, ny, nz) = grid.dimensions();
    let mut source = Array3::<f64>::zeros((nx, ny, nz));
    if let Some((i, j, k)) = grid.coordinates_to_indices(
        source_position[0].max(0.0),
        source_position[1].max(0.0),
        source_position[2].max(0.0),
    ) {
        source[[i, j, k]] = 1e6;
    }

    let fluence = solver.solve(&source)?;
    Ok(flatten_kji(&fluence))
}

fn optical_property_map_to_array3(map: &OpticalPropertyMap) -> Array3<OpticalPropertyData> {
    map.properties().clone()
}

fn flatten_kji(field: &Array3<f64>) -> Vec<f64> {
    let [nx, ny, nz] = field.shape();
    let mut out = Vec::with_capacity(nx * ny * nz);
    for k in 0..nz {
        for j in 0..ny {
            for i in 0..nx {
                out.push(field[[i, j, k]]);
            }
        }
    }
    out
}
