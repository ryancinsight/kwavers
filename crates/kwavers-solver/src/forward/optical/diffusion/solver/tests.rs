//! Value-semantic regression tests for the diffusion solver.

use super::{analytical, DiffusionSolver, DiffusionSolverConfig};
use kwavers_domain::grid::Grid;
use kwavers_domain::medium::properties::OpticalPropertyData;
use anyhow::Result;
use ndarray::Array3;

#[test]
fn test_analytical_infinite_medium() {
    let tissue = OpticalPropertyData::soft_tissue();
    let power = 1.0;

    let distances = [0.001, 0.005, 0.01, 0.02];
    for &r in &distances {
        let fluence = analytical::infinite_medium_point_source(r, power, tissue);
        assert!(fluence > 0.0);
        assert!(fluence.is_finite());
    }

    let fluence_near = analytical::infinite_medium_point_source(0.01, power, tissue);
    let fluence_far = analytical::infinite_medium_point_source(0.02, power, tissue);
    assert!(
        fluence_near > fluence_far,
        "Fluence should decay with distance"
    );
}

#[test]
fn test_solver_uniform_medium() -> Result<()> {
    let grid = Grid::new(20, 20, 20, 1e-3, 1e-3, 1e-3)?;
    let tissue = OpticalPropertyData::soft_tissue();

    let config = DiffusionSolverConfig {
        max_iterations: 1000,
        tolerance: 1e-4,
        boundary_parameter: 2.0,
        boundary_conditions: None,
        verbose: false,
    };

    let solver = DiffusionSolver::uniform(grid.clone(), tissue, config)?;

    let (nx, ny, nz) = grid.dimensions();
    let mut source = Array3::zeros((nx, ny, nz));
    source[[nx / 2, ny / 2, nz / 2]] = 1e6;

    let fluence = solver.solve(&source)?;

    assert!(
        fluence.iter().all(|&x| x >= 0.0),
        "Fluence must be non-negative"
    );
    assert!(
        fluence.iter().any(|&x| x > 0.0),
        "Fluence should be non-zero"
    );

    let max_idx = fluence
        .indexed_iter()
        .max_by(|(_, a), (_, b)| a.total_cmp(b))
        .unwrap()
        .0;
    let dist_from_center = ((max_idx.0 as isize - nx as isize / 2).pow(2)
        + (max_idx.1 as isize - ny as isize / 2).pow(2)
        + (max_idx.2 as isize - nz as isize / 2).pow(2)) as f64;
    assert!(
        dist_from_center < 10.0,
        "Maximum fluence should be near source location"
    );

    Ok(())
}

#[test]
fn test_solver_symmetry() -> Result<()> {
    let grid = Grid::new(30, 30, 30, 1e-3, 1e-3, 1e-3)?;
    let tissue = OpticalPropertyData::soft_tissue();

    let config = DiffusionSolverConfig {
        max_iterations: 2000,
        tolerance: 1e-5,
        boundary_parameter: 2.0,
        boundary_conditions: None,
        verbose: false,
    };

    let solver = DiffusionSolver::uniform(grid.clone(), tissue, config)?;

    let (nx, ny, nz) = grid.dimensions();
    let mut source = Array3::zeros((nx, ny, nz));
    let center = (nx / 2, ny / 2, nz / 2);
    source[[center.0, center.1, center.2]] = 1e6;

    let fluence = solver.solve(&source)?;

    let r_test = 5;
    let test_points = [
        (center.0 + r_test, center.1, center.2),
        (center.0 - r_test, center.1, center.2),
        (center.0, center.1 + r_test, center.2),
        (center.0, center.1 - r_test, center.2),
    ];

    let fluence_values: Vec<f64> = test_points
        .iter()
        .filter(|&&(i, j, k)| i < nx && j < ny && k < nz)
        .map(|&(i, j, k)| fluence[[i, j, k]])
        .collect();

    if fluence_values.len() >= 2 {
        let mean = fluence_values.iter().sum::<f64>() / fluence_values.len() as f64;
        let max_deviation = fluence_values
            .iter()
            .map(|&f| (f - mean).abs() / mean)
            .max_by(|a, b| a.total_cmp(b))
            .unwrap_or(0.0);

        assert!(
            max_deviation < 0.2,
            "Symmetry check failed: max deviation = {:.1}%",
            max_deviation * 100.0
        );
    }

    Ok(())
}

#[test]
fn test_heterogeneous_medium() -> Result<()> {
    let grid = Grid::new(20, 20, 20, 1e-3, 1e-3, 1e-3)?;
    let (nx, ny, nz) = grid.dimensions();

    let tissue = OpticalPropertyData::soft_tissue();
    let tumor = OpticalPropertyData::tumor();

    let mut optical_map = Array3::from_elem((nx, ny, nz), tissue);

    let center = (nx / 2, ny / 2, nz / 2);
    let radius = 5;
    for i in 0..nx {
        for j in 0..ny {
            for k in 0..nz {
                let dist_sq = (i as isize - center.0 as isize).pow(2)
                    + (j as isize - center.1 as isize).pow(2)
                    + (k as isize - center.2 as isize).pow(2);
                if dist_sq <= (radius as isize).pow(2) {
                    optical_map[[i, j, k]] = tumor;
                }
            }
        }
    }

    let config = DiffusionSolverConfig {
        max_iterations: 2000,
        tolerance: 1e-4,
        boundary_parameter: 2.0,
        boundary_conditions: None,
        verbose: false,
    };

    let solver = DiffusionSolver::new(grid, optical_map, config)?;

    let mut source = Array3::zeros((nx, ny, nz));
    source[[center.0, center.1, center.2]] = 1e6;

    let fluence = solver.solve(&source)?;

    assert!(fluence[[center.0, center.1, center.2]] > 0.0);
    assert!(fluence.iter().all(|&x| x >= 0.0));

    Ok(())
}
