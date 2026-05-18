//! Tests for the FEM Helmholtz frequency-domain backend.

use super::backend::FemHelmholtzBackend;
use crate::domain::grid::Grid;
use crate::domain::medium::HomogeneousMedium;
use crate::simulation::backends::acoustic::FrequencyDomainAcousticBackend;
use crate::solver::forward::helmholtz::fem::{FemHelmholtzConfig, FemPreconditionerType};
use ndarray::arr2;
use num_complex::Complex64;

fn medium_for(grid: &Grid) -> HomogeneousMedium {
    HomogeneousMedium::new(1000.0, 1500.0, 0.0, 0.0, grid)
}

#[test]
fn fem_helmholtz_backend_solves_frequency_domain_pressure() {
    let grid = Grid::new(2, 2, 2, 1.0, 1.0, 1.0).unwrap();
    let medium = medium_for(&grid);
    let config = FemHelmholtzConfig {
        wavenumber: 1.0,
        radiation_boundary: false,
        preconditioner: FemPreconditionerType::None,
        tolerance: 1.0e-10,
        ..Default::default()
    };
    let mut backend = FemHelmholtzBackend::from_grid(&grid, &medium, config).unwrap();
    backend.add_nodal_load(0, Complex64::new(1.0, 0.0)).unwrap();

    backend.solve().unwrap();

    assert_eq!(backend.mesh_size(), (8, 6));
    assert_eq!(backend.wavenumber(), 1.0);
    let norm: f64 = backend.pressure_solution().iter().map(|u| u.norm()).sum();
    assert!(norm > 0.0);
}

#[test]
fn fem_helmholtz_backend_interpolates_solved_pressure() {
    let grid = Grid::new(2, 2, 2, 1.0, 1.0, 1.0).unwrap();
    let medium = medium_for(&grid);
    let config = FemHelmholtzConfig {
        wavenumber: 1.0,
        radiation_boundary: false,
        preconditioner: FemPreconditionerType::None,
        tolerance: 1.0e-10,
        ..Default::default()
    };
    let mut backend = FemHelmholtzBackend::from_grid(&grid, &medium, config).unwrap();
    backend.add_nodal_load(0, Complex64::new(1.0, 0.0)).unwrap();
    backend.solve().unwrap();

    let points = arr2(&[[0.0, 0.0, 0.0], [2.0, 2.0, 2.0]]);
    let values = backend.interpolate_pressure(points.view()).unwrap();

    assert_eq!(values.len(), 2);
    assert!(values[0].norm() > 0.0);
    assert_eq!(values[1], Complex64::new(0.0, 0.0));
}

#[test]
fn fem_helmholtz_backend_rejects_invalid_nodal_load() {
    let grid = Grid::new(2, 2, 2, 1.0, 1.0, 1.0).unwrap();
    let medium = medium_for(&grid);
    let config = FemHelmholtzConfig {
        wavenumber: 0.0,
        radiation_boundary: false,
        ..Default::default()
    };
    let mut backend = FemHelmholtzBackend::from_grid(&grid, &medium, config).unwrap();

    assert!(backend.add_nodal_load(8, Complex64::new(1.0, 0.0)).is_err());
    assert!(backend
        .add_nodal_load(0, Complex64::new(f64::NAN, 0.0))
        .is_err());
}
