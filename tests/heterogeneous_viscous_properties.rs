use kwavers::grid::Grid;
use kwavers::medium::heterogeneous::HeterogeneousMedium;
use kwavers::medium::viscous::ViscousProperties;

#[test]
fn test_heterogeneous_viscous_properties_interpolated() {
    let grid = Grid::new(4, 4, 4, 1e-3, 1e-3, 1e-3).unwrap();
    let mut medium = HeterogeneousMedium::new(4, 4, 4, true);

    // Uniform fields for deterministic checks
    medium.density.fill(1000.0);
    medium.viscosity.fill(1.0e-3);
    medium.shear_viscosity_coeff.fill(2.0e-3);
    medium.bulk_viscosity_coeff.fill(5.0e-3);

    // A mid-domain continuous coordinate
    let x = 1.5e-3; // halfway between indices 1 and 2
    let y = 1.5e-3;
    let z = 1.5e-3;

    // Validate dynamic viscosity
    let mu = medium.viscosity(x, y, z, &grid);
    assert!((mu - 1.0e-3).abs() < 1e-12);

    // Validate shear viscosity coefficient retrieval
    let mu_s = medium.shear_viscosity(x, y, z, &grid);
    assert!((mu_s - 2.0e-3).abs() < 1e-12);

    // Validate bulk viscosity coefficient retrieval
    let mu_b = medium.bulk_viscosity(x, y, z, &grid);
    assert!((mu_b - 5.0e-3).abs() < 1e-12);

    // Validate kinematic viscosity ν = μ / ρ
    let nu = medium.kinematic_viscosity(x, y, z, &grid);
    assert!((nu - 1.0e-6).abs() < 1e-12);
}

#[test]
fn test_heterogeneous_viscous_properties_nearest_neighbor() {
    let grid = Grid::new(4, 4, 4, 1e-3, 1e-3, 1e-3).unwrap();
    let mut medium = HeterogeneousMedium::new(4, 4, 4, false);

    medium.density.fill(1000.0);
    medium.viscosity.fill(1.2e-3);
    medium.shear_viscosity_coeff.fill(2.4e-3);
    medium.bulk_viscosity_coeff.fill(3.6e-3);

    // Choose a coordinate that maps to a valid nearest neighbor index
    let x = 2.0e-3; // nearest neighbor round to index 2
    let y = 1.0e-3; // index 1
    let z = 0.0;    // index 0

    let mu = medium.viscosity(x, y, z, &grid);
    assert!((mu - 1.2e-3).abs() < 1e-12);

    let mu_s = medium.shear_viscosity(x, y, z, &grid);
    assert!((mu_s - 2.4e-3).abs() < 1e-12);

    let mu_b = medium.bulk_viscosity(x, y, z, &grid);
    assert!((mu_b - 3.6e-3).abs() < 1e-12);

    let nu = medium.kinematic_viscosity(x, y, z, &grid);
    assert!((nu - 1.2e-6).abs() < 1e-12);
}
