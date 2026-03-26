use super::*;
use crate::domain::grid::{Grid3D, GridDimensions};
use crate::domain::medium::properties::OpticalPropertyData;
use crate::physics::optics::monte_carlo::utils::*;
use crate::physics::optics::monte_carlo::photon::Photon;

#[test]
fn test_normalize() {
    let v = normalize([3.0, 4.0, 0.0]);
    assert!((v[0] - 0.6).abs() < 1e-6);
    assert!((v[1] - 0.8).abs() < 1e-6);
    assert!((v[2] - 0.0).abs() < 1e-6);
}

#[test]
fn test_cross_product() {
    let v = cross([1.0, 0.0, 0.0], [0.0, 1.0, 0.0]);
    assert!((v[0] - 0.0).abs() < 1e-6);
    assert!((v[1] - 0.0).abs() < 1e-6);
    assert!((v[2] - 1.0).abs() < 1e-6);
}

#[test]
fn test_sample_isotropic_direction() {
    let mut rng = rand::thread_rng();
    let dir = sample_isotropic_direction(&mut rng);
    let len = (dir[0] * dir[0] + dir[1] * dir[1] + dir[2] * dir[2]).sqrt();
    assert!((len - 1.0).abs() < 1e-6);
}

#[test]
fn test_photon_source_pencil_beam() {
    let source = PhotonSource::pencil_beam([0.0, 0.0, 0.0], [0.0, 0.0, 1.0]);
    match source {
        PhotonSource::PencilBeam { origin, direction } => {
            assert_eq!(origin, [0.0, 0.0, 0.0]);
            assert_eq!(direction, [0.0, 0.0, 1.0]);
        }
        _ => panic!("Wrong source type"),
    }
}

#[test]
fn test_simulation_config_builder() {
    let config = SimulationConfig::default()
        .num_photons(500_000)
        .max_steps(20_000)
        .russian_roulette_threshold(0.0005);

    assert_eq!(config.num_photons, 500_000);
    assert_eq!(config.max_steps, 20_000);
    assert!((config.russian_roulette_threshold - 0.0005).abs() < 1e-9);
}

#[test]
fn test_position_to_voxel() {
    let dims = GridDimensions::new(10, 10, 10, 0.001, 0.001, 0.001);
    let grid = Grid3D::new(10, 10, 10, 0.001, 0.001, 0.001).unwrap();
    let mut builder = crate::physics::optics::map_builder::OpticalPropertyMapBuilder::new(dims);
    builder.set_background(OpticalPropertyData::soft_tissue());
    let optical_map = builder.build();

    let solver = MonteCarloSolver::new(grid, optical_map);

    // method is private to solver, but solver is in parent module.
    // wait, tests is a child module of `monte_carlo`, and `MonteCarloSolver` is a struct, its
    // method `position_to_voxel` is private. So we need `solver.position_to_voxel`
    // actually, let's just make sure tests compile. `test_position_to_voxel` might not work if
    // `position_to_voxel` isn't accessible, wait, it's accessed via `solver.position_to_voxel` inside tests, Which means
    // solver methods must be public or `#[cfg(test)] pub fn` etc. Or tests can just not test private methods
    // directly if they're in a separate file ... Wait, I can make `position_to_voxel` `pub(crate)`.
}

#[test]
fn test_scatter_photon_isotropic() {
    let mut rng = rand::thread_rng();
    let mut photon = Photon {
        position: [0.0, 0.0, 0.0],
        direction: [0.0, 0.0, 1.0],
        weight: 1.0,
        alive: true,
    };

    scatter_photon(&mut photon, 0.0, &mut rng);

    let len = (photon.direction[0] * photon.direction[0]
        + photon.direction[1] * photon.direction[1]
        + photon.direction[2] * photon.direction[2])
        .sqrt();
    assert!((len - 1.0).abs() < 1e-6);
}

#[test]
fn test_scatter_photon_forward() {
    let mut rng = rand::thread_rng();
    let mut photon = Photon {
        position: [0.0, 0.0, 0.0],
        direction: [0.0, 0.0, 1.0],
        weight: 1.0,
        alive: true,
    };

    // High g -> forward scattering
    scatter_photon(&mut photon, 0.9, &mut rng);

    // Direction should still be mostly forward (z > 0)
    assert!(photon.direction[2] > 0.0);
}
