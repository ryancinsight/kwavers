//! Verification that sampling reproduces the medium's own property values.

use super::*;
use crate::homogeneous::HomogeneousMedium;

fn grid() -> Grid {
    Grid::new(4, 3, 2, 1.0e-4, 1.0e-4, 1.0e-4).expect("valid grid")
}

/// Every sampled field equals what the medium reports at that cell — the whole
/// contract of this type. A field that silently kept its zero initialization
/// (the failure mode when a property is added to the struct but not to the
/// sampling loop) is exactly what this catches.
#[test]
fn sampling_reproduces_medium_properties() {
    let grid = grid();
    let medium = HomogeneousMedium::water(&grid);
    let fields = MaterialFields::sample(&medium, &grid);

    let shape = [grid.nx, grid.ny, grid.nz];
    for field in [
        &fields.rho0,
        &fields.c0,
        &fields.alpha0_db,
        &fields.alpha_power,
        &fields.nonlinearity,
    ] {
        assert_eq!(field.shape(), shape);
    }

    for k in 0..grid.nz {
        for j in 0..grid.ny {
            for i in 0..grid.nx {
                let (x, y, z) = grid.indices_to_coordinates(i, j, k);
                let index = [i, j, k];
                assert_eq!(fields.rho0[index], medium.density(i, j, k));
                assert_eq!(fields.c0[index], medium.sound_speed(i, j, k));
                assert_eq!(
                    fields.alpha0_db[index],
                    medium.alpha_coefficient(x, y, z, &grid)
                );
                assert_eq!(
                    fields.alpha_power[index],
                    medium.alpha_power(x, y, z, &grid)
                );
                assert_eq!(fields.nonlinearity[index], medium.nonlinearity(i, j, k));
            }
        }
    }

    // Water is a real medium: positive density, speed and exponent.
    assert!(fields.rho0[[0, 0, 0]] > 0.0);
    assert!(fields.c0[[0, 0, 0]] > 0.0);
    assert!(fields.alpha_power[[0, 0, 0]] > 0.0);
}

#[test]
fn bulk_modulus_is_rho_c_squared() {
    let grid = grid();
    let medium = HomogeneousMedium::water(&grid);
    let fields = MaterialFields::sample(&medium, &grid);
    let modulus = fields.bulk_modulus();

    assert_eq!(modulus.shape(), fields.rho0.shape());
    for k in 0..grid.nz {
        for j in 0..grid.ny {
            for i in 0..grid.nx {
                let index = [i, j, k];
                let expected = fields.rho0[index] * fields.c0[index] * fields.c0[index];
                assert_eq!(modulus[index], expected);
            }
        }
    }
}

#[test]
fn losslessness_tracks_the_absorption_prefactor() {
    let grid = grid();
    let mut fields = MaterialFields::new((grid.nx, grid.ny, grid.nz));
    assert!(fields.is_lossless(), "a zeroed field set is lossless");

    fields.alpha0_db[[2, 1, 0]] = 0.5;
    assert!(
        !fields.is_lossless(),
        "a single absorbing cell makes the medium lossy"
    );
}
