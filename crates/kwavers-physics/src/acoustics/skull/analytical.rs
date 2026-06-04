use kwavers_core::error::{KwaversError, KwaversResult};
use kwavers_grid::Grid;
use ndarray::Array3;
/// Generate spherical skull.
/// # Errors
/// - Returns [`Err`] if an internal constraint is violated.
///
pub fn generate_spherical_skull(
    grid: &Grid,
    thickness: f64,
    radius: f64,
) -> KwaversResult<Array3<f64>> {
    let mut mask = Array3::zeros((grid.nx, grid.ny, grid.nz));

    let cx = grid.nx as f64 / 2.0;
    let cy = grid.ny as f64 / 2.0;
    let cz = grid.nz as f64 / 2.0;

    let inner_radius = radius - thickness / grid.dx;
    let outer_radius = radius;

    for i in 0..grid.nx {
        for j in 0..grid.ny {
            for k in 0..grid.nz {
                let r = (k as f64 - cz)
                    .mul_add(
                        k as f64 - cz,
                        (j as f64 - cy).mul_add(j as f64 - cy, (i as f64 - cx).powi(2)),
                    )
                    .sqrt();

                if r >= inner_radius && r <= outer_radius {
                    mask[[i, j, k]] = 1.0;
                }
            }
        }
    }

    Ok(mask)
}
/// Generate ellipsoidal skull.
/// # Errors
/// - Returns [`KwaversError::InvalidInput`] if the precondition for invalid or out-of-range input parameters is violated.
///
pub fn generate_ellipsoidal_skull(
    grid: &Grid,
    thickness: f64,
    params: &[f64],
) -> KwaversResult<Array3<f64>> {
    if params.len() < 3 {
        return Err(KwaversError::InvalidInput(
            "Ellipsoid requires 3 radii".to_owned(),
        ));
    }

    let mut mask = Array3::zeros((grid.nx, grid.ny, grid.nz));

    let cx = grid.nx as f64 / 2.0;
    let cy = grid.ny as f64 / 2.0;
    let cz = grid.nz as f64 / 2.0;

    let (rx, ry, rz) = (params[0], params[1], params[2]);
    let thickness_x = thickness / grid.dx;

    for i in 0..grid.nx {
        for j in 0..grid.ny {
            for k in 0..grid.nz {
                let dx = (i as f64 - cx) / rx;
                let dy = (j as f64 - cy) / ry;
                let dz = (k as f64 - cz) / rz;

                let r = (dx * dx + dy * dy + dz * dz).sqrt();

                let inner_r = 1.0 - thickness_x / rx;
                if r >= inner_r && r <= 1.0 {
                    mask[[i, j, k]] = 1.0;
                }
            }
        }
    }

    Ok(mask)
}
