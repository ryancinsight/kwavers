use kwavers_core::error::{ConfigError, KwaversError, KwaversResult};
use ndarray::Array3;

/// Create a 2D circular disc mask
///
/// Generates a binary mask with `true` inside disc, `false` outside (`makeDisc`).
///
/// # Mathematical Definition
///
/// For each grid point $(x_i, y_j, z_k)$:
/// $\text{mask}(i,j,k) = \text{true}$ if $\sqrt{(x_i - x_c)^2 + (y_j - y_c)^2} \leq r$.
/// # Errors
/// - Returns [`KwaversError::Config`] if the precondition for a Config-class constraint is violated.
///
pub fn make_disc(
    dim: (usize, usize, usize),
    spacing: (f64, f64, f64),
    center: [f64; 3],
    radius: f64,
) -> KwaversResult<Array3<bool>> {
    if radius <= 0.0 {
        return Err(KwaversError::Config(ConfigError::InvalidValue {
            parameter: "radius".to_owned(),
            value: radius.to_string(),
            constraint: "Radius must be positive".to_owned(),
        }));
    }

    let (nx, ny, nz) = dim;
    let (dx, dy, _dz) = spacing;

    let mut mask = Array3::from_elem((nx, ny, nz), false);
    let radius_sq = radius * radius;

    for i in 0..nx {
        let x = i as f64 * dx;
        let dx_sq = (x - center[0]).powi(2);

        for j in 0..ny {
            let y = j as f64 * dy;
            let dy_sq = (y - center[1]).powi(2);

            let dist_sq = dx_sq + dy_sq;

            if dist_sq <= radius_sq + 1e-10 {
                for k in 0..nz {
                    mask[[i, j, k]] = true;
                }
            }
        }
    }

    Ok(mask)
}

/// Create a 3D spherical ball mask
///
/// Generates a binary mask with `true` inside a spherical region.
/// # Errors
/// - Returns [`KwaversError::Config`] if the precondition for a Config-class constraint is violated.
///
pub fn make_ball(
    dim: (usize, usize, usize),
    spacing: (f64, f64, f64),
    center: [f64; 3],
    radius: f64,
) -> KwaversResult<Array3<bool>> {
    if radius <= 0.0 {
        return Err(KwaversError::Config(ConfigError::InvalidValue {
            parameter: "radius".to_owned(),
            value: radius.to_string(),
            constraint: "Radius must be positive".to_owned(),
        }));
    }

    let (nx, ny, nz) = dim;
    let (dx, dy, dz) = spacing;

    let mut mask = Array3::from_elem((nx, ny, nz), false);
    let radius_sq = radius * radius;

    for i in 0..nx {
        let x = i as f64 * dx;
        let dx_sq = (x - center[0]).powi(2);

        for j in 0..ny {
            let y = j as f64 * dy;
            let dy_sq = (y - center[1]).powi(2);

            for k in 0..nz {
                let z = k as f64 * dz;
                let dz_sq = (z - center[2]).powi(2);

                let dist_sq = dx_sq + dy_sq + dz_sq;

                if dist_sq <= radius_sq + 1e-10 {
                    mask[[i, j, k]] = true;
                }
            }
        }
    }

    Ok(mask)
}

/// Create a 3D spherical mask (alias for [`make_ball`])
/// # Errors
/// - Returns [`Err`] if an internal constraint is violated.
///
#[inline]
pub fn make_sphere(
    dim: (usize, usize, usize),
    spacing: (f64, f64, f64),
    center: [f64; 3],
    radius: f64,
) -> KwaversResult<Array3<bool>> {
    make_ball(dim, spacing, center, radius)
}
