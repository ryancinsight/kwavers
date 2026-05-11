use crate::core::error::{ConfigError, KwaversError, KwaversResult};
use ndarray::Array3;

/// Create a 2D circle outline (shell) mask.
///
/// Generates a binary mask with `true` on the circle perimeter, `false` elsewhere.
/// # Errors
/// - Returns [`KwaversError::Config`] if the precondition for a Config-class constraint is violated.
///
pub fn make_circle(
    dim: (usize, usize, usize),
    spacing: (f64, f64, f64),
    center: [f64; 3],
    radius: f64,
    thickness: usize,
) -> KwaversResult<Array3<bool>> {
    if radius <= 0.0 {
        return Err(KwaversError::Config(ConfigError::InvalidValue {
            parameter: "radius".to_owned(),
            value: radius.to_string(),
            constraint: "Radius must be positive".to_owned(),
        }));
    }
    if thickness == 0 {
        return Err(KwaversError::Config(ConfigError::InvalidValue {
            parameter: "thickness".to_owned(),
            value: thickness.to_string(),
            constraint: "Thickness must be at least 1".to_owned(),
        }));
    }

    let (nx, ny, nz) = dim;
    let (dx, dy, _dz) = spacing;

    let mut mask = Array3::from_elem((nx, ny, nz), false);
    let half_thickness = thickness as f64 * dx * 0.5;

    for i in 0..nx {
        let x = i as f64 * dx;
        let dx_sq = (x - center[0]).powi(2);

        for j in 0..ny {
            let y = j as f64 * dy;
            let dist = (y - center[1]).mul_add(y - center[1], dx_sq).sqrt();

            if (dist - radius).abs() <= half_thickness {
                for k in 0..nz {
                    mask[[i, j, k]] = true;
                }
            }
        }
    }

    Ok(mask)
}
