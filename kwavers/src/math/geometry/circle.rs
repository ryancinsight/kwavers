use crate::core::error::{ConfigError, KwaversError, KwaversResult};
use ndarray::Array3;

/// Create a 2D circle outline (shell) mask.
///
/// Generates a binary mask with `true` on the circle perimeter, `false` elsewhere.
pub fn make_circle(
    dim: (usize, usize, usize),
    spacing: (f64, f64, f64),
    center: [f64; 3],
    radius: f64,
    thickness: usize,
) -> KwaversResult<Array3<bool>> {
    if radius <= 0.0 {
        return Err(KwaversError::Config(ConfigError::InvalidValue {
            parameter: "radius".to_string(),
            value: radius.to_string(),
            constraint: "Radius must be positive".to_string(),
        }));
    }
    if thickness == 0 {
        return Err(KwaversError::Config(ConfigError::InvalidValue {
            parameter: "thickness".to_string(),
            value: thickness.to_string(),
            constraint: "Thickness must be at least 1".to_string(),
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
            let dist = (dx_sq + (y - center[1]).powi(2)).sqrt();

            if (dist - radius).abs() <= half_thickness {
                for k in 0..nz {
                    mask[[i, j, k]] = true;
                }
            }
        }
    }

    Ok(mask)
}
