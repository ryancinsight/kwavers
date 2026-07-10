use leto::Array3;

use kwavers_core::error::{KwaversError, KwaversResult, PhysicsError};
use kwavers_physics::therapy::microbubble::Position3D;

/// Sample acoustic field at bubble position
///
/// Extracts local acoustic properties from 3D field arrays using
/// central-difference gradient estimation.
///
/// # Returns
///
/// - `pressure`: Local pressure (Pa)
/// - `pressure_gradient`: (∂P/∂x, ∂P/∂y, ∂P/∂z) [Pa/m]
/// # Errors
/// - Returns [`KwaversError::Physics`] if the precondition for a Physics-class constraint is violated.
///
pub fn sample_acoustic_field_at_position(
    position: &Position3D,
    pressure_field: &Array3<f64>,
    grid_spacing: (f64, f64, f64),
) -> KwaversResult<(f64, (f64, f64, f64))> {
    let [nx, ny, nz] = pressure_field.shape();
    let (dx, dy, dz) = grid_spacing;

    let ix = (position.x / dx).round() as usize;
    let iy = (position.y / dy).round() as usize;
    let iz = (position.z / dz).round() as usize;

    if ix >= nx || iy >= ny || iz >= nz {
        return Err(KwaversError::Physics(PhysicsError::InvalidParameter {
            parameter: "position".to_owned(),
            value: 0.0,
            reason: "bubble position outside grid domain".to_owned(),
        }));
    }

    let pressure = pressure_field[[ix, iy, iz]];

    let grad_x = if ix > 0 && ix < nx - 1 {
        (pressure_field[[ix + 1, iy, iz]] - pressure_field[[ix - 1, iy, iz]]) / (2.0 * dx)
    } else {
        0.0
    };

    let grad_y = if iy > 0 && iy < ny - 1 {
        (pressure_field[[ix, iy + 1, iz]] - pressure_field[[ix, iy - 1, iz]]) / (2.0 * dy)
    } else {
        0.0
    };

    let grad_z = if iz > 0 && iz < nz - 1 {
        (pressure_field[[ix, iy, iz + 1]] - pressure_field[[ix, iy, iz - 1]]) / (2.0 * dz)
    } else {
        0.0
    };

    Ok((pressure, (grad_x, grad_y, grad_z)))
}
