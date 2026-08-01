use leto::Array3;

use aequitas::systems::si::quantities::{Length, Pressure, PressureGradient};
use kwavers_core::error::{KwaversError, KwaversResult, PhysicsError};
use kwavers_physics::therapy::microbubble::{Position3D, PressureGradient3D};

/// Sample acoustic field at bubble position
///
/// Extracts local acoustic properties from 3D field arrays using
/// central-difference gradient estimation.
///
/// # Returns
///
/// - `pressure`: Local pressure (Pa)
/// - `pressure_gradient`: Cartesian pressure gradient [Pa/m]
/// # Errors
/// - Returns [`KwaversError::Physics`] if the precondition for a Physics-class constraint is violated.
///
pub fn sample_acoustic_field_at_position(
    position: &Position3D,
    pressure_field: &Array3<f64>,
    grid_spacing: (Length<f64>, Length<f64>, Length<f64>),
) -> KwaversResult<(Pressure<f64>, PressureGradient3D)> {
    let [nx, ny, nz] = pressure_field.shape();
    let (dx, dy, dz) = (
        grid_spacing.0.into_base(),
        grid_spacing.1.into_base(),
        grid_spacing.2.into_base(),
    );

    let ix = (position.x.into_base() / dx).round() as usize;
    let iy = (position.y.into_base() / dy).round() as usize;
    let iz = (position.z.into_base() / dz).round() as usize;

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

    Ok((
        Pressure::from_base(pressure),
        PressureGradient3D::new(
            PressureGradient::from_base(grad_x),
            PressureGradient::from_base(grad_y),
            PressureGradient::from_base(grad_z),
        ),
    ))
}
