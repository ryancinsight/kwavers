//! Acoustic-to-thermal coupling source terms.

use ndarray::Array3;

/// Compute volumetric heat source from acoustic absorption [W/m^3].
#[must_use]
pub fn acoustic_heat_source(
    pressure: &Array3<f64>,
    velocity_x: &Array3<f64>,
    velocity_y: &Array3<f64>,
    velocity_z: &Array3<f64>,
    density: &Array3<f64>,
    sound_speed: &Array3<f64>,
    absorption: &Array3<f64>,
) -> Array3<f64> {
    let (nx, ny, nz) = pressure.dim();
    let mut q = Array3::zeros((nx, ny, nz));
    for i in 0..nx {
        for j in 0..ny {
            for k in 0..nz {
                let rho = density[[i, j, k]];
                let c = sound_speed[[i, j, k]];
                if rho > 0.0 && c > 0.0 {
                    let p = pressure[[i, j, k]];
                    let vx = velocity_x[[i, j, k]];
                    let vy = velocity_y[[i, j, k]];
                    let vz = velocity_z[[i, j, k]];
                    let alpha = absorption[[i, j, k]];
                    let energy_density =
                        0.5 * rho * (vx * vx + vy * vy + vz * vz) + p * p / (2.0 * rho * c * c);
                    q[[i, j, k]] = 2.0 * alpha * c * energy_density;
                }
            }
        }
    }
    q
}
