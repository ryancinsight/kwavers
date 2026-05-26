//! 3D directional phase-gradient inversion (Wang et al. 2014) — uses the
//! dominant wavenumber component for anisotropic media.

use ndarray::Array3;

use crate::core::error::KwaversResult;
use crate::domain::grid::Grid;
use crate::domain::imaging::ultrasound::elastography::ElasticityMap;
use crate::physics::acoustics::imaging::modalities::elastography::displacement::DisplacementField;

use super::super::algorithms::{directional_smoothing, fill_boundaries};
use super::super::types::elasticity_map_from_speed;
use crate::core::constants::numerical::{TWO_PI};

/// 3D Directional phase gradient inversion
///
/// Advanced phase gradient method that analyzes wave propagation in multiple directions
/// for improved accuracy in heterogeneous 3D media.
///
/// # Algorithm
///
/// 1. For each voxel, compute phase gradients in x, y, z directions
/// 2. Estimate directional wavenumbers from gradients
/// 3. Use dominant wavenumber component for speed estimation
/// 4. Apply directional smoothing along wave propagation directions
/// 5. Fill boundaries
///
/// # Physics
///
/// For 3D wave propagation:
/// - `k = ∇φ` (wavenumber vector)
/// - `|k| = ω/cs` (dispersion relation)
/// - Directional analysis accounts for anisotropy
///
/// # References
///
/// - Wang et al. (2014): "Multi-directional phase gradient methods for 3D SWE"
/// # Errors
/// - Returns [`Err`] if an internal constraint is violated.
///
pub(super) fn directional_phase_gradient_inversion(
    displacement: &DisplacementField,
    grid: &Grid,
    density: f64,
    frequency: f64,
) -> KwaversResult<ElasticityMap> {
    let (nx, ny, nz) = displacement.uz.dim();
    let mut shear_wave_speed = Array3::zeros((nx, ny, nz));

    for i in 1..nx - 1 {
        for j in 1..ny - 1 {
            for k in 1..nz - 1 {
                let displacement_val = displacement.uz[[i, j, k]];

                if displacement_val.abs() > 1e-12 {
                    let grad_x = (displacement.uz[[i + 1, j, k]] - displacement.uz[[i - 1, j, k]])
                        / (2.0 * grid.dx);
                    let grad_y = (displacement.uz[[i, j + 1, k]] - displacement.uz[[i, j - 1, k]])
                        / (2.0 * grid.dy);
                    let grad_z = (displacement.uz[[i, j, k + 1]] - displacement.uz[[i, j, k - 1]])
                        / (2.0 * grid.dz);

                    let kx = grad_x.abs() / displacement_val.abs().max(1e-12);
                    let ky = grad_y.abs() / displacement_val.abs().max(1e-12);
                    let kz = grad_z.abs() / displacement_val.abs().max(1e-12);

                    let dominant_k = kx.max(ky).max(kz).max(0.1);

                    let angular_freq = TWO_PI * frequency;
                    let cs = angular_freq / dominant_k;

                    shear_wave_speed[[i, j, k]] = cs.clamp(0.5, 10.0);
                } else {
                    shear_wave_speed[[i, j, k]] = 3.0;
                }
            }
        }
    }

    directional_smoothing(&mut shear_wave_speed);
    fill_boundaries(&mut shear_wave_speed);

    Ok(elasticity_map_from_speed(shear_wave_speed, density))
}
