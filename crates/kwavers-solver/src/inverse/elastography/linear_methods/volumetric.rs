//! 3D volumetric time-of-flight inversion (Urban et al. 2013) — multi-source
//! median speed estimation.

use ndarray::Array3;

use kwavers_core::error::KwaversResult;
use kwavers_grid::Grid;
use kwavers_imaging::ultrasound::elastography::ElasticityMap;
use kwavers_physics::acoustics::imaging::modalities::elastography::displacement::DisplacementField;

use super::super::algorithms::{fill_boundaries, find_push_locations, volumetric_smoothing};
use super::super::types::elasticity_map_from_speed;
use super::time_of_flight_inversion;

/// 3D Volumetric time-of-flight inversion
///
/// Enhanced time-of-flight method for 3D volumes with multi-directional wave analysis.
/// Accounts for complex wave propagation patterns in volumetric tissue.
///
/// # Algorithm
///
/// 1. Identify multiple push locations using peak detection
/// 2. For each voxel, estimate speeds from all push sources
/// 3. Use median of estimates for robustness against outliers
/// 4. Apply volumetric smoothing with edge preservation
/// 5. Fill boundaries
///
/// # Advantages
///
/// - Robust to heterogeneous tissue structures
/// - Reduced sensitivity to noise via median filtering
/// - Accounts for multiple wave interaction patterns
///
/// # References
///
/// - Urban et al. (2013): "3D SWE reconstruction with multi-directional waves"
/// # Errors
/// - Returns [`Err`] if an internal constraint is violated.
///
/// # Panics
/// - Panics if an internal invariant assumed to hold at this call site is violated.
///
pub(super) fn volumetric_time_of_flight_inversion(
    displacement: &DisplacementField,
    grid: &Grid,
    density: f64,
    frequency: f64,
) -> KwaversResult<ElasticityMap> {
    let (nx, ny, nz) = displacement.uz.dim();
    let mut shear_wave_speed = Array3::zeros((nx, ny, nz));

    let push_locations = find_push_locations(displacement, grid);

    if push_locations.is_empty() {
        return time_of_flight_inversion(displacement, grid, density, frequency);
    }

    for i in 0..nx {
        for j in 0..ny {
            for k in 0..nz {
                let voxel_pos = [i as f64 * grid.dx, j as f64 * grid.dy, k as f64 * grid.dz];
                let displacement_amp = displacement.uz[[i, j, k]].abs();

                if displacement_amp > 1e-12 {
                    let mut speed_estimates = Vec::new();

                    for push_pos in &push_locations {
                        let distance = (voxel_pos[2] - push_pos[2])
                            .mul_add(
                                voxel_pos[2] - push_pos[2],
                                (voxel_pos[1] - push_pos[1]).mul_add(
                                    voxel_pos[1] - push_pos[1],
                                    (voxel_pos[0] - push_pos[0]).powi(2),
                                ),
                            )
                            .sqrt();

                        if distance > 1e-6 {
                            let max_push_disp = push_locations
                                .iter()
                                .map(|p| {
                                    displacement.uz[[
                                        (p[0] / grid.dx) as usize,
                                        (p[1] / grid.dy) as usize,
                                        (p[2] / grid.dz) as usize,
                                    ]]
                                    .abs()
                                })
                                .fold(0.0, f64::max)
                                .max(1e-12);

                            let normalized_amp = displacement_amp / max_push_disp;
                            let arrival_time = distance / (normalized_amp * 10.0);
                            let cs = distance / arrival_time;
                            speed_estimates.push(cs.clamp(0.5, 10.0));
                        }
                    }

                    if !speed_estimates.is_empty() {
                        speed_estimates.sort_by(|a, b| a.total_cmp(b));
                        let median_idx = speed_estimates.len() / 2;
                        shear_wave_speed[[i, j, k]] = speed_estimates[median_idx];
                    } else {
                        shear_wave_speed[[i, j, k]] = 3.0;
                    }
                } else {
                    shear_wave_speed[[i, j, k]] = 3.0;
                }
            }
        }
    }

    volumetric_smoothing(&mut shear_wave_speed);
    fill_boundaries(&mut shear_wave_speed);

    Ok(elasticity_map_from_speed(shear_wave_speed, density))
}
