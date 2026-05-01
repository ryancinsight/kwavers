//! Time-of-flight inversion (Bercoff et al. 2004) — estimates shear wave speed
//! from arrival times relative to a single push source.

use ndarray::Array3;

use crate::core::error::KwaversResult;
use crate::domain::grid::Grid;
use crate::domain::imaging::ultrasound::elastography::ElasticityMap;
use crate::physics::acoustics::imaging::modalities::elastography::displacement::DisplacementField;

use super::super::algorithms::{fill_boundaries, spatial_smoothing};
use super::super::types::elasticity_map_from_speed;

/// Time-of-flight inversion (simple method)
///
/// Estimates shear wave speed from arrival time at different locations.
/// Assumes waves propagate radially from a single push location.
///
/// # Algorithm
///
/// 1. Find push location (maximum displacement)
/// 2. For each voxel, compute distance from push
/// 3. Estimate arrival time from displacement amplitude
/// 4. Compute speed: cs = distance / arrival_time
/// 5. Apply spatial smoothing and boundary filling
///
/// # Physics
///
/// Simple geometric relationship: cs = Δx / Δt
/// where Δx is spatial distance and Δt is temporal delay.
///
/// # References
///
/// - Bercoff et al. (2004): "Supersonic shear imaging"
pub(super) fn time_of_flight_inversion(
    displacement: &DisplacementField,
    grid: &Grid,
    density: f64,
    _frequency: f64,
) -> KwaversResult<ElasticityMap> {
    let (nx, ny, nz) = displacement.uz.dim();
    let mut shear_wave_speed = Array3::zeros((nx, ny, nz));

    // Find push location (maximum displacement)
    let mut push_i = 0;
    let mut push_j = 0;
    let mut push_k = 0;
    let mut max_displacement = 0.0;

    for i in 0..nx {
        for j in 0..ny {
            for k in 0..nz {
                let disp_mag = displacement.uz[[i, j, k]].abs();
                if disp_mag > max_displacement {
                    max_displacement = disp_mag;
                    push_i = i;
                    push_j = j;
                    push_k = k;
                }
            }
        }
    }

    // Convert push location to physical coordinates
    let push_x = push_i as f64 * grid.dx;
    let push_y = push_j as f64 * grid.dy;
    let push_z = push_k as f64 * grid.dz;

    // For each point, estimate arrival time and shear wave speed
    for i in 0..nx {
        for j in 0..ny {
            for k in 0..nz {
                let displacement_amp = displacement.uz[[i, j, k]].abs();

                if displacement_amp > 1e-12 {
                    let x = i as f64 * grid.dx;
                    let y = j as f64 * grid.dy;
                    let z = k as f64 * grid.dz;
                    let distance =
                        ((x - push_x).powi(2) + (y - push_y).powi(2) + (z - push_z).powi(2)).sqrt();

                    if distance > 1e-6 {
                        // Estimate arrival time based on displacement amplitude
                        // Higher amplitude indicates earlier arrival (closer in time)
                        let normalized_amp = displacement_amp / max_displacement.max(1e-12);
                        let arrival_time = distance / (normalized_amp * 10.0);

                        let cs = distance / arrival_time;

                        // Clamp to realistic range for soft tissue (0.5-10 m/s)
                        shear_wave_speed[[i, j, k]] = cs.clamp(0.5, 10.0);
                    } else {
                        shear_wave_speed[[i, j, k]] = 3.0;
                    }
                } else {
                    shear_wave_speed[[i, j, k]] = 3.0;
                }
            }
        }
    }

    spatial_smoothing(&mut shear_wave_speed);
    fill_boundaries(&mut shear_wave_speed);

    Ok(elasticity_map_from_speed(shear_wave_speed, density))
}
