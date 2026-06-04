//! Acoustic radiation force push-location detection.
//!
//! References:
//! - Bercoff et al. (2004). *Supersonic shear imaging*.
//! - Palmeri et al. (2008). *Quantifying hepatic shear modulus in vivo*.

use kwavers_grid::Grid;
use kwavers_physics::acoustics::imaging::modalities::elastography::displacement::DisplacementField;

/// Find push locations (27-neighbourhood local maxima) in a displacement field.
///
/// # Algorithm
///
/// 1. Threshold = 30 % of the global maximum `|uz|`.
/// 2. For each interior voxel above threshold, accept it if no neighbour in the
///    27-voxel cube has a larger `|uz|`.
/// 3. Return the accepted voxel centres in physical coordinates.
///
/// Returns a `Vec<[f64; 3]>` of `[x, y, z]` coordinates in metres.
#[must_use]
pub fn find_push_locations(displacement: &DisplacementField, grid: &Grid) -> Vec<[f64; 3]> {
    let (nx, ny, nz) = displacement.uz.dim();
    let mut locations = Vec::new();
    let threshold = displacement.uz.iter().copied().fold(0.0, f64::max) * 0.3;

    for i in 1..nx - 1 {
        for j in 1..ny - 1 {
            for k in 1..nz - 1 {
                let val = displacement.uz[[i, j, k]].abs();
                if val <= threshold {
                    continue;
                }

                let mut is_local_max = true;
                'neighbor_loop: for di in -1..=1i32 {
                    for dj in -1..=1i32 {
                        for dk in -1..=1i32 {
                            if di == 0 && dj == 0 && dk == 0 {
                                continue;
                            }
                            let ii = (i as i32 + di) as usize;
                            let jj = (j as i32 + dj) as usize;
                            let kk = (k as i32 + dk) as usize;

                            if ii < nx
                                && jj < ny
                                && kk < nz
                                && displacement.uz[[ii, jj, kk]].abs() > val
                            {
                                is_local_max = false;
                                break 'neighbor_loop;
                            }
                        }
                    }
                }

                if is_local_max {
                    locations.push([i as f64 * grid.dx, j as f64 * grid.dy, k as f64 * grid.dz]);
                }
            }
        }
    }

    locations
}
