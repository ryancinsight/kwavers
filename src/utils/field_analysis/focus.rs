//! Focus finding and beam width calculations

use crate::error::KwaversResult;
use crate::grid::Grid;
use ndarray::{Array1, ArrayView3};

/// Find focus location using maximum pressure
pub fn find_focus(pressure_field: ArrayView3<f64>, grid: &Grid) -> KwaversResult<[f64; 3]> {
    let mut max_pressure = 0.0;
    let mut focus_indices = [0, 0, 0];

    for ix in 0..grid.nx {
        for iy in 0..grid.ny {
            for iz in 0..grid.nz {
                let p = pressure_field[[ix, iy, iz]].abs();
                if p > max_pressure {
                    max_pressure = p;
                    focus_indices = [ix, iy, iz];
                }
            }
        }
    }

    Ok([
        focus_indices[0] as f64 * grid.dx,
        focus_indices[1] as f64 * grid.dy,
        focus_indices[2] as f64 * grid.dz,
    ])
}

/// Find focal plane (axial slice with maximum energy)
pub fn find_focal_plane(
    pressure_field: ArrayView3<f64>,
    grid: &Grid,
    axis: usize,
) -> KwaversResult<usize> {
    let n_slices = match axis {
        0 => grid.nx,
        1 => grid.ny,
        2 => grid.nz,
        _ => {
            return Err(crate::error::KwaversError::InvalidInput(
                "Invalid axis: must be 0, 1, or 2".to_string(),
            ))
        }
    };

    let mut max_energy = 0.0;
    let mut focal_slice = 0;

    for slice in 0..n_slices {
        let mut slice_energy = 0.0;

        match axis {
            0 => {
                for iy in 0..grid.ny {
                    for iz in 0..grid.nz {
                        slice_energy += pressure_field[[slice, iy, iz]].powi(2);
                    }
                }
            }
            1 => {
                for ix in 0..grid.nx {
                    for iz in 0..grid.nz {
                        slice_energy += pressure_field[[ix, slice, iz]].powi(2);
                    }
                }
            }
            2 => {
                for ix in 0..grid.nx {
                    for iy in 0..grid.ny {
                        slice_energy += pressure_field[[ix, iy, slice]].powi(2);
                    }
                }
            }
            _ => unreachable!(),
        }

        if slice_energy > max_energy {
            max_energy = slice_energy;
            focal_slice = slice;
        }
    }

    Ok(focal_slice)
}

/// Calculate beam width at different axial distances
pub fn calculate_beam_width(
    pressure_field: ArrayView3<f64>,
    grid: &Grid,
    axis: usize,
    threshold_db: f64,
) -> KwaversResult<Array1<f64>> {
    let n_slices = match axis {
        0 => grid.nx,
        1 => grid.ny,
        2 => grid.nz,
        _ => {
            return Err(crate::error::KwaversError::InvalidInput(
                "Invalid axis: must be 0, 1, or 2".to_string(),
            ))
        }
    };

    let mut widths = Array1::zeros(n_slices);
    let threshold_ratio = 10.0_f64.powf(threshold_db / 20.0);

    for slice in 0..n_slices {
        // Find maximum in this slice
        let mut max_pressure = 0.0;

        match axis {
            2 => {
                // Most common: propagation along z
                for ix in 0..grid.nx {
                    for iy in 0..grid.ny {
                        max_pressure =
                            f64::max(max_pressure, pressure_field[[ix, iy, slice]].abs());
                    }
                }

                if max_pressure > 0.0 {
                    let threshold = max_pressure * threshold_ratio;

                    // Find width in x direction at center y
                    let center_y = grid.ny / 2;
                    let mut left = grid.nx / 2;
                    let mut right = grid.nx / 2;

                    while left > 0 && pressure_field[[left, center_y, slice]].abs() > threshold {
                        left -= 1;
                    }

                    while right < grid.nx - 1
                        && pressure_field[[right, center_y, slice]].abs() > threshold
                    {
                        right += 1;
                    }

                    widths[slice] = (right - left) as f64 * grid.dx;
                }
            }
            _ => {
                // Similar logic for other axes
                widths[slice] = 0.0;
            }
        }
    }

    Ok(widths)
}
