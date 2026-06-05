//! Focus finding and beam width calculations

use super::validation::{invalid_parameter, validate_pressure_field_domain};
use kwavers_core::error::{KwaversError, KwaversResult};
use kwavers_grid::Grid;
use ndarray::{Array1, ArrayView3};

/// Find focus location using maximum pressure
/// # Errors
/// - Returns [`Err`] if an internal constraint is violated.
///
pub fn find_focus(pressure_field: ArrayView3<f64>, grid: &Grid) -> KwaversResult<[f64; 3]> {
    validate_pressure_field_domain(pressure_field, grid)?;

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
/// # Errors
/// - Returns [`KwaversError::InvalidInput`] if the precondition for invalid or out-of-range input parameters is violated.
///
pub fn find_focal_plane(
    pressure_field: ArrayView3<f64>,
    grid: &Grid,
    axis: usize,
) -> KwaversResult<usize> {
    let n_slices = axis_slice_count(grid, axis)?;
    validate_pressure_field_domain(pressure_field, grid)?;

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
            _ => {
                // Invalid axis - return error
                return Err(KwaversError::InvalidInput(format!(
                    "Invalid axis {axis} for focus finding, expected 0, 1, or 2"
                )));
            }
        }

        if slice_energy > max_energy {
            max_energy = slice_energy;
            focal_slice = slice;
        }
    }

    Ok(focal_slice)
}

/// Calculate beam width at different axial distances
/// # Errors
/// - Returns [`Err`] if an internal constraint is violated.
///
pub fn calculate_beam_width(
    pressure_field: ArrayView3<f64>,
    grid: &Grid,
    axis: usize,
    threshold_db: f64,
) -> KwaversResult<Array1<f64>> {
    let n_slices = axis_slice_count(grid, axis)?;
    validate_pressure_field_domain(pressure_field, grid)?;
    if !threshold_db.is_finite() {
        return Err(invalid_parameter(
            "threshold_db",
            threshold_db,
            "must be finite",
        ));
    }

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

fn axis_slice_count(grid: &Grid, axis: usize) -> KwaversResult<usize> {
    match axis {
        0 => Ok(grid.nx),
        1 => Ok(grid.ny),
        2 => Ok(grid.nz),
        _ => Err(KwaversError::InvalidInput(
            "Invalid axis: must be 0, 1, or 2".to_owned(),
        )),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use kwavers_grid::Grid;
    use ndarray::Array3;

    fn small_grid() -> Grid {
        Grid::new(8, 8, 8, 1e-3, 1e-3, 1e-3).unwrap()
    }

    // ── find_focus ────────────────────────────────────────────────────────────

    /// Spike at [4,3,2] → find_focus returns (4·dx, 3·dy, 2·dz).
    #[test]
    fn find_focus_locates_pressure_spike() {
        let grid = small_grid();
        let mut field = Array3::<f64>::zeros((8, 8, 8));
        field[[4, 3, 2]] = 1000.0;

        let loc = find_focus(field.view(), &grid).unwrap();

        assert!((loc[0] - 4.0 * grid.dx).abs() < 1e-14, "focus x");
        assert!((loc[1] - 3.0 * grid.dy).abs() < 1e-14, "focus y");
        assert!((loc[2] - 2.0 * grid.dz).abs() < 1e-14, "focus z");
    }

    /// Zero field → find_focus returns origin (max stays at 0 → indices [0,0,0]).
    #[test]
    fn find_focus_returns_origin_for_zero_field() {
        let grid = small_grid();
        let field = Array3::<f64>::zeros((8, 8, 8));
        let loc = find_focus(field.view(), &grid).unwrap();
        assert_eq!(loc, [0.0, 0.0, 0.0]);
    }

    #[test]
    fn find_focus_rejects_shape_mismatch() {
        let grid = small_grid();
        let field = Array3::<f64>::zeros((7, 8, 8));

        let err = find_focus(field.view(), &grid).unwrap_err();
        let message = err.to_string();

        assert!(
            message.contains("Dimension mismatch"),
            "unexpected error: {message}"
        );
        assert!(message.contains("(8, 8, 8)"), "expected shape: {message}");
        assert!(message.contains("(7, 8, 8)"), "actual shape: {message}");
    }

    #[test]
    fn find_focus_rejects_nonfinite_pressure() {
        let grid = small_grid();
        let mut field = Array3::<f64>::zeros((8, 8, 8));
        field[[6, 5, 4]] = f64::NAN;

        let err = find_focus(field.view(), &grid).unwrap_err();
        let message = err.to_string();

        assert!(message.contains("nonfinite"), "unexpected error: {message}");
        assert!(
            message.contains("[6, 5, 4]"),
            "sample index must be reported: {message}"
        );
    }

    // ── find_focal_plane ──────────────────────────────────────────────────────

    /// All energy concentrated in z-slice 5 → find_focal_plane(axis=2) = 5.
    #[test]
    fn find_focal_plane_locates_energy_slice_along_z() {
        let grid = small_grid();
        let mut field = Array3::<f64>::zeros((8, 8, 8));
        for ix in 0..8 {
            for iy in 0..8 {
                field[[ix, iy, 5]] = 10.0;
            }
        }

        let plane = find_focal_plane(field.view(), &grid, 2).unwrap();
        assert_eq!(plane, 5, "focal z-plane must be 5");
    }

    /// Invalid axis (3) → Err.
    #[test]
    fn find_focal_plane_errors_for_invalid_axis() {
        let grid = small_grid();
        let field = Array3::<f64>::zeros((8, 8, 8));

        let err = find_focal_plane(field.view(), &grid, 3).unwrap_err();
        let message = err.to_string();

        assert!(
            message.contains("Invalid axis"),
            "unexpected error: {message}"
        );
    }

    // ── calculate_beam_width ──────────────────────────────────────────────────

    /// Invalid axis → Err.
    #[test]
    fn calculate_beam_width_errors_for_invalid_axis() {
        let grid = small_grid();
        let field = Array3::<f64>::zeros((8, 8, 8));

        let err = calculate_beam_width(field.view(), &grid, 5, -6.0).unwrap_err();
        let message = err.to_string();

        assert!(
            message.contains("Invalid axis"),
            "unexpected error: {message}"
        );
    }

    /// Zero field → all slice widths are 0.0 (no pressure above threshold).
    #[test]
    fn calculate_beam_width_zero_for_zero_field() {
        let grid = small_grid();
        let field = Array3::<f64>::zeros((8, 8, 8));
        let widths = calculate_beam_width(field.view(), &grid, 2, -6.0).unwrap();
        assert_eq!(widths.len(), grid.nz);
        assert!(
            widths.iter().all(|&w| w == 0.0),
            "zero field must yield zero beam widths"
        );
    }

    #[test]
    fn calculate_beam_width_rejects_nonfinite_threshold() {
        let grid = small_grid();
        let field = Array3::<f64>::zeros((8, 8, 8));

        let err = calculate_beam_width(field.view(), &grid, 2, f64::NAN).unwrap_err();
        let message = err.to_string();

        assert!(
            message.contains("threshold_db"),
            "unexpected error: {message}"
        );
        assert!(message.contains("finite"), "domain reason: {message}");
    }
}
