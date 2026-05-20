//! Field metrics calculation

use crate::core::error::KwaversResult;
use crate::domain::grid::Grid;
use ndarray::ArrayView3;

/// Field analysis metrics
#[derive(Debug, Clone)]
pub struct FieldMetrics {
    /// Peak pressure location [x, y, z] (m)
    pub peak_location: [f64; 3],
    /// Peak pressure value (Pa)
    pub peak_pressure: f64,
    /// Focal distance (m)
    pub focal_distance: f64,
    /// Beam width at focus (m)
    pub beam_width: f64,
    /// Beam divergence angle (radians)
    pub divergence_angle: f64,
    /// Total acoustic energy stored in the pressure field (J)
    ///
    /// Computed as `∑ p²/(2ρc²) · ΔV`, the acoustic potential-energy
    /// density integrated over the simulation volume.  Units are joules.
    /// Note: this is NOT radiated power in watts; for radiated power use a
    /// surface integral of intensity over a closed boundary.
    pub total_power: f64,
    /// Spatial peak intensity (W/m²)
    pub spatial_peak_intensity: f64,
}

/// Calculate comprehensive field metrics
/// # Errors
/// - Propagates any [`KwaversError`] returned by called functions.
///
pub fn calculate_field_metrics(
    pressure_field: ArrayView3<f64>,
    grid: &Grid,
    density: f64,
    sound_speed: f64,
) -> KwaversResult<FieldMetrics> {
    // Find peak pressure
    let (peak_location, peak_pressure) = find_peak_pressure(pressure_field, grid)?;

    // Calculate focal distance
    let focal_distance = peak_location[2]
        .mul_add(
            peak_location[2],
            peak_location[1].mul_add(peak_location[1], peak_location[0].powi(2)),
        )
        .sqrt();

    // Calculate beam width at focus
    let beam_width = calculate_beam_width_at_location(
        pressure_field,
        grid,
        peak_location,
        peak_pressure * 0.707, // -3dB point
    );

    // Calculate divergence angle
    let divergence_angle = if focal_distance > 0.0 {
        (beam_width / (2.0 * focal_distance)).atan()
    } else {
        0.0
    };

    // Calculate total power and intensity
    let impedance = density * sound_speed;
    let mut total_power = 0.0;
    let mut max_intensity = 0.0;

    for ix in 0..grid.nx {
        for iy in 0..grid.ny {
            for iz in 0..grid.nz {
                let p = pressure_field[[ix, iy, iz]];
                // Acoustic intensity: I = p²/(2ρc)  [W/m²]
                let intensity = p.powi(2) / (2.0 * impedance);
                // Acoustic potential energy density: e = p²/(2ρc²)  [J/m³]
                // Integrating e·dV over the volume gives total stored
                // acoustic energy in joules (not power in watts).
                let energy_density = intensity / sound_speed;
                total_power += energy_density * grid.dx * grid.dy * grid.dz;
                max_intensity = f64::max(max_intensity, intensity);
            }
        }
    }

    Ok(FieldMetrics {
        peak_location,
        peak_pressure,
        focal_distance,
        beam_width,
        divergence_angle,
        total_power,
        spatial_peak_intensity: max_intensity,
    })
}

/// Find peak pressure location in field
/// # Errors
/// - Returns [`Err`] if an internal constraint is violated.
///
pub fn find_peak_pressure(
    pressure_field: ArrayView3<f64>,
    grid: &Grid,
) -> KwaversResult<([f64; 3], f64)> {
    let mut max_pressure = 0.0;
    let mut max_location = [0, 0, 0];

    for ix in 0..grid.nx {
        for iy in 0..grid.ny {
            for iz in 0..grid.nz {
                let p = pressure_field[[ix, iy, iz]].abs();
                if p > max_pressure {
                    max_pressure = p;
                    max_location = [ix, iy, iz];
                }
            }
        }
    }

    let location = [
        max_location[0] as f64 * grid.dx,
        max_location[1] as f64 * grid.dy,
        max_location[2] as f64 * grid.dz,
    ];

    Ok((location, max_pressure))
}

/// Calculate beam width at a specific location
fn calculate_beam_width_at_location(
    pressure_field: ArrayView3<f64>,
    grid: &Grid,
    location: [f64; 3],
    threshold: f64,
) -> f64 {
    // Find indices closest to location
    let ix = (location[0] / grid.dx).round() as usize;
    let iy = (location[1] / grid.dy).round() as usize;
    let iz = (location[2] / grid.dz).round() as usize;

    // Measure width in x and y directions

    // X direction
    let mut left = ix;
    let mut right = ix;

    while left > 0 && pressure_field[[left, iy, iz]].abs() > threshold {
        left -= 1;
    }

    while right < grid.nx - 1 && pressure_field[[right, iy, iz]].abs() > threshold {
        right += 1;
    }

    let width_x = (right - left) as f64 * grid.dx;

    // Y direction
    let mut bottom = iy;
    let mut top = iy;

    while bottom > 0 && pressure_field[[ix, bottom, iz]].abs() > threshold {
        bottom -= 1;
    }

    while top < grid.ny - 1 && pressure_field[[ix, top, iz]].abs() > threshold {
        top += 1;
    }

    let width_y = (top - bottom) as f64 * grid.dy;

    // Return average width
    (width_x + width_y) / 2.0
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::domain::grid::Grid;
    use ndarray::Array3;

    fn small_grid() -> Grid {
        Grid::new(8, 8, 8, 1e-3, 1e-3, 1e-3).unwrap()
    }

    /// `find_peak_pressure` returns position of the maximum |pressure| cell.
    ///
    /// Spike at [4,4,4]: location = 4·dx × (4·dy) × (4·dz) = (4e-3, 4e-3, 4e-3).
    #[test]
    fn find_peak_pressure_locates_spike_cell() {
        let grid = small_grid();
        let mut field = Array3::<f64>::zeros((8, 8, 8));
        field[[4, 4, 4]] = 500.0;

        let (loc, peak) = find_peak_pressure(field.view(), &grid).unwrap();

        assert!((peak - 500.0).abs() < 1e-12, "peak pressure must be 500 Pa");
        assert!((loc[0] - 4.0 * grid.dx).abs() < 1e-14, "x location");
        assert!((loc[1] - 4.0 * grid.dy).abs() < 1e-14, "y location");
        assert!((loc[2] - 4.0 * grid.dz).abs() < 1e-14, "z location");
    }

    /// Uniform zero field: peak = 0 and location = origin (all-zero search result).
    #[test]
    fn find_peak_pressure_zero_for_zero_field() {
        let grid = small_grid();
        let field = Array3::<f64>::zeros((8, 8, 8));
        let (_loc, peak) = find_peak_pressure(field.view(), &grid).unwrap();
        assert_eq!(peak, 0.0, "peak must be 0 for zero field");
    }

    /// `calculate_field_metrics` returns finite, positive values for a physical field.
    #[test]
    fn calculate_field_metrics_positive_for_nonzero_field() {
        let grid = small_grid();
        let mut field = Array3::<f64>::zeros((8, 8, 8));
        field[[4, 4, 4]] = 1e5;

        let metrics = calculate_field_metrics(field.view(), &grid, 1000.0, 1500.0).unwrap();

        assert!(
            metrics.peak_pressure > 0.0,
            "peak_pressure must be positive"
        );
        assert!(
            metrics.total_power >= 0.0,
            "total_power must be non-negative"
        );
        assert!(
            metrics.spatial_peak_intensity > 0.0,
            "spatial_peak_intensity positive"
        );
    }
}
