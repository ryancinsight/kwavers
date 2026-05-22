//! Field metrics calculation

use super::pressure::{acoustic_impedance, harmonic_peak_intensity};
use super::validation::{validate_pressure_field_domain, validation_error};
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
    validate_pressure_field_domain(pressure_field, grid)?;
    let Some(impedance) = acoustic_impedance(density, sound_speed) else {
        return Err(validation_error(format!(
            "Acoustic impedance requires positive finite density and sound_speed, got density={density}, sound_speed={sound_speed}"
        )));
    };

    // Find peak pressure
    let (peak_location, peak_pressure) = find_peak_pressure_unchecked(pressure_field, grid);

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

    // Calculate total stored acoustic energy and intensity
    let mut total_power = 0.0;
    let mut max_intensity = 0.0;

    for ix in 0..grid.nx {
        for iy in 0..grid.ny {
            for iz in 0..grid.nz {
                let p = pressure_field[[ix, iy, iz]];
                // Acoustic intensity: I = p²/(2ρc)  [W/m²]
                let intensity = harmonic_peak_intensity(p, impedance);
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
    validate_pressure_field_domain(pressure_field, grid)?;
    Ok(find_peak_pressure_unchecked(pressure_field, grid))
}

fn find_peak_pressure_unchecked(pressure_field: ArrayView3<f64>, grid: &Grid) -> ([f64; 3], f64) {
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

    (location, max_pressure)
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
    use crate::core::constants::fundamental::SOUND_SPEED_WATER_SIM;

    fn small_grid() -> Grid {
        Grid::new(8, 8, 8, 1e-3, 1e-3, 1e-3).unwrap()
    }

    fn assert_close(actual: f64, expected: f64, tolerance: f64, context: &str) {
        assert!(
            (actual - expected).abs() <= tolerance,
            "{context}: expected {expected:.12e}, got {actual:.12e}"
        );
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

    /// Peak search uses pressure magnitude, so rarefactional extrema are not
    /// hidden by smaller positive compressional samples.
    #[test]
    fn find_peak_pressure_uses_signed_pressure_magnitude() {
        let grid = small_grid();
        let mut field = Array3::<f64>::zeros((8, 8, 8));
        field[[2, 3, 4]] = 500.0;
        field[[5, 6, 7]] = -700.0;

        let (loc, peak) = find_peak_pressure(field.view(), &grid).unwrap();

        assert_close(peak, 700.0, 1e-12, "rarefactional peak magnitude");
        assert_close(loc[0], 5.0 * grid.dx, 1e-14, "x location");
        assert_close(loc[1], 6.0 * grid.dy, 1e-14, "y location");
        assert_close(loc[2], 7.0 * grid.dz, 1e-14, "z location");
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

        let metrics = calculate_field_metrics(field.view(), &grid, 1000.0, SOUND_SPEED_WATER_SIM).unwrap();
        let expected_intensity = 1e5_f64.powi(2) / (2.0 * 1000.0 * SOUND_SPEED_WATER_SIM);
        let expected_energy = expected_intensity / SOUND_SPEED_WATER_SIM * grid.dx * grid.dy * grid.dz;

        assert_close(metrics.peak_pressure, 1e5, 1e-9, "peak pressure");
        assert_close(metrics.peak_location[0], 4.0 * grid.dx, 1e-14, "peak x");
        assert_close(
            metrics.spatial_peak_intensity,
            expected_intensity,
            1e-9,
            "spatial peak intensity",
        );
        assert_close(
            metrics.total_power,
            expected_energy,
            1e-18,
            "stored acoustic energy",
        );
    }

    #[test]
    fn calculate_field_metrics_rejects_invalid_impedance_domain() {
        let grid = small_grid();
        let mut field = Array3::<f64>::zeros((8, 8, 8));
        field[[4, 4, 4]] = 1e5;

        let err = calculate_field_metrics(field.view(), &grid, 0.0, SOUND_SPEED_WATER_SIM).unwrap_err();
        let message = err.to_string();

        assert!(
            message.contains("Acoustic impedance"),
            "unexpected error: {message}"
        );
        assert!(
            message.contains("density=0"),
            "density must be reported: {message}"
        );
    }

    #[test]
    fn calculate_field_metrics_rejects_nonfinite_pressure_samples() {
        let grid = small_grid();
        let mut field = Array3::<f64>::zeros((8, 8, 8));
        field[[1, 2, 3]] = f64::NAN;

        let err = calculate_field_metrics(field.view(), &grid, 1000.0, SOUND_SPEED_WATER_SIM).unwrap_err();
        let message = err.to_string();

        assert!(message.contains("nonfinite"), "unexpected error: {message}");
        assert!(
            message.contains("[1, 2, 3]"),
            "sample index must be reported: {message}"
        );
    }

    #[test]
    fn find_peak_pressure_rejects_shape_mismatch() {
        let grid = small_grid();
        let field = Array3::<f64>::zeros((7, 8, 8));

        let err = find_peak_pressure(field.view(), &grid).unwrap_err();
        let message = err.to_string();

        assert!(
            message.contains("Dimension mismatch"),
            "unexpected error: {message}"
        );
        assert!(message.contains("(8, 8, 8)"), "expected shape: {message}");
        assert!(message.contains("(7, 8, 8)"), "actual shape: {message}");
    }
}
