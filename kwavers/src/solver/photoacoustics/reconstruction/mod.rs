use crate::core::error::KwaversResult;
use crate::domain::imaging::photoacoustic::{PhotoacousticScenario, PhotoacousticSignalSet};
use ndarray::Array3;

mod benchmarks;
mod line_sensor_fft;
mod planar_sensor_fft;
mod time_reversal;
mod validation;
mod workspace;

pub use benchmarks::ReconstructionBenchmarkCase;
pub use line_sensor_fft::LineSensorFftReconstruction;
pub use planar_sensor_fft::PlanarSensorFftReconstruction;
pub use time_reversal::TimeReversalReconstruction;
pub use validation::ReconstructionValidationCase;
pub use workspace::ReconstructionWorkspace;

/// Canonical reconstruction-side signal sampling and image recovery model.
///
/// # Numerical Algorithm
/// Detector signals are sampled by trilinear interpolation from propagated
/// pressure snapshots. Reconstruction dispatch then selects a geometry-specific
/// inverse algorithm: planar FFT, line FFT, or time reversal.
#[derive(Debug, Default)]
pub struct PhotoacousticReconstructionModel;

impl PhotoacousticReconstructionModel {
    /// Sample signals.
    /// # Errors
    /// - Returns [`Err`] if an internal constraint is violated.
    ///
    pub fn sample_signals(
        &self,
        scenario: &PhotoacousticScenario,
        pressure_fields: &[Array3<f64>],
        time_points: &[f64],
    ) -> KwaversResult<PhotoacousticSignalSet> {
        let mut workspace = ReconstructionWorkspace::new(
            time_points.len(),
            scenario.sensor_positions_m.len(),
            (scenario.grid.nx, scenario.grid.ny, scenario.grid.nz),
        );
        for (d_idx, position) in scenario.sensor_positions_m.iter().enumerate() {
            let x_idx = position[0] / scenario.grid.dx;
            let y_idx = position[1] / scenario.grid.dy;
            let z_idx = position[2] / scenario.grid.dz;
            for (t_idx, field) in pressure_fields.iter().enumerate() {
                workspace.sensor_data[[t_idx, d_idx]] =
                    Self::interpolate_detector_signal(field, x_idx, y_idx, z_idx);
            }
        }

        let sampling_frequency_hz = if time_points.len() > 1 {
            1.0 / (time_points[1] - time_points[0])
        } else {
            0.0
        };

        Ok(PhotoacousticSignalSet {
            sensor_positions: scenario.sensor_positions_m.clone(),
            sensor_data: workspace.sensor_data,
            sampling_frequency_hz,
        })
    }
    /// Reconstruct.
    /// # Errors
    /// - Returns [`Err`] if an internal constraint is violated.
    ///
    pub fn reconstruct(
        &self,
        scenario: &PhotoacousticScenario,
        signals: &PhotoacousticSignalSet,
    ) -> KwaversResult<Array3<f64>> {
        let geometry = if is_planar(&signals.sensor_positions) {
            "planar"
        } else if is_line(&signals.sensor_positions) {
            "line"
        } else {
            "general"
        };
        let _validation_case = ReconstructionValidationCase {
            name: "canonical_reconstruction_dispatch",
            geometry,
        };

        if is_planar(&signals.sensor_positions) {
            return PlanarSensorFftReconstruction.reconstruct(scenario, signals);
        }
        if is_line(&signals.sensor_positions) {
            return LineSensorFftReconstruction.reconstruct(scenario, signals);
        }
        TimeReversalReconstruction.reconstruct(scenario, signals)
    }

    #[must_use]
    pub fn interpolate_detector_signal(
        field: &Array3<f64>,
        x_det: f64,
        y_det: f64,
        z_det: f64,
    ) -> f64 {
        let (nx, ny, nz) = field.dim();
        let x_clamp = x_det.clamp(0.0, (nx - 1) as f64);
        let y_clamp = y_det.clamp(0.0, (ny - 1) as f64);
        let z_clamp = z_det.clamp(0.0, (nz - 1) as f64);

        let x_floor = x_clamp.floor() as usize;
        let y_floor = y_clamp.floor() as usize;
        let z_floor = z_clamp.floor() as usize;
        let x_ceil = (x_floor + 1).min(nx - 1);
        let y_ceil = (y_floor + 1).min(ny - 1);
        let z_ceil = (z_floor + 1).min(nz - 1);

        let x_weight = x_clamp - x_floor as f64;
        let y_weight = y_clamp - y_floor as f64;
        let z_weight = z_clamp - z_floor as f64;

        let c000 = field[[x_floor, y_floor, z_floor]];
        let c001 = field[[x_floor, y_floor, z_ceil]];
        let c010 = field[[x_floor, y_ceil, z_floor]];
        let c011 = field[[x_floor, y_ceil, z_ceil]];
        let c100 = field[[x_ceil, y_floor, z_floor]];
        let c101 = field[[x_ceil, y_floor, z_ceil]];
        let c110 = field[[x_ceil, y_ceil, z_floor]];
        let c111 = field[[x_ceil, y_ceil, z_ceil]];

        (c111 * x_weight * y_weight).mul_add(
            z_weight,
            (c110 * x_weight * y_weight).mul_add(
                1.0 - z_weight,
                (c101 * x_weight * (1.0 - y_weight)).mul_add(
                    z_weight,
                    (c100 * x_weight * (1.0 - y_weight)).mul_add(
                        1.0 - z_weight,
                        (c011 * (1.0 - x_weight) * y_weight).mul_add(
                            z_weight,
                            (c010 * (1.0 - x_weight) * y_weight).mul_add(
                                1.0 - z_weight,
                                (c000 * (1.0 - x_weight) * (1.0 - y_weight)).mul_add(
                                    1.0 - z_weight,
                                    c001 * (1.0 - x_weight) * (1.0 - y_weight) * z_weight,
                                ),
                            ),
                        ),
                    ),
                ),
            ),
        )
    }
}

fn is_planar(sensor_positions: &[[f64; 3]]) -> bool {
    sensor_positions.first().is_some_and(|first| {
        sensor_positions
            .iter()
            .all(|p| (p[2] - first[2]).abs() < 1e-12)
    })
}

fn is_line(sensor_positions: &[[f64; 3]]) -> bool {
    sensor_positions.first().is_some_and(|first| {
        sensor_positions
            .iter()
            .all(|p| (p[1] - first[1]).abs() < 1e-12 && (p[2] - first[2]).abs() < 1e-12)
    })
}

#[cfg(test)]
mod tests {
    use super::PhotoacousticReconstructionModel;
    use ndarray::Array3;

    #[test]
    fn canonical_interpolator_matches_corner_and_midpoint_values() {
        let field = Array3::from_shape_fn((2, 2, 2), |(i, j, k)| (i + j + k) as f64);

        let corner =
            PhotoacousticReconstructionModel::interpolate_detector_signal(&field, 0.0, 0.0, 0.0);
        let midpoint =
            PhotoacousticReconstructionModel::interpolate_detector_signal(&field, 0.5, 0.5, 0.5);

        assert!((corner - 0.0).abs() < 1e-12);
        assert!((midpoint - 1.5).abs() < 1e-12);
    }
}
