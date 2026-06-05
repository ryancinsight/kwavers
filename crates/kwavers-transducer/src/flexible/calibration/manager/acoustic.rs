//! Acoustic peak detection and quality metric helpers for [`CalibrationManager`].

use super::CalibrationManager;
use kwavers_core::error::KwaversResult;
use ndarray::{Array2, Array3};

impl CalibrationManager {
    /// Detect local pressure-field peaks above 3 × RMS, separated by at least
    /// `wavelength / 2` grid cells.
    pub(super) fn extract_peaks(
        &self,
        pressure_field: &Array3<f64>,
        wavelength: f64,
    ) -> KwaversResult<Vec<[f64; 3]>> {
        let (nx, ny, nz) = pressure_field.dim();
        let mut peaks = Vec::new();

        let rms: f64 =
            pressure_field.iter().map(|&x| x * x).sum::<f64>().sqrt() / (nx * ny * nz) as f64;
        let threshold = 3.0 * rms;
        let min_separation = (wavelength / 2.0) as usize;

        for i in min_separation..(nx - min_separation) {
            for j in min_separation..(ny - min_separation) {
                for k in min_separation..(nz - min_separation) {
                    let val = pressure_field[[i, j, k]].abs();

                    if val > threshold {
                        let mut is_max = true;
                        'outer: for di in 0..=2 {
                            for dj in 0..=2 {
                                for dk in 0..=2 {
                                    if di == 1 && dj == 1 && dk == 1 {
                                        continue;
                                    }
                                    let neighbor =
                                        pressure_field[[i + di - 1, j + dj - 1, k + dk - 1]].abs();
                                    if neighbor > val {
                                        is_max = false;
                                        break 'outer;
                                    }
                                }
                            }
                        }

                        if is_max {
                            peaks.push([i as f64, j as f64, k as f64]);
                        }
                    }
                }
            }
        }

        Ok(peaks)
    }

    /// Match detected peaks to known reflectors by nearest-neighbour Euclidean distance.
    pub(super) fn match_reflectors(
        &self,
        peaks: &[[f64; 3]],
        reflectors: &[[f64; 3]],
    ) -> KwaversResult<Vec<(usize, usize)>> {
        let mut correspondences = Vec::new();

        for (i, peak) in peaks.iter().enumerate() {
            let mut min_dist = f64::INFINITY;
            let mut best_match = 0;

            for (j, reflector) in reflectors.iter().enumerate() {
                let dist = (peak[2] - reflector[2])
                    .mul_add(
                        peak[2] - reflector[2],
                        (peak[1] - reflector[1])
                            .mul_add(peak[1] - reflector[1], (peak[0] - reflector[0]).powi(2)),
                    )
                    .sqrt();

                if dist < min_dist {
                    min_dist = dist;
                    best_match = j;
                }
            }

            correspondences.push((i, best_match));
        }

        Ok(correspondences)
    }

    /// Construct element position array from peak–reflector correspondences.
    pub(super) fn estimate_positions(
        &self,
        correspondences: &[(usize, usize)],
        reflectors: &[[f64; 3]],
    ) -> KwaversResult<Array2<f64>> {
        let num_elements = correspondences.len();
        let mut positions = Array2::zeros((num_elements, 3));

        for (i, &(_, reflector_idx)) in correspondences.iter().enumerate() {
            if reflector_idx < reflectors.len() {
                positions[[i, 0]] = reflectors[reflector_idx][0];
                positions[[i, 1]] = reflectors[reflector_idx][1];
                positions[[i, 2]] = reflectors[reflector_idx][2];
            }
        }

        Ok(positions)
    }

    /// Update `quality_metrics` from correspondence ratio.
    pub(super) fn update_quality_metrics(
        &mut self,
        positions: &Array2<f64>,
        correspondences: &[(usize, usize)],
    ) {
        let num_correspondences = correspondences.len();
        let num_elements = positions.nrows();
        let correspondence_ratio = num_correspondences as f64 / num_elements.max(1) as f64;

        self.data.quality_metrics.position_uncertainty = 1e-3 / correspondence_ratio;
        self.data.quality_metrics.orientation_uncertainty = 1e-2 / correspondence_ratio;
        self.data.quality_metrics.confidence = correspondence_ratio.min(1.0);
    }
}
