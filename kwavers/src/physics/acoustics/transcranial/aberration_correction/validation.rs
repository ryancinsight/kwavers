//! Correction validation: field simulation and performance metrics

use super::phase_correction::{PhaseCorrection, TranscranialAberrationCorrection};
use crate::core::error::KwaversResult;
use ndarray::Array3;
use num_complex::Complex;

/// Validation results for aberration correction
#[derive(Debug)]
pub struct CorrectionValidation {
    /// Focal intensity (W/cm²)
    pub focal_intensity: f64,
    /// Sidelobe level (dB below main lobe)
    pub sidelobe_level_db: f64,
    /// Focal spot volume (cm³)
    pub focal_spot_size: f64,
}

impl TranscranialAberrationCorrection {
    /// Validate correction performance
    pub fn validate_correction(
        &self,
        correction: &PhaseCorrection,
        skull_model: &ndarray::Array3<f64>,
        transducer_positions: &[[f64; 3]],
        target_point: &[f64; 3],
    ) -> KwaversResult<CorrectionValidation> {
        let corrected_field = self.simulate_corrected_field(
            correction,
            skull_model,
            transducer_positions,
            target_point,
        )?;

        let focal_intensity = self.calculate_focal_intensity(&corrected_field, target_point);
        let sidelobe_level = self.calculate_sidelobe_level(&corrected_field, target_point);

        Ok(CorrectionValidation {
            focal_intensity,
            sidelobe_level_db: 10.0 * sidelobe_level.log10(),
            focal_spot_size: self.calculate_focal_spot_size(&corrected_field, target_point),
        })
    }

    /// Simulate acoustic field with phase correction applied
    fn simulate_corrected_field(
        &self,
        correction: &PhaseCorrection,
        _skull_model: &ndarray::Array3<f64>,
        transducer_positions: &[[f64; 3]],
        _target_point: &[f64; 3],
    ) -> KwaversResult<Array3<f64>> {
        let (nx, ny, nz) = self.grid.dimensions();
        let mut field = Array3::zeros((nx, ny, nz));

        for k in 0..nz {
            for j in 0..ny {
                for i in 0..nx {
                    let x = i as f64 * self.grid.dx;
                    let y = j as f64 * self.grid.dy;
                    let z = k as f64 * self.grid.dz;

                    let mut total_field = Complex::new(0.0, 0.0);

                    for (elem_idx, &elem_pos) in transducer_positions.iter().enumerate() {
                        let dx = x - elem_pos[0];
                        let dy = y - elem_pos[1];
                        let dz = z - elem_pos[2];
                        let distance = (dx * dx + dy * dy + dz * dz).sqrt();

                        if distance > 0.0 {
                            let phase = correction.phases.get(elem_idx).unwrap_or(&0.0);
                            let amplitude = correction.amplitudes.get(elem_idx).unwrap_or(&1.0);

                            let k_wave =
                                2.0 * std::f64::consts::PI * self.frequency / self.reference_speed;
                            let uncorrected_phase = k_wave * distance;
                            let corrected_phase = uncorrected_phase + phase;

                            let contribution = Complex::from_polar(*amplitude, corrected_phase);
                            total_field += contribution / distance;
                        }
                    }

                    field[[i, j, k]] = total_field.norm_sqr();
                }
            }
        }

        Ok(field)
    }

    /// Calculate focal intensity at target point
    fn calculate_focal_intensity(&self, field: &Array3<f64>, target_point: &[f64; 3]) -> f64 {
        let ix = ((target_point[0] / self.grid.dx) as usize).min(self.grid.nx - 1);
        let iy = ((target_point[1] / self.grid.dy) as usize).min(self.grid.ny - 1);
        let iz = ((target_point[2] / self.grid.dz) as usize).min(self.grid.nz - 1);

        field[[ix, iy, iz]]
    }

    /// Calculate sidelobe level relative to main lobe
    fn calculate_sidelobe_level(&self, field: &Array3<f64>, target_point: &[f64; 3]) -> f64 {
        let focal_intensity = self.calculate_focal_intensity(field, target_point);
        if focal_intensity == 0.0 {
            return 0.0;
        }

        let mut max_sidelobe: f64 = 0.0;
        let focal_ix = ((target_point[0] / self.grid.dx) as usize).min(self.grid.nx - 1);
        let focal_iy = ((target_point[1] / self.grid.dy) as usize).min(self.grid.ny - 1);
        let focal_iz = ((target_point[2] / self.grid.dz) as usize).min(self.grid.nz - 1);

        for k in 0..self.grid.nz {
            for j in 0..self.grid.ny {
                for i in 0..self.grid.nx {
                    let distance_from_focus = (((i as i32 - focal_ix as i32).pow(2)
                        + (j as i32 - focal_iy as i32).pow(2)
                        + (k as i32 - focal_iz as i32).pow(2))
                        as f64)
                        .sqrt();

                    if distance_from_focus > 3.0 {
                        max_sidelobe = max_sidelobe.max(field[[i, j, k]]);
                    }
                }
            }
        }

        max_sidelobe / focal_intensity
    }

    /// Calculate focal spot size (-6dB volume)
    fn calculate_focal_spot_size(&self, field: &Array3<f64>, target_point: &[f64; 3]) -> f64 {
        let focal_intensity = self.calculate_focal_intensity(field, target_point);
        let threshold = focal_intensity / 4.0;

        let mut volume = 0.0;
        for &intensity in field.iter() {
            if intensity >= threshold {
                volume += self.grid.dx * self.grid.dy * self.grid.dz;
            }
        }

        volume
    }
}
