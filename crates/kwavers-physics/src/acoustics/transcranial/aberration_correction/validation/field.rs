use super::super::phase_correction::{PhaseCorrection, TranscranialAberrationCorrection};
use eunomia::Complex;
use kwavers_core::constants::numerical::TWO_PI;
use kwavers_core::error::KwaversResult;
use leto::Array3;

impl TranscranialAberrationCorrection {
    /// Simulate acoustic field with phase correction applied.
    /// # Errors
    /// - Returns [`Err`] if an internal constraint is violated.
    ///
    pub(super) fn simulate_corrected_field(
        &self,
        correction: &PhaseCorrection,
        _skull_model: &leto::Array3<f64>,
        transducer_positions: &[[f64; 3]],
        _target_point: &[f64; 3],
    ) -> KwaversResult<Array3<f64>> {
        let (nx, ny, nz) = self.grid.dimensions();
        let mut field = Array3::zeros([nx, ny, nz]);
        let k_wave = TWO_PI * self.frequency / self.reference_speed;

        for k in 0..nz {
            let z = k as f64 * self.grid.dz;
            for j in 0..ny {
                let y = j as f64 * self.grid.dy;
                for i in 0..nx {
                    let x = i as f64 * self.grid.dx;
                    let mut total_field = Complex::new(0.0, 0.0);

                    for (elem_idx, &elem_pos) in transducer_positions.iter().enumerate() {
                        let dx = x - elem_pos[0];
                        let dy = y - elem_pos[1];
                        let dz = z - elem_pos[2];
                        let distance = dz.mul_add(dz, dx.mul_add(dx, dy * dy)).sqrt();

                        if distance > 0.0 {
                            let phase = correction.phases.get(elem_idx).unwrap_or(&0.0);
                            let amplitude = correction.amplitudes.get(elem_idx).unwrap_or(&1.0);
                            let corrected_phase = k_wave * distance + phase;
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
}
