use leto::{
    Array3,
    Array4,
};

use kwavers_core::error::KwaversResult;

use super::NeuralBeamformer;

impl NeuralBeamformer {
    /// Traditional delay-and-sum beamforming.
    ///
    /// Computes baseline image using geometric focusing delays.
    /// # Errors
    /// - Returns [`Err`] if an internal constraint is violated.
    ///
    pub(super) fn traditional_beamforming(
        &self,
        rf_data: &Array4<f32>,
        steering_angles: &[f64],
    ) -> KwaversResult<Array3<f32>> {
        let [frames, channels, samples, _] = rf_data.shape();
        let num_angles = steering_angles.len();

        let mut image = Array3::<f32>::zeros((frames, num_angles, samples));

        let positions = &self.config.sensor_geometry.positions;
        let c = self.config.sensor_geometry.sound_speed;

        for f in 0..frames {
            for (a_idx, &angle) in steering_angles.iter().enumerate() {
                let delays: Vec<f64> = positions
                    .iter()
                    .map(|pos| pos[0] * angle.sin() / c)
                    .collect();

                // Linear-interpolated DAS: reduces truncation error from O(1/f_s) to O(1/f_s²).
                // Reference: Thomenius KE (1996). "Evolution of ultrasound beamformers." IEEE UFFC Symp. Eq. (7).
                for s in 0..samples {
                    let mut sum = 0.0_f32;
                    let mut count = 0usize;

                    for ch in 0..channels.min(positions.len()) {
                        let delay_f64 = delays[ch] * self.config.sensor_geometry.sampling_frequency;
                        let delay_floor = delay_f64.floor() as isize;
                        let frac = (delay_f64 - delay_floor as f64) as f32;

                        let idx0 = s as isize + delay_floor;
                        let idx1 = idx0 + 1;

                        match (usize::try_from(idx0), usize::try_from(idx1)) {
                            (Ok(i0), Ok(i1)) if i1 < samples => {
                                let s0 = rf_data[[f, ch, i0, 0]];
                                let s1 = rf_data[[f, ch, i1, 0]];
                                sum += frac.mul_add(s1 - s0, s0);
                                count += 1;
                            }
                            (Ok(i0), _) if i0 < samples => {
                                sum += rf_data[[f, ch, i0, 0]];
                                count += 1;
                            }
                            _ => {}
                        }
                    }

                    if count > 0 {
                        image[[f, a_idx, s]] = sum / count as f32;
                    }
                }
            }
        }

        Ok(image)
    }

    /// Assess signal quality using coherence factor.
    ///
    /// CF = |Sum(s_i)|² / (N · Sum(|s_i|²)) ∈ [0, 1].
    /// # Errors
    /// - Returns [`Err`] if an internal constraint is violated.
    ///
    pub(super) fn assess_signal_quality(&self, rf_data: &Array4<f32>) -> KwaversResult<f64> {
        let [frames, channels, samples, _] = rf_data.shape();

        if channels == 0 || samples == 0 {
            return Ok(0.0);
        }

        let mut total_cf = 0.0;
        let mut count = 0;

        let stride = 1.max(samples / 100);

        for f in 0..frames {
            for s in (0..samples).step_by(stride) {
                let mut sum_sig = 0.0;
                let mut sum_sq_energy = 0.0;

                for c in 0..channels {
                    let val = rf_data[[f, c, s, 0]];
                    sum_sig += val;
                    sum_sq_energy += val * val;
                }

                if sum_sq_energy > 1e-10 {
                    let coherent_energy = sum_sig * sum_sig;
                    let incoherent_energy = channels as f32 * sum_sq_energy;
                    total_cf += (coherent_energy / incoherent_energy) as f64;
                }
                count += 1;
            }
        }

        Ok(if count > 0 {
            total_cf / count as f64
        } else {
            0.0
        })
    }
}
