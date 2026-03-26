//! Scattered acoustic field calculation from bubble clouds
//!
//! ## Mathematical Foundation
//!
//! Scattered pressure from spherical bubble (Rayleigh scattering):
//! ```text
//! p_s(r) = A · cos(kr) / r
//! ```
//!
//! Near resonance: enhanced scattering amplitude ∝ R³ · 10⁶
//! Off-resonance: scattering amplitude ∝ R³ · 10³

use super::simulator::CloudDynamics;
use crate::core::error::KwaversResult;
use ndarray::Array3;
use std::collections::HashMap;

/// Scattered acoustic field from bubble cloud
#[derive(Debug)]
pub struct ScatteredField {
    /// Fundamental frequency scattered pressure
    pub fundamental: Array3<f64>,
    /// Harmonic components (frequency -> pressure field)
    pub harmonics: HashMap<usize, Array3<f64>>,
    /// Center frequency (Hz)
    pub frequency: f64,
}

impl CloudDynamics {
    /// Calculate scattered acoustic field from bubble cloud
    pub fn calculate_scattered_field(&self, frequency: f64) -> KwaversResult<ScatteredField> {
        if self.bubbles.is_empty() {
            return Ok(ScatteredField {
                fundamental: Array3::<f64>::zeros((64, 64, 64)),
                harmonics: HashMap::new(),
                frequency,
            });
        }

        let nx = 64;
        let ny = 64;
        let nz = 64;

        let mut scattered_pressure = Array3::<f64>::zeros((nx, ny, nz));
        let harmonics = HashMap::new();

        let k: f64 = 2.0 * std::f64::consts::PI * frequency / 1500.0;

        for (bubble_idx, bubble) in self.bubbles.iter().enumerate() {
            if bubble_idx >= 1000 {
                break;
            }

            let bubble_pos = &bubble.position;

            // Calculate scattering amplitude based on bubble resonance
            let resonance_freq = bubble.properties.resonance_frequency(101325.0, 1000.0);
            let scattering_amplitude = if (frequency - resonance_freq).abs() < 0.1 * resonance_freq
            {
                bubble.current_radius.powi(3) * 1e6
            } else {
                bubble.current_radius.powi(3) * 1e3
            };

            for i in 0..nx {
                for j in 0..ny {
                    for kz in 0..nz {
                        let x = (i as f64 - nx as f64 / 2.0) * 1e-4;
                        let y = (j as f64 - ny as f64 / 2.0) * 1e-4;
                        let z = (kz as f64 - nz as f64 / 2.0) * 1e-4;

                        let dx = x - bubble_pos[0];
                        let dy = y - bubble_pos[1];
                        let dz = z - bubble_pos[2];
                        let distance = (dx * dx + dy * dy + dz * dz).sqrt();

                        if distance > 1e-6 {
                            let phase: f64 = k * distance;
                            let contribution =
                                scattering_amplitude * (phase.cos() / distance).clamp(-1e6, 1e6);

                            scattered_pressure[[i, j, kz]] += contribution;
                        }
                    }
                }
            }
        }

        let bubble_count = self.bubbles.len().max(1) as f64;
        scattered_pressure.mapv_inplace(|x| x / bubble_count);

        Ok(ScatteredField {
            fundamental: scattered_pressure,
            harmonics,
            frequency,
        })
    }
}
