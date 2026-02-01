//! Aberration Correction for Transcranial Ultrasound
//!
//! Implements phase correction algorithms to compensate for skull-induced
//! phase aberrations in transcranial focused ultrasound.

use crate::core::error::KwaversResult;
use crate::domain::grid::Grid;
use crate::physics::acoustics::analytical::patterns::phase_shifting::core::wrap_phase;
use ndarray::Array3;
use num_complex::Complex;

/// Phase correction data for transducer elements
#[derive(Debug, Clone)]
pub struct PhaseCorrection {
    /// Correction phases for each transducer element (radians)
    pub phases: Vec<f64>,
    /// Correction amplitudes (normalized)
    pub amplitudes: Vec<f64>,
    /// Expected focal gain improvement (dB)
    pub focal_gain_db: f64,
    /// Correction quality metric (0-1, higher is better)
    pub quality_metric: f64,
}

/// Transcranial aberration correction system
#[derive(Debug)]
pub struct TranscranialAberrationCorrection {
    /// Computational grid
    grid: Grid,
    /// Operating frequency (Hz)
    frequency: f64,
    /// Reference sound speed (m/s)
    reference_speed: f64,
    /// Number of transducer elements
    _num_elements: usize,
}

impl TranscranialAberrationCorrection {
    /// Create new aberration correction system
    pub fn new(grid: &Grid) -> KwaversResult<Self> {
        Ok(Self {
            grid: grid.clone(),
            frequency: 650e3,        // Default brain therapy frequency
            reference_speed: 1500.0, // Water speed
            _num_elements: 1024,     // Default hemispherical array
        })
    }

    /// Calculate phase correction from skull model
    pub fn calculate_correction(
        &self,
        skull_model: &crate::physics::skull::CTBasedSkullModel,
        transducer_positions: &[[f64; 3]],
        target_point: &[f64; 3],
    ) -> KwaversResult<PhaseCorrection> {
        println!(
            "Calculating aberration correction for {} transducer elements",
            transducer_positions.len()
        );

        // Step 1: Calculate propagation path through skull for each element
        let path_delays =
            self.calculate_path_delays(skull_model, transducer_positions, target_point)?;

        // Step 2: Convert delays to phases
        let wavenumbers = self.calculate_wavenumbers(&path_delays);
        let mut phases = Vec::with_capacity(transducer_positions.len());

        for &k in &wavenumbers {
            // Phase conjugation: exp(-j*k*delay)
            // For focusing, we want to advance the phase by the delay amount
            phases.push(-k); // Negative for phase advance
        }

        // Step 3: Optimize amplitudes for uniform intensity
        let amplitudes = self.optimize_amplitudes(&path_delays);

        // Step 4: Estimate correction quality
        let quality_metric = self.estimate_correction_quality(&path_delays, &phases);

        // Step 5: Calculate expected focal gain
        let focal_gain_db = self.calculate_focal_gain(&path_delays);

        Ok(PhaseCorrection {
            phases,
            amplitudes,
            focal_gain_db,
            quality_metric,
        })
    }

    /// Calculate propagation delays through skull
    fn calculate_path_delays(
        &self,
        skull_model: &crate::physics::skull::CTBasedSkullModel,
        transducer_positions: &[[f64; 3]],
        target_point: &[f64; 3],
    ) -> KwaversResult<Vec<f64>> {
        let mut delays = Vec::with_capacity(transducer_positions.len());

        for &transducer_pos in transducer_positions {
            // Calculate straight-line path from transducer to target
            let path_vector = [
                target_point[0] - transducer_pos[0],
                target_point[1] - transducer_pos[1],
                target_point[2] - transducer_pos[2],
            ];

            let path_length =
                (path_vector[0].powi(2) + path_vector[1].powi(2) + path_vector[2].powi(2)).sqrt();

            // Sample skull properties along the path
            let num_samples = 100;
            let mut total_delay = 0.0;

            for i in 0..num_samples {
                let t = i as f64 / (num_samples - 1) as f64;
                let point = [
                    transducer_pos[0] + t * path_vector[0],
                    transducer_pos[1] + t * path_vector[1],
                    transducer_pos[2] + t * path_vector[2],
                ];

                // Convert to grid coordinates
                let ix = ((point[0] / self.grid.dx) as usize).min(self.grid.nx - 1);
                let iy = ((point[1] / self.grid.dy) as usize).min(self.grid.ny - 1);
                let iz = ((point[2] / self.grid.dz) as usize).min(self.grid.nz - 1);

                // Get local sound speed from skull model
                let local_speed = skull_model.sound_speed(ix, iy, iz);

                // Accumulate phase delay: k * ds = (2πf/c) * ds
                let ds = path_length / num_samples as f64;
                let k_local = 2.0 * std::f64::consts::PI * self.frequency / local_speed;
                total_delay += k_local * ds;
            }

            // Subtract reference delay (straight line at reference speed)
            let reference_delay =
                2.0 * std::f64::consts::PI * self.frequency * path_length / self.reference_speed;
            let aberration_delay = total_delay - reference_delay;

            delays.push(aberration_delay);
        }

        Ok(delays)
    }

    /// Calculate wavenumbers from delays
    fn calculate_wavenumbers(&self, delays: &[f64]) -> Vec<f64> {
        delays.to_vec()
    }

    /// Optimize element amplitudes for uniform focal intensity
    fn optimize_amplitudes(&self, delays: &[f64]) -> Vec<f64> {
        // Simple amplitude optimization based on attenuation
        // In practice, this would use more sophisticated optimization
        let max_delay = delays.iter().cloned().fold(0.0_f64, f64::max);
        let min_delay = delays.iter().cloned().fold(f64::INFINITY, f64::min);

        delays
            .iter()
            .map(|&delay| {
                // Compensate for attenuation (higher delay = more attenuation)
                let delay_range = max_delay - min_delay;
                if delay_range > 0.0 {
                    1.0 + (max_delay - delay) / delay_range // Boost weaker elements
                } else {
                    1.0
                }
            })
            .collect()
    }

    /// Estimate correction quality (0-1 scale)
    fn estimate_correction_quality(&self, delays: &[f64], phases: &[f64]) -> f64 {
        // Calculate residual phase error after correction
        let mut residual_errors = Vec::new();

        for (&delay, &phase) in delays.iter().zip(phases.iter()) {
            // Use SSOT wrap_phase function to wrap to [-π, π] range
            let residual_wrapped = wrap_phase(delay + phase);
            residual_errors.push(residual_wrapped.abs());
        }

        // Quality metric: 1 / (1 + mean_residual_error)
        let mean_residual = residual_errors.iter().sum::<f64>() / residual_errors.len() as f64;
        1.0 / (1.0 + mean_residual)
    }

    /// Calculate expected focal gain improvement
    fn calculate_focal_gain(&self, delays: &[f64]) -> f64 {
        // Estimate focal gain based on delay distribution
        // Reference: Clement & Hynynen (2002)

        let max_delay = delays.iter().cloned().fold(0.0_f64, f64::max);
        let min_delay = delays.iter().cloned().fold(f64::INFINITY, f64::min);
        let delay_range = max_delay - min_delay;

        // Focal gain improvement ≈ 20*log10(2π / delay_range_radians)
        // Simplified approximation
        // TODO_AUDIT: P1 - Advanced Transcranial Aberration Correction - Implement full aberration correction with adaptive optics and time-reversal focusing
        // DEPENDS ON: physics/acoustics/transcranial/adaptive_optics.rs, physics/acoustics/transcranial/time_reversal.rs, physics/acoustics/transcranial/phase_conjugation.rs
        // MISSING: Time-reversal mirror for perfect focusing through aberrating media
        // MISSING: Phase conjugation for real-time aberration compensation
        // MISSING: Adaptive optics with deformable mirror correction
        // MISSING: Multi-element array optimization for skull transmission
        // MISSING: Patient-specific skull acoustic property characterization
        // SEVERITY: CRITICAL (enables transcranial therapeutic ultrasound)
        // THEOREM: Time-reversal invariance: Wave equation ∂²u/∂t² - c²∇²u = 0 is time-reversal symmetric
        // THEOREM: Phase conjugation: If u(t) is aberrated wave, then u*(-t) focuses perfectly at source
        // REFERENCES: Fink (1992) IEEE Trans Ultrason Ferroelectr Freq Control; Tanter et al. (2007) Nat Rev Drug Discov
        if delay_range > 0.0 {
            20.0 * (2.0 * std::f64::consts::PI / delay_range).log10()
        } else {
            0.0 // No aberrations
        }
    }

    /// Apply time-reversal aberration correction
    pub fn apply_time_reversal_correction(
        &self,
        _measured_field: &Array3<Complex<f64>>,
        transducer_positions: &[[f64; 3]],
    ) -> KwaversResult<PhaseCorrection> {
        // Time-reversal acoustics for aberration correction
        // Reference: Aubry et al. (2003)

        let mut phases = vec![0.0; transducer_positions.len()];
        let amplitudes = vec![1.0; transducer_positions.len()];

        // Simplified: assume point target and calculate phase conjugation
        // In practice, this would involve:
        // 1. Measure scattered field from hydrophone
        // 2. Time-reverse the signal
        // 3. Back-propagate to get element phases

        println!("Applying time-reversal aberration correction");

        // Placeholder implementation
        for i in 0..transducer_positions.len() {
            // Simple geometric phase calculation
            let dist_to_origin = (transducer_positions[i][0].powi(2)
                + transducer_positions[i][1].powi(2)
                + transducer_positions[i][2].powi(2))
            .sqrt();
            let k = 2.0 * std::f64::consts::PI * self.frequency / self.reference_speed;
            phases[i] = -k * dist_to_origin;
        }

        Ok(PhaseCorrection {
            phases,
            amplitudes,
            focal_gain_db: 10.0, // Typical improvement
            quality_metric: 0.8,
        })
    }

    /// Adaptive aberration correction using iterative optimization
    pub fn adaptive_correction(
        &mut self,
        initial_correction: &PhaseCorrection,
        feedback_signal: &[f64],
        learning_rate: f64,
    ) -> PhaseCorrection {
        let mut new_phases = initial_correction.phases.clone();
        let new_amplitudes = initial_correction.amplitudes.clone();

        // Simple gradient descent on feedback signal
        // In practice, this would use more sophisticated optimization
        for i in 0..new_phases.len() {
            if i < feedback_signal.len() {
                // Adjust phase based on feedback
                let gradient = feedback_signal[i] - 1.0; // Target = 1.0 (normalized)
                new_phases[i] -= learning_rate * gradient;
            }
        }

        PhaseCorrection {
            phases: new_phases,
            amplitudes: new_amplitudes,
            focal_gain_db: initial_correction.focal_gain_db * 1.1, // Slight improvement
            quality_metric: (initial_correction.quality_metric + 0.1).min(1.0),
        }
    }

    /// Validate correction performance
    pub fn validate_correction(
        &self,
        correction: &PhaseCorrection,
        skull_model: &crate::physics::skull::CTBasedSkullModel,
        transducer_positions: &[[f64; 3]],
        target_point: &[f64; 3],
    ) -> KwaversResult<CorrectionValidation> {
        // Simulate corrected field
        let corrected_field = self.simulate_corrected_field(
            correction,
            skull_model,
            transducer_positions,
            target_point,
        )?;

        // Calculate metrics
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
        _skull_model: &crate::physics::skull::CTBasedSkullModel,
        transducer_positions: &[[f64; 3]],
        _target_point: &[f64; 3],
    ) -> KwaversResult<Array3<f64>> {
        let (nx, ny, nz) = self.grid.dimensions();
        let mut field = Array3::zeros((nx, ny, nz));

        // Simplified field simulation with phase correction
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
                            // Apply phase correction
                            let phase = correction.phases.get(elem_idx).unwrap_or(&0.0);
                            let amplitude = correction.amplitudes.get(elem_idx).unwrap_or(&1.0);

                            let k =
                                2.0 * std::f64::consts::PI * self.frequency / self.reference_speed;
                            let uncorrected_phase = k * distance;
                            let corrected_phase = uncorrected_phase + phase;

                            let contribution = Complex::from_polar(*amplitude, corrected_phase);
                            total_field += contribution / distance; // Spherical spreading
                        }
                    }

                    field[[i, j, k]] = total_field.norm_sqr(); // Intensity
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

        // Find maximum sidelobe
        let mut max_sidelobe: f64 = 0.0;
        let focal_ix = ((target_point[0] / self.grid.dx) as usize).min(self.grid.nx - 1);
        let focal_iy = ((target_point[1] / self.grid.dy) as usize).min(self.grid.ny - 1);
        let focal_iz = ((target_point[2] / self.grid.dz) as usize).min(self.grid.nz - 1);

        for k in 0..self.grid.nz {
            for j in 0..self.grid.ny {
                for i in 0..self.grid.nx {
                    // Skip points near focal region
                    let distance_from_focus = (((i as i32 - focal_ix as i32).pow(2)
                        + (j as i32 - focal_iy as i32).pow(2)
                        + (k as i32 - focal_iz as i32).pow(2))
                        as f64)
                        .sqrt();

                    if distance_from_focus > 3.0 {
                        // Outside focal region
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
        let threshold = focal_intensity / 4.0; // -6dB

        let mut volume = 0.0;
        for &intensity in field.iter() {
            if intensity >= threshold {
                volume += self.grid.dx * self.grid.dy * self.grid.dz;
            }
        }

        volume
    }
}

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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::physics::skull::CTBasedSkullModel;

    #[test]
    fn test_aberration_corrector_creation() {
        let grid = Grid::new(32, 32, 32, 0.002, 0.002, 0.002).unwrap();
        let corrector = TranscranialAberrationCorrection::new(&grid);
        assert!(corrector.is_ok());
    }

    #[test]
    fn test_phase_correction_calculation() {
        let grid = Grid::new(16, 16, 16, 0.005, 0.005, 0.005).unwrap();
        let corrector = TranscranialAberrationCorrection::new(&grid).unwrap();

        // Create simple skull model
        let ct_data = Array3::from_elem((16, 16, 16), 400.0); // Bone HU
        let skull_model = CTBasedSkullModel::from_ct_data(&ct_data).unwrap();

        // Simple transducer array
        let transducer_positions = vec![[0.0, 0.0, 0.08], [0.02, 0.0, 0.08], [0.0, 0.02, 0.08]];

        let target_point = [0.0, 0.0, 0.0]; // Brain center

        let correction =
            corrector.calculate_correction(&skull_model, &transducer_positions, &target_point);

        assert!(correction.is_ok());
        let corr = correction.unwrap();
        assert_eq!(corr.phases.len(), transducer_positions.len());
        assert_eq!(corr.amplitudes.len(), transducer_positions.len());
    }

    #[test]
    fn test_time_reversal_correction() {
        let grid = Grid::new(16, 16, 16, 0.005, 0.005, 0.005).unwrap();
        let corrector = TranscranialAberrationCorrection::new(&grid).unwrap();

        let transducer_positions = vec![[0.0, 0.0, 0.08], [0.02, 0.0, 0.08]];

        let measured_field = Array3::from_elem((16, 16, 16), Complex::new(1.0, 0.0));

        let correction =
            corrector.apply_time_reversal_correction(&measured_field, &transducer_positions);

        assert!(correction.is_ok());
    }
}
