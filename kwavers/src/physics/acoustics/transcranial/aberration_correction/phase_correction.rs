//! Phase correction computation for transcranial aberration
//!
//! ## Mathematical Foundation
//!
//! Phase aberration from skull: Δφ = ∫ (k_skull - k_water) ds
//! Phase conjugation: Apply -Δφ to each transducer element
//!
//! ## References
//!
//! - Clement & Hynynen (2002) PMB 47(8):1219-1235

use crate::core::error::KwaversResult;
use crate::domain::grid::Grid;
use crate::physics::acoustics::analytical::patterns::phase_shifting::core::wrap_phase;
use log::info;

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
    pub(crate) grid: Grid,
    /// Operating frequency (Hz)
    pub(crate) frequency: f64,
    /// Reference sound speed (m/s)
    pub(crate) reference_speed: f64,
    /// Number of transducer elements
    pub(crate) _num_elements: usize,
}

impl TranscranialAberrationCorrection {
    /// Create new aberration correction system
    pub fn new(grid: &Grid) -> KwaversResult<Self> {
        Ok(Self {
            grid: grid.clone(),
            frequency: 650e3,
            reference_speed: 1500.0,
            _num_elements: 1024,
        })
    }

    /// Calculate phase correction from skull model
    pub fn calculate_correction(
        &self,
        skull_ct_data: &ndarray::Array3<f64>,
        transducer_positions: &[[f64; 3]],
        target_point: &[f64; 3],
    ) -> KwaversResult<PhaseCorrection> {
        info!(
            "Calculating aberration correction for {} transducer elements",
            transducer_positions.len()
        );

        let path_delays =
            self.calculate_path_delays(skull_ct_data, transducer_positions, target_point)?;

        let wavenumbers = self.calculate_wavenumbers(&path_delays);
        let mut phases = Vec::with_capacity(transducer_positions.len());

        for &k in &wavenumbers {
            phases.push(-k);
        }

        let amplitudes = self.optimize_amplitudes(&path_delays);
        let quality_metric = self.estimate_correction_quality(&path_delays, &phases);
        let focal_gain_db = self.calculate_focal_gain(&path_delays);

        Ok(PhaseCorrection {
            phases,
            amplitudes,
            focal_gain_db,
            quality_metric,
        })
    }

    /// Calculate propagation delays through skull
    pub(crate) fn calculate_path_delays(
        &self,
        skull_ct_data: &ndarray::Array3<f64>,
        transducer_positions: &[[f64; 3]],
        target_point: &[f64; 3],
    ) -> KwaversResult<Vec<f64>> {
        let mut delays = Vec::with_capacity(transducer_positions.len());

        for &transducer_pos in transducer_positions {
            let path_vector = [
                target_point[0] - transducer_pos[0],
                target_point[1] - transducer_pos[1],
                target_point[2] - transducer_pos[2],
            ];

            let path_length =
                (path_vector[0].powi(2) + path_vector[1].powi(2) + path_vector[2].powi(2)).sqrt();

            let num_samples = 100;
            let mut total_delay = 0.0;

            for i in 0..num_samples {
                let t = i as f64 / (num_samples - 1) as f64;
                let point = [
                    transducer_pos[0] + t * path_vector[0],
                    transducer_pos[1] + t * path_vector[1],
                    transducer_pos[2] + t * path_vector[2],
                ];

                let ix = ((point[0] / self.grid.dx) as usize).min(self.grid.nx - 1);
                let iy = ((point[1] / self.grid.dy) as usize).min(self.grid.ny - 1);
                let iz = ((point[2] / self.grid.dz) as usize).min(self.grid.nz - 1);

                let hu = skull_ct_data[[ix, iy, iz]];
                let local_speed = crate::domain::imaging::medical::CTImageLoader::hu_to_sound_speed(hu);

                let ds = path_length / num_samples as f64;
                let k_local = 2.0 * std::f64::consts::PI * self.frequency / local_speed;
                total_delay += k_local * ds;
            }

            let reference_delay =
                2.0 * std::f64::consts::PI * self.frequency * path_length / self.reference_speed;
            let aberration_delay = total_delay - reference_delay;

            delays.push(aberration_delay);
        }

        Ok(delays)
    }

    /// Calculate wavenumbers from delays
    pub(crate) fn calculate_wavenumbers(&self, delays: &[f64]) -> Vec<f64> {
        delays.to_vec()
    }

    /// Optimize element amplitudes for uniform focal intensity
    pub(crate) fn optimize_amplitudes(&self, delays: &[f64]) -> Vec<f64> {
        let max_delay = delays.iter().cloned().fold(0.0_f64, f64::max);
        let min_delay = delays.iter().cloned().fold(f64::INFINITY, f64::min);

        delays
            .iter()
            .map(|&delay| {
                let delay_range = max_delay - min_delay;
                if delay_range > 0.0 {
                    1.0 + (max_delay - delay) / delay_range
                } else {
                    1.0
                }
            })
            .collect()
    }

    /// Estimate correction quality (0-1 scale)
    pub(crate) fn estimate_correction_quality(&self, delays: &[f64], phases: &[f64]) -> f64 {
        let mut residual_errors = Vec::new();

        for (&delay, &phase) in delays.iter().zip(phases.iter()) {
            let residual_wrapped = wrap_phase(delay + phase);
            residual_errors.push(residual_wrapped.abs());
        }

        let mean_residual = residual_errors.iter().sum::<f64>() / residual_errors.len() as f64;
        1.0 / (1.0 + mean_residual)
    }

    /// Calculate expected focal gain improvement
    pub(crate) fn calculate_focal_gain(&self, delays: &[f64]) -> f64 {
        let max_delay = delays.iter().cloned().fold(0.0_f64, f64::max);
        let min_delay = delays.iter().cloned().fold(f64::INFINITY, f64::min);
        let delay_range = max_delay - min_delay;

        if delay_range > 0.0 {
            20.0 * (2.0 * std::f64::consts::PI / delay_range).log10()
        } else {
            0.0
        }
    }
}
