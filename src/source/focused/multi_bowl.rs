//! Multi-element bowl array implementation
//!
//! Provides arrays of bowl transducers for complex field synthesis.

use super::bowl::{BowlConfig, BowlTransducer};
use crate::{error::KwaversResult, grid::Grid};
use ndarray::{Array3, Zip};
use std::f64::consts::PI;

/// Multi-element bowl array (makeMultiBowl equivalent)
pub struct MultiBowlArray {
    /// Individual bowl transducers
    bowls: Vec<BowlTransducer>,
    /// Relative amplitudes for each bowl
    amplitudes: Vec<f64>,
    /// Relative phases for each bowl
    phases: Vec<f64>,
}

impl MultiBowlArray {
    /// Create a new multi-bowl array
    pub fn new(configs: Vec<BowlConfig>) -> KwaversResult<Self> {
        let n_bowls = configs.len();
        let mut bowls = Vec::with_capacity(n_bowls);
        let mut amplitudes = Vec::with_capacity(n_bowls);
        let mut phases = Vec::with_capacity(n_bowls);

        for config in configs {
            amplitudes.push(config.amplitude);
            phases.push(config.phase);
            bowls.push(BowlTransducer::new(config)?);
        }

        Ok(Self {
            bowls,
            amplitudes,
            phases,
        })
    }

    /// Generate combined source from all bowls
    ///
    /// This method combines the contributions from all bowl transducers,
    /// applying both amplitude scaling and phase shifts. The phase shifts
    /// are crucial for beam steering and complex field synthesis.
    pub fn generate_source(&self, grid: &Grid, time: f64) -> KwaversResult<Array3<f64>> {
        let mut combined_source = grid.create_field();

        // Add contributions from each bowl
        for (i, bowl) in self.bowls.iter().enumerate() {
            // Generate source for this bowl at the current time
            // Note: We need to adjust the time to account for the phase offset
            let omega = 2.0 * PI * bowl.config.frequency;
            let phase_offset = self.phases[i] - bowl.config.phase; // Relative phase
            let time_offset = phase_offset / omega; // Convert phase to time offset

            let bowl_source = bowl.generate_source(grid, time + time_offset)?;

            // Apply relative amplitude
            let scale = self.amplitudes[i] / bowl.config.amplitude;

            Zip::from(&mut combined_source)
                .and(&bowl_source)
                .for_each(|c, &b| *c += scale * b);
        }

        Ok(combined_source)
    }

    /// Set beam steering parameters
    pub fn set_beam_steering(&mut self, focus: [f64; 3]) {
        // Update focus for all bowls
        for bowl in &mut self.bowls {
            bowl.config.focus = focus;
        }
    }

    /// Apply apodization (amplitude shading)
    pub fn apply_apodization(&mut self, apodization_type: ApodizationType) {
        let n = self.bowls.len();

        match apodization_type {
            ApodizationType::Uniform => {
                self.amplitudes = vec![1.0; n];
            }
            ApodizationType::Hamming => {
                for i in 0..n {
                    let x = i as f64 / (n - 1) as f64;
                    self.amplitudes[i] = 0.54 - 0.46 * (2.0 * PI * x).cos();
                }
            }
            ApodizationType::Hanning => {
                for i in 0..n {
                    let x = i as f64 / (n - 1) as f64;
                    self.amplitudes[i] = 0.5 * (1.0 - (2.0 * PI * x).cos());
                }
            }
            ApodizationType::Gaussian(sigma) => {
                let center = (n - 1) as f64 / 2.0;
                for i in 0..n {
                    let x = (i as f64 - center) / center;
                    self.amplitudes[i] = (-x * x / (2.0 * sigma * sigma)).exp();
                }
            }
        }
    }
}

/// Apodization types for multi-element arrays
#[derive(Debug, Clone, Copy)]
pub enum ApodizationType {
    /// Uniform weighting (no apodization)
    Uniform,
    /// Hamming window
    Hamming,
    /// Hanning (Hann) window
    Hanning,
    /// Gaussian window with specified sigma
    Gaussian(f64),
}
