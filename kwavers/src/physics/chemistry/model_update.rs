//! ChemicalModel update orchestration

use super::model::{ChemicalModel, ChemicalModelState};
use super::parameters::ChemicalUpdateParams;
use crate::core::error::KwaversResult;
use std::time::Instant;

impl ChemicalModel {
    /// Update chemical reactions based on acoustic field
    pub fn update(&mut self, params: &ChemicalUpdateParams) -> KwaversResult<()> {
        let start = Instant::now();
        self.state = ChemicalModelState::Running;

        // Update radical initiation
        let radical_start = Instant::now();
        self.radical_initiation.update_radicals(
            params.pressure,
            params.light,
            params.bubble_radius,
            params.grid,
            params.dt,
            params.medium,
            params.frequency,
        );
        self.metrics
            .set_computation_time(radical_start.elapsed().as_secs_f64() * 1000.0);

        // Update reaction kinetics if enabled
        if self.enable_kinetics {
            if let Some(ref mut kinetics) = self.kinetics {
                let kinetics_start = Instant::now();
                kinetics.update_reactions(
                    &self.radical_initiation.radical_concentration,
                    params.temperature,
                    params.grid,
                    params.dt,
                    params.medium,
                );
                let kinetics_time = kinetics_start.elapsed().as_secs_f64() * 1000.0;
                self.metrics
                    .set_computation_time(self.metrics.computation_time_ms + kinetics_time);
            }
        }

        // Update photochemical effects if enabled
        if self.enable_photochemical {
            if let Some(ref mut photochemical) = self.photochemical {
                let photo_start = Instant::now();
                photochemical.update_photochemical(
                    params.light,
                    params.emission_spectrum,
                    params.bubble_radius,
                    params.temperature,
                    params.grid,
                    params.dt,
                    params.medium,
                );
                let photo_time = photo_start.elapsed().as_secs_f64() * 1000.0;
                self.metrics
                    .set_computation_time(self.metrics.computation_time_ms + photo_time);
            }
        }

        self.computation_time = start.elapsed();
        self.update_count += 1;
        self.state = ChemicalModelState::Completed;
        self.metrics.increment_reactions(1);

        Ok(())
    }
}
