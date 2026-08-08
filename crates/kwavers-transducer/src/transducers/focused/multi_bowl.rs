//! Multi-element bowl array implementation
//!
//! Provides arrays of bowl transducers for complex field synthesis.

use super::bowl::{BowlConfig, BowlTransducer};
use super::validation::{field_validation_error, validate_finite_field};
use aequitas::systems::si::quantities::{Angle, Length, Pressure, Time};
use aequitas::systems::si::units::{Hertz, Pascal, Radian, Second};
use kwavers_core::constants::numerical::TWO_PI;
use kwavers_core::error::KwaversResult;
use kwavers_grid::Grid;
use leto::Array3;
use moirai_parallel::{enumerate_mut_with, Adaptive};

/// Multi-element bowl array (makeMultiBowl equivalent)
#[derive(Debug)]
pub struct MultiBowlArray {
    /// Individual bowl transducers
    pub(crate) bowls: Vec<BowlTransducer>,
    /// Relative amplitudes for each bowl
    amplitudes: Vec<Pressure<f64>>,
    /// Relative phases for each bowl
    phases: Vec<Angle<f64>>,
}

impl MultiBowlArray {
    /// Create a new multi-bowl array
    /// # Errors
    /// - Propagates any [`kwavers_core::error::KwaversError`] returned by called functions.
    ///
    pub fn new(configs: Vec<BowlConfig>) -> KwaversResult<Self> {
        let mut bowls = Vec::with_capacity(configs.len());

        for config in configs {
            bowls.push(BowlTransducer::new(config)?);
        }

        Self::from_bowls(bowls)
    }

    /// Create a multi-bowl source from already constructed bowl transducers.
    ///
    /// This constructor keeps hemispherical, annular, and bounded polar-span
    /// bowl layouts inside the generic source domain instead of introducing
    /// anatomy-specific aggregate source types.
    ///
    /// # Errors
    ///
    /// Returns [`kwavers_core::error::KwaversError::Validation`] when the array
    /// contains no bowls.
    pub fn from_bowls(bowls: Vec<BowlTransducer>) -> KwaversResult<Self> {
        validate_bowl_count(bowls.len())?;
        let amplitudes = bowls
            .iter()
            .map(|bowl| bowl.config.amplitude)
            .collect::<Vec<_>>();
        let phases = bowls
            .iter()
            .map(|bowl| bowl.config.phase)
            .collect::<Vec<_>>();

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
    /// # Errors
    /// - Propagates any [`kwavers_core::error::KwaversError`] returned by called functions.
    ///
    pub fn generate_source(&self, grid: &Grid, time: Time<f64>) -> KwaversResult<Array3<f64>> {
        let mut combined_source = Array3::zeros([grid.nx, grid.ny, grid.nz]);

        // Add contributions from each bowl
        for (i, bowl) in self.bowls.iter().enumerate() {
            // Generate source for this bowl at the current time
            // Note: We need to adjust the time to account for the phase offset
            let omega = TWO_PI * bowl.config.frequency.in_unit::<Hertz>();
            validate_finite_field(
                "multi_bowl_amplitude",
                self.amplitudes[i].in_unit::<Pascal>(),
            )?;
            validate_finite_field("multi_bowl_phase", self.phases[i].in_unit::<Radian>())?;
            let phase_offset =
                self.phases[i].in_unit::<Radian>() - bowl.config.phase.in_unit::<Radian>(); // Relative phase
            let time_offset = phase_offset / omega; // Convert phase to time offset

            let bowl_source = bowl.generate_source(
                grid,
                Time::from_unit::<Second>(time.in_unit::<Second>() + time_offset),
            )?;

            // Apply relative amplitude
            let scale = amplitude_scale(
                self.amplitudes[i].in_unit::<Pascal>(),
                bowl.config.amplitude.in_unit::<Pascal>(),
            );

            let combined_data = combined_source
                .as_slice_mut()
                .expect("invariant: freshly allocated Array3 is contiguous");
            let bowl_data = bowl_source
                .as_slice()
                .expect("invariant: generated bowl source Array3 is contiguous");
            enumerate_mut_with::<Adaptive, _, _>(combined_data, |idx, c| {
                *c += scale * bowl_data[idx];
            });
        }

        Ok(combined_source)
    }

    /// Set beam steering parameters
    pub fn set_beam_steering(&mut self, focus: [Length<f64>; 3]) -> KwaversResult<()> {
        // Update focus for all bowls
        for bowl in &mut self.bowls {
            bowl.set_focus(focus)?;
        }
        Ok(())
    }

    /// Apply dimensionless apodization weights to the original drive pressures.
    ///
    /// # Theorem
    ///
    /// Each bowl source is linear in its drive pressure `A_i`. Therefore an
    /// apodized array field satisfies `p = sum_i w_i p_i` by storing the target
    /// absolute pressure `w_i A_i` and scaling each generated bowl field by
    /// `(w_i A_i) / A_i`. Zero-drive bowls preserve the unique zero field.
    pub fn apply_apodization(&mut self, apodization_type: ApodizationType) {
        let n = self.bowls.len();
        self.amplitudes = self
            .bowls
            .iter()
            .zip((0..n).map(|i| apodization_type.weights(i, n)))
            .map(|(bowl, weight)| {
                Pressure::from_unit::<Pascal>(bowl.config.amplitude.in_unit::<Pascal>() * weight)
            })
            .collect();
    }
}

pub use kwavers_math::ApodizationType;

fn validate_bowl_count(count: usize) -> KwaversResult<()> {
    if count > 0 {
        Ok(())
    } else {
        Err(field_validation_error(
            "bowl_count",
            count.to_string(),
            "must be at least one",
        ))
    }
}

fn amplitude_scale(target_amplitude: f64, source_amplitude: f64) -> f64 {
    if source_amplitude == 0.0 {
        0.0
    } else {
        target_amplitude / source_amplitude
    }
}

#[cfg(test)]
mod tests;
