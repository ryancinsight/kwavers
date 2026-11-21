//! Field coupling strategies for multi-physics simulations
//!
//! This module implements different strategies for coupling fields between
//! different physics domains (acoustic, optical, thermal).

mod coupling_ops;
mod material;

pub use material::{FieldCouplingMaterial, FieldCouplingMedium, PhotoelasticCoefficient};
#[cfg(test)]
mod tests;

use aequitas::systems::si::quantities::{Frequency, Time};
use kwavers_core::error::{KwaversError, KwaversResult};
use kwavers_field::indices::PRESSURE_IDX;
use leto::Array3;
use material::FieldCouplingGains;

/// Field coupling strategy
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum FieldCouplingStrategy {
    /// No coupling between fields (independent simulations)
    None,
    /// Weak coupling (sequential updates)
    Weak,
    /// Strong coupling (iterative updates)
    Strong,
    /// Adaptive coupling (adjusts based on field gradients)
    Adaptive,
}

/// Field coupler for multi-physics interactions
#[derive(Debug)]
pub struct MultiphysicsFieldCoupler {
    /// Coupling strategy
    strategy: FieldCouplingStrategy,
    /// Coupling strength parameters
    coupling_strength: f64,
    /// Maximum iterations for strong coupling
    max_iterations: usize,
    /// Tolerance for convergence
    tolerance: f64,
}

impl MultiphysicsFieldCoupler {
    /// Create a new field coupler
    #[must_use]
    pub fn new(strategy: FieldCouplingStrategy) -> Self {
        Self {
            strategy,
            coupling_strength: 1.0,
            max_iterations: 10,
            tolerance: 1e-6,
        }
    }

    /// Couple fields using the nominal water and tissue coefficients and
    /// a fixed acoustic intensity loss of 0.5 m⁻¹.
    ///
    /// Use [`Self::couple_fields_with_medium`] to evaluate a spectral loss law.
    ///
    /// # Errors
    /// - Propagates any [`crate::KwaversError`] returned by called functions.
    ///
    pub fn couple_fields(&self, fields: &mut [Array3<f64>], dt: f64) -> KwaversResult<()> {
        if self.strategy == FieldCouplingStrategy::None {
            return Ok(());
        }
        coupling_ops::validate_coupled_field_set(fields)?;
        let gains = FieldCouplingMaterial::nominal()?.gains(Time::from_base(dt))?;
        self.apply_fields(fields, &|_| gains)
    }

    /// Update collocated fields using the local material and acoustic frequency.
    ///
    /// Coefficients describe the supplied medium at each grid cell. The
    /// stored fields carry pressure in pascals, intensity in watts per
    /// square metre, and temperature in degrees Celsius.
    ///
    /// # Errors
    /// Rejects invalid time, frequency, geometry, or material before mutating
    /// the fields. Each cell is sampled once per call, before any iteration.
    pub fn couple_fields_with_medium(
        &self,
        fields: &mut [Array3<f64>],
        duration: Time,
        acoustic_frequency: Frequency,
        medium: &impl FieldCouplingMedium,
    ) -> KwaversResult<()> {
        if self.strategy == FieldCouplingStrategy::None {
            return Ok(());
        }
        let seconds = duration.into_base();
        let hertz = acoustic_frequency.into_base();
        if !seconds.is_finite() || seconds < 0.0 || !hertz.is_finite() || hertz <= 0.0 {
            return Err(KwaversError::InvalidInput(
                "field coupling requires nonnegative finite time and positive finite frequency"
                    .into(),
            ));
        }
        coupling_ops::validate_coupled_field_set(fields)?;
        let shape = fields[PRESSURE_IDX].shape();
        let mut gains = Vec::new();
        gains
            .try_reserve_exact(fields[PRESSURE_IDX].len())
            .map_err(|error| KwaversError::InvalidInput(error.to_string()))?;
        for x in 0..shape[0] {
            for y in 0..shape[1] {
                for z in 0..shape[2] {
                    gains.push(
                        medium
                            .material_at([x, y, z], acoustic_frequency)?
                            .gains(duration)?,
                    );
                }
            }
        }
        let gains = Array3::from_shape_vec(shape, gains)
            .map_err(|error| KwaversError::DimensionMismatch(error.to_string()))?;
        self.apply_fields(fields, &|cell| gains[cell])
    }

    fn apply_fields(
        &self,
        fields: &mut [Array3<f64>],
        gains: &impl Fn([usize; 3]) -> FieldCouplingGains,
    ) -> KwaversResult<()> {
        match self.strategy {
            FieldCouplingStrategy::None => Ok(()),
            FieldCouplingStrategy::Weak => {
                Self::apply_weak_coupling(fields, gains, self.coupling_strength)
            }
            FieldCouplingStrategy::Strong => self.apply_strong_coupling(fields, gains),
            FieldCouplingStrategy::Adaptive => Self::apply_adaptive_coupling(fields, gains),
        }
    }
}
