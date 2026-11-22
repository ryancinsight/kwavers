//! Spectral phase rotation for a calibrated power-law medium.
use super::PowerLawAbsorption;
use aequitas::systems::si::quantities::{Frequency, Time, Velocity, Wavenumber};
use eunomia::Complex64;
use kwavers_core::constants::numerical::TWO_PI;
use kwavers_core::error::{KwaversError, KwaversResult};
use leto::Array3;

/// Apply the power-law phase correction at the reference acoustic speed.
#[derive(Debug)]
pub struct AbsorptionDispersionCorrection {
    law: PowerLawAbsorption,
    reference_speed: Velocity,
}

impl AbsorptionDispersionCorrection {
    /// Bind a power law to the speed used by the reference spectral grid.
    ///
    /// # Errors
    /// Rejects a nonfinite or nonpositive reference speed.
    pub fn new(law: PowerLawAbsorption, reference_speed: Velocity) -> KwaversResult<Self> {
        if !reference_speed.into_base().is_finite() || reference_speed.into_base() <= 0.0 {
            return Err(KwaversError::InvalidInput(
                "Reference speed must be finite and positive".into(),
            ));
        }
        Ok(Self {
            law,
            reference_speed,
        })
    }

    /// Rotate by exp(i k (cₚ−c₀) Δt) at f = |k| c₀/(2π).
    /// Wavenumbers are signed angular wavenumbers in radians per metre.
    ///
    /// # Errors
    /// Rejects shape or layout errors, invalid timestep, nonfinite wavenumbers
    /// or phase, and invalid dispersion before modifying any spectral sample.
    pub fn apply_k_space(
        &self,
        spectrum: &mut Array3<Complex64>,
        wavenumbers: &Array3<Wavenumber>,
        dt: Time,
    ) -> KwaversResult<()> {
        if spectrum.shape() != wavenumbers.shape()
            || !dt.into_base().is_finite()
            || dt.into_base() < 0.0
        {
            return Err(KwaversError::InvalidInput(
                "Dispersion requires matching shapes and a finite nonnegative timestep".into(),
            ));
        }
        let mut phases = Vec::with_capacity(wavenumbers.len());
        for &k in wavenumbers.iter() {
            let frequency =
                Wavenumber::from_base(k.into_base().abs()) * (self.reference_speed / TWO_PI);
            let speed = self.law.phase_velocity(frequency, self.reference_speed)?;
            let angle = (k * (speed - self.reference_speed) * dt).into_base();
            if !angle.is_finite() {
                return Err(KwaversError::InvalidInput(
                    "Spectral phase shift exceeds scalar range".into(),
                ));
            }
            phases.push(angle);
        }
        let samples = spectrum
            .try_iter_mut()
            .map_err(|e| KwaversError::InvalidInput(e.to_string()))?;
        for (sample, angle) in samples.zip(phases) {
            if angle != 0.0 {
                *sample *= Complex64::from_polar(1.0, angle);
            }
        }
        Ok(())
    }

    /// Angular wavenumber k = 2πf/cₚ, in radians per metre.
    ///
    /// # Errors
    /// Rejects an invalid phase branch or an unrepresentable wavenumber.
    pub fn dispersion_relation(&self, frequency: Frequency) -> KwaversResult<Wavenumber> {
        let speed = self.law.phase_velocity(frequency, self.reference_speed)?;
        if frequency.into_base() == 0.0 {
            return Ok(Wavenumber::from_base(0.0));
        }
        let k = (frequency / speed) * TWO_PI;
        if !k.into_base().is_finite() {
            return Err(KwaversError::InvalidInput(
                "Angular wavenumber exceeds scalar range".into(),
            ));
        }
        Ok(k)
    }
}
