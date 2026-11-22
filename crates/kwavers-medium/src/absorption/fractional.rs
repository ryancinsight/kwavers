//! Fractional time derivatives and calibrated spatial absorption.
use super::PowerLawAbsorption;
use aequitas::systems::si::quantities::{
    Dimensionless, Frequency, Length, ReciprocalLength, Time, Velocity, Wavenumber,
};
use eunomia::Complex64;
use kwavers_core::constants::numerical::TWO_PI;
use kwavers_core::error::{KwaversError, KwaversResult};
use leto::Array3;

mod derivative;
pub use derivative::FractionalDerivative;

/// Homogeneous Fourier-mode damping with amplitude loss calibrated at a reference frequency.
#[derive(Debug, Clone, Copy)]
pub struct FractionalLaplacian {
    law: PowerLawAbsorption,
}

impl FractionalLaplacian {
    /// Construct α(f) = α_ref (f/f_ref)^y.
    ///
    /// # Errors
    /// Requires finite calibration, positive reference frequency and nonnegative loss and order.
    pub fn new(
        reference_loss: ReciprocalLength,
        reference_frequency: Frequency,
        exponent: Dimensionless,
    ) -> KwaversResult<Self> {
        Ok(Self {
            law: PowerLawAbsorption::new(reference_loss, reference_frequency, exponent, false)?,
        })
    }

    /// Return the calibrated amplitude loss per unit length.
    ///
    /// # Errors
    /// Rejects invalid frequencies and unrepresentable attenuation.
    pub fn absorption_at_frequency(&self, frequency: Frequency) -> KwaversResult<ReciprocalLength> {
        self.law.absorption_at_frequency(frequency)
    }

    /// Bind the propagation distance c Δt once for a set of modes.
    ///
    /// # Errors
    /// Requires positive finite speed, finite nonnegative time and representable distance.
    pub fn step(&self, speed: Velocity, elapsed: Time) -> KwaversResult<FractionalAbsorptionStep> {
        if !speed.into_base().is_finite()
            || speed.into_base() <= 0.0
            || !elapsed.into_base().is_finite()
            || elapsed.into_base() < 0.0
        {
            return Err(KwaversError::InvalidInput(
                "Absorption requires positive finite speed and finite nonnegative elapsed time"
                    .into(),
            ));
        }
        let distance = speed * elapsed;
        if !distance.into_base().is_finite()
            || (distance.into_base() == 0.0 && elapsed.into_base() != 0.0)
        {
            return Err(KwaversError::InvalidInput(
                "Propagation distance is outside the representable range".into(),
            ));
        }
        Ok(FractionalAbsorptionStep {
            model: *self,
            speed,
            distance,
        })
    }
}

/// Exact amplitude evolution of dp̂/dt = −c α(c|k|/(2π)) p̂ for a fixed timestep.
#[derive(Debug, Clone, Copy)]
pub struct FractionalAbsorptionStep {
    model: FractionalLaplacian,
    speed: Velocity,
    distance: Length,
}

impl FractionalAbsorptionStep {
    /// Evaluate exp(−α·distance). For y=0 the calibrated loss also applies at DC.
    ///
    /// # Errors
    /// Rejects negative or nonfinite magnitudes and unrepresentable modal frequencies or losses.
    #[inline]
    pub fn gain(&self, magnitude: Wavenumber) -> KwaversResult<Dimensionless> {
        if !magnitude.into_base().is_finite() || magnitude.into_base() < 0.0 {
            return Err(KwaversError::InvalidInput(
                "Angular wavenumber magnitude must be finite and nonnegative".into(),
            ));
        }
        if self.distance.into_base() == 0.0 {
            return Ok(Dimensionless::from_base(1.0));
        }
        let omega = self.speed * magnitude;
        // If the product overflows, dividing first can still recover a finite frequency.
        let frequency = if omega.into_base().is_finite() {
            omega / TWO_PI
        } else {
            (self.speed / TWO_PI) * magnitude
        };
        if frequency.into_base() == 0.0 && magnitude.into_base() > 0.0 {
            return Err(KwaversError::InvalidInput(
                "Positive modal frequency is below the representable range".into(),
            ));
        }
        let loss = self.model.absorption_at_frequency(frequency)?;
        Ok(Dimensionless::from_base(
            (-(loss * self.distance).into_base()).exp(),
        ))
    }

    /// Multiply each complex Fourier coefficient by its real modal gain.
    ///
    /// # Errors
    /// Shapes and all modes are checked before the first coefficient is modified.
    pub fn apply_k_space(
        &self,
        spectrum: &mut Array3<Complex64>,
        magnitudes: &Array3<Wavenumber>,
    ) -> KwaversResult<()> {
        if spectrum.shape() != magnitudes.shape() {
            return Err(KwaversError::InvalidInput(
                "Wavenumber and spectrum shapes must match".into(),
            ));
        }
        let gains = magnitudes
            .iter()
            .map(|&k| self.gain(k))
            .collect::<KwaversResult<Vec<_>>>()?;
        let samples = spectrum
            .try_iter_mut()
            .map_err(|e| KwaversError::InvalidInput(e.to_string()))?;
        for (sample, gain) in samples.zip(gains) {
            *sample *= gain.into_base();
        }
        Ok(())
    }
}
