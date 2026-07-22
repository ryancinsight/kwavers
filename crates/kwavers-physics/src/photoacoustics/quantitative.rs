use aequitas::systems::si::{quantities::ReciprocalLength, units::JoulePerSquareMeter};
use hyperion::{
    coefficient::{Absorption, EffectiveAttenuation, InteractionCoefficient},
    quantity::{EnergyFluence, PathLength},
    TransportError,
};
use thiserror::Error;

/// Failure while converting a photoacoustic signal to an absorption estimate.
#[derive(Clone, Copy, Debug, Error, PartialEq)]
pub enum QuantitativePhotoacousticError {
    /// Hyperion rejected an optical input or derived coefficient.
    #[error(transparent)]
    Transport(#[from] TransportError<f64>),
    /// The signal amplitude is negative or non-finite.
    #[error("photoacoustic signal must be finite and non-negative, got {value}")]
    InvalidSignal {
        /// Rejected pressure amplitude in pascals.
        value: f64,
    },
    /// The Gruneisen coefficient is non-positive or non-finite.
    #[error("Gruneisen coefficient must be finite and positive, got {value}")]
    InvalidGrueneisen {
        /// Rejected dimensionless coefficient.
        value: f64,
    },
    /// A zero fluence cannot support quantitative absorption recovery.
    #[error("energy fluence must be positive for absorption recovery")]
    ZeroFluence,
}

/// Return the depth-biased absorption inferred without fluence compensation.
///
/// The inference law is `mu_a_tilde = mu_a exp(-mu_eff z)`. Hyperion owns
/// optical depth and transmission; this function owns their photoacoustic
/// interpretation as an apparent absorption coefficient.
///
/// # Errors
///
/// Returns a Hyperion transport error when optical depth or the derived
/// coefficient is non-finite.
pub fn apparent_absorption(
    absorption: InteractionCoefficient<f64, Absorption>,
    effective_attenuation: InteractionCoefficient<f64, EffectiveAttenuation>,
    depth: PathLength<f64>,
) -> Result<InteractionCoefficient<f64, Absorption>, TransportError<f64>> {
    let transmission = effective_attenuation
        .optical_depth(depth)?
        .transmission()
        .into_quantity()
        .into_base();
    InteractionCoefficient::new(ReciprocalLength::from_base(
        absorption.into_quantity().into_base() * transmission,
    ))
}

/// Recover absorption from a Gruneisen-weighted photoacoustic signal.
///
/// Uses `mu_a = signal / (Gamma F)`, where pressure and energy-fluence units
/// imply reciprocal length because one pascal equals one joule per cubic metre.
///
/// # Errors
///
/// Rejects invalid signal/Gruneisen values, zero fluence, and any non-finite
/// absorption result.
pub fn compensate_fluence(
    signal_pa: f64,
    gruneisen: f64,
    fluence: EnergyFluence<f64>,
) -> Result<InteractionCoefficient<f64, Absorption>, QuantitativePhotoacousticError> {
    if !signal_pa.is_finite() || signal_pa < 0.0 {
        return Err(QuantitativePhotoacousticError::InvalidSignal { value: signal_pa });
    }
    if !gruneisen.is_finite() || gruneisen <= 0.0 {
        return Err(QuantitativePhotoacousticError::InvalidGrueneisen { value: gruneisen });
    }
    let fluence_value = fluence.in_unit::<JoulePerSquareMeter>();
    if fluence_value == 0.0 {
        return Err(QuantitativePhotoacousticError::ZeroFluence);
    }
    Ok(InteractionCoefficient::new(ReciprocalLength::from_base(
        signal_pa / (gruneisen * fluence_value),
    ))?)
}

#[cfg(test)]
mod tests {
    use aequitas::systems::si::{
        quantities::{EnergyPerArea, Length, ReciprocalLength},
        units::PerMeter,
    };
    use hyperion::{
        coefficient::{Absorption, EffectiveAttenuation, InteractionCoefficient},
        quantity::{EnergyFluence, PathLength},
    };

    use super::{apparent_absorption, compensate_fluence};

    #[test]
    fn apparent_absorption_has_one_over_e_bias_at_one_penetration_depth() {
        let absorption =
            InteractionCoefficient::<_, Absorption>::new(ReciprocalLength::from_base(1.0)).unwrap();
        let attenuation = InteractionCoefficient::<_, EffectiveAttenuation>::new(
            ReciprocalLength::from_base(2.0),
        )
        .unwrap();
        let depth = PathLength::new(Length::from_base(0.5)).unwrap();
        let apparent = apparent_absorption(absorption, attenuation, depth)
            .unwrap()
            .in_unit::<PerMeter>();
        let expected = 1.0 / std::f64::consts::E;
        // One exponential evaluation and one multiply are bounded here by
        // sixteen binary64 epsilons around the unit-scale reference.
        assert!((apparent - expected).abs() <= 16.0 * f64::EPSILON);
    }

    #[test]
    fn fluence_compensation_recovers_absorption_and_rejects_zero_fluence() {
        let fluence = EnergyFluence::new(EnergyPerArea::from_base(1_500.0)).unwrap();
        let recovered = compensate_fluence(2_250.0, 0.2, fluence)
            .unwrap()
            .in_unit::<PerMeter>();
        assert_eq!(recovered, 7.5);

        let zero = EnergyFluence::new(EnergyPerArea::from_base(0.0)).unwrap();
        assert!(compensate_fluence(1.0, 0.2, zero).is_err());
    }
}
