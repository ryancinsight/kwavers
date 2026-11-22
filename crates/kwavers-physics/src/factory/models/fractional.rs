//! Wire migration for homogeneous fractional absorption calibrations.
use aequitas::systems::si::quantities::{Dimensionless, Frequency, ReciprocalLength};
use serde::{Deserialize, Deserializer};

pub(super) fn read_calibration<'de, D: Deserializer<'de>>(
    deserializer: D,
) -> Result<(ReciprocalLength, Frequency, Dimensionless), D::Error> {
    if !deserializer.is_human_readable() {
        return Deserialize::deserialize(deserializer);
    }
    #[derive(Deserialize)]
    #[serde(untagged, deny_unknown_fields)]
    enum Calibration {
        Reference {
            reference_loss: ReciprocalLength,
            reference_frequency: Frequency,
            exponent: Dimensionless,
        },
        Angular {
            alpha0: f64,
            exponent: Dimensionless,
        },
    }
    Ok(match Calibration::deserialize(deserializer)? {
        Calibration::Reference {
            reference_loss,
            reference_frequency,
            exponent,
        } => (reference_loss, reference_frequency, exponent),
        // At ω=1 rad/s, α=α₀·1^y has the same numerical value for every order.
        Calibration::Angular { alpha0, exponent } => (
            ReciprocalLength::from_base(alpha0),
            Frequency::from_base(std::f64::consts::TAU.recip()),
            exponent,
        ),
    })
}
