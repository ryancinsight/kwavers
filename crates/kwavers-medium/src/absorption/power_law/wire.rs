use super::*;

#[derive(Serialize, Deserialize)]
pub(super) struct PowerLawWire {
    alpha_0: f64,
    y: f64,
    f_ref: f64,
    dispersion_correction: bool,
}

impl TryFrom<PowerLawWire> for PowerLawAbsorption {
    type Error = KwaversError;

    fn try_from(wire: PowerLawWire) -> KwaversResult<Self> {
        Self::new(
            reference_loss_from_decibels_per_centimetre(wire.alpha_0),
            Frequency::from_base(wire.f_ref),
            Dimensionless::from_base(wire.y),
            wire.dispersion_correction,
        )
    }
}

impl From<PowerLawAbsorption> for PowerLawWire {
    fn from(law: PowerLawAbsorption) -> Self {
        Self {
            alpha_0: law.reference_loss.into_base() / (DB_TO_NP / CM_TO_M),
            y: law.exponent.into_base(),
            f_ref: law.reference_frequency.into_base(),
            dispersion_correction: law.dispersion_correction,
        }
    }
}
