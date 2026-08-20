use aequitas::systems::si::quantities::{
    AcousticImpedance, Dimensionless, Frequency, Length, MassDensity, ReciprocalLength, Velocity,
};
use aequitas::systems::si::units::Megahertz;
use core::mem::size_of;
use kwavers_core::constants::acoustic_parameters::{
    BONE_DENSITY, DENSITY_SKULL_SUTURE, DENSITY_SKULL_TRABECULAR, SHEAR_SPEED_SKULL_CORTICAL,
    SHEAR_SPEED_SKULL_TRABECULAR, SOUND_SPEED_SKULL_CORTICAL, SOUND_SPEED_SKULL_SUTURE,
    SOUND_SPEED_SKULL_TRABECULAR,
};
use kwavers_core::error::{KwaversError, KwaversResult};

/// Validated acoustic configuration for a skull-bone phase.
///
/// This is the skull workflow's material-configuration single source of truth.
/// Physical values remain Aequitas quantities until a numerical kernel extracts
/// their base-unit representation.
///
/// # Examples
///
/// ```
/// use kwavers_physics::acoustics::skull::AcousticSkullProperties;
///
/// let cortical = AcousticSkullProperties::cortical();
/// assert_eq!(cortical.sound_speed().into_base(), 3100.0);
/// ```
#[derive(Debug, Clone, PartialEq)]
#[must_use]
pub struct AcousticSkullProperties {
    sound_speed: Velocity<f64>,
    density: MassDensity<f64>,
    attenuation_at_one_megahertz: ReciprocalLength<f64>,
    thickness: Length<f64>,
    shear_speed: Option<Velocity<f64>>,
}

impl Default for AcousticSkullProperties {
    fn default() -> Self {
        Self::cortical()
    }
}

impl AcousticSkullProperties {
    /// Constructs a validated skull-bone configuration.
    ///
    /// # Errors
    ///
    /// Returns [`KwaversError::InvalidInput`] when a required value is
    /// non-finite or non-positive, or when attenuation is negative.
    pub fn new(
        sound_speed: Velocity<f64>,
        density: MassDensity<f64>,
        attenuation_at_one_megahertz: ReciprocalLength<f64>,
        thickness: Length<f64>,
        shear_speed: Option<Velocity<f64>>,
    ) -> KwaversResult<Self> {
        validate_positive("skull sound speed", *sound_speed.as_base())?;
        validate_positive("skull density", *density.as_base())?;
        validate_non_negative(
            "skull attenuation at 1 MHz",
            *attenuation_at_one_megahertz.as_base(),
        )?;
        validate_positive("skull thickness", *thickness.as_base())?;
        if let Some(speed) = shear_speed {
            validate_positive("skull shear speed", *speed.as_base())?;
        }

        Ok(Self {
            sound_speed,
            density,
            attenuation_at_one_megahertz,
            thickness,
            shear_speed,
        })
    }

    /// Returns the canonical adult cortical-skull configuration.
    pub const fn cortical() -> Self {
        Self {
            sound_speed: Velocity::from_base(SOUND_SPEED_SKULL_CORTICAL),
            density: MassDensity::from_base(BONE_DENSITY),
            attenuation_at_one_megahertz: ReciprocalLength::from_base(60.0),
            thickness: Length::from_base(0.007),
            shear_speed: Some(Velocity::from_base(SHEAR_SPEED_SKULL_CORTICAL)),
        }
    }

    /// Returns the canonical trabecular-skull configuration.
    pub const fn trabecular() -> Self {
        Self {
            sound_speed: Velocity::from_base(SOUND_SPEED_SKULL_TRABECULAR),
            density: MassDensity::from_base(DENSITY_SKULL_TRABECULAR),
            attenuation_at_one_megahertz: ReciprocalLength::from_base(40.0),
            thickness: Length::from_base(0.005),
            shear_speed: Some(Velocity::from_base(SHEAR_SPEED_SKULL_TRABECULAR)),
        }
    }

    /// Returns the canonical cranial-suture configuration.
    pub const fn suture() -> Self {
        Self {
            sound_speed: Velocity::from_base(SOUND_SPEED_SKULL_SUTURE),
            density: MassDensity::from_base(DENSITY_SKULL_SUTURE),
            attenuation_at_one_megahertz: ReciprocalLength::from_base(20.0),
            thickness: Length::from_base(0.002),
            shear_speed: None,
        }
    }

    /// Creates skull properties for a named bone type.
    ///
    /// # Errors
    ///
    /// Returns [`KwaversError::InvalidInput`] when `bone_type` is not
    /// `"cortical"`, `"trabecular"`, or `"suture"`.
    pub fn from_bone_type(bone_type: &str) -> KwaversResult<Self> {
        match bone_type {
            "cortical" => Ok(Self::cortical()),
            "trabecular" => Ok(Self::trabecular()),
            "suture" => Ok(Self::suture()),
            _ => Err(KwaversError::InvalidInput(format!(
                "unknown skull bone type: {bone_type}"
            ))),
        }
    }

    /// Returns the longitudinal sound speed.
    #[must_use]
    pub const fn sound_speed(&self) -> Velocity<f64> {
        self.sound_speed
    }

    /// Returns the mass density.
    #[must_use]
    pub const fn density(&self) -> MassDensity<f64> {
        self.density
    }

    /// Returns attenuation at the 1 MHz reference frequency.
    #[must_use]
    pub const fn attenuation_at_one_megahertz(&self) -> ReciprocalLength<f64> {
        self.attenuation_at_one_megahertz
    }

    /// Returns the representative skull thickness.
    #[must_use]
    pub const fn thickness(&self) -> Length<f64> {
        self.thickness
    }

    /// Returns the shear-wave speed when the phase supports shear propagation.
    #[must_use]
    pub const fn shear_speed(&self) -> Option<Velocity<f64>> {
        self.shear_speed
    }

    /// Returns acoustic impedance `Z = rho * c`.
    #[must_use]
    pub fn acoustic_impedance(&self) -> AcousticImpedance<f64> {
        self.density * self.sound_speed
    }

    /// Calculates the intensity transmission coefficient at normal incidence.
    ///
    /// The lossless planar-interface law is
    /// `T_I = 4 Z_1 Z_2 / (Z_1 + Z_2)^2`.
    ///
    /// # Errors
    ///
    /// Returns [`KwaversError::InvalidInput`] when the incident impedance is
    /// non-finite or non-positive.
    pub fn transmission_coefficient(
        &self,
        incident_impedance: AcousticImpedance<f64>,
    ) -> KwaversResult<Dimensionless<f64>> {
        let incident = *incident_impedance.as_base();
        validate_positive("incident acoustic impedance", incident)?;
        let skull = *self.acoustic_impedance().as_base();
        Ok(Dimensionless::from_base(
            4.0 * incident * skull / (incident + skull).powi(2),
        ))
    }

    /// Calculates the reciprocal-length attenuation at `frequency`.
    ///
    /// The skull presets use a linear frequency law referenced to 1 MHz.
    ///
    /// # Errors
    ///
    /// Returns [`KwaversError::InvalidInput`] when frequency is non-finite or
    /// negative.
    pub fn attenuation_at_frequency(
        &self,
        frequency: Frequency<f64>,
    ) -> KwaversResult<ReciprocalLength<f64>> {
        let frequency_megahertz = frequency.in_unit::<Megahertz>();
        validate_non_negative("attenuation frequency", frequency_megahertz)?;
        Ok(ReciprocalLength::from_base(
            *self.attenuation_at_one_megahertz.as_base() * frequency_megahertz,
        ))
    }
}

fn validate_positive(name: &str, value: f64) -> KwaversResult<()> {
    if value.is_finite() && value > 0.0 {
        Ok(())
    } else {
        Err(KwaversError::InvalidInput(format!(
            "{name} must be finite and positive; got {value}"
        )))
    }
}

fn validate_non_negative(name: &str, value: f64) -> KwaversResult<()> {
    if value.is_finite() && value >= 0.0 {
        Ok(())
    } else {
        Err(KwaversError::InvalidInput(format!(
            "{name} must be finite and non-negative; got {value}"
        )))
    }
}

const _: () = assert!(
    size_of::<AcousticSkullProperties>()
        == size_of::<Velocity<f64>>()
            + size_of::<MassDensity<f64>>()
            + size_of::<ReciprocalLength<f64>>()
            + size_of::<Length<f64>>()
            + size_of::<Option<Velocity<f64>>>()
);
