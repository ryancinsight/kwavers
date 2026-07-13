use kwavers_core::constants::{
    ACOUSTIC_ABSORPTION_TISSUE, DENSITY_WATER_NOMINAL, GRUNEISEN_WATER_37C,
    OPTICAL_ABSORPTION_TISSUE_NIR, REDUCED_SCATTERING_TISSUE_NIR, SOUND_SPEED_TISSUE,
    SPECIFIC_HEAT_WATER, THERMAL_CONDUCTIVITY_WATER,
};
use kwavers_core::error::{KwaversError, KwaversResult, ValidationError};

/// Physical coefficients for the coupled acoustic-optical-thermal system.
///
/// Contains material properties needed to evaluate the PDE residuals. Default
/// values correspond to soft biological tissue at 37 C.
#[derive(Debug, Clone)]
pub struct PhysicsCoefficients {
    /// Speed of sound [m/s].
    pub sound_speed: f64,
    /// Mass density [kg/m^3].
    pub density: f64,
    /// Specific heat capacity [J/(kg*K)].
    pub specific_heat: f64,
    /// Thermal conductivity [W/(m*K)].
    pub thermal_conductivity: f64,
    /// Optical absorption coefficient mu_a [1/m].
    pub optical_absorption: f64,
    /// Reduced scattering coefficient mu_s' [1/m].
    pub reduced_scattering: f64,
    /// Acoustic absorption coefficient [Np/m].
    pub acoustic_absorption: f64,
    /// Gruneisen parameter Gamma for photoacoustic source p0 = Gamma*mu_a*Phi.
    ///
    /// For water at 37 C: Gamma ~= 0.12. Treating Gamma = 1 overestimates
    /// photoacoustic amplitude by about 8x.
    ///
    /// Reference: Jacques, S.L. (1993). Appl. Opt. 32(13), 2447-2454.
    pub gruneisen: f64,
}

impl Default for PhysicsCoefficients {
    fn default() -> Self {
        Self {
            sound_speed: SOUND_SPEED_TISSUE,
            density: DENSITY_WATER_NOMINAL,
            specific_heat: SPECIFIC_HEAT_WATER,
            thermal_conductivity: THERMAL_CONDUCTIVITY_WATER,
            optical_absorption: OPTICAL_ABSORPTION_TISSUE_NIR,
            reduced_scattering: REDUCED_SCATTERING_TISSUE_NIR,
            acoustic_absorption: ACOUSTIC_ABSORPTION_TISSUE,
            gruneisen: GRUNEISEN_WATER_37C,
        }
    }
}

impl PhysicsCoefficients {
    /// Thermal diffusivity `kappa = k / (rho * c_p)`.
    #[must_use]
    pub fn thermal_diffusivity(&self) -> f64 {
        self.thermal_conductivity / (self.density * self.specific_heat)
    }

    /// Optical diffusion coefficient `D = 1 / (3 * (mu_a + mu_s'))`.
    #[must_use]
    pub fn optical_diffusion(&self) -> f64 {
        1.0 / (3.0 * (self.optical_absorption + self.reduced_scattering))
    }

    /// Validate the coefficient domain required by residual assembly.
    ///
    /// Positivity is required for denominators in thermal diffusivity,
    /// acoustic intensity, and heat-capacity conversion. Absorption and
    /// scattering coefficients are nonnegative; at least one optical transport
    /// coefficient must be positive so diffusion has a finite denominator.
    ///
    /// # Errors
    /// - Returns [`crate::KwaversError::Validation`] if a coefficient is nonfinite or
    ///   outside the physical domain required by the PDE residual.
    pub(in crate::multiphysics::monolithic) fn validate(&self) -> KwaversResult<()> {
        validate_positive("PhysicsCoefficients::sound_speed", self.sound_speed)?;
        validate_positive("PhysicsCoefficients::density", self.density)?;
        validate_positive("PhysicsCoefficients::specific_heat", self.specific_heat)?;
        validate_nonnegative(
            "PhysicsCoefficients::thermal_conductivity",
            self.thermal_conductivity,
        )?;
        validate_nonnegative(
            "PhysicsCoefficients::optical_absorption",
            self.optical_absorption,
        )?;
        validate_nonnegative(
            "PhysicsCoefficients::reduced_scattering",
            self.reduced_scattering,
        )?;
        validate_nonnegative(
            "PhysicsCoefficients::acoustic_absorption",
            self.acoustic_absorption,
        )?;

        if !self.gruneisen.is_finite() {
            return Err(KwaversError::Validation(ValidationError::InvalidValue {
                parameter: "PhysicsCoefficients::gruneisen".to_owned(),
                value: self.gruneisen,
                reason: "must be finite".to_owned(),
            }));
        }

        let optical_transport = self.optical_absorption + self.reduced_scattering;
        if optical_transport <= 0.0 {
            return Err(KwaversError::Validation(
                ValidationError::InvalidParameter {
                    parameter: "PhysicsCoefficients::optical_transport".to_owned(),
                    reason: "optical_absorption + reduced_scattering must be positive".to_owned(),
                },
            ));
        }

        Ok(())
    }
}

fn validate_positive(parameter: &str, value: f64) -> KwaversResult<()> {
    if !value.is_finite() || value <= 0.0 {
        return Err(KwaversError::Validation(ValidationError::InvalidValue {
            parameter: parameter.to_owned(),
            value,
            reason: "must be finite and positive".to_owned(),
        }));
    }

    Ok(())
}

fn validate_nonnegative(parameter: &str, value: f64) -> KwaversResult<()> {
    if !value.is_finite() || value < 0.0 {
        return Err(KwaversError::Validation(ValidationError::InvalidValue {
            parameter: parameter.to_owned(),
            value,
            reason: "must be finite and nonnegative".to_owned(),
        }));
    }

    Ok(())
}
