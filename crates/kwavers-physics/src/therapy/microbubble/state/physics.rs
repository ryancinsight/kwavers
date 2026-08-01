use super::MicrobubbleState;
use aequitas::systems::si::quantities::{Area, Dimensionless, Energy, Frequency, Mass, Volume};
use kwavers_core::constants::fundamental::{ATMOSPHERIC_PRESSURE, DENSITY_WATER_NOMINAL};
use kwavers_core::constants::numerical::{FOUR_PI, TWO_PI};
use kwavers_core::constants::thermodynamic::HEAT_CAPACITY_RATIO_DIATOMIC;
use kwavers_core::error::{KwaversError, KwaversResult, ValidationError};

impl MicrobubbleState {
    #[must_use]
    pub fn volume(&self) -> Volume<f64> {
        Volume::from_base((4.0 / 3.0) * std::f64::consts::PI * self.radius.into_base().powi(3))
    }

    #[must_use]
    pub fn surface_area(&self) -> Area<f64> {
        Area::from_base(FOUR_PI * self.radius.into_base().powi(2))
    }

    #[must_use]
    pub fn compression_ratio(&self) -> Dimensionless<f64> {
        Dimensionless::from_base(self.radius.into_base() / self.radius_equilibrium.into_base())
    }

    #[must_use]
    pub fn is_compressed(&self) -> bool {
        self.radius < self.radius_equilibrium
    }

    #[must_use]
    pub fn is_expanded(&self) -> bool {
        self.radius > self.radius_equilibrium
    }

    /// Inertial cavitation criterion: compression ratio > 2 (radius doubles).
    #[must_use]
    pub fn is_cavitating(&self) -> bool {
        self.compression_ratio().into_base() > 2.0
    }

    /// Kinetic energy of oscillating bubble wall (J).
    #[must_use]
    pub fn kinetic_energy(&self) -> Energy<f64> {
        let mass_effective = FOUR_PI * DENSITY_WATER_NOMINAL * self.radius.into_base().powi(3);
        Energy::from_base(0.5 * mass_effective * self.wall_velocity.into_base().powi(2))
    }

    /// Potential energy relative to equilibrium (J).
    ///
    /// Brennen (1995), §4.1, Eq. (4.7): E_pot = P₀V₀{(V/V₀)−1 + [(V₀/V)^{γ−1}−1]/(γ−1)}
    #[must_use]
    pub fn potential_energy(&self) -> Energy<f64> {
        let r0 = self.radius_equilibrium.into_base().max(f64::EPSILON);
        let r = self.radius.into_base().max(f64::EPSILON);
        let p0 = ATMOSPHERIC_PRESSURE;
        let gamma = HEAT_CAPACITY_RATIO_DIATOMIC;

        let v0 = (4.0 / 3.0) * std::f64::consts::PI * r0.powi(3);
        let v = (4.0 / 3.0) * std::f64::consts::PI * r.powi(3);
        let v_ratio = v / v0;

        let ambient_term = v_ratio - 1.0;
        let gas_term = ((1.0 / v_ratio).powf(gamma - 1.0) - 1.0) / (gamma - 1.0);

        Energy::from_base(p0 * v0 * (ambient_term + gas_term))
    }

    #[must_use]
    pub fn total_energy(&self) -> Energy<f64> {
        self.kinetic_energy() + self.potential_energy()
    }

    /// Minnaert resonance frequency: f₀ = (1/2πR₀)√(3γP₀/ρ) (Hz).
    #[must_use]
    pub fn resonance_frequency(&self) -> Frequency<f64> {
        let numerator =
            3.0 * HEAT_CAPACITY_RATIO_DIATOMIC * ATMOSPHERIC_PRESSURE / DENSITY_WATER_NOMINAL;
        Frequency::from_base(numerator.sqrt() / (TWO_PI * self.radius_equilibrium.into_base()))
    }

    #[must_use]
    pub fn drug_mass(&self) -> Mass<f64> {
        Mass::from_base(self.drug_concentration.into_base() * self.volume().into_base())
    }

    /// Drug remaining fraction.
    /// # Errors
    /// - Returns [`Err`] if an internal constraint is violated.
    ///
    #[must_use]
    pub fn drug_remaining_fraction(&self) -> Dimensionless<f64> {
        let initial_mass = self.drug_concentration.into_base()
            * (4.0 / 3.0)
            * std::f64::consts::PI
            * self.radius_equilibrium.into_base().powi(3);
        let fraction = if initial_mass > 0.0 {
            1.0 - (self.drug_released_total.into_base() / initial_mass)
        } else {
            0.0
        };
        Dimensionless::from_base(fraction)
    }
    /// Validate.
    /// # Errors
    /// - Returns [`KwaversError::Validation`] if the precondition for a Validation-class constraint is violated.
    ///
    pub fn validate(&self) -> KwaversResult<()> {
        if self.radius.into_base() <= 0.0 {
            return Err(KwaversError::Validation(ValidationError::InvalidValue {
                parameter: "radius".to_owned(),
                value: self.radius.into_base(),
                reason: "must be positive".to_owned(),
            }));
        }
        if self.radius_equilibrium.into_base() <= 0.0 {
            return Err(KwaversError::Validation(ValidationError::InvalidValue {
                parameter: "radius_equilibrium".to_owned(),
                value: self.radius_equilibrium.into_base(),
                reason: "must be positive".to_owned(),
            }));
        }
        if self.temperature.into_base() <= 0.0 {
            return Err(KwaversError::Validation(ValidationError::InvalidValue {
                parameter: "temperature".to_owned(),
                value: self.temperature.into_base(),
                reason: "must be positive (Kelvin)".to_owned(),
            }));
        }
        if self.pressure_internal.into_base() < 0.0 {
            return Err(KwaversError::Validation(ValidationError::InvalidValue {
                parameter: "pressure_internal".to_owned(),
                value: self.pressure_internal.into_base(),
                reason: "must be non-negative".to_owned(),
            }));
        }
        if self.drug_concentration.into_base() < 0.0 {
            return Err(KwaversError::Validation(ValidationError::InvalidValue {
                parameter: "drug_concentration".to_owned(),
                value: self.drug_concentration.into_base(),
                reason: "must be non-negative".to_owned(),
            }));
        }

        Ok(())
    }
}
