use super::MicrobubbleState;
use crate::core::constants::fundamental::{ATMOSPHERIC_PRESSURE, DENSITY_WATER_NOMINAL};
use crate::core::error::{KwaversError, KwaversResult, ValidationError};

impl MicrobubbleState {
    #[must_use]
    pub fn volume(&self) -> f64 {
        (4.0 / 3.0) * std::f64::consts::PI * self.radius.powi(3)
    }

    #[must_use]
    pub fn surface_area(&self) -> f64 {
        4.0 * std::f64::consts::PI * self.radius.powi(2)
    }

    #[must_use]
    pub fn compression_ratio(&self) -> f64 {
        self.radius / self.radius_equilibrium
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
        self.compression_ratio() > 2.0
    }

    /// Kinetic energy of oscillating bubble wall (J).
    #[must_use]
    pub fn kinetic_energy(&self) -> f64 {
        let mass_effective =
            4.0 * std::f64::consts::PI * DENSITY_WATER_NOMINAL * self.radius.powi(3);
        0.5 * mass_effective * self.wall_velocity.powi(2)
    }

    /// Potential energy relative to equilibrium (J).
    ///
    /// Brennen (1995), §4.1, Eq. (4.7): E_pot = P₀V₀{(V/V₀)−1 + [(V₀/V)^{γ−1}−1]/(γ−1)}
    #[must_use]
    pub fn potential_energy(&self) -> f64 {
        const POLYTROPIC_INDEX: f64 = 1.4;

        let r0 = self.radius_equilibrium.max(f64::EPSILON);
        let r = self.radius.max(f64::EPSILON);
        let p0 = ATMOSPHERIC_PRESSURE;
        let gamma = POLYTROPIC_INDEX;

        let v0 = (4.0 / 3.0) * std::f64::consts::PI * r0.powi(3);
        let v = (4.0 / 3.0) * std::f64::consts::PI * r.powi(3);
        let v_ratio = v / v0;

        let ambient_term = v_ratio - 1.0;
        let gas_term = ((1.0 / v_ratio).powf(gamma - 1.0) - 1.0) / (gamma - 1.0);

        p0 * v0 * (ambient_term + gas_term)
    }

    #[must_use]
    pub fn total_energy(&self) -> f64 {
        self.kinetic_energy() + self.potential_energy()
    }

    /// Minnaert resonance frequency: f₀ = (1/2πR₀)√(3γP₀/ρ) (Hz).
    #[must_use]
    pub fn resonance_frequency(&self) -> f64 {
        const POLYTROPIC_INDEX: f64 = 1.4;

        let numerator = 3.0 * POLYTROPIC_INDEX * ATMOSPHERIC_PRESSURE / DENSITY_WATER_NOMINAL;
        numerator.sqrt() / (2.0 * std::f64::consts::PI * self.radius_equilibrium)
    }

    #[must_use]
    pub fn drug_mass(&self) -> f64 {
        self.drug_concentration * self.volume()
    }

    /// Drug remaining fraction.
    /// # Errors
    /// - Returns [`Err`] if an internal constraint is violated.
    ///
    #[must_use]
    pub fn drug_remaining_fraction(&self) -> f64 {
        let initial_mass = self.drug_concentration
            * (4.0 / 3.0)
            * std::f64::consts::PI
            * self.radius_equilibrium.powi(3);
        if initial_mass > 0.0 {
            1.0 - (self.drug_released_total / initial_mass)
        } else {
            0.0
        }
    }
    /// Validate.
    /// # Errors
    /// - Returns [`KwaversError::Validation`] if the precondition for a Validation-class constraint is violated.
    ///
    pub fn validate(&self) -> KwaversResult<()> {
        if self.radius <= 0.0 {
            return Err(KwaversError::Validation(ValidationError::InvalidValue {
                parameter: "radius".to_owned(),
                value: self.radius,
                reason: "must be positive".to_owned(),
            }));
        }
        if self.radius_equilibrium <= 0.0 {
            return Err(KwaversError::Validation(ValidationError::InvalidValue {
                parameter: "radius_equilibrium".to_owned(),
                value: self.radius_equilibrium,
                reason: "must be positive".to_owned(),
            }));
        }
        if self.temperature <= 0.0 {
            return Err(KwaversError::Validation(ValidationError::InvalidValue {
                parameter: "temperature".to_owned(),
                value: self.temperature,
                reason: "must be positive (Kelvin)".to_owned(),
            }));
        }
        if self.pressure_internal < 0.0 {
            return Err(KwaversError::Validation(ValidationError::InvalidValue {
                parameter: "pressure_internal".to_owned(),
                value: self.pressure_internal,
                reason: "must be non-negative".to_owned(),
            }));
        }
        if self.drug_concentration < 0.0 {
            return Err(KwaversError::Validation(ValidationError::InvalidValue {
                parameter: "drug_concentration".to_owned(),
                value: self.drug_concentration,
                reason: "must be non-negative".to_owned(),
            }));
        }

        Ok(())
    }
}
