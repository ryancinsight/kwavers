//! `MicrobubblePopulation` — bubble population with size distribution and scattering.

use super::microbubble::{Microbubble, CeusSizeDistribution};
use crate::core::error::{KwaversError, KwaversResult, ValidationError};

/// Population of microbubbles with size distribution
#[derive(Debug, Clone)]
pub struct MicrobubblePopulation {
    /// Reference microbubble (mean properties)
    pub reference_bubble: Microbubble,
    /// Size distribution parameters (log-normal)
    pub size_distribution: CeusSizeDistribution,
    /// Initial concentration (bubbles/m³)
    pub concentration: f64,
}

impl MicrobubblePopulation {
    /// Create new microbubble population
    /// # Errors
    /// - Returns [`Err`] if an internal constraint is violated.
    ///
    pub fn new(concentration: f64, mean_diameter: f64) -> KwaversResult<Self> {
        let reference_bubble = Microbubble::new(mean_diameter / 2.0, 1.5, 0.8);

        // Typical log-normal distribution for contrast agents
        let size_distribution = CeusSizeDistribution {
            mean_radius: reference_bubble.radius_eq,
            std_dev: reference_bubble.radius_eq * 0.3, // 30% coefficient of variation
        };

        Ok(Self {
            reference_bubble,
            size_distribution,
            concentration: concentration * 1e6, // Convert bubbles/mL to bubbles/m³
        })
    }

    /// Global concentration (simplified)
    /// # Errors
    /// - Returns [`Err`] if an internal constraint is violated.
    ///
    #[must_use]
    pub fn get_concentration(&self) -> f64 {
        self.concentration
    }
    /// Effective scattering.
    /// # Errors
    /// - Returns [`KwaversError::Validation`] if the precondition for a Validation-class constraint is violated.
    /// - Propagates any [`KwaversError`] returned by called functions.
    ///
    pub fn effective_scattering(
        &self,
        frequency: f64,
        ambient_pressure: f64,
        liquid_density: f64,
        sound_speed: f64,
        liquid_viscosity: f64,
    ) -> KwaversResult<f64> {
        if !frequency.is_finite() || frequency <= 0.0 {
            return Err(KwaversError::Validation(ValidationError::InvalidValue {
                parameter: "frequency".to_owned(),
                value: frequency,
                reason: "must be positive and finite".to_owned(),
            }));
        }
        if !ambient_pressure.is_finite() || ambient_pressure <= 0.0 {
            return Err(KwaversError::Validation(ValidationError::InvalidValue {
                parameter: "ambient_pressure".to_owned(),
                value: ambient_pressure,
                reason: "must be positive and finite".to_owned(),
            }));
        }
        if !liquid_density.is_finite() || liquid_density <= 0.0 {
            return Err(KwaversError::Validation(ValidationError::InvalidValue {
                parameter: "liquid_density".to_owned(),
                value: liquid_density,
                reason: "must be positive and finite".to_owned(),
            }));
        }
        if !sound_speed.is_finite() || sound_speed <= 0.0 {
            return Err(KwaversError::Validation(ValidationError::InvalidValue {
                parameter: "sound_speed".to_owned(),
                value: sound_speed,
                reason: "must be positive and finite".to_owned(),
            }));
        }
        if !liquid_viscosity.is_finite() || liquid_viscosity < 0.0 {
            return Err(KwaversError::Validation(ValidationError::InvalidValue {
                parameter: "liquid_viscosity".to_owned(),
                value: liquid_viscosity,
                reason: "must be non-negative and finite".to_owned(),
            }));
        }

        let mean_radius = self.size_distribution.mean_radius;
        let std_dev = self.size_distribution.std_dev;
        if !mean_radius.is_finite() || mean_radius <= 0.0 {
            return Err(KwaversError::Validation(ValidationError::InvalidValue {
                parameter: "mean_radius".to_owned(),
                value: mean_radius,
                reason: "must be positive and finite".to_owned(),
            }));
        }
        if !std_dev.is_finite() || std_dev < 0.0 {
            return Err(KwaversError::Validation(ValidationError::InvalidValue {
                parameter: "std_dev".to_owned(),
                value: std_dev,
                reason: "must be non-negative and finite".to_owned(),
            }));
        }

        let shell_elasticity = self.reference_bubble.shell_elasticity;
        let shell_viscosity = self.reference_bubble.shell_viscosity;

        let scattering_at_radius = |radius_eq: f64| -> KwaversResult<f64> {
            if !radius_eq.is_finite() || radius_eq <= 0.0 {
                return Err(KwaversError::Validation(ValidationError::InvalidValue {
                    parameter: "radius_eq".to_owned(),
                    value: radius_eq,
                    reason: "must be positive and finite".to_owned(),
                }));
            }

            let diameter_um = radius_eq * 2.0 * 1e6;
            let base_freq_mhz = 6.0 / diameter_um;
            let shell_factor = (shell_elasticity / 1000.0).mul_add(0.2, 1.0);
            let f0_hz = base_freq_mhz * shell_factor * 1e6;

            if !f0_hz.is_finite() || f0_hz <= 0.0 {
                return Err(KwaversError::Validation(ValidationError::InvalidValue {
                    parameter: "resonance_frequency".to_owned(),
                    value: f0_hz,
                    reason: "must be positive and finite".to_owned(),
                }));
            }

            let omega = 2.0 * std::f64::consts::PI * frequency;
            let omega0 = 2.0 * std::f64::consts::PI * f0_hz;

            let effective_viscosity = liquid_viscosity + shell_viscosity;
            let delta_visc = 4.0 * effective_viscosity / (liquid_density * radius_eq * radius_eq);
            let delta_rad = omega0 * omega0 * radius_eq / sound_speed;
            let delta = delta_visc + delta_rad;

            let denom =
                (delta * omega).mul_add(delta * omega, (omega0 * omega0 - omega * omega).powi(2));
            if denom <= 0.0 || !denom.is_finite() {
                return Err(KwaversError::Validation(ValidationError::InvalidValue {
                    parameter: "scattering_denom".to_owned(),
                    value: denom,
                    reason: "must be positive and finite".to_owned(),
                }));
            }

            let sigma = 4.0 * std::f64::consts::PI * omega.powi(4) * radius_eq.powi(2) / denom;
            if !sigma.is_finite() || sigma < 0.0 {
                return Err(KwaversError::Validation(ValidationError::InvalidValue {
                    parameter: "scattering_cross_section".to_owned(),
                    value: sigma,
                    reason: "must be non-negative and finite".to_owned(),
                }));
            }
            Ok(sigma)
        };

        if std_dev == 0.0 {
            return scattering_at_radius(mean_radius);
        }

        let variance = std_dev * std_dev;
        let sigma2_ln = (variance / (mean_radius * mean_radius)).ln_1p();
        let sigma_ln = sigma2_ln.sqrt();
        if !sigma_ln.is_finite() || sigma_ln <= 0.0 {
            return scattering_at_radius(mean_radius);
        }
        let mu_ln = 0.5f64.mul_add(-sigma2_ln, mean_radius.ln());

        let a = 6.0f64.mul_add(-sigma_ln, mu_ln);
        let b = 6.0f64.mul_add(sigma_ln, mu_ln);
        let n: usize = 128;
        let h = (b - a) / n as f64;
        let norm = 1.0 / (sigma_ln * (2.0 * std::f64::consts::PI).sqrt());

        let mut sum = 0.0;
        for i in 0..=n {
            let y = h.mul_add(i as f64, a);
            let z = (y - mu_ln) / sigma_ln;
            let pdf = norm * (-0.5 * z * z).exp();
            let radius = y.exp();
            let integrand = scattering_at_radius(radius)? * pdf;
            let weight = if i == 0 || i == n {
                1.0
            } else if i % 2 == 1 {
                4.0
            } else {
                2.0
            };
            sum += weight * integrand;
        }

        let result = (h / 3.0) * sum;
        if !result.is_finite() || result < 0.0 {
            return Err(KwaversError::Validation(ValidationError::InvalidValue {
                parameter: "effective_scattering".to_owned(),
                value: result,
                reason: "must be non-negative and finite".to_owned(),
            }));
        }

        Ok(result)
    }
}
