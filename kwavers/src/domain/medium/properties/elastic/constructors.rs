use crate::core::constants::acoustic_parameters::BONE_DENSITY;
use super::ElasticPropertyData;

impl ElasticPropertyData {
    /// Construct from Lamé parameters with validation.
    ///
    /// # Errors
    ///
    /// Returns error if parameters violate physical constraints.
    pub fn new(density: f64, lambda: f64, mu: f64) -> Result<Self, String> {
        if density <= 0.0 {
            return Err(format!("Density must be positive, got {}", density));
        }
        if lambda < 0.0 {
            return Err(format!("Lamé lambda must be non-negative, got {}", lambda));
        }
        if mu <= 0.0 {
            return Err(format!("Shear modulus mu must be positive, got {}", mu));
        }
        let nu = lambda / (2.0 * (lambda + mu));
        if nu <= -1.0 || nu >= 0.5 {
            return Err(format!("Poisson's ratio {} violates bounds (-1, 0.5)", nu));
        }
        Ok(Self {
            density,
            lambda,
            mu,
        })
    }

    /// Construct from engineering parameters (Young's modulus E, Poisson's ratio ν).
    ///
    /// # Panics
    ///
    /// Panics if parameters are unphysical.
    #[must_use]
    pub fn from_engineering(density: f64, youngs_modulus: f64, poisson_ratio: f64) -> Self {
        Self::try_from_engineering(density, youngs_modulus, poisson_ratio)
            .expect("Invalid engineering parameters")
    }

    /// Fallible version of `from_engineering`.
    ///
    /// # Errors
    ///
    /// Returns error if parameters violate physical constraints.
    pub fn try_from_engineering(
        density: f64,
        youngs_modulus: f64,
        poisson_ratio: f64,
    ) -> Result<Self, String> {
        if density <= 0.0 {
            return Err(format!("Density must be positive, got {}", density));
        }
        if youngs_modulus <= 0.0 {
            return Err(format!(
                "Young's modulus must be positive, got {}",
                youngs_modulus
            ));
        }
        if poisson_ratio <= -1.0 || poisson_ratio >= 0.5 {
            return Err(format!(
                "Poisson's ratio must be in (-1, 0.5), got {}",
                poisson_ratio
            ));
        }
        let lambda = youngs_modulus * poisson_ratio
            / ((1.0 + poisson_ratio) * 2.0f64.mul_add(-poisson_ratio, 1.0));
        let mu = youngs_modulus / (2.0 * (1.0 + poisson_ratio));
        Ok(Self {
            density,
            lambda,
            mu,
        })
    }

    /// Construct from wave speeds (inverse problem).
    ///
    /// # Panics
    ///
    /// Panics if parameters are unphysical.
    #[must_use]
    pub fn from_wave_speeds(density: f64, p_speed: f64, s_speed: f64) -> Self {
        Self::try_from_wave_speeds(density, p_speed, s_speed)
            .expect("Invalid wave speed parameters")
    }

    /// Fallible version of `from_wave_speeds`.
    ///
    /// Recovers Lamé parameters from measured wave speeds:
    /// ```text
    /// μ = ρ c_s²
    /// λ = ρ c_p² - 2μ
    /// ```
    ///
    /// # Errors
    ///
    /// Returns error if `s_speed >= p_speed` or any speed/density is non-positive.
    pub fn try_from_wave_speeds(density: f64, p_speed: f64, s_speed: f64) -> Result<Self, String> {
        if density <= 0.0 {
            return Err(format!("Density must be positive, got {}", density));
        }
        if p_speed <= 0.0 {
            return Err(format!("P-wave speed must be positive, got {}", p_speed));
        }
        if s_speed <= 0.0 {
            return Err(format!("S-wave speed must be positive, got {}", s_speed));
        }
        if s_speed >= p_speed {
            return Err(format!(
                "S-wave speed ({}) must be less than P-wave speed ({})",
                s_speed, p_speed
            ));
        }
        let mu = density * s_speed * s_speed;
        let lambda = (density * p_speed).mul_add(p_speed, -(2.0 * mu));
        Self::new(density, lambda, mu)
    }

    /// Steel properties (generic)
    #[must_use]
    pub fn steel() -> Self {
        Self::from_engineering(7850.0, 200e9, 0.3)
    }

    /// Aluminum properties (generic)
    #[must_use]
    pub fn aluminum() -> Self {
        Self::from_engineering(2700.0, 69e9, 0.33)
    }

    /// Bone properties (cortical bone)
    #[must_use]
    pub fn bone() -> Self {
        Self::from_engineering(BONE_DENSITY, 17e9, 0.3)
    }
}
