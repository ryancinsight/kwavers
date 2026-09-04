use super::ElasticPropertyData;
use aequitas::systems::si::quantities::{Dimensionless, MassDensity, Pressure, Velocity};
use kwavers_core::constants::acoustic_parameters::BONE_DENSITY;
use proteus::elastic::IsotropicModuli;

impl ElasticPropertyData {
    /// Construct from Lamé parameters with validation.
    ///
    /// The positive-definite domain `mu > 0` and `K = lambda + 2mu/3 > 0` is
    /// validated by `proteus::elastic::IsotropicModuli::from_lame`. This admits
    /// auxetic solids (`lambda < 0` while `K > 0`); callers that need the
    /// stricter `lambda >= 0` must check the returned value themselves —
    /// `HomogeneousMedium::set_lame_parameters` does.
    ///
    /// # Errors
    ///
    /// Returns error if `density <= 0` or the moduli fall outside the
    /// provider's positive-definite domain.
    pub fn new(density: f64, lambda: f64, mu: f64) -> Result<Self, String> {
        if density <= 0.0 {
            return Err(format!("Density must be positive, got {}", density));
        }
        IsotropicModuli::<f64>::from_lame(Pressure::from_base(lambda), Pressure::from_base(mu))
            .map_err(|e| format!("Invalid Lamé moduli: {e}"))?;
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

    /// Fallible version of `from_engineering`. Delegates the
    /// `mu = E / (2(1 + nu))` and `lambda = E nu / ((1 + nu)(1 - 2nu))`
    /// identity to `proteus::elastic::IsotropicModuli::from_young_poisson`,
    /// which accepts the full `nu in (-1, 1/2)` domain — including the
    /// auxetic regime `nu < 0` that kwavers's old check rejected.
    ///
    /// # Errors
    ///
    /// Returns error if `density <= 0`, `youngs_modulus <= 0`, `nu`
    /// is outside `(-1, 1/2)`, or the resulting moduli are not finite.
    pub fn try_from_engineering(
        density: f64,
        youngs_modulus: f64,
        poisson_ratio: f64,
    ) -> Result<Self, String> {
        if density <= 0.0 {
            return Err(format!("Density must be positive, got {}", density));
        }
        let moduli = IsotropicModuli::<f64>::from_young_poisson(
            Pressure::from_base(youngs_modulus),
            Dimensionless::from_base(poisson_ratio),
        )
        .map_err(|e| format!("Invalid engineering parameters: {e}"))?;
        let lambda = *moduli.lame_lambda().as_base();
        let mu = *moduli.shear_modulus().as_base();
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

    /// Fallible version of `from_wave_speeds`. Delegates the
    /// `μ = ρ·c_s²` and `λ = ρ·(c_p² − 2·c_s²)` identity to
    /// `proteus::elastic::IsotropicModuli::from_wave_speeds`, which
    /// enforces `K = λ + 2μ/3 > 0` (positive-definite) but admits the
    /// auxetic regime `λ < 0` that the previous kwavers-only check
    /// rejected. The kwavers-specific `c_s ≥ c_p` sanity bound is
    /// preserved at the call site; see
    /// `HomogeneousMedium::elastic_homogeneous` for the constructor that
    /// rejects auxetic media outright.
    ///
    /// # Errors
    ///
    /// Returns error if `density <= 0`, `p_speed <= 0`, `c_s >= c_p`,
    /// or the provider's positive-definite domain fails.
    pub fn try_from_wave_speeds(density: f64, p_speed: f64, s_speed: f64) -> Result<Self, String> {
        if density <= 0.0 {
            return Err(format!("Density must be positive, got {}", density));
        }
        if p_speed <= 0.0 {
            return Err(format!("P-wave speed must be positive, got {}", p_speed));
        }
        if s_speed < 0.0 {
            return Err(format!("S-wave speed must be non-negative, got {}", s_speed));
        }
        if s_speed >= p_speed {
            return Err(format!(
                "S-wave speed ({}) must be less than P-wave speed ({})",
                s_speed, p_speed
            ));
        }
        let moduli = IsotropicModuli::<f64>::from_wave_speeds(
            Velocity::from_base(p_speed),
            Velocity::from_base(s_speed),
            MassDensity::from_base(density),
        )
        .map_err(|e| format!("Invalid wave speeds: {e}"))?;
        let lambda = *moduli.lame_lambda().as_base();
        let mu = *moduli.shear_modulus().as_base();
        Ok(Self {
            density,
            lambda,
            mu,
        })
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
