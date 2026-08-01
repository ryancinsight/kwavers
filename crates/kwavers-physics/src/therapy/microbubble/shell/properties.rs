use super::state::ShellState;
use aequitas::systems::si::quantities::{
    Dimensionless, DynamicViscosity, Length, Pressure, SurfaceTension,
};
use kwavers_core::constants::cavitation::SURFACE_TENSION_WATER;
use kwavers_core::error::{KwaversError, KwaversResult, ValidationError};
use std::fmt;

/// Marmottant shell properties and state.
///
/// ## Domain Invariants
///
/// 1. **Radius ordering**: 0 < R_buckling < R₀ < R_rupture
/// 2. **Elasticity**: κ_s ≥ 0
/// 3. **Viscosity**: μ_shell ≥ 0
///
/// ## References
///
/// - Marmottant et al. (2005): JASA 118(6):3499–3505
/// - Doinikov et al. (2011): nonlinear viscous stress in encapsulating shells
#[derive(Debug, Clone)]
pub struct MarmottantShellProperties {
    /// Equilibrium radius (m)
    pub radius_equilibrium: Length<f64>,
    /// Buckling radius (m)
    pub radius_buckling: Length<f64>,
    /// Rupture radius (m)
    pub radius_rupture: Length<f64>,
    /// Shell elastic modulus κ_s [N/m]
    pub elasticity: SurfaceTension<f64>,
    /// Shell viscosity μ_shell [Pa·s]
    pub viscosity: DynamicViscosity<f64>,
    /// Water surface tension (ruptured state) [N/m]
    pub surface_tension_water: SurfaceTension<f64>,
    /// Current shell state
    pub state: ShellState,
    /// Whether rupture has occurred (irreversible)
    pub has_ruptured: bool,
}

impl MarmottantShellProperties {
    /// Create new Marmottant shell properties.
    /// # Errors
    /// - Returns [`KwaversError::Validation`] if the precondition for a Validation-class constraint is violated.
    ///
    pub fn new(
        radius_equilibrium: Length<f64>,
        elasticity: SurfaceTension<f64>,
        viscosity: DynamicViscosity<f64>,
        buckling_ratio: f64,
        rupture_ratio: f64,
    ) -> KwaversResult<Self> {
        let radius_equilibrium_value = radius_equilibrium.into_base();
        let elasticity_value = elasticity.into_base();
        let viscosity_value = viscosity.into_base();

        if radius_equilibrium_value <= 0.0 {
            return Err(KwaversError::Validation(ValidationError::InvalidValue {
                parameter: "radius_equilibrium".to_owned(),
                value: radius_equilibrium_value,
                reason: "must be positive".to_owned(),
            }));
        }
        if elasticity_value < 0.0 {
            return Err(KwaversError::Validation(ValidationError::InvalidValue {
                parameter: "elasticity".to_owned(),
                value: elasticity_value,
                reason: "must be non-negative".to_owned(),
            }));
        }
        if viscosity_value < 0.0 {
            return Err(KwaversError::Validation(ValidationError::InvalidValue {
                parameter: "viscosity".to_owned(),
                value: viscosity_value,
                reason: "must be non-negative".to_owned(),
            }));
        }
        if buckling_ratio <= 0.0 || buckling_ratio >= 1.0 {
            return Err(KwaversError::Validation(ValidationError::InvalidValue {
                parameter: "buckling_ratio".to_owned(),
                value: buckling_ratio,
                reason: "must be in (0, 1)".to_owned(),
            }));
        }
        if rupture_ratio <= 1.0 {
            return Err(KwaversError::Validation(ValidationError::InvalidValue {
                parameter: "rupture_ratio".to_owned(),
                value: rupture_ratio,
                reason: "must be > 1".to_owned(),
            }));
        }

        let radius_buckling = radius_equilibrium_value * buckling_ratio;
        let radius_rupture = radius_equilibrium_value * rupture_ratio;

        Ok(Self {
            radius_equilibrium,
            radius_buckling: Length::from_base(radius_buckling),
            radius_rupture: Length::from_base(radius_rupture),
            elasticity,
            viscosity,
            surface_tension_water: SurfaceTension::from_base(SURFACE_TENSION_WATER),
            state: ShellState::Elastic,
            has_ruptured: false,
        })
    }

    /// Create typical SonoVue-like shell.
    /// # Errors
    /// - Returns [`Err`] if an internal constraint is violated.
    ///
    pub fn sono_vue(radius_equilibrium: Length<f64>) -> KwaversResult<Self> {
        Self::new(
            radius_equilibrium,
            SurfaceTension::from_base(0.5),
            DynamicViscosity::from_base(0.8e-9),
            0.85,
            1.6,
        )
    }

    /// Create typical Definity-like shell.
    /// # Errors
    /// - Returns [`Err`] if an internal constraint is violated.
    ///
    pub fn definity(radius_equilibrium: Length<f64>) -> KwaversResult<Self> {
        Self::new(
            radius_equilibrium,
            SurfaceTension::from_base(1.0),
            DynamicViscosity::from_base(1.2e-9),
            0.90,
            1.8,
        )
    }

    /// Create drug-delivery shell (weaker for easier rupture).
    /// # Errors
    /// - Returns [`Err`] if an internal constraint is violated.
    ///
    pub fn drug_delivery(radius_equilibrium: Length<f64>) -> KwaversResult<Self> {
        Self::new(
            radius_equilibrium,
            SurfaceTension::from_base(0.3),
            DynamicViscosity::from_base(0.5e-9),
            0.80,
            1.4,
        )
    }

    /// Calculate surface tension χ(R) (Marmottant 2005, eq. 1).
    ///
    /// ```text
    /// χ(R) = ⎧ 0                          R < R_buckling
    ///        ⎨ κ_s(R²/R_buckling² − 1)    R_buckling ≤ R ≤ R_rupture
    ///        ⎩ σ_water                    R > R_rupture
    /// ```
    ///
    /// The elastic regime is referenced to **R_buckling** (Marmottant 2005), so
    /// χ(R_buckling) = 0 — the surface tension is continuous and non-negative
    /// throughout. (A prior R_equilibrium reference gave χ(R_buckling) =
    /// κ_s(R_buckling²/R₀² − 1) < 0, an unphysical negative surface tension over
    /// `R ∈ [R_buckling, R₀)`; this matches the canonical
    /// `bubble_dynamics::encapsulated::MarmottantModel`.)
    #[must_use]
    pub fn surface_tension(&self, radius: Length<f64>) -> SurfaceTension<f64> {
        let radius_value = radius.into_base();
        let radius_buckling = self.radius_buckling.into_base();
        let radius_rupture = self.radius_rupture.into_base();
        if radius < self.radius_buckling {
            SurfaceTension::from_base(0.0)
        } else if radius_value <= radius_rupture {
            let r_b_sq = radius_buckling * radius_buckling;
            let r_sq = radius_value * radius_value;
            SurfaceTension::from_base(self.elasticity.into_base() * (r_sq / r_b_sq - 1.0))
        } else {
            self.surface_tension_water
        }
    }

    /// Calculate d(χ)/dR.
    ///
    /// Elastic regime: d(χ)/dR = 2κ_s·R/R_buckling² (referenced to R_buckling
    /// per Marmottant 2005, consistent with [`Self::surface_tension`]).
    #[must_use]
    pub fn surface_tension_derivative(&self, radius: Length<f64>) -> Pressure<f64> {
        let radius_value = radius.into_base();
        let radius_buckling = self.radius_buckling.into_base();
        if radius_value < radius_buckling {
            Pressure::from_base(0.0)
        } else if radius_value <= self.radius_rupture.into_base() {
            let r_b_sq = radius_buckling * radius_buckling;
            Pressure::from_base(2.0 * self.elasticity.into_base() * radius_value / r_b_sq)
        } else {
            Pressure::from_base(0.0)
        }
    }

    /// Calculate shell contribution to bubble wall pressure.
    ///
    /// P_shell = 2χ(R)/R + 4μ_shell·(Ṙ/R)
    #[must_use]
    pub fn pressure_contribution(
        &self,
        radius: Length<f64>,
        wall_velocity: aequitas::systems::si::quantities::Velocity<f64>,
    ) -> Pressure<f64> {
        let radius_value = radius.into_base();
        if radius_value <= 0.0 {
            return Pressure::from_base(0.0);
        }
        let chi = self.surface_tension(radius).into_base();
        let elastic_term = 2.0 * chi / radius_value;
        let viscous_term =
            4.0 * self.viscosity.into_base() * wall_velocity.into_base() / radius_value;
        Pressure::from_base(elastic_term + viscous_term)
    }

    /// Update shell state based on current radius.
    ///
    /// Rupture is irreversible.
    pub fn update_state(&mut self, radius: Length<f64>) {
        if self.has_ruptured {
            self.state = ShellState::Ruptured;
            return;
        }
        if radius > self.radius_rupture {
            self.state = ShellState::Ruptured;
            self.has_ruptured = true;
        } else if radius < self.radius_buckling {
            self.state = ShellState::Buckled;
        } else {
            self.state = ShellState::Elastic;
        }
    }

    /// Check if shell is in elastic regime.
    #[must_use]
    pub fn is_elastic(&self) -> bool {
        self.state == ShellState::Elastic
    }

    /// Check if shell is buckled.
    #[must_use]
    pub fn is_buckled(&self) -> bool {
        self.state == ShellState::Buckled
    }

    /// Check if shell has ruptured.
    #[must_use]
    pub fn is_ruptured(&self) -> bool {
        self.state == ShellState::Ruptured
    }

    /// Calculate shell strain (R/R₀ − 1).
    /// # Errors
    /// - Returns [`Err`] if an internal constraint is violated.
    ///
    #[must_use]
    pub fn strain(&self, radius: Length<f64>) -> Dimensionless<f64> {
        Dimensionless::from_base(radius.into_base() / self.radius_equilibrium.into_base() - 1.0)
    }

    /// Calculate shell stress (approximately χ(R)).
    /// # Errors
    /// - Returns [`Err`] if an internal constraint is violated.
    ///
    #[must_use]
    pub fn stress(&self, radius: Length<f64>) -> SurfaceTension<f64> {
        self.surface_tension(radius)
    }

    /// Validate shell properties.
    /// # Errors
    /// - Returns [`KwaversError::Validation`] if the precondition for a Validation-class constraint is violated.
    ///
    pub fn validate(&self) -> KwaversResult<()> {
        if self.radius_equilibrium.into_base() <= 0.0 {
            return Err(KwaversError::Validation(ValidationError::InvalidValue {
                parameter: "radius_equilibrium".to_owned(),
                value: self.radius_equilibrium.into_base(),
                reason: "must be positive".to_owned(),
            }));
        }
        if self.radius_buckling >= self.radius_equilibrium {
            return Err(KwaversError::Validation(ValidationError::InvalidValue {
                parameter: "radius_buckling".to_owned(),
                value: self.radius_buckling.into_base(),
                reason: "must be < radius_equilibrium".to_owned(),
            }));
        }
        if self.radius_rupture <= self.radius_equilibrium {
            return Err(KwaversError::Validation(ValidationError::InvalidValue {
                parameter: "radius_rupture".to_owned(),
                value: self.radius_rupture.into_base(),
                reason: "must be > radius_equilibrium".to_owned(),
            }));
        }
        if self.elasticity.into_base() < 0.0 {
            return Err(KwaversError::Validation(ValidationError::InvalidValue {
                parameter: "elasticity".to_owned(),
                value: self.elasticity.into_base(),
                reason: "must be non-negative".to_owned(),
            }));
        }
        Ok(())
    }
}

impl fmt::Display for MarmottantShellProperties {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "MarmottantShell(κ_s={:.2}N/m, μ={:.2e}Pa·s, R₀={:.2}μm, state={})",
            self.elasticity.into_base(),
            self.viscosity.into_base(),
            self.radius_equilibrium.into_base() * 1e6,
            self.state
        )
    }
}
