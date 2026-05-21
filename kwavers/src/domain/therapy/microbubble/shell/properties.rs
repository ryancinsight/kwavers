use super::state::ShellState;
use crate::core::constants::cavitation::SURFACE_TENSION_WATER;
use crate::core::error::{KwaversError, KwaversResult, ValidationError};
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
    pub radius_equilibrium: f64,
    /// Buckling radius (m)
    pub radius_buckling: f64,
    /// Rupture radius (m)
    pub radius_rupture: f64,
    /// Shell elastic modulus κ_s [N/m]
    pub elasticity: f64,
    /// Shell viscosity μ_shell [Pa·s]
    pub viscosity: f64,
    /// Water surface tension (ruptured state) [N/m]
    pub surface_tension_water: f64,
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
        radius_equilibrium: f64,
        elasticity: f64,
        viscosity: f64,
        buckling_ratio: f64,
        rupture_ratio: f64,
    ) -> KwaversResult<Self> {
        if radius_equilibrium <= 0.0 {
            return Err(KwaversError::Validation(ValidationError::InvalidValue {
                parameter: "radius_equilibrium".to_owned(),
                value: radius_equilibrium,
                reason: "must be positive".to_owned(),
            }));
        }
        if elasticity < 0.0 {
            return Err(KwaversError::Validation(ValidationError::InvalidValue {
                parameter: "elasticity".to_owned(),
                value: elasticity,
                reason: "must be non-negative".to_owned(),
            }));
        }
        if viscosity < 0.0 {
            return Err(KwaversError::Validation(ValidationError::InvalidValue {
                parameter: "viscosity".to_owned(),
                value: viscosity,
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

        let radius_buckling = radius_equilibrium * buckling_ratio;
        let radius_rupture = radius_equilibrium * rupture_ratio;

        Ok(Self {
            radius_equilibrium,
            radius_buckling,
            radius_rupture,
            elasticity,
            viscosity,
            surface_tension_water: SURFACE_TENSION_WATER,
            state: ShellState::Elastic,
            has_ruptured: false,
        })
    }

    /// Create typical SonoVue-like shell.
    /// # Errors
    /// - Returns [`Err`] if an internal constraint is violated.
    ///
    pub fn sono_vue(radius_equilibrium: f64) -> KwaversResult<Self> {
        Self::new(radius_equilibrium, 0.5, 0.8e-9, 0.85, 1.6)
    }

    /// Create typical Definity-like shell.
    /// # Errors
    /// - Returns [`Err`] if an internal constraint is violated.
    ///
    pub fn definity(radius_equilibrium: f64) -> KwaversResult<Self> {
        Self::new(radius_equilibrium, 1.0, 1.2e-9, 0.90, 1.8)
    }

    /// Create drug-delivery shell (weaker for easier rupture).
    /// # Errors
    /// - Returns [`Err`] if an internal constraint is violated.
    ///
    pub fn drug_delivery(radius_equilibrium: f64) -> KwaversResult<Self> {
        Self::new(radius_equilibrium, 0.3, 0.5e-9, 0.80, 1.4)
    }

    /// Calculate surface tension χ(R) (Marmottant 2005).
    ///
    /// ```text
    /// χ(R) = ⎧ 0                      R < R_buckling
    ///        ⎨ κ_s(R²/R₀² − 1)       R_buckling ≤ R ≤ R_rupture
    ///        ⎩ σ_water                R > R_rupture
    /// ```
    #[must_use]
    pub fn surface_tension(&self, radius: f64) -> f64 {
        if radius < self.radius_buckling {
            0.0
        } else if radius <= self.radius_rupture {
            let r0_sq = self.radius_equilibrium * self.radius_equilibrium;
            let r_sq = radius * radius;
            self.elasticity * (r_sq / r0_sq - 1.0)
        } else {
            self.surface_tension_water
        }
    }

    /// Calculate d(χ)/dR.
    ///
    /// Elastic regime: d(χ)/dR = 2κ_s·R/R₀²
    #[must_use]
    pub fn surface_tension_derivative(&self, radius: f64) -> f64 {
        if radius < self.radius_buckling {
            0.0
        } else if radius <= self.radius_rupture {
            let r0_sq = self.radius_equilibrium * self.radius_equilibrium;
            2.0 * self.elasticity * radius / r0_sq
        } else {
            0.0
        }
    }

    /// Calculate shell contribution to bubble wall pressure.
    ///
    /// P_shell = 2χ(R)/R + 4μ_shell·(Ṙ/R)
    #[must_use]
    pub fn pressure_contribution(&self, radius: f64, wall_velocity: f64) -> f64 {
        if radius <= 0.0 {
            return 0.0;
        }
        let chi = self.surface_tension(radius);
        let elastic_term = 2.0 * chi / radius;
        let viscous_term = 4.0 * self.viscosity * wall_velocity / radius;
        elastic_term + viscous_term
    }

    /// Update shell state based on current radius.
    ///
    /// Rupture is irreversible.
    pub fn update_state(&mut self, radius: f64) {
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
    pub fn strain(&self, radius: f64) -> f64 {
        radius / self.radius_equilibrium - 1.0
    }

    /// Calculate shell stress (approximately χ(R)).
    /// # Errors
    /// - Returns [`Err`] if an internal constraint is violated.
    ///
    #[must_use]
    pub fn stress(&self, radius: f64) -> f64 {
        self.surface_tension(radius)
    }

    /// Validate shell properties.
    /// # Errors
    /// - Returns [`KwaversError::Validation`] if the precondition for a Validation-class constraint is violated.
    ///
    pub fn validate(&self) -> KwaversResult<()> {
        if self.radius_equilibrium <= 0.0 {
            return Err(KwaversError::Validation(ValidationError::InvalidValue {
                parameter: "radius_equilibrium".to_owned(),
                value: self.radius_equilibrium,
                reason: "must be positive".to_owned(),
            }));
        }
        if self.radius_buckling >= self.radius_equilibrium {
            return Err(KwaversError::Validation(ValidationError::InvalidValue {
                parameter: "radius_buckling".to_owned(),
                value: self.radius_buckling,
                reason: "must be < radius_equilibrium".to_owned(),
            }));
        }
        if self.radius_rupture <= self.radius_equilibrium {
            return Err(KwaversError::Validation(ValidationError::InvalidValue {
                parameter: "radius_rupture".to_owned(),
                value: self.radius_rupture,
                reason: "must be > radius_equilibrium".to_owned(),
            }));
        }
        if self.elasticity < 0.0 {
            return Err(KwaversError::Validation(ValidationError::InvalidValue {
                parameter: "elasticity".to_owned(),
                value: self.elasticity,
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
            self.elasticity,
            self.viscosity,
            self.radius_equilibrium * 1e6,
            self.state
        )
    }
}
