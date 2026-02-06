//! Marmottant Shell Model for Coated Microbubbles
//!
//! Implementation of the Marmottant et al. (2005) model for large amplitude
//! oscillations of lipid-coated microbubbles used in medical ultrasound.
//!
//! ## Physical Model
//!
//! The Marmottant model describes the nonlinear mechanical behavior of
//! phospholipid shells coating microbubbles. The shell can exist in three
//! distinct mechanical states:
//!
//! ### 1. Buckled State (R < R_buckling)
//! - Shell under compression, folded/buckled
//! - Zero surface tension: χ = 0
//! - No elastic resistance to compression
//!
//! ### 2. Elastic State (R_buckling ≤ R ≤ R_rupture)
//! - Shell stretched, elastic resistance active
//! - Linear elastic regime: χ(R) = κ_s * (R²/R₀² - 1)
//! - κ_s: Shell elastic modulus [N/m]
//!
//! ### 3. Ruptured State (R > R_rupture)
//! - Shell ruptured, behaves like free bubble
//! - Water surface tension: χ = σ_water ≈ 0.072 N/m
//! - Irreversible transition
//!
//! ## Mathematical Specification
//!
//! Surface tension as function of radius:
//!
//! ```text
//!            ⎧ 0                           R < R_buckling
//!            ⎪
//! χ(R) =     ⎨ κ_s (R²/R₀² - 1)           R_buckling ≤ R ≤ R_rupture
//!            ⎪
//!            ⎩ σ_water                     R > R_rupture
//! ```
//!
//! Shell contribution to bubble wall pressure:
//!
//! ```text
//! P_shell = 2χ(R)/R + 4κ_shell·(Ṙ/R)
//! ```
//!
//! where:
//! - χ(R): Surface tension [N/m]
//! - κ_shell: Shell viscosity [Pa·s]
//! - Ṙ: Bubble wall velocity [m/s]
//!
//! ## References
//!
//! - Marmottant et al. (2005): "A model for large amplitude oscillations of
//!   coated bubbles accounting for buckling and rupture", JASA 118(6):3499-3505
//! - Doinikov et al. (2011): "Modeling of nonlinear viscous stress in
//!   encapsulating shells of lipid-coated contrast agent microbubbles"
//! - Sijl et al. (2010): "Subharmonic behavior of phospholipid-coated
//!   ultrasound contrast agent microbubbles"

use crate::core::error::{KwaversError, KwaversResult, ValidationError};
use std::fmt;

/// Shell mechanical state according to Marmottant model
///
/// Represents the current mechanical configuration of the lipid shell.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ShellState {
    /// Shell is buckled/compressed (R < R_buckling)
    Buckled,
    /// Shell is in elastic regime (R_buckling ≤ R ≤ R_rupture)
    Elastic,
    /// Shell has ruptured (R > R_rupture)
    Ruptured,
}

impl fmt::Display for ShellState {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ShellState::Buckled => write!(f, "Buckled"),
            ShellState::Elastic => write!(f, "Elastic"),
            ShellState::Ruptured => write!(f, "Ruptured"),
        }
    }
}

/// Marmottant shell properties and state
///
/// Encapsulates all parameters and state needed for the Marmottant model
/// of lipid-coated microbubbles.
///
/// ## Domain Invariants
///
/// 1. **Radius Ordering**: 0 < R_buckling < R₀ < R_rupture
/// 2. **Elasticity**: κ_s ≥ 0
/// 3. **Viscosity**: μ_shell ≥ 0
/// 4. **State Consistency**: State transitions follow physical laws
///
/// ## Typical Values
///
/// For phospholipid-coated contrast agents:
/// - κ_s ≈ 0.5-1.5 N/m (shell elasticity)
/// - μ_shell ≈ 0.5-2.0 × 10⁻⁹ Pa·s (shell viscosity)
/// - R_buckling/R₀ ≈ 0.8-0.9 (10-20% compression)
/// - R_rupture/R₀ ≈ 1.5-2.0 (50-100% expansion)
#[derive(Debug, Clone)]
pub struct MarmottantShellProperties {
    /// Equilibrium radius [m]
    pub radius_equilibrium: f64,

    /// Buckling radius [m]
    pub radius_buckling: f64,

    /// Rupture radius [m]
    pub radius_rupture: f64,

    /// Shell elastic modulus [N/m]
    pub elasticity: f64,

    /// Shell viscosity [Pa·s]
    pub viscosity: f64,

    /// Water surface tension (for ruptured state) [N/m]
    pub surface_tension_water: f64,

    /// Current shell state
    pub state: ShellState,

    /// Whether rupture has occurred (irreversible)
    pub has_ruptured: bool,
}

impl MarmottantShellProperties {
    /// Create new Marmottant shell properties
    ///
    /// # Arguments
    ///
    /// - `radius_equilibrium`: Equilibrium radius [m]
    /// - `elasticity`: Shell elastic modulus κ_s [N/m]
    /// - `viscosity`: Shell viscosity μ_shell [Pa·s]
    /// - `buckling_ratio`: R_buckling/R₀ (typically 0.8-0.9)
    /// - `rupture_ratio`: R_rupture/R₀ (typically 1.5-2.0)
    ///
    /// # Returns
    ///
    /// New shell properties with initial elastic state
    ///
    /// # Validation
    ///
    /// - Radius must be positive
    /// - Elasticity and viscosity must be non-negative
    /// - Buckling ratio must be in (0, 1)
    /// - Rupture ratio must be > 1
    pub fn new(
        radius_equilibrium: f64,
        elasticity: f64,
        viscosity: f64,
        buckling_ratio: f64,
        rupture_ratio: f64,
    ) -> KwaversResult<Self> {
        // Validate inputs
        if radius_equilibrium <= 0.0 {
            return Err(KwaversError::Validation(ValidationError::InvalidValue {
                parameter: "radius_equilibrium".to_string(),
                value: radius_equilibrium,
                reason: "must be positive".to_string(),
            }));
        }
        if elasticity < 0.0 {
            return Err(KwaversError::Validation(ValidationError::InvalidValue {
                parameter: "elasticity".to_string(),
                value: elasticity,
                reason: "must be non-negative".to_string(),
            }));
        }
        if viscosity < 0.0 {
            return Err(KwaversError::Validation(ValidationError::InvalidValue {
                parameter: "viscosity".to_string(),
                value: viscosity,
                reason: "must be non-negative".to_string(),
            }));
        }
        if buckling_ratio <= 0.0 || buckling_ratio >= 1.0 {
            return Err(KwaversError::Validation(ValidationError::InvalidValue {
                parameter: "buckling_ratio".to_string(),
                value: buckling_ratio,
                reason: "must be in (0, 1)".to_string(),
            }));
        }
        if rupture_ratio <= 1.0 {
            return Err(KwaversError::Validation(ValidationError::InvalidValue {
                parameter: "rupture_ratio".to_string(),
                value: rupture_ratio,
                reason: "must be > 1".to_string(),
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
            surface_tension_water: 0.072, // N/m at body temperature
            state: ShellState::Elastic,   // Initially at equilibrium (elastic)
            has_ruptured: false,
        })
    }

    /// Create typical SonoVue-like shell
    pub fn sono_vue(radius_equilibrium: f64) -> KwaversResult<Self> {
        Self::new(
            radius_equilibrium,
            0.5,    // N/m
            0.8e-9, // Pa·s
            0.85,   // 15% compression to buckle
            1.6,    // 60% expansion to rupture
        )
    }

    /// Create typical Definity-like shell
    pub fn definity(radius_equilibrium: f64) -> KwaversResult<Self> {
        Self::new(
            radius_equilibrium,
            1.0,    // N/m (stiffer)
            1.2e-9, // Pa·s
            0.90,   // 10% compression to buckle
            1.8,    // 80% expansion to rupture
        )
    }

    /// Create drug-delivery shell (weaker for easier rupture)
    pub fn drug_delivery(radius_equilibrium: f64) -> KwaversResult<Self> {
        Self::new(
            radius_equilibrium,
            0.3,    // N/m (weak)
            0.5e-9, // Pa·s
            0.80,   // 20% compression to buckle
            1.4,    // 40% expansion to rupture (easy to break)
        )
    }

    /// Calculate surface tension χ(R) according to Marmottant model
    ///
    /// # Arguments
    ///
    /// - `radius`: Current bubble radius [m]
    ///
    /// # Returns
    ///
    /// Surface tension [N/m]
    ///
    /// # Mathematical Specification
    ///
    /// ```text
    /// χ(R) = ⎧ 0                           R < R_buckling
    ///        ⎨ κ_s (R²/R₀² - 1)           R_buckling ≤ R ≤ R_rupture
    ///        ⎩ σ_water                     R > R_rupture
    /// ```
    #[must_use]
    pub fn surface_tension(&self, radius: f64) -> f64 {
        if radius < self.radius_buckling {
            // Buckled state: zero surface tension
            0.0
        } else if radius <= self.radius_rupture {
            // Elastic state: linear elastic response
            let r0_sq = self.radius_equilibrium * self.radius_equilibrium;
            let r_sq = radius * radius;
            self.elasticity * (r_sq / r0_sq - 1.0)
        } else {
            // Ruptured state: behaves like free bubble
            self.surface_tension_water
        }
    }

    /// Calculate d(χ)/dR (derivative of surface tension)
    ///
    /// Needed for pressure calculations and stability analysis.
    ///
    /// # Returns
    ///
    /// d(χ)/dR [N/m²]
    #[must_use]
    pub fn surface_tension_derivative(&self, radius: f64) -> f64 {
        if radius < self.radius_buckling {
            // Buckled: constant zero
            0.0
        } else if radius <= self.radius_rupture {
            // Elastic: d(χ)/dR = 2κ_s·R/R₀²
            let r0_sq = self.radius_equilibrium * self.radius_equilibrium;
            2.0 * self.elasticity * radius / r0_sq
        } else {
            // Ruptured: constant surface tension
            0.0
        }
    }

    /// Calculate shell contribution to bubble wall pressure
    ///
    /// # Arguments
    ///
    /// - `radius`: Current radius [m]
    /// - `wall_velocity`: dR/dt [m/s]
    ///
    /// # Returns
    ///
    /// Pressure contribution [Pa]
    ///
    /// # Formula
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

    /// Update shell state based on current radius
    ///
    /// Handles state transitions according to Marmottant model.
    /// Rupture is irreversible.
    ///
    /// # State Transition Rules
    ///
    /// 1. Buckled → Elastic: R ≥ R_buckling
    /// 2. Elastic → Buckled: R < R_buckling
    /// 3. Elastic → Ruptured: R > R_rupture (irreversible)
    /// 4. Ruptured → (no transitions, permanent)
    pub fn update_state(&mut self, radius: f64) {
        if self.has_ruptured {
            // Once ruptured, always ruptured
            self.state = ShellState::Ruptured;
            return;
        }

        if radius > self.radius_rupture {
            // Rupture occurs (irreversible)
            self.state = ShellState::Ruptured;
            self.has_ruptured = true;
        } else if radius < self.radius_buckling {
            // Shell buckles
            self.state = ShellState::Buckled;
        } else {
            // Elastic regime
            self.state = ShellState::Elastic;
        }
    }

    /// Check if shell is in elastic regime
    #[must_use]
    pub fn is_elastic(&self) -> bool {
        self.state == ShellState::Elastic
    }

    /// Check if shell is buckled
    #[must_use]
    pub fn is_buckled(&self) -> bool {
        self.state == ShellState::Buckled
    }

    /// Check if shell has ruptured
    #[must_use]
    pub fn is_ruptured(&self) -> bool {
        self.state == ShellState::Ruptured
    }

    /// Calculate shell strain (R/R₀ - 1)
    #[must_use]
    pub fn strain(&self, radius: f64) -> f64 {
        radius / self.radius_equilibrium - 1.0
    }

    /// Calculate shell stress (approximately χ(R))
    #[must_use]
    pub fn stress(&self, radius: f64) -> f64 {
        self.surface_tension(radius)
    }

    /// Validate shell properties
    pub fn validate(&self) -> KwaversResult<()> {
        if self.radius_equilibrium <= 0.0 {
            return Err(KwaversError::Validation(ValidationError::InvalidValue {
                parameter: "radius_equilibrium".to_string(),
                value: self.radius_equilibrium,
                reason: "must be positive".to_string(),
            }));
        }
        if self.radius_buckling >= self.radius_equilibrium {
            return Err(KwaversError::Validation(ValidationError::InvalidValue {
                parameter: "radius_buckling".to_string(),
                value: self.radius_buckling,
                reason: "must be < radius_equilibrium".to_string(),
            }));
        }
        if self.radius_rupture <= self.radius_equilibrium {
            return Err(KwaversError::Validation(ValidationError::InvalidValue {
                parameter: "radius_rupture".to_string(),
                value: self.radius_rupture,
                reason: "must be > radius_equilibrium".to_string(),
            }));
        }
        if self.elasticity < 0.0 {
            return Err(KwaversError::Validation(ValidationError::InvalidValue {
                parameter: "elasticity".to_string(),
                value: self.elasticity,
                reason: "must be non-negative".to_string(),
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_create_shell() {
        let shell = MarmottantShellProperties::new(1.0e-6, 1.0, 1.0e-9, 0.85, 1.6).unwrap();
        assert!(shell.validate().is_ok());
        assert_eq!(shell.state, ShellState::Elastic);
        assert!(!shell.has_ruptured);
    }

    #[test]
    fn test_sono_vue_shell() {
        let shell = MarmottantShellProperties::sono_vue(1.25e-6).unwrap();
        assert!(shell.validate().is_ok());
        assert_eq!(shell.radius_equilibrium, 1.25e-6);
    }

    #[test]
    fn test_surface_tension_buckled() {
        let shell = MarmottantShellProperties::new(1.0e-6, 1.0, 1.0e-9, 0.85, 1.6).unwrap();
        let r_buckled = 0.8e-6; // < R_buckling
        let chi = shell.surface_tension(r_buckled);
        assert_eq!(chi, 0.0);
    }

    #[test]
    fn test_surface_tension_elastic() {
        let shell = MarmottantShellProperties::new(1.0e-6, 1.0, 1.0e-9, 0.85, 1.6).unwrap();
        let r = 1.1e-6; // Expanded, elastic regime

        let chi = shell.surface_tension(r);
        let expected = 1.0 * ((1.1 * 1.1) / (1.0 * 1.0) - 1.0);
        assert!((chi - expected).abs() < 1e-10);
    }

    #[test]
    fn test_surface_tension_ruptured() {
        let shell = MarmottantShellProperties::new(1.0e-6, 1.0, 1.0e-9, 0.85, 1.6).unwrap();
        let r_rupture = 2.0e-6; // > R_rupture
        let chi = shell.surface_tension(r_rupture);
        assert_eq!(chi, 0.072);
    }

    #[test]
    fn test_state_transitions() {
        let mut shell = MarmottantShellProperties::new(1.0e-6, 1.0, 1.0e-9, 0.85, 1.6).unwrap();

        // Start in elastic
        assert_eq!(shell.state, ShellState::Elastic);

        // Compress to buckle
        shell.update_state(0.8e-6);
        assert_eq!(shell.state, ShellState::Buckled);
        assert!(shell.is_buckled());

        // Return to elastic
        shell.update_state(1.0e-6);
        assert_eq!(shell.state, ShellState::Elastic);
        assert!(shell.is_elastic());

        // Expand to rupture
        shell.update_state(2.0e-6);
        assert_eq!(shell.state, ShellState::Ruptured);
        assert!(shell.is_ruptured());
        assert!(shell.has_ruptured);

        // Rupture is irreversible
        shell.update_state(1.0e-6);
        assert_eq!(shell.state, ShellState::Ruptured);
        assert!(shell.has_ruptured);
    }

    #[test]
    fn test_pressure_contribution() {
        let shell = MarmottantShellProperties::new(1.0e-6, 1.0, 1.0e-9, 0.85, 1.6).unwrap();
        let radius = 1.0e-6;
        let velocity = 10.0; // m/s

        let p = shell.pressure_contribution(radius, velocity);
        // Should have both elastic and viscous terms
        assert!(p > 0.0);
    }

    #[test]
    fn test_strain_calculation() {
        let shell = MarmottantShellProperties::new(1.0e-6, 1.0, 1.0e-9, 0.85, 1.6).unwrap();

        // At equilibrium
        assert_eq!(shell.strain(1.0e-6), 0.0);

        // Compressed 10%
        assert!((shell.strain(0.9e-6) + 0.1).abs() < 1e-10);

        // Expanded 20%
        assert!((shell.strain(1.2e-6) - 0.2).abs() < 1e-10);
    }

    #[test]
    fn test_validation_invalid_buckling_ratio() {
        let result = MarmottantShellProperties::new(1.0e-6, 1.0, 1.0e-9, 1.5, 1.6);
        assert!(result.is_err());
    }

    #[test]
    fn test_validation_invalid_rupture_ratio() {
        let result = MarmottantShellProperties::new(1.0e-6, 1.0, 1.0e-9, 0.85, 0.9);
        assert!(result.is_err());
    }

    #[test]
    fn test_surface_tension_derivative() {
        let shell = MarmottantShellProperties::new(1.0e-6, 1.0, 1.0e-9, 0.85, 1.6).unwrap();

        // In elastic regime
        let r = 1.0e-6;
        let dchi_dr = shell.surface_tension_derivative(r);
        let expected = 2.0 * 1.0 * r / (1.0e-6 * 1.0e-6);
        assert!((dchi_dr - expected).abs() < 1e-10);

        // Buckled: should be zero
        assert_eq!(shell.surface_tension_derivative(0.8e-6), 0.0);

        // Ruptured: should be zero (constant)
        assert_eq!(shell.surface_tension_derivative(2.0e-6), 0.0);
    }

    #[test]
    fn test_drug_delivery_shell() {
        let shell = MarmottantShellProperties::drug_delivery(2.0e-6).unwrap();
        assert!(shell.validate().is_ok());
        // Should have lower rupture threshold than imaging agents
        assert!(shell.radius_rupture < 2.0e-6 * 1.5);
    }
}
