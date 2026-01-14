//! Drug Payload and Release Kinetics
//!
//! Implementation of drug encapsulation and release mechanisms for therapeutic
//! microbubbles used in targeted drug delivery applications.
//!
//! ## Physical Model
//!
//! Therapeutic microbubbles can encapsulate drugs in several configurations:
//! - **Surface-loaded**: Drugs attached to shell surface
//! - **Shell-embedded**: Drugs within lipid bilayer
//! - **Core-encapsulated**: Drugs in aqueous core or gas phase
//!
//! ## Release Mechanisms
//!
//! 1. **Passive Diffusion**: Continuous slow release through intact shell
//! 2. **Shell Rupture**: Burst release when shell breaks (cavitation)
//! 3. **Ultrasound-Enhanced**: Increased permeability during oscillation
//!
//! ## Mathematical Model
//!
//! First-order release kinetics with shell-state dependency:
//!
//! ```text
//! dC/dt = -k_release · C · P(shell_state)
//! ```
//!
//! where:
//! - C: Drug concentration in bubble [kg/m³]
//! - k_release: Base release rate constant [1/s]
//! - P(shell_state): Permeability factor (0-1 for intact, 1 for ruptured)
//!
//! Permeability enhancement during oscillation:
//!
//! ```text
//! P(σ) = P₀ · (1 + α · σ²)
//! ```
//!
//! where:
//! - σ = (R - R₀)/R₀: Shell strain
//! - α: Strain enhancement factor
//! - P₀: Baseline permeability
//!
//! ## References
//!
//! - Stride & Coussios (2010): "Nucleation, mapping and control of cavitation
//!   for drug delivery", Phys Med Biol 55(23):R127
//! - Ferrara et al. (2007): "Ultrasound microbubble contrast agents",
//!   Nat Rev Drug Discov 6(5):347-356
//! - Sirsi & Borden (2014): "State-of-the-art materials for ultrasound-triggered
//!   drug delivery", Adv Drug Deliv Rev 72:3-14

use crate::core::error::{KwaversError, KwaversResult, ValidationError};
use crate::domain::therapy::microbubble::shell::ShellState;
use std::fmt;

/// Drug loading configuration
///
/// Describes how drug is loaded into/onto the microbubble.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DrugLoadingMode {
    /// Drug on shell surface (easy release)
    SurfaceAttached,
    /// Drug in shell lipid bilayer (moderate release)
    ShellEmbedded,
    /// Drug in bubble core (slow release, burst on rupture)
    CoreEncapsulated,
}

impl fmt::Display for DrugLoadingMode {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            DrugLoadingMode::SurfaceAttached => write!(f, "Surface"),
            DrugLoadingMode::ShellEmbedded => write!(f, "Shell"),
            DrugLoadingMode::CoreEncapsulated => write!(f, "Core"),
        }
    }
}

/// Drug payload properties and release state
///
/// Represents the therapeutic drug cargo carried by a microbubble and
/// tracks its release kinetics.
///
/// ## Domain Invariants
///
/// 1. **Conservation**: initial_mass = remaining_mass + released_mass
/// 2. **Positivity**: All masses and concentrations ≥ 0
/// 3. **Physical Bounds**: Release rate must be finite and non-negative
#[derive(Debug, Clone)]
pub struct DrugPayload {
    /// Initial drug mass loaded [kg]
    pub initial_mass: f64,

    /// Current drug concentration in bubble [kg/m³]
    pub concentration: f64,

    /// Cumulative drug released to environment [kg]
    pub released_mass: f64,

    /// Drug loading configuration
    pub loading_mode: DrugLoadingMode,

    /// Base release rate constant [1/s]
    pub release_rate_constant: f64,

    /// Strain enhancement factor (dimensionless)
    pub strain_enhancement_factor: f64,

    /// Baseline shell permeability (dimensionless, 0-1)
    pub baseline_permeability: f64,
}

impl DrugPayload {
    /// Create new drug payload
    ///
    /// # Arguments
    ///
    /// - `concentration`: Initial drug concentration [kg/m³]
    /// - `bubble_volume`: Initial bubble volume [m³]
    /// - `loading_mode`: How drug is loaded
    /// - `release_rate_constant`: Base release rate [1/s]
    ///
    /// # Returns
    ///
    /// New drug payload with zero release
    pub fn new(
        concentration: f64,
        bubble_volume: f64,
        loading_mode: DrugLoadingMode,
        release_rate_constant: f64,
    ) -> KwaversResult<Self> {
        if concentration < 0.0 {
            return Err(KwaversError::Validation(ValidationError::InvalidValue {
                parameter: "concentration".to_string(),
                value: concentration,
                reason: "must be non-negative".to_string(),
            }));
        }
        if bubble_volume < 0.0 {
            return Err(KwaversError::Validation(ValidationError::InvalidValue {
                parameter: "bubble_volume".to_string(),
                value: bubble_volume,
                reason: "must be non-negative".to_string(),
            }));
        }
        if release_rate_constant < 0.0 {
            return Err(KwaversError::Validation(ValidationError::InvalidValue {
                parameter: "release_rate_constant".to_string(),
                value: release_rate_constant,
                reason: "must be non-negative".to_string(),
            }));
        }

        let initial_mass = concentration * bubble_volume;

        // Set permeability based on loading mode
        let (baseline_permeability, strain_enhancement_factor) = match loading_mode {
            DrugLoadingMode::SurfaceAttached => (0.1, 2.0), // Easy release, moderate enhancement
            DrugLoadingMode::ShellEmbedded => (0.05, 5.0),  // Moderate release, strong enhancement
            DrugLoadingMode::CoreEncapsulated => (0.01, 10.0), // Slow release, very strong enhancement
        };

        Ok(Self {
            initial_mass,
            concentration,
            released_mass: 0.0,
            loading_mode,
            release_rate_constant,
            strain_enhancement_factor,
            baseline_permeability,
        })
    }

    /// Create typical doxorubicin-loaded microbubble
    ///
    /// Doxorubicin (chemotherapy drug) typical loading: 10-100 μg/mL
    pub fn doxorubicin(bubble_volume: f64) -> KwaversResult<Self> {
        const CONCENTRATION: f64 = 50.0; // kg/m³ (50 μg/mL)
        const RELEASE_RATE: f64 = 0.01; // 1/s (fast release)

        Self::new(
            CONCENTRATION,
            bubble_volume,
            DrugLoadingMode::ShellEmbedded,
            RELEASE_RATE,
        )
    }

    /// Create typical gene therapy microbubble (plasmid DNA)
    pub fn gene_therapy(bubble_volume: f64) -> KwaversResult<Self> {
        const CONCENTRATION: f64 = 10.0; // kg/m³
        const RELEASE_RATE: f64 = 0.005; // 1/s (slower release)

        Self::new(
            CONCENTRATION,
            bubble_volume,
            DrugLoadingMode::CoreEncapsulated,
            RELEASE_RATE,
        )
    }

    /// Calculate current permeability factor based on shell state
    ///
    /// # Arguments
    ///
    /// - `shell_state`: Current shell mechanical state
    /// - `shell_strain`: Current shell strain (R/R₀ - 1)
    ///
    /// # Returns
    ///
    /// Permeability factor [0-1 for intact, 1.0 for ruptured]
    ///
    /// # Model
    ///
    /// ```text
    /// P(state, strain) = {
    ///   P₀ · (1 + α · strain²)    if Elastic/Buckled
    ///   1.0                        if Ruptured
    /// }
    /// ```
    #[must_use]
    pub fn permeability_factor(&self, shell_state: ShellState, shell_strain: f64) -> f64 {
        match shell_state {
            ShellState::Ruptured => {
                // Complete release
                1.0
            }
            ShellState::Elastic | ShellState::Buckled => {
                // Strain-enhanced permeability
                let enhancement = 1.0 + self.strain_enhancement_factor * shell_strain.abs().powi(2);
                (self.baseline_permeability * enhancement).min(1.0)
            }
        }
    }

    /// Update drug release for timestep
    ///
    /// Uses first-order kinetics with shell-state dependent permeability.
    ///
    /// # Arguments
    ///
    /// - `bubble_volume`: Current bubble volume [m³]
    /// - `shell_state`: Current shell state
    /// - `shell_strain`: Current shell strain
    /// - `dt`: Time step [s]
    ///
    /// # Returns
    ///
    /// Mass of drug released during this timestep [kg]
    ///
    /// # Equations
    ///
    /// ```text
    /// dC/dt = -k · C · P(state, strain)
    /// C(t+dt) = C(t) · exp(-k · P · dt)
    /// Released = (C(t) - C(t+dt)) · V
    /// ```
    pub fn update_release(
        &mut self,
        bubble_volume: f64,
        shell_state: ShellState,
        shell_strain: f64,
        dt: f64,
    ) -> KwaversResult<f64> {
        if dt < 0.0 {
            return Err(KwaversError::Validation(ValidationError::InvalidValue {
                parameter: "dt".to_string(),
                value: dt,
                reason: "must be non-negative".to_string(),
            }));
        }

        if self.concentration <= 0.0 || bubble_volume <= 0.0 {
            // No drug left to release
            return Ok(0.0);
        }

        // Calculate effective release rate
        let permeability = self.permeability_factor(shell_state, shell_strain);
        let k_effective = self.release_rate_constant * permeability;

        // Exponential decay (exact solution for first-order kinetics)
        let decay_factor = (-k_effective * dt).exp();
        let new_concentration = self.concentration * decay_factor;

        // Mass released this timestep
        let current_mass = self.concentration * bubble_volume;
        let new_mass = new_concentration * bubble_volume;
        let released_this_step = current_mass - new_mass;

        // Update state
        self.concentration = new_concentration;
        self.released_mass += released_this_step;

        Ok(released_this_step)
    }

    /// Remaining drug mass [kg]
    #[must_use]
    pub fn remaining_mass(&self, bubble_volume: f64) -> f64 {
        self.concentration * bubble_volume
    }

    /// Fraction of drug released [0-1]
    #[must_use]
    pub fn release_fraction(&self) -> f64 {
        if self.initial_mass > 0.0 {
            self.released_mass / self.initial_mass
        } else {
            0.0
        }
    }

    /// Check if payload is depleted
    #[must_use]
    pub fn is_depleted(&self) -> bool {
        self.concentration < 1e-10 // Effectively zero
    }

    /// Validate drug payload state
    pub fn validate(&self) -> KwaversResult<()> {
        if self.concentration < 0.0 {
            return Err(KwaversError::Validation(ValidationError::InvalidValue {
                parameter: "concentration".to_string(),
                value: self.concentration,
                reason: "must be non-negative".to_string(),
            }));
        }
        if self.released_mass < 0.0 {
            return Err(KwaversError::Validation(ValidationError::InvalidValue {
                parameter: "released_mass".to_string(),
                value: self.released_mass,
                reason: "must be non-negative".to_string(),
            }));
        }
        if self.release_rate_constant < 0.0 {
            return Err(KwaversError::Validation(ValidationError::InvalidValue {
                parameter: "release_rate_constant".to_string(),
                value: self.release_rate_constant,
                reason: "must be non-negative".to_string(),
            }));
        }

        Ok(())
    }
}

impl fmt::Display for DrugPayload {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "DrugPayload({}, C={:.2}kg/m³, released={:.1}%)",
            self.loading_mode,
            self.concentration,
            self.release_fraction() * 100.0
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_create_drug_payload() {
        let volume = 1e-15; // 1 femtoliter
        let concentration = 50.0; // kg/m³
        let payload =
            DrugPayload::new(concentration, volume, DrugLoadingMode::ShellEmbedded, 0.01).unwrap();

        assert_eq!(payload.concentration, concentration);
        assert_eq!(payload.released_mass, 0.0);
        assert_eq!(payload.release_fraction(), 0.0);
        assert!(!payload.is_depleted());
        assert!(payload.validate().is_ok());
    }

    #[test]
    fn test_doxorubicin_payload() {
        let volume = 1e-15;
        let payload = DrugPayload::doxorubicin(volume).unwrap();

        assert_eq!(payload.concentration, 50.0);
        assert_eq!(payload.loading_mode, DrugLoadingMode::ShellEmbedded);
        assert!(payload.validate().is_ok());
    }

    #[test]
    fn test_permeability_ruptured() {
        let volume = 1e-15;
        let payload = DrugPayload::new(50.0, volume, DrugLoadingMode::ShellEmbedded, 0.01).unwrap();

        let perm = payload.permeability_factor(ShellState::Ruptured, 0.5);
        assert_eq!(perm, 1.0); // Full permeability
    }

    #[test]
    fn test_permeability_elastic_no_strain() {
        let volume = 1e-15;
        let payload = DrugPayload::new(50.0, volume, DrugLoadingMode::ShellEmbedded, 0.01).unwrap();

        let perm = payload.permeability_factor(ShellState::Elastic, 0.0);
        assert_eq!(perm, payload.baseline_permeability);
    }

    #[test]
    fn test_permeability_enhanced_by_strain() {
        let volume = 1e-15;
        let payload = DrugPayload::new(50.0, volume, DrugLoadingMode::ShellEmbedded, 0.01).unwrap();

        let strain = 0.5; // 50% strain
        let perm_strained = payload.permeability_factor(ShellState::Elastic, strain);
        let perm_unstrained = payload.permeability_factor(ShellState::Elastic, 0.0);

        assert!(perm_strained > perm_unstrained);
    }

    #[test]
    fn test_release_no_permeability() {
        let volume = 1e-15;
        let mut payload =
            DrugPayload::new(50.0, volume, DrugLoadingMode::CoreEncapsulated, 0.0).unwrap();

        let released = payload
            .update_release(volume, ShellState::Elastic, 0.0, 1.0)
            .unwrap();

        // Zero release rate -> no release
        assert_eq!(released, 0.0);
        assert_eq!(payload.concentration, 50.0);
    }

    #[test]
    fn test_release_exponential_decay() {
        let volume = 1e-15;
        let concentration = 100.0;
        let k = 0.1; // 1/s
        let mut payload =
            DrugPayload::new(concentration, volume, DrugLoadingMode::SurfaceAttached, k).unwrap();

        let dt = 1.0; // 1 second
        let permeability = payload.permeability_factor(ShellState::Elastic, 0.0);

        payload
            .update_release(volume, ShellState::Elastic, 0.0, dt)
            .unwrap();

        // Should follow: C(t) = C(0) * exp(-k * P * dt)
        let expected = concentration * (-k * permeability * dt).exp();
        assert!((payload.concentration - expected).abs() < 1e-10);
    }

    #[test]
    fn test_release_mass_conservation() {
        let volume = 1e-15;
        let concentration = 100.0;
        let mut payload =
            DrugPayload::new(concentration, volume, DrugLoadingMode::ShellEmbedded, 0.1).unwrap();

        let initial_mass = payload.initial_mass;

        // Release over multiple steps
        for _ in 0..10 {
            payload
                .update_release(volume, ShellState::Elastic, 0.2, 0.1)
                .unwrap();
        }

        let remaining = payload.remaining_mass(volume);
        let total = remaining + payload.released_mass;

        // Mass should be conserved (within numerical tolerance)
        assert!((total - initial_mass).abs() / initial_mass < 1e-10);
    }

    #[test]
    fn test_release_complete_on_rupture() {
        let volume = 1e-15;
        let mut payload =
            DrugPayload::new(100.0, volume, DrugLoadingMode::CoreEncapsulated, 0.5).unwrap();

        // Ruptured shell -> full permeability
        // Run enough timesteps to achieve >99% release
        // With k=0.5 and dt=0.1, after 50 steps: exp(-0.5*0.1*50) = exp(-2.5) ≈ 0.08
        // So 1 - 0.08 = 0.92 (92% released)
        // Need more steps: after 100 steps: exp(-5) ≈ 0.007, so 99.3% released
        for _ in 0..100 {
            payload
                .update_release(volume, ShellState::Ruptured, 1.0, 0.1)
                .unwrap();
        }

        // Should be nearly depleted (>99%)
        assert!(
            payload.release_fraction() > 0.99,
            "Expected >99% release, got {:.2}%",
            payload.release_fraction() * 100.0
        );
    }

    #[test]
    fn test_release_fraction() {
        let volume = 1e-15;
        let mut payload =
            DrugPayload::new(100.0, volume, DrugLoadingMode::ShellEmbedded, 0.1).unwrap();

        assert_eq!(payload.release_fraction(), 0.0);

        // Release 50%
        let half_life = (2.0_f64.ln()) / 0.1; // t_1/2 = ln(2)/k
        let permeability = payload.permeability_factor(ShellState::Elastic, 0.0);
        let dt = half_life / permeability;

        payload
            .update_release(volume, ShellState::Elastic, 0.0, dt)
            .unwrap();

        assert!((payload.release_fraction() - 0.5).abs() < 0.01);
    }

    #[test]
    fn test_is_depleted() {
        let volume = 1e-15;
        let mut payload =
            DrugPayload::new(1e-9, volume, DrugLoadingMode::ShellEmbedded, 1.0).unwrap();

        assert!(!payload.is_depleted());

        // Release everything
        for _ in 0..100 {
            payload
                .update_release(volume, ShellState::Ruptured, 1.0, 0.1)
                .unwrap();
        }

        assert!(payload.is_depleted());
    }

    #[test]
    fn test_validation_negative_concentration() {
        let volume = 1e-15;
        let result = DrugPayload::new(-10.0, volume, DrugLoadingMode::ShellEmbedded, 0.1);
        assert!(result.is_err());
    }

    #[test]
    fn test_loading_mode_display() {
        assert_eq!(format!("{}", DrugLoadingMode::SurfaceAttached), "Surface");
        assert_eq!(format!("{}", DrugLoadingMode::ShellEmbedded), "Shell");
        assert_eq!(format!("{}", DrugLoadingMode::CoreEncapsulated), "Core");
    }
}
