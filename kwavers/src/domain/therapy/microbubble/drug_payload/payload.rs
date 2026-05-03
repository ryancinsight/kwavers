//! Drug payload properties and release kinetics.
//!
//! ## Mathematical Model
//!
//! First-order release kinetics with shell-state dependency:
//!
//! ```text
//! dC/dt = -k_release · C · P(shell_state)
//! ```
//!
//! ## References
//!
//! - Stride & Coussios (2010): "Nucleation, mapping and control of cavitation
//!   for drug delivery", Phys Med Biol 55(23):R127
//! - Ferrara et al. (2007): "Ultrasound microbubble contrast agents",
//!   Nat Rev Drug Discov 6(5):347-356

use super::loading_mode::DrugLoadingMode;
use crate::core::error::{KwaversError, KwaversResult, ValidationError};
use crate::domain::therapy::microbubble::shell::ShellState;
use std::fmt;

/// Drug payload properties and release state
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

        let (baseline_permeability, strain_enhancement_factor) = match loading_mode {
            DrugLoadingMode::SurfaceAttached => (0.1, 2.0),
            DrugLoadingMode::ShellEmbedded => (0.05, 5.0),
            DrugLoadingMode::CoreEncapsulated => (0.01, 10.0),
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
    pub fn doxorubicin(bubble_volume: f64) -> KwaversResult<Self> {
        const CONCENTRATION: f64 = 50.0;
        const RELEASE_RATE: f64 = 0.01;
        Self::new(
            CONCENTRATION,
            bubble_volume,
            DrugLoadingMode::ShellEmbedded,
            RELEASE_RATE,
        )
    }

    /// Create typical gene therapy microbubble (plasmid DNA)
    pub fn gene_therapy(bubble_volume: f64) -> KwaversResult<Self> {
        const CONCENTRATION: f64 = 10.0;
        const RELEASE_RATE: f64 = 0.005;
        Self::new(
            CONCENTRATION,
            bubble_volume,
            DrugLoadingMode::CoreEncapsulated,
            RELEASE_RATE,
        )
    }

    /// Calculate current permeability factor based on shell state
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
            ShellState::Ruptured => 1.0,
            ShellState::Elastic | ShellState::Buckled => {
                let enhancement = 1.0 + self.strain_enhancement_factor * shell_strain.abs().powi(2);
                (self.baseline_permeability * enhancement).min(1.0)
            }
        }
    }

    /// Update drug release for timestep
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
            return Ok(0.0);
        }

        let permeability = self.permeability_factor(shell_state, shell_strain);
        let k_effective = self.release_rate_constant * permeability;

        let decay_factor = (-k_effective * dt).exp();
        let new_concentration = self.concentration * decay_factor;

        let current_mass = self.concentration * bubble_volume;
        let new_mass = new_concentration * bubble_volume;
        let released_this_step = current_mass - new_mass;

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
        self.concentration < 1e-10
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
