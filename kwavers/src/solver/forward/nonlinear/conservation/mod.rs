//! Conservation Diagnostics for Nonlinear Solvers
//!
//! This module implements conservation law monitoring and diagnostics for nonlinear
//! acoustic solvers (KZK, Westervelt, Kuznetsov equations). It provides real-time
//! tracking of energy, momentum, and mass conservation violations to ensure physical
//! correctness of numerical solutions.
//!
//! # Mathematical Foundation
//!
//! ## Energy Conservation (Acoustic)
//!
//! Total acoustic energy density:
//! ```text
//! E = E_kinetic + E_potential
//!   = (ρ₀/2)|u|² + p²/(2ρ₀c₀²)
//! ```
//!
//! Energy balance equation:
//! ```text
//! ∂E/∂t + ∇·S = -αE + Q_source
//! ```
//!
//! Where:
//! - S = p·u: Energy flux (acoustic Poynting vector)
//! - α: Absorption coefficient
//! - Q_source: External energy input
//!
//! ## Momentum Conservation
//!
//! Momentum density: ρu
//! ```text
//! ∂(ρu)/∂t + ∇·(ρu⊗u + pI) = f_body + f_viscous
//! ```
//!
//! ## Mass Conservation
//!
//! Continuity equation:
//! ```text
//! ∂ρ/∂t + ∇·(ρu) = S_mass
//! ```
//!
//! # Numerical Conservation
//!
//! For a well-posed numerical scheme, conservation errors should satisfy:
//! - Absolute error: |ΔE| < ε_abs (typically 10⁻⁸ per step)
//! - Relative error: |ΔE/E₀| < ε_rel (typically 10⁻⁶ cumulative)
//! - Rate: |dE/dt| should match analytical predictions
//!
//! # References
//!
//! - LeVeque (2002) "Finite Volume Methods for Hyperbolic Problems"
//! - Toro (2009) "Riemann Solvers and Numerical Methods"
//! - Hamilton & Blackstock (1998) "Nonlinear Acoustics"
//! - Pierce (1989) "Acoustics: An Introduction to Its Physical Principles"

use std::fmt;

/// Conservation law types being monitored
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ConservationLaw {
    /// Energy conservation (kinetic + potential)
    Energy,
    /// Momentum conservation (x, y, z components)
    Momentum,
    /// Mass conservation (continuity)
    Mass,
}

impl fmt::Display for ConservationLaw {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ConservationLaw::Energy => write!(f, "Energy"),
            ConservationLaw::Momentum => write!(f, "Momentum"),
            ConservationLaw::Mass => write!(f, "Mass"),
        }
    }
}

/// Conservation violation severity levels
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum ViolationSeverity {
    /// Acceptable (within numerical tolerance)
    Acceptable,
    /// Warning (approaching tolerance limits)
    Warning,
    /// Error (exceeds acceptable tolerance)
    Error,
    /// Critical (solution likely invalid)
    Critical,
}

impl fmt::Display for ViolationSeverity {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ViolationSeverity::Acceptable => write!(f, "ACCEPTABLE"),
            ViolationSeverity::Warning => write!(f, "WARNING"),
            ViolationSeverity::Error => write!(f, "ERROR"),
            ViolationSeverity::Critical => write!(f, "CRITICAL"),
        }
    }
}

/// Conservation diagnostic result
#[derive(Debug, Clone)]
pub struct ConservationDiagnostic {
    /// Conservation law being checked
    pub law: ConservationLaw,
    /// Initial value (reference)
    pub initial_value: f64,
    /// Current value
    pub current_value: f64,
    /// Absolute change: ΔQ = Q_current - Q_initial
    pub absolute_change: f64,
    /// Relative change: ΔQ/Q_initial
    pub relative_change: f64,
    /// Severity of violation
    pub severity: ViolationSeverity,
    /// Time step number
    pub step: usize,
    /// Simulation time
    pub time: f64,
}

/// Conservation tolerance parameters
#[derive(Debug, Clone, Copy)]
pub struct ConservationTolerances {
    /// Absolute tolerance for conservation errors
    pub absolute_tolerance: f64,
    /// Relative tolerance for conservation errors
    pub relative_tolerance: f64,
    /// Check interval (number of time steps)
    pub check_interval: usize,
}

/// Trait for conservation diagnostics in nonlinear solvers
pub trait ConservationDiagnostics {
    /// Calculate total acoustic energy
    ///
    /// E = ∫∫∫ [ρ₀/2 |u|² + p²/(2ρ₀c₀²)] dV
    ///
    /// # Returns
    ///
    /// Total energy in Joules
    fn calculate_total_energy(&self) -> f64;

    /// Calculate total momentum
    ///
    /// P = ∫∫∫ ρ₀ u dV
    ///
    /// # Returns
    ///
    /// Total momentum (Px, Py, Pz) in kg·m/s
    fn calculate_total_momentum(&self) -> (f64, f64, f64);

    /// Calculate total mass
    ///
    /// M = ∫∫∫ ρ dV
    ///
    /// # Returns
    ///
    /// Total mass in kg
    fn calculate_total_mass(&self) -> f64;

    /// Check energy conservation
    ///
    /// Compares current energy to initial reference value.
    fn check_energy_conservation(
        &self,
        initial_energy: f64,
        step: usize,
        time: f64,
        tolerances: &ConservationTolerances,
    ) -> ConservationDiagnostic {
        let current_energy = self.calculate_total_energy();
        ConservationDiagnostic::new(
            ConservationLaw::Energy,
            initial_energy,
            current_energy,
            step,
            time,
            tolerances,
        )
    }

    /// Check all conservation laws
    ///
    /// Returns diagnostics for energy, momentum, and mass conservation.
    fn check_all_conservation(
        &self,
        initial_energy: f64,
        initial_momentum: (f64, f64, f64),
        initial_mass: f64,
        step: usize,
        time: f64,
        tolerances: &ConservationTolerances,
    ) -> Vec<ConservationDiagnostic> {
        let mut diagnostics = Vec::new();

        diagnostics.push(self.check_energy_conservation(initial_energy, step, time, tolerances));

        let current_momentum = self.calculate_total_momentum();
        let initial_momentum_mag =
            (initial_momentum.0.powi(2) + initial_momentum.1.powi(2) + initial_momentum.2.powi(2))
                .sqrt();
        let current_momentum_mag =
            (current_momentum.0.powi(2) + current_momentum.1.powi(2) + current_momentum.2.powi(2))
                .sqrt();
        diagnostics.push(ConservationDiagnostic::new(
            ConservationLaw::Momentum,
            initial_momentum_mag,
            current_momentum_mag,
            step,
            time,
            tolerances,
        ));

        let current_mass = self.calculate_total_mass();
        diagnostics.push(ConservationDiagnostic::new(
            ConservationLaw::Mass,
            initial_mass,
            current_mass,
            step,
            time,
            tolerances,
        ));

        diagnostics
    }
}

/// Conservation tracking state
#[derive(Debug, Clone)]
pub struct ConservationTracker {
    /// Initial energy (reference)
    pub initial_energy: f64,
    /// Initial momentum (reference)
    pub initial_momentum: (f64, f64, f64),
    /// Initial mass (reference)
    pub initial_mass: f64,
    /// History of diagnostics
    pub history: Vec<ConservationDiagnostic>,
    /// Tolerance settings
    pub tolerances: ConservationTolerances,
    /// Maximum violation severity encountered
    pub max_severity: ViolationSeverity,
}

/// Conservation summary statistics
#[derive(Debug, Clone)]
pub struct ConservationSummary {
    /// Total number of conservation checks performed
    pub total_checks: usize,
    /// Maximum violation severity encountered
    pub max_severity: ViolationSeverity,
    /// Maximum energy conservation error (relative)
    pub max_energy_error: f64,
    /// Final energy conservation error (relative)
    pub final_energy_error: f64,
}

impl fmt::Display for ConservationSummary {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "Conservation Summary: {} checks, Max severity: {}, Max energy error: {:.2e}%, Final energy error: {:.2e}%",
            self.total_checks,
            self.max_severity,
            self.max_energy_error * 100.0,
            self.final_energy_error * 100.0
        )
    }
}

mod diagnostic;
pub mod helpers;
#[cfg(test)]
mod tests;
mod tolerances;
mod tracker;
