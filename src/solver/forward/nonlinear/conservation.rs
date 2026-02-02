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

use crate::core::error::{KwaversError, KwaversResult};
use ndarray::Array3;
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

impl ConservationDiagnostic {
    /// Create new diagnostic result
    pub fn new(
        law: ConservationLaw,
        initial_value: f64,
        current_value: f64,
        step: usize,
        time: f64,
        tolerances: &ConservationTolerances,
    ) -> Self {
        let absolute_change = current_value - initial_value;
        let relative_change = if initial_value.abs() > 1e-15 {
            absolute_change / initial_value
        } else {
            0.0
        };

        let severity =
            Self::assess_severity(absolute_change.abs(), relative_change.abs(), tolerances);

        Self {
            law,
            initial_value,
            current_value,
            absolute_change,
            relative_change,
            severity,
            step,
            time,
        }
    }

    /// Assess violation severity based on tolerances
    fn assess_severity(
        abs_error: f64,
        rel_error: f64,
        tolerances: &ConservationTolerances,
    ) -> ViolationSeverity {
        if abs_error < tolerances.absolute_tolerance && rel_error < tolerances.relative_tolerance {
            ViolationSeverity::Acceptable
        } else if abs_error < tolerances.absolute_tolerance * 10.0
            && rel_error < tolerances.relative_tolerance * 10.0
        {
            ViolationSeverity::Warning
        } else if abs_error < tolerances.absolute_tolerance * 100.0
            && rel_error < tolerances.relative_tolerance * 100.0
        {
            ViolationSeverity::Error
        } else {
            ViolationSeverity::Critical
        }
    }

    /// Check if violation is acceptable
    pub fn is_acceptable(&self) -> bool {
        self.severity == ViolationSeverity::Acceptable
    }

    /// Check if violation requires action
    pub fn requires_action(&self) -> bool {
        self.severity >= ViolationSeverity::Error
    }
}

impl fmt::Display for ConservationDiagnostic {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "[{}] {} Conservation: ΔQ = {:.3e} ({:.2e}%), Severity: {}",
            self.step,
            self.law,
            self.absolute_change,
            self.relative_change * 100.0,
            self.severity
        )
    }
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

impl Default for ConservationTolerances {
    fn default() -> Self {
        Self {
            absolute_tolerance: 1e-8, // 10⁻⁸ per step
            relative_tolerance: 1e-6, // 0.0001% cumulative
            check_interval: 100,      // Check every 100 steps
        }
    }
}

impl ConservationTolerances {
    /// Strict tolerances (for validation and testing)
    pub fn strict() -> Self {
        Self {
            absolute_tolerance: 1e-10,
            relative_tolerance: 1e-8,
            check_interval: 10,
        }
    }

    /// Relaxed tolerances (for production runs)
    pub fn relaxed() -> Self {
        Self {
            absolute_tolerance: 1e-6,
            relative_tolerance: 1e-4,
            check_interval: 1000,
        }
    }
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

        // Energy conservation
        diagnostics.push(self.check_energy_conservation(initial_energy, step, time, tolerances));

        // Momentum conservation (use magnitude for simplicity)
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

        // Mass conservation
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

impl ConservationTracker {
    /// Create new conservation tracker
    pub fn new(
        initial_energy: f64,
        initial_momentum: (f64, f64, f64),
        initial_mass: f64,
        tolerances: ConservationTolerances,
    ) -> Self {
        Self {
            initial_energy,
            initial_momentum,
            initial_mass,
            history: Vec::new(),
            tolerances,
            max_severity: ViolationSeverity::Acceptable,
        }
    }

    /// Update conservation diagnostics
    pub fn update<T: ConservationDiagnostics>(
        &mut self,
        solver: &T,
        step: usize,
        time: f64,
    ) -> Vec<ConservationDiagnostic> {
        // Only check at specified intervals
        if step % self.tolerances.check_interval != 0 {
            return Vec::new();
        }

        let diagnostics = solver.check_all_conservation(
            self.initial_energy,
            self.initial_momentum,
            self.initial_mass,
            step,
            time,
            &self.tolerances,
        );

        // Update max severity
        for diag in &diagnostics {
            if diag.severity > self.max_severity {
                self.max_severity = diag.severity;
            }
        }

        // Store in history
        self.history.extend(diagnostics.clone());

        diagnostics
    }

    /// Get summary statistics
    pub fn summary(&self) -> ConservationSummary {
        let energy_violations: Vec<_> = self
            .history
            .iter()
            .filter(|d| d.law == ConservationLaw::Energy)
            .collect();

        let max_energy_error = energy_violations
            .iter()
            .map(|d| d.relative_change.abs())
            .fold(0.0, f64::max);

        ConservationSummary {
            total_checks: self.history.len() / 3, // 3 laws per check
            max_severity: self.max_severity,
            max_energy_error,
            final_energy_error: energy_violations
                .last()
                .map(|d| d.relative_change.abs())
                .unwrap_or(0.0),
        }
    }

    /// Check if solution is valid (all violations acceptable)
    pub fn is_solution_valid(&self) -> bool {
        self.max_severity <= ViolationSeverity::Warning
    }

    /// Get critical violations
    pub fn critical_violations(&self) -> Vec<&ConservationDiagnostic> {
        self.history
            .iter()
            .filter(|d| d.severity == ViolationSeverity::Critical)
            .collect()
    }
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

/// Helper functions for conservation calculations
pub mod helpers {
    use super::*;

    /// Calculate acoustic energy density at a point
    ///
    /// e = (ρ₀/2)|u|² + p²/(2ρ₀c₀²)
    #[inline]
    pub fn acoustic_energy_density(
        pressure: f64,
        velocity: (f64, f64, f64),
        density: f64,
        sound_speed: f64,
    ) -> f64 {
        let kinetic =
            0.5 * density * (velocity.0.powi(2) + velocity.1.powi(2) + velocity.2.powi(2));
        let potential = pressure.powi(2) / (2.0 * density * sound_speed.powi(2));
        kinetic + potential
    }

    /// Calculate acoustic intensity (magnitude of energy flux)
    ///
    /// I = p·u (W/m²)
    #[inline]
    pub fn acoustic_intensity(pressure: f64, velocity: (f64, f64, f64)) -> f64 {
        let flux_x = pressure * velocity.0;
        let flux_y = pressure * velocity.1;
        let flux_z = pressure * velocity.2;
        (flux_x.powi(2) + flux_y.powi(2) + flux_z.powi(2)).sqrt()
    }

    /// Integrate field over volume (trapezoidal rule)
    pub fn integrate_field(field: &Array3<f64>, dx: f64, dy: f64, dz: f64) -> f64 {
        let dv = dx * dy * dz;
        field.sum() * dv
    }

    /// Calculate RMS (root mean square) of field
    pub fn field_rms(field: &Array3<f64>) -> f64 {
        let n = field.len() as f64;
        let sum_squares = field.iter().map(|&x| x * x).sum::<f64>();
        (sum_squares / n).sqrt()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_conservation_diagnostic_severity() {
        let tolerances = ConservationTolerances::default();

        // Acceptable violation
        let diag = ConservationDiagnostic::new(
            ConservationLaw::Energy,
            1000.0,
            1000.0 + 1e-9,
            100,
            0.1,
            &tolerances,
        );
        assert_eq!(diag.severity, ViolationSeverity::Acceptable);
        assert!(diag.is_acceptable());
        assert!(!diag.requires_action());

        // Critical violation
        let diag_critical = ConservationDiagnostic::new(
            ConservationLaw::Energy,
            1000.0,
            2000.0,
            100,
            0.1,
            &tolerances,
        );
        assert_eq!(diag_critical.severity, ViolationSeverity::Critical);
        assert!(diag_critical.requires_action());
    }

    #[test]
    fn test_conservation_tracker() {
        let tolerances = ConservationTolerances {
            check_interval: 1, // Check every step for testing
            ..Default::default()
        };

        let mut tracker = ConservationTracker::new(1000.0, (0.0, 0.0, 0.0), 100.0, tolerances);

        // Simulate energy drift
        struct MockSolver {
            energy: f64,
        }

        impl ConservationDiagnostics for MockSolver {
            fn calculate_total_energy(&self) -> f64 {
                self.energy
            }
            fn calculate_total_momentum(&self) -> (f64, f64, f64) {
                (0.0, 0.0, 0.0)
            }
            fn calculate_total_mass(&self) -> f64 {
                100.0
            }
        }

        let solver = MockSolver {
            energy: 1000.0 + 1e-9,
        }; // Tiny energy drift (acceptable)
        let diagnostics = tracker.update(&solver, 1, 0.001);

        assert_eq!(diagnostics.len(), 3); // Energy, momentum, mass
        assert_eq!(diagnostics[0].law, ConservationLaw::Energy);
        assert!(diagnostics[0].is_acceptable());

        // Test with larger drift (should trigger warning)
        let solver_warning = MockSolver { energy: 1001.0 }; // 0.1% drift
        let diagnostics_warning = tracker.update(&solver_warning, 2, 0.002);
        assert!(diagnostics_warning[0].severity >= ViolationSeverity::Warning);
    }

    #[test]
    fn test_energy_density_calculation() {
        let density = 1000.0; // kg/m³
        let sound_speed = 1500.0; // m/s
        let pressure = 1000.0; // Pa
        let velocity = (0.1, 0.0, 0.0); // m/s

        let energy_density =
            helpers::acoustic_energy_density(pressure, velocity, density, sound_speed);

        // Should be sum of kinetic and potential energy densities
        let kinetic = 0.5 * density * 0.1_f64.powi(2);
        let potential = pressure.powi(2) / (2.0 * density * sound_speed.powi(2));
        let expected = kinetic + potential;

        assert!((energy_density - expected).abs() < 1e-10);
    }

    #[test]
    fn test_conservation_tolerances() {
        let default = ConservationTolerances::default();
        assert!(default.absolute_tolerance > 0.0);
        assert!(default.relative_tolerance > 0.0);

        let strict = ConservationTolerances::strict();
        assert!(strict.absolute_tolerance < default.absolute_tolerance);

        let relaxed = ConservationTolerances::relaxed();
        assert!(relaxed.absolute_tolerance > default.absolute_tolerance);
    }

    #[test]
    fn test_field_integration() {
        let field = Array3::<f64>::ones((10, 10, 10));
        let dx = 0.1;
        let dy = 0.1;
        let dz = 0.1;

        let integral = helpers::integrate_field(&field, dx, dy, dz);
        let expected = 10.0 * 10.0 * 10.0 * 0.1 * 0.1 * 0.1; // Volume

        assert!((integral - expected).abs() < 1e-10);
    }

    #[test]
    fn test_field_rms() {
        let field = Array3::<f64>::from_elem((5, 5, 5), 2.0);
        let rms = helpers::field_rms(&field);
        assert!((rms - 2.0).abs() < 1e-10);
    }
}
