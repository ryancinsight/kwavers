use std::fmt;

use super::{ConservationDiagnostic, ConservationLaw, ConservationTolerances, ViolationSeverity};

impl ConservationDiagnostic {
    #[must_use] 
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

    /// Is acceptable.
    /// # Errors
    /// - Returns [`Err`] if an internal constraint is violated.
    ///
    #[must_use] 
    pub fn is_acceptable(&self) -> bool {
        self.severity == ViolationSeverity::Acceptable
    }

    /// Requires action.
    /// # Errors
    /// - Returns [`Err`] if an internal constraint is violated.
    ///
    #[must_use] 
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
