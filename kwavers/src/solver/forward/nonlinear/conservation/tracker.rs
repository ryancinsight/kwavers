use super::{
    ConservationDiagnostic, ConservationDiagnostics, ConservationLaw, ConservationSummary,
    ConservationTolerances, ConservationTracker, ViolationSeverity,
};

impl ConservationTracker {
    #[must_use] 
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

    pub fn update<T: ConservationDiagnostics>(
        &mut self,
        solver: &T,
        step: usize,
        time: f64,
    ) -> Vec<ConservationDiagnostic> {
        if !step.is_multiple_of(self.tolerances.check_interval) {
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

        for diag in &diagnostics {
            if diag.severity > self.max_severity {
                self.max_severity = diag.severity;
            }
        }

        self.history.extend(diagnostics.clone());

        diagnostics
    }

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
            total_checks: self.history.len() / 3,
            max_severity: self.max_severity,
            max_energy_error,
            final_energy_error: energy_violations
                .last()
                .map_or(0.0, |d| d.relative_change.abs()),
        }
    }

    #[must_use] 
    pub fn is_solution_valid(&self) -> bool {
        self.max_severity <= ViolationSeverity::Warning
    }

    #[must_use] 
    pub fn critical_violations(&self) -> Vec<&ConservationDiagnostic> {
        self.history
            .iter()
            .filter(|d| d.severity == ViolationSeverity::Critical)
            .collect()
    }
}
