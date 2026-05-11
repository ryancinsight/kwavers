//! `ConservationChecker` — initialization, per-step verification, and reset.

use std::collections::HashMap;

use log::{debug, warn};
use ndarray::Array3;

use crate::core::error::KwaversResult;
use crate::domain::grid::Grid;

use super::types::{ConservationLaw, ConservationResult, ConservedQuantity};

/// Conservation law checker for multiphysics simulations.
#[derive(Debug)]
pub struct ConservationChecker {
    /// Initial conserved quantities (keyed by field name).
    initial_quantities: HashMap<String, ConservedQuantity>,
    /// Relative error tolerance.
    tolerance: f64,
    /// Grid for volume-element computation.
    grid: Grid,
    /// Number of checks performed.
    pub(super) check_count: u64,
    /// Enable verbose diagnostic output.
    verbose: bool,
}

impl ConservationChecker {
    /// Create a new conservation checker.
    ///
    /// # Arguments
    /// * `grid`      — Computational grid.
    /// * `tolerance` — Relative error tolerance (typical: `1e-6`).
    pub fn new(grid: Grid, tolerance: f64) -> Self {
        Self {
            initial_quantities: HashMap::new(),
            tolerance,
            grid,
            check_count: 0,
            verbose: false,
        }
    }

    /// Initialize conservation checker with baseline field integrals.
    ///
    /// Establishes reference values for all named fields. Call once at
    /// simulation start before the first `check`.
    ///
    /// # Errors
    /// Always returns `Ok`; signature matches callers that propagate errors.
    pub fn initialize(
        &mut self,
        fields: &HashMap<String, Array3<f64>>,
        field_names: &[&str],
    ) -> KwaversResult<HashMap<String, ConservedQuantity>> {
        self.initial_quantities.clear();

        for name in field_names {
            if let Some(field) = fields.get(*name) {
                let integral = self.compute_integral(field);
                let magnitude = field.iter().map(|v| v.abs()).fold(0.0, f64::max);

                self.initial_quantities.insert(
                    name.to_string(),
                    ConservedQuantity {
                        name: name.to_string(),
                        integral,
                        magnitude,
                        time: 0.0,
                    },
                );
            }
        }

        if self.verbose {
            debug!(
                "Conservation checker initialized with {} fields",
                field_names.len()
            );
        }

        Ok(self.initial_quantities.clone())
    }

    /// Verify conservation laws against baseline integrals.
    ///
    /// Returns one `ConservationResult` per initialized field name.
    ///
    /// # Errors
    /// Always returns `Ok`; signature matches callers that propagate errors.
    pub fn check(
        &mut self,
        fields: &HashMap<String, Array3<f64>>,
        _time: f64,
    ) -> KwaversResult<HashMap<String, ConservationResult>> {
        let mut results = HashMap::new();

        for (name, initial) in &self.initial_quantities {
            if let Some(field) = fields.get(name) {
                let current_integral = self.compute_integral(field);
                let absolute_change = (current_integral - initial.integral).abs();
                let relative_error = if initial.integral.abs() > 1e-15 {
                    absolute_change / initial.integral.abs()
                } else {
                    0.0
                };

                let passed = relative_error <= self.tolerance;
                let error_message = if !passed {
                    Some(format!(
                        "Conservation violation for {}: relative error {:.3e} exceeds tolerance {:.3e}",
                        name, relative_error, self.tolerance
                    ))
                } else {
                    None
                };

                let result = ConservationResult {
                    law: self.infer_law_from_name(name),
                    initial_value: initial.integral,
                    current_value: current_integral,
                    absolute_change,
                    relative_error,
                    tolerance: self.tolerance,
                    passed,
                    error_message: error_message.clone(),
                };

                results.insert(name.clone(), result);

                if self.verbose {
                    if let Some(msg) = error_message.as_ref() {
                        warn!("{}", msg);
                    }
                }
            }
        }

        self.check_count += 1;
        Ok(results)
    }

    /// Return number of `check` calls performed since construction or `reset`.
    pub fn check_count(&self) -> u64 {
        self.check_count
    }

    /// Reset checker to initial state (clears baselines and count).
    pub fn reset(&mut self) {
        self.initial_quantities.clear();
        self.check_count = 0;
    }

    /// Enable or disable verbose diagnostic warnings.
    pub fn set_verbose(&mut self, verbose: bool) {
        self.verbose = verbose;
    }

    // ── Private helpers ───────────────────────────────────────────────────────

    /// Volume integral `∫f dV` via rectangular quadrature.
    fn compute_integral(&self, field: &Array3<f64>) -> f64 {
        let dv = self.grid.dx * self.grid.dy * self.grid.dz;
        field.iter().sum::<f64>() * dv
    }

    /// Infer `ConservationLaw` from a field name by keyword matching.
    pub(super) fn infer_law_from_name(&self, name: &str) -> ConservationLaw {
        let lower = name.to_lowercase();
        if lower.contains("mass") || lower.contains("rho") || lower.contains("density") {
            ConservationLaw::Mass
        } else if lower.contains("momentum") || lower.contains("velocity") {
            ConservationLaw::Momentum
        } else if lower.contains("energy") || lower.contains("thermal") {
            ConservationLaw::Energy
        } else {
            ConservationLaw::Charge
        }
    }
}
