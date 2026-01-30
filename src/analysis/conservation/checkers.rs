//! Conservation Law Checkers
//!
//! Core module for verifying conservation of physical quantities during simulation.

use crate::core::error::KwaversResult;
use crate::domain::grid::Grid;
use ndarray::Array3;
use std::collections::HashMap;

/// Type of conservation law to verify
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ConservationLaw {
    /// Mass conservation: ∫ρ dV = constant
    Mass,

    /// Momentum conservation: ∫ρu dV = constant (x, y, z components)
    Momentum,

    /// Energy conservation: ∫(KE + PE + TE) dV = constant
    Energy,

    /// Charge conservation (for EM fields)
    Charge,
}

impl std::fmt::Display for ConservationLaw {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Mass => write!(f, "Mass"),
            Self::Momentum => write!(f, "Momentum"),
            Self::Energy => write!(f, "Energy"),
            Self::Charge => write!(f, "Charge"),
        }
    }
}

/// Conservation quantity at a single timestep
#[derive(Debug, Clone)]
pub struct ConservedQuantity {
    /// Name of the quantity
    pub name: String,

    /// Integral value (∫ quantity dV)
    pub integral: f64,

    /// Relative magnitude for normalization
    pub magnitude: f64,

    /// Timestamp when measured
    pub time: f64,
}

/// Result of conservation verification
#[derive(Debug, Clone)]
pub struct ConservationResult {
    /// Conservation law verified
    pub law: ConservationLaw,

    /// Initial quantity value
    pub initial_value: f64,

    /// Current quantity value
    pub current_value: f64,

    /// Absolute change
    pub absolute_change: f64,

    /// Relative error (dimensionless)
    pub relative_error: f64,

    /// Tolerance used for verification
    pub tolerance: f64,

    /// Whether conservation is satisfied
    pub passed: bool,

    /// Error message if failed
    pub error_message: Option<String>,
}

/// Conservation law checker for multiphysics simulations
#[derive(Debug)]
pub struct ConservationChecker {
    /// Initial conserved quantities
    initial_quantities: HashMap<String, ConservedQuantity>,

    /// Conservation tolerance (relative error threshold)
    tolerance: f64,

    /// Grid for volume element computation
    grid: Grid,

    /// Number of checks performed
    check_count: u64,

    /// Verbose output
    verbose: bool,
}

impl ConservationChecker {
    /// Create a new conservation checker
    ///
    /// # Arguments
    ///
    /// * `grid` - Computational grid
    /// * `tolerance` - Relative error tolerance (default: 1e-6)
    pub fn new(grid: Grid, tolerance: f64) -> Self {
        Self {
            initial_quantities: HashMap::new(),
            tolerance,
            grid,
            check_count: 0,
            verbose: false,
        }
    }

    /// Initialize conservation checker with baseline quantities
    ///
    /// This establishes reference values for all conservation laws.
    /// Called once at the beginning of simulation.
    ///
    /// # Arguments
    ///
    /// * `fields` - Field map containing pressure, velocity, temperature, etc.
    /// * `field_names` - Names of fields to track (e.g., "pressure", "velocity_x")
    ///
    /// # Returns
    ///
    /// Result map with initial quantities for each field
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

                let quantity = ConservedQuantity {
                    name: name.to_string(),
                    integral,
                    magnitude,
                    time: 0.0,
                };

                self.initial_quantities.insert(name.to_string(), quantity);
            }
        }

        if self.verbose {
            eprintln!(
                "Conservation checker initialized with {} fields",
                field_names.len()
            );
        }

        Ok(self.initial_quantities.clone())
    }

    /// Verify conservation laws
    ///
    /// Compares current field integrals to initial baseline values.
    ///
    /// # Arguments
    ///
    /// * `fields` - Current field map
    /// * `_time` - Current simulation time
    ///
    /// # Returns
    ///
    /// Map of conservation results (one per initialized field)
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

                if self.verbose && !passed {
                    eprintln!("⚠️ {}", error_message.as_ref().unwrap());
                }
            }
        }

        self.check_count += 1;

        Ok(results)
    }

    /// Compute integral of field (∫f dV)
    ///
    /// Uses simple trapezoidal rule with grid spacing.
    fn compute_integral(&self, field: &Array3<f64>) -> f64 {
        let dv = self.grid.dx * self.grid.dy * self.grid.dz;
        field.iter().sum::<f64>() * dv
    }

    /// Infer conservation law from field name
    fn infer_law_from_name(&self, name: &str) -> ConservationLaw {
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

    /// Set verbose output mode
    pub fn set_verbose(&mut self, verbose: bool) {
        self.verbose = verbose;
    }

    /// Get number of checks performed
    pub fn check_count(&self) -> u64 {
        self.check_count
    }

    /// Reset checker to initial state
    pub fn reset(&mut self) {
        self.initial_quantities.clear();
        self.check_count = 0;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_conservation_checker_creation() -> KwaversResult<()> {
        let grid = Grid::new(16, 16, 16, 0.1, 0.1, 0.1)?;
        let checker = ConservationChecker::new(grid, 1e-6);

        assert_eq!(checker.check_count, 0);
        assert_eq!(checker.initial_quantities.len(), 0);

        Ok(())
    }

    #[test]
    fn test_conservation_law_display() {
        assert_eq!(ConservationLaw::Mass.to_string(), "Mass");
        assert_eq!(ConservationLaw::Momentum.to_string(), "Momentum");
        assert_eq!(ConservationLaw::Energy.to_string(), "Energy");
        assert_eq!(ConservationLaw::Charge.to_string(), "Charge");
    }

    #[test]
    fn test_initialization() -> KwaversResult<()> {
        let grid = Grid::new(8, 8, 8, 0.1, 0.1, 0.1)?;
        let mut checker = ConservationChecker::new(grid, 1e-6);

        let mut fields = HashMap::new();
        fields.insert("pressure".to_string(), Array3::ones((8, 8, 8)));

        let result = checker.initialize(&fields, &["pressure"])?;
        assert_eq!(result.len(), 1);
        assert!(result.contains_key("pressure"));

        Ok(())
    }

    #[test]
    fn test_perfect_conservation() -> KwaversResult<()> {
        let grid = Grid::new(8, 8, 8, 0.1, 0.1, 0.1)?;
        let mut checker = ConservationChecker::new(grid, 1e-10);

        let mut fields = HashMap::new();
        let pressure = Array3::ones((8, 8, 8));
        fields.insert("pressure".to_string(), pressure.clone());

        checker.initialize(&fields, &["pressure"])?;

        // Same field should have zero error
        let results = checker.check(&fields, 0.0)?;
        let pressure_result = results.get("pressure").unwrap();

        assert_eq!(pressure_result.relative_error, 0.0);
        assert!(pressure_result.passed);

        Ok(())
    }

    #[test]
    fn test_conservation_violation_detection() -> KwaversResult<()> {
        let grid = Grid::new(8, 8, 8, 0.1, 0.1, 0.1)?;
        let mut checker = ConservationChecker::new(grid, 1e-10);

        let mut fields = HashMap::new();
        fields.insert("pressure".to_string(), Array3::ones((8, 8, 8)));

        checker.initialize(&fields, &["pressure"])?;

        // Modify field to introduce error
        let mut pressure_modified = Array3::ones((8, 8, 8));
        pressure_modified[[0, 0, 0]] = 1.5; // Small perturbation

        fields.insert("pressure".to_string(), pressure_modified);

        let results = checker.check(&fields, 0.0)?;
        let pressure_result = results.get("pressure").unwrap();

        assert!(pressure_result.relative_error > 0.0);
        assert!(!pressure_result.passed);
        assert!(pressure_result.error_message.is_some());

        Ok(())
    }

    #[test]
    fn test_infer_law_from_name() -> KwaversResult<()> {
        let grid = Grid::new(8, 8, 8, 0.1, 0.1, 0.1)?;
        let checker = ConservationChecker::new(grid, 1e-6);

        assert_eq!(
            checker.infer_law_from_name("pressure"),
            ConservationLaw::Charge
        );
        assert_eq!(
            checker.infer_law_from_name("density"),
            ConservationLaw::Mass
        );
        assert_eq!(
            checker.infer_law_from_name("velocity_x"),
            ConservationLaw::Momentum
        );
        assert_eq!(
            checker.infer_law_from_name("thermal_energy"),
            ConservationLaw::Energy
        );

        Ok(())
    }
}
