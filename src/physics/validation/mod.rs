//! Physics validation module with domain-based organization
//! 
//! Validates numerical implementations against analytical solutions
//! and published benchmarks from peer-reviewed literature.

pub mod wave_equations;
pub mod nonlinear;
pub mod materials;
pub mod numerical_methods;
pub mod conservation;

// Re-export validation traits
pub use wave_equations::WaveEquationValidator;
pub use nonlinear::NonlinearValidator;
pub use materials::MaterialValidator;
pub use numerical_methods::NumericalValidator;
pub use conservation::ConservationValidator;

/// Common validation result type
#[derive(Debug, Clone)]
pub struct ValidationMetrics {
    pub l2_error: f64,
    pub linf_error: f64,
    pub relative_error: f64,
    pub convergence_rate: Option<f64>,
}

impl ValidationMetrics {
    pub fn new(l2: f64, linf: f64, relative: f64) -> Self {
        Self {
            l2_error: l2,
            linf_error: linf,
            relative_error: relative,
            convergence_rate: None,
        }
    }
    
    pub fn passes_tolerance(&self, tol: f64) -> bool {
        self.relative_error < tol
    }
}