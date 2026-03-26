//! Result types for Epstein-Plesset stability analysis

/// Epstein-Plesset stability analysis results
#[derive(Debug, Clone, PartialEq)]
pub struct StabilityAnalysis {
    /// Natural resonance frequency [Hz]
    pub resonance_frequency: f64,
    /// Stability parameter (δω²) - determines oscillation type
    pub stability_parameter: f64,
    /// Damping coefficient [1/s]
    pub damping_coefficient: f64,
    /// Quality factor (dimensionless)
    pub quality_factor: f64,
    /// Is the oscillation stable?
    pub is_stable: bool,
    /// Oscillation classification
    pub oscillation_type: OscillationType,
}

/// Classification of bubble oscillation behavior
#[derive(Debug, Clone, PartialEq)]
pub enum OscillationType {
    /// Stable harmonic oscillation (δω² > 0)
    StableHarmonic,
    /// Marginally stable (aperiodic, δω² = 0)
    Marginal,
    /// Unstable oscillation (δω² < 0, exponential growth)
    Unstable,
}

/// Stability boundary analysis
#[derive(Debug, Clone, PartialEq)]
pub struct StabilityBoundary {
    /// Critical surface tension for stability [N/m]
    pub critical_surface_tension: f64,
    /// Critical viscosity for stability [Pa·s]
    pub critical_viscosity: f64,
    /// Current surface tension [N/m]
    pub current_sigma: f64,
    /// Current viscosity [Pa·s]
    pub current_mu: f64,
}

impl StabilityBoundary {
    /// Check if current parameters are in stable region
    #[must_use]
    pub fn is_currently_stable(&self) -> bool {
        self.current_sigma >= self.critical_surface_tension
            && self.current_mu <= self.critical_viscosity
    }
}

/// Predicted amplitude evolution
#[derive(Debug, Clone, PartialEq)]
pub enum AmplitudeEvolution {
    /// Exponential decay to equilibrium
    Decaying {
        initial_amplitude: f64,
        final_amplitude: f64,
        decay_rate: f64,
        time_constant: f64,
    },
    /// Exponential growth (unstable)
    Growing {
        initial_amplitude: f64,
        final_amplitude: f64,
        growth_rate: f64,
        time_constant: f64,
    },
    /// Constant amplitude (marginally stable)
    Constant { amplitude: f64 },
}

/// Validation results for Epstein-Plesset implementation
#[derive(Debug, Clone, PartialEq)]
pub struct ValidationResults {
    /// Relative error in resonance frequency calculation
    pub resonance_frequency_error: f64,
    /// Quality factor is in reasonable range
    pub quality_factor_valid: bool,
    /// Stability parameter is physically reasonable
    pub stability_parameter_valid: bool,
    /// All validation tests passed
    pub all_tests_passed: bool,
}
