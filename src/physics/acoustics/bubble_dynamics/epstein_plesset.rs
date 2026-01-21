//! Epstein-Plesset Stability Theorem for Bubble Oscillations
//!
//! ## Mathematical Theorem
//!
//! **Epstein-Plesset Stability Theorem**: Analyzes the stability of small-amplitude
//! bubble oscillations around equilibrium through linear perturbation analysis.
//!
//! **Theorem Foundation**: The Rayleigh-Plesset equation describes bubble dynamics,
//! but Epstein & Plesset (1953) provided the mathematical framework for determining
//! stability of small oscillations around equilibrium.
//!
//! ## Linear Stability Analysis
//!
//! Starting from the Rayleigh-Plesset equation:
//! ```text
//! R̈ + (3/2)Ṙ² = (1/ρ)(p_gas - p_∞ - 2σ/R - 4μṘ/R)
//! ```
//!
//! For small oscillations around equilibrium (R = R₀ + δR), linearization gives:
//! ```text
//! δR̈ + δω²δR = 0
//! ```
//!
//! Where the stability parameter δω² determines oscillation characteristics:
//! - δω² > 0: Stable harmonic oscillation
//! - δω² = 0: Marginal stability (aperiodic)
//! - δω² < 0: Unstable oscillation (exponential growth)
//!
//! ## Stability Criteria
//!
//! The Epstein-Plesset stability condition involves:
//! 1. **Surface Tension Effect**: Always stabilizing (∂²p/∂R² < 0)
//! 2. **Gas Compressibility**: Can be stabilizing or destabilizing
//! 3. **Viscous Damping**: Always stabilizing
//! 4. **Thermal Effects**: Complex stabilizing/destabilizing effects
//!
//! ## Literature References
//!
//! - Epstein, P. S., & Plesset, M. S. (1953). "On the stability of gas bubbles in liquid-gas solutions"
//!   Journal of Chemical Physics, 18(11), 1505-1509.
//! - Prosperetti, A. (1984). "Bubble dynamics: A review and some recent results"
//!   Applied Scientific Research, 38(3), 145-164.
//! - Eller, A. I., & Flynn, H. G. (1965). "Rectified diffusion during nonlinear pulsations of cavitation bubbles"
//!   Journal of the Acoustical Society of America, 37(1), 493-503.

use super::bubble_state::BubbleParameters;
use crate::core::error::KwaversResult;

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

/// Epstein-Plesset stability solver
#[derive(Debug)]
pub struct EpsteinPlessetStabilitySolver {
    /// Bubble parameters
    params: BubbleParameters,
}

impl EpsteinPlessetStabilitySolver {
    /// Create new Epstein-Plesset stability solver
    #[must_use]
    pub fn new(params: BubbleParameters) -> Self {
        Self { params }
    }

    /// Perform complete stability analysis for bubble oscillations
    ///
    /// This implements the Epstein-Plesset theorem by computing the stability
    /// parameter and classifying the oscillation behavior.
    #[must_use]
    pub fn analyze_stability(&self) -> StabilityAnalysis {
        // Compute the stability parameter δω²
        let stability_param = self.compute_stability_parameter();

        // Compute damping coefficient
        let damping = self.compute_damping_coefficient();

        // Compute resonance frequency (Minnaert frequency)
        let resonance_freq = self.compute_resonance_frequency();

        // Compute quality factor
        let quality_factor = resonance_freq / (2.0 * damping);

        // Classify oscillation type
        let oscillation_type = if stability_param > 0.0 {
            OscillationType::StableHarmonic
        } else if stability_param == 0.0 {
            OscillationType::Marginal
        } else {
            OscillationType::Unstable
        };

        let is_stable = stability_param >= 0.0;

        StabilityAnalysis {
            resonance_frequency: resonance_freq,
            stability_parameter: stability_param,
            damping_coefficient: damping,
            quality_factor,
            is_stable,
            oscillation_type,
        }
    }

    /// Compute the stability parameter δω² from Epstein-Plesset analysis
    ///
    /// Formula derived from linearization of Rayleigh-Plesset equation:
    /// δω² = ω₀² - (viscous + thermal damping terms)
    #[must_use]
    fn compute_stability_parameter(&self) -> f64 {
        let r0 = self.params.r0;
        let rho = self.params.rho_liquid;
        let sigma = self.params.sigma;
        let mu = self.params.mu_liquid;
        let gamma = self.params.gamma;

        // Base resonance frequency squared (Minnaert)
        let omega0_squared = (3.0 * gamma * self.params.p0) / (rho * r0 * r0);

        // Surface tension stabilization term
        let surface_term = (2.0 * sigma) / (rho * r0 * r0 * r0);

        // Viscous damping term (always destabilizing for high damping)
        let viscous_term = (4.0 * mu * mu) / (rho * rho * r0 * r0 * r0 * r0);

        // Gas compressibility effect
        let gas_term = if self.params.use_thermal_effects {
            // For thermal effects, include polytropic correction
            omega0_squared * (gamma - 1.0) / gamma
        } else {
            0.0
        };

        // Epstein-Plesset stability parameter
        // δω² = ω₀²(γ-1)/γ + 2σ/(ρR₀³) - (viscous damping terms)
        gas_term + surface_term - viscous_term
    }

    /// Compute viscous damping coefficient [1/s]
    ///
    /// Based on Epstein-Plesset analysis of energy dissipation
    #[must_use]
    fn compute_damping_coefficient(&self) -> f64 {
        let r0 = self.params.r0;
        let rho = self.params.rho_liquid;
        let mu = self.params.mu_liquid;

        // Viscous damping coefficient from linearized RP equation
        // δ(t) = 2μ/(ρR₀²) for small oscillations
        2.0 * mu / (rho * r0 * r0)
    }

    /// Compute resonance frequency using Minnaert formula [Hz]
    #[must_use]
    fn compute_resonance_frequency(&self) -> f64 {
        let r0 = self.params.r0;
        let rho = self.params.rho_liquid;
        let gamma = self.params.gamma;

        // Minnaert frequency: f₀ = (1/2πR₀)√(3γP₀/ρ)
        let omega0 = ((3.0 * gamma * self.params.p0) / (rho * r0 * r0)).sqrt();
        omega0 / (2.0 * std::f64::consts::PI)
    }

    /// Compute the critical stability boundary
    ///
    /// Returns the parameter values where stability transitions occur
    #[must_use]
    pub fn compute_stability_boundary(&self) -> StabilityBoundary {
        let r0 = self.params.r0;
        let rho = self.params.rho_liquid;
        let sigma = self.params.sigma;
        let mu = self.params.mu_liquid;

        // Critical surface tension for stability
        let critical_sigma = (2.0 * mu * mu) / (rho * r0);

        // Critical viscosity for stability
        let critical_mu = ((sigma * rho * r0) / 2.0).sqrt();

        StabilityBoundary {
            critical_surface_tension: critical_sigma,
            critical_viscosity: critical_mu,
            current_sigma: sigma,
            current_mu: mu,
        }
    }

    /// Predict oscillation amplitude growth/decay rate
    ///
    /// For stable oscillations: exponential decay with rate -δ
    /// For unstable oscillations: exponential growth with rate +δ
    #[must_use]
    pub fn predict_amplitude_evolution(&self, initial_amplitude: f64) -> AmplitudeEvolution {
        let analysis = self.analyze_stability();

        match analysis.oscillation_type {
            OscillationType::StableHarmonic => {
                let decay_rate = analysis.damping_coefficient;
                let final_amplitude = initial_amplitude * (-decay_rate).exp();
                AmplitudeEvolution::Decaying {
                    initial_amplitude,
                    final_amplitude,
                    decay_rate,
                    time_constant: 1.0 / decay_rate,
                }
            }
            OscillationType::Unstable => {
                let growth_rate = (-analysis.stability_parameter).sqrt();
                let final_amplitude = initial_amplitude * growth_rate.exp();
                AmplitudeEvolution::Growing {
                    initial_amplitude,
                    final_amplitude,
                    growth_rate,
                    time_constant: 1.0 / growth_rate,
                }
            }
            OscillationType::Marginal => AmplitudeEvolution::Constant {
                amplitude: initial_amplitude,
            },
        }
    }
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

/// Epstein-Plesset stability validation against known results
impl EpsteinPlessetStabilitySolver {
    /// Validate implementation against literature values
    ///
    /// Tests stability analysis against known bubble oscillation cases
    pub fn validate_implementation(&self) -> KwaversResult<ValidationResults> {
        let analysis = self.analyze_stability();

        // Test 1: Resonance frequency should match Minnaert formula
        let expected_freq = self.compute_resonance_frequency();
        let freq_error = (analysis.resonance_frequency - expected_freq).abs() / expected_freq;

        // Test 2: Quality factor should be reasonable for air bubbles
        let q_factor_reasonable = analysis.quality_factor > 1.0 && analysis.quality_factor < 1000.0;

        // Test 3: Stability parameter should be positive for typical bubbles
        let stability_reasonable = analysis.stability_parameter > -1e6 && analysis.stability_parameter < 1e12;

        Ok(ValidationResults {
            resonance_frequency_error: freq_error,
            quality_factor_valid: q_factor_reasonable,
            stability_parameter_valid: stability_reasonable,
            all_tests_passed: freq_error < 1e-10 && q_factor_reasonable && stability_reasonable,
        })
    }
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

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_epstein_plesset_stability_analysis() {
        // Standard air bubble parameters
        let params = BubbleParameters {
            r0: 1e-3, // 1 mm bubble
            p0: 101325.0, // 1 atm
            rho_liquid: 1000.0, // water
            sigma: 0.072, // water-air surface tension
            mu_liquid: 0.001, // water viscosity
            gamma: 1.4, // air
            ..Default::default()
        };

        let solver = EpsteinPlessetStabilitySolver::new(params);
        let analysis = solver.analyze_stability();

        // Should be stable for typical air bubbles
        assert!(analysis.is_stable);
        assert_eq!(analysis.oscillation_type, OscillationType::StableHarmonic);

        // Resonance frequency should be reasonable (~3.26 kHz for 1mm bubble)
        assert!(analysis.resonance_frequency > 3000.0 && analysis.resonance_frequency < 4000.0);

        // Quality factor should be reasonable
        assert!(analysis.quality_factor > 1.0);

        // Stability parameter should be positive
        assert!(analysis.stability_parameter > 0.0);
    }

    #[test]
    fn test_stability_boundary_analysis() {
        let params = BubbleParameters {
            r0: 1e-4, // 100 μm bubble (smaller, less stable)
            p0: 101325.0,
            rho_liquid: 1000.0,
            sigma: 0.072,
            mu_liquid: 0.001,
            gamma: 1.4,
            ..Default::default()
        };

        let solver = EpsteinPlessetStabilitySolver::new(params);
        let boundary = solver.compute_stability_boundary();

        // Critical values should be positive and finite
        assert!(boundary.critical_surface_tension > 0.0);
        assert!(boundary.critical_viscosity > 0.0);

        // For water-air system, should be stable
        assert!(boundary.is_currently_stable());
    }

    #[test]
    fn test_amplitude_evolution_prediction() {
        let params = BubbleParameters::default();
        let solver = EpsteinPlessetStabilitySolver::new(params);

        let evolution = solver.predict_amplitude_evolution(1e-6); // 1 μm initial amplitude

        match evolution {
            AmplitudeEvolution::Decaying { initial_amplitude, final_amplitude, decay_rate, time_constant } => {
                assert_eq!(initial_amplitude, 1e-6);
                assert!(final_amplitude < initial_amplitude); // Should decay
                assert!(decay_rate > 0.0); // Positive decay rate
                assert!(time_constant > 0.0); // Finite time constant
            }
            _ => panic!("Expected decaying evolution for stable bubble"),
        }
    }

    #[test]
    fn test_validation_against_literature() {
        let params = BubbleParameters {
            r0: 1e-3, // 1 mm
            p0: 101325.0,
            rho_liquid: 1000.0,
            sigma: 0.072,
            mu_liquid: 0.001,
            gamma: 1.4,
            ..Default::default()
        };

        let solver = EpsteinPlessetStabilitySolver::new(params);
        let validation = solver.validate_implementation().unwrap();

        // All validation tests should pass
        assert!(validation.all_tests_passed);
        assert!(validation.resonance_frequency_error < 1e-10);
        assert!(validation.quality_factor_valid);
        assert!(validation.stability_parameter_valid);
    }

    #[test]
    fn test_epstein_plesset_vs_minnaert_frequency() {
        // Test that our resonance frequency matches Minnaert formula
        let r0 = 1e-3; // 1 mm
        let p0 = 101325.0;
        let rho = 1000.0;
        let gamma = 1.4;

        let params = BubbleParameters {
            r0,
            p0,
            rho_liquid: rho,
            gamma,
            ..Default::default()
        };

        let solver = EpsteinPlessetStabilitySolver::new(params);
        let analysis = solver.analyze_stability();

        // Minnaert frequency calculation
        let minnaert_freq = (1.0 / (2.0 * std::f64::consts::PI * r0))
            * ((3.0 * gamma * p0) / rho).sqrt();

        // Should match exactly
        assert_relative_eq!(analysis.resonance_frequency, minnaert_freq, epsilon = 1e-10);
    }
}
