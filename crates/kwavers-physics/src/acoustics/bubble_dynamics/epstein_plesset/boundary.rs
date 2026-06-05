//! Stability boundary and amplitude evolution predictions

use super::solver::EpsteinPlessetStabilitySolver;
use super::types::{AmplitudeEvolution, OscillationType, StabilityBoundary};

impl EpsteinPlessetStabilitySolver {
    /// Compute the critical stability boundary
    #[must_use]
    pub fn compute_stability_boundary(&self) -> StabilityBoundary {
        let r0 = self.params.r0;
        let rho = self.params.rho_liquid;
        let sigma = self.params.sigma;
        let mu = self.params.mu_liquid;

        let critical_sigma = (2.0 * mu * mu) / (rho * r0);
        let critical_mu = ((sigma * rho * r0) / 2.0).sqrt();

        StabilityBoundary {
            critical_surface_tension: critical_sigma,
            critical_viscosity: critical_mu,
            current_sigma: sigma,
            current_mu: mu,
        }
    }

    /// Predict oscillation amplitude growth/decay rate
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
