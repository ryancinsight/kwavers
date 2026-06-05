//! Shelled-microbubble dissolution with a gas-permeation resistance.

use super::epstein_plesset::EpsteinPlessetDissolution;
use super::traits::{sealed, DissolutionModel, GasDiffusionParams};

/// Encapsulated-microbubble dissolution: Epstein–Plesset diffusion in series
/// with the lipid/protein shell's finite gas permeability (Sarkar 2009).
///
/// ## Model
///
/// The shell adds a surface resistance to gas transport in series with the
/// liquid-side diffusive resistance. The diffusive interfacial transfer
/// coefficient `D/R` and the shell permeation coefficient `k_s` [m/s] combine in
/// series, so the Epstein–Plesset radius rate is reduced by the factor
/// ```text
///   1 / (1 + D/(R·k_s))
/// ```
/// * `k_s → ∞` (no shell): recovers bare Epstein–Plesset.
/// * `k_s → 0` (impermeable shell): `dR/dt → 0` — the microbubble is stabilised.
///
/// This is why coated contrast agents (SonoVue, Definity) persist far longer
/// than free bubbles of the same size, and it sets the residual-bubble
/// dissolution time τ_d that governs inter-pulse shielding.
///
/// ## Reference
/// Sarkar K, Katiyar A, Jain P (2009). *Ann. Biomed. Eng.* 37, 2196
/// (gas permeation resistance of encapsulated microbubbles).
#[derive(Debug, Clone, Copy)]
pub struct ShellPermeationDissolution {
    core: EpsteinPlessetDissolution,
    /// Shell gas-permeation coefficient `k_s` [m/s] (lipid ≈ 1e-7–1e-5).
    pub shell_permeability: f64,
}

impl ShellPermeationDissolution {
    /// Build from gas-diffusion parameters and a shell permeation coefficient.
    #[must_use]
    pub fn new(params: GasDiffusionParams, shell_permeability_m_s: f64) -> Self {
        Self {
            core: EpsteinPlessetDissolution::new(params),
            shell_permeability: shell_permeability_m_s.max(0.0),
        }
    }

    /// Typical lipid-shell SonoVue/Definity-like microbubble (`k_s ≈ 1e-6 m/s`).
    #[must_use]
    pub fn lipid_shell(params: GasDiffusionParams) -> Self {
        Self::new(params, 1.0e-6)
    }
}

impl sealed::Sealed for ShellPermeationDissolution {}

impl DissolutionModel for ShellPermeationDissolution {
    fn radius_rate(&self, radius_m: f64, time_s: f64) -> f64 {
        let r = radius_m.max(1e-15);
        let bare = self.core.radius_rate(r, time_s);
        // Series diffusion + shell-permeation resistance.
        let resistance = if self.shell_permeability > 0.0 {
            1.0 + self.core.params().diffusivity / (r * self.shell_permeability)
        } else {
            f64::INFINITY
        };
        bare / resistance
    }

    fn params(&self) -> &GasDiffusionParams {
        self.core.params()
    }

    fn dissolution_time(&self, r0_m: f64) -> Option<f64> {
        // Free-bubble estimate scaled by the shell resistance at R₀ (the shell
        // slows dissolution by ≈ 1 + D/(R₀ k_s)); used for integrator step/window
        // sizing — the exact value comes from numerical integration.
        self.core.dissolution_time(r0_m).map(|t| {
            let resistance = if self.shell_permeability > 0.0 {
                1.0 + self.core.params().diffusivity / (r0_m.max(1e-15) * self.shell_permeability)
            } else {
                1.0
            };
            t * resistance
        })
    }
}
