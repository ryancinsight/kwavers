use super::shell_model::EncapsulatedShellModel;
use crate::acoustics::bubble_dynamics::bubble_state::{BubbleParameters, BubbleState};
use kwavers_core::error::KwaversResult;

/// Sarkar, Shi, Chatterjee & Forsberg (2005) interfacial encapsulated-bubble model.
///
/// Sarkar uses **interfacial (surface) rheology** rather than a distributed shell:
/// shell elasticity is folded into a radius-dependent effective surface tension
/// (like Marmottant but without buckling/rupture branches), and shell damping is
/// a surface dilatational viscosity:
///
/// ```text
/// σ(R)      = σ0 + E_s (R²/R0² − 1)          interfacial dilatational elasticity E_s [N/m]
/// S_viscous = 4 κ_s Ṙ/R²                      surface dilatational viscosity κ_s [kg/s]
/// ```
///
/// The elastic stress enters through the Laplace term `2σ(R)/R`; there is no
/// separate elastic shell stress. With the thin-shell identities `E_s = 3 G_s d`
/// and `κ_s = 3 μ_s d`, `4 κ_s = 12 μ_s d` recovers the Church/Hoff viscous
/// prefactor — the consistency check between the interfacial and distributed forms.
///
/// # Evidence tier
/// Literature (Sarkar et al. 2005; Doinikov & Bouakaz 2011) — the `σ(R)` form and
/// the `4 κ_s Ṙ/R²` viscous term are unambiguous across the literature. Validated
/// here by the analytic equilibrium balance and damping/restoring-sign properties.
#[derive(Debug, Clone)]
pub struct SarkarModel {
    params: BubbleParameters,
    /// Reference interfacial tension σ0 [N/m] (value of σ at R = R0).
    sigma_0: f64,
    /// Interfacial dilatational elasticity E_s [N/m].
    elasticity: f64,
    /// Interfacial dilatational (surface) viscosity κ_s [kg/s].
    surface_viscosity: f64,
}

impl SarkarModel {
    /// Create a Sarkar model.
    ///
    /// # Arguments
    /// * `params` — bubble/liquid parameters.
    /// * `sigma_0` — reference interfacial tension σ0 [N/m].
    /// * `elasticity` — interfacial dilatational elasticity E_s [N/m].
    /// * `surface_viscosity` — interfacial dilatational viscosity κ_s [kg/s].
    #[must_use]
    pub fn new(params: BubbleParameters, sigma_0: f64, elasticity: f64, surface_viscosity: f64) -> Self {
        Self {
            params,
            sigma_0,
            elasticity,
            surface_viscosity,
        }
    }

    /// Effective interfacial tension σ(R) = σ0 + E_s (R²/R0² − 1) [N/m].
    #[must_use]
    pub fn surface_tension(&self, r: f64) -> f64 {
        let r0 = self.params.r0;
        self.elasticity
            .mul_add((r / r0).powi(2) - 1.0, self.sigma_0)
    }

    /// Calculate bubble-wall acceleration (delegates to the shared RP driver).
    /// # Errors
    /// - Returns [`Err`] if an internal constraint is violated.
    pub fn calculate_acceleration(
        &self,
        state: &mut BubbleState,
        p_acoustic: f64,
        t: f64,
    ) -> KwaversResult<f64> {
        EncapsulatedShellModel::acceleration(self, state, p_acoustic, t)
    }
}

impl EncapsulatedShellModel for SarkarModel {
    fn params(&self) -> &BubbleParameters {
        &self.params
    }

    fn equilibrium_gas_pressure(&self) -> f64 {
        // σ(R0) = σ0, so the static balance reference is p0 + 2σ0/R0.
        self.params.p0 + 2.0 * self.sigma_0 / self.params.r0
    }

    fn effective_surface_tension(&self, r: f64) -> f64 {
        self.surface_tension(r)
    }

    fn shell_stress(&self, r: f64, v: f64) -> f64 {
        // Surface dilatational viscosity: 4 κ_s Ṙ/R². No separate elastic stress
        // (elasticity is in σ(R)).
        4.0 * self.surface_viscosity * v / (r * r)
    }
}
