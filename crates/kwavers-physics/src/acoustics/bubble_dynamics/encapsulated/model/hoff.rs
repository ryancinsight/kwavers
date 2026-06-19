use super::super::shell::ShellProperties;
use super::shell_model::EncapsulatedShellModel;
use crate::acoustics::bubble_dynamics::bubble_state::{BubbleParameters, BubbleState};
use kwavers_core::error::KwaversResult;

/// Hoff, Sontum & Hovem (2000) thin-shell encapsulated-bubble model.
///
/// Like Church (1995) it parameterizes the shell by its shear modulus `G_s`,
/// shear viscosity `μ_s`, and thickness `d`, but its **elastic restoring stress
/// is linear in the relative displacement** `[1 − R0/R]` rather than Church's
/// quadratic strain `[(R/R0)² − 1]`. The viscous damping stress is identical to
/// Church (`12 μ_s d Ṙ/R²`), so with `G_s = 0` the two models coincide exactly.
///
/// # Equations (Doinikov & Bouakaz 2011 review of Hoff et al. 2000)
/// ```text
/// S_elastic = 12 G_s (d/R) [1 − R0/R]
/// S_viscous = 12 μ_s (d/R) Ṙ/R = 12 μ_s d Ṙ/R²
/// ```
/// This is the common thin-shell linearization. Hoff's original incompressible
/// shell additionally scales both terms by `(R0/R)³` (constant shell volume);
/// that refinement is not applied here and is noted as a follow-up.
///
/// # Evidence tier
/// Literature-recall (Hoff 2000 / Doinikov & Bouakaz 2011); the viscous term and
/// the `G_s = 0` reduction to Church are differentially verified in tests.
#[derive(Debug, Clone)]
pub struct HoffModel {
    params: BubbleParameters,
    shell: ShellProperties,
}

impl HoffModel {
    /// Create a Hoff model from bubble parameters and shell properties.
    #[must_use]
    pub fn new(params: BubbleParameters, mut shell: ShellProperties) -> Self {
        shell.compute_critical_radii(params.r0, params.p0);
        Self { params, shell }
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

    /// Get shell properties.
    #[must_use]
    pub fn shell_properties(&self) -> &ShellProperties {
        &self.shell
    }
}

impl EncapsulatedShellModel for HoffModel {
    fn params(&self) -> &BubbleParameters {
        &self.params
    }

    fn equilibrium_gas_pressure(&self) -> f64 {
        self.params.p0 + self.params.surface_tension_pressure(self.params.r0)
    }

    fn effective_surface_tension(&self, _r: f64) -> f64 {
        self.params.sigma
    }

    fn shell_stress(&self, r: f64, v: f64) -> f64 {
        let r0 = self.params.r0;
        let d = self.shell.thickness;
        let g = self.shell.shear_modulus;
        let mu_s = self.shell.shear_viscosity;
        // Elastic: 12 G_s (d/R)[1 − R0/R]; viscous: 12 μ_s (d/R) Ṙ/R.
        let shell_elastic = 12.0 * g * (d / r) * (1.0 - r0 / r);
        let shell_viscous = 12.0 * mu_s * (d / r) * v / r;
        shell_elastic + shell_viscous
    }
}
