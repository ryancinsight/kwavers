use super::super::shell::ShellProperties;
use super::shell_model::EncapsulatedShellModel;
use crate::acoustics::bubble_dynamics::bubble_state::{BubbleParameters, BubbleState};
use kwavers_core::error::KwaversResult;

/// Church model for encapsulated bubbles with elastic shell
///
/// Implements the linearized shell model from Church (1995) which adds
/// shell elasticity and viscosity terms to the Rayleigh-Plesset equation.
#[derive(Debug, Clone)]
pub struct ChurchModel {
    params: BubbleParameters,
    shell: ShellProperties,
}

impl ChurchModel {
    /// Create new Church model with shell properties
    #[must_use]
    pub fn new(params: BubbleParameters, mut shell: ShellProperties) -> Self {
        // Compute critical radii for the shell
        shell.compute_critical_radii(params.r0, params.p0);

        Self { params, shell }
    }

    /// Calculate bubble wall acceleration with shell effects (Church model).
    ///
    /// Delegates to the shared [`EncapsulatedShellModel`] Rayleigh-Plesset driver;
    /// the Church-specific pieces are its trait methods below.
    /// # Errors
    /// - Returns [`Err`] if an internal constraint is violated.
    ///
    pub fn calculate_acceleration(
        &self,
        state: &mut BubbleState,
        p_acoustic: f64,
        t: f64,
    ) -> KwaversResult<f64> {
        EncapsulatedShellModel::acceleration(self, state, p_acoustic, t)
    }

    /// Get shell properties
    #[must_use]
    pub fn shell_properties(&self) -> &ShellProperties {
        &self.shell
    }
}

impl EncapsulatedShellModel for ChurchModel {
    fn params(&self) -> &BubbleParameters {
        &self.params
    }

    fn equilibrium_gas_pressure(&self) -> f64 {
        // p_eq = p0 + 2σ/R0 (Young-Laplace at equilibrium).
        self.params.p0 + self.params.surface_tension_pressure(self.params.r0)
    }

    fn effective_surface_tension(&self, _r: f64) -> f64 {
        // Constant liquid surface tension; shell elasticity is carried by the
        // separate shell-stress term, not by σ.
        self.params.sigma
    }

    fn shell_stress(&self, r: f64, v: f64) -> f64 {
        let r0 = self.params.r0;
        let d = self.shell.thickness;
        let g = self.shell.shear_modulus;
        let mu_s = self.shell.shear_viscosity;
        // Church (1995): elastic 12 G (d/R)[(R/R₀)² − 1] + viscous 12 μ_s (d/R) Ṙ/R.
        let shell_elastic = 12.0 * g * (d / r) * (r / r0).mul_add(r / r0, -1.0);
        let shell_viscous = 12.0 * mu_s * (d / r) * v / r;
        shell_elastic + shell_viscous
    }
}
