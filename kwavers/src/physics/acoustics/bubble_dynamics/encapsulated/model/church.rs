use super::super::shell::ShellProperties;
use crate::core::error::KwaversResult;
use crate::physics::acoustics::bubble_dynamics::bubble_state::{BubbleParameters, BubbleState};

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

    /// Calculate bubble wall acceleration with shell effects (Church model)
    pub fn calculate_acceleration(
        &self,
        state: &mut BubbleState,
        p_acoustic: f64,
        t: f64,
    ) -> KwaversResult<f64> {
        let r = state.radius;
        let v = state.wall_velocity;
        let r0 = self.params.r0;

        // Acoustic forcing
        let omega = 2.0 * std::f64::consts::PI * self.params.driving_frequency;
        let p_acoustic_inst = p_acoustic * (omega * t).sin();
        let p_inf = self.params.p0 + p_acoustic_inst;

        // Internal gas pressure (polytropic)
        let gamma = state.gas_species.gamma();
        let p_eq = self.params.p0 + 2.0 * self.params.sigma / r0;
        let p_gas = p_eq * (r0 / r).powf(3.0 * gamma);

        // Standard Rayleigh-Plesset terms
        let surface_tension = 2.0 * self.params.sigma / r;
        let viscous_stress = 4.0 * self.params.mu_liquid * v / r;

        // Shell elasticity term: 12G(d/R)[(R/R₀)² - 1]
        // This represents the elastic restoring force from shell deformation
        let d = self.shell.thickness;
        let g = self.shell.shear_modulus;
        let shell_elastic = 12.0 * g * (d / r) * ((r / r0).powi(2) - 1.0);

        // Shell viscosity term: 12μ_s(d/R)(dR/dt)/R
        // This represents the viscous damping from shell material
        let mu_s = self.shell.shear_viscosity;
        let shell_viscous = 12.0 * mu_s * (d / r) * v / r;

        // Net pressure difference
        let net_pressure =
            p_gas - p_inf - surface_tension - viscous_stress - shell_elastic - shell_viscous;

        // Solve for R̈ from Rayleigh-Plesset equation
        let accel = (net_pressure / (self.params.rho_liquid * r)) - (1.5 * v * v / r);

        // Update state
        state.wall_acceleration = accel;
        state.pressure_internal = p_gas;
        state.pressure_liquid = p_inf;

        Ok(accel)
    }

    /// Get shell properties
    #[must_use]
    pub fn shell_properties(&self) -> &ShellProperties {
        &self.shell
    }
}
