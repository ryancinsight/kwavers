use super::super::shell::ShellProperties;
use crate::core::error::KwaversResult;
use crate::physics::acoustics::bubble_dynamics::bubble_state::{BubbleParameters, BubbleState};

/// Marmottant model for encapsulated bubbles with buckling/rupture
///
/// Implements the nonlinear shell model from Marmottant et al. (2005) which
/// accounts for:
/// - Buckling behavior at small radii (R < R_buckling)
/// - Elastic regime with variable surface tension
/// - Rupture at large radii (R > R_rupture)
#[derive(Debug, Clone)]
pub struct MarmottantModel {
    params: BubbleParameters,
    shell: ShellProperties,
    /// Elastic compression modulus χ [N/m]
    chi: f64,
}

impl MarmottantModel {
    /// Create new Marmottant model with shell properties
    ///
    /// # Arguments
    /// * `params` - Bubble parameters
    /// * `shell` - Shell properties
    /// * `chi` - Elastic compression modulus χ [N/m]
    #[must_use]
    pub fn new(params: BubbleParameters, mut shell: ShellProperties, chi: f64) -> Self {
        // Compute critical radii for the shell
        shell.compute_critical_radii(params.r0, params.p0);

        Self { params, shell, chi }
    }

    /// Calculate effective surface tension based on Marmottant model
    #[must_use]
    pub fn surface_tension(&self, radius: f64) -> f64 {
        let r_b = self.shell.r_buckling;
        let r_r = self.shell.r_rupture;

        if radius <= r_b {
            // Buckled state: no surface tension
            0.0
        } else if radius <= r_r {
            // Elastic regime: variable surface tension
            self.chi * (radius.powi(2) - r_b.powi(2)) / radius.powi(2)
        } else {
            // Ruptured state: water surface tension
            0.0728 // Water surface tension at 20°C [N/m]
        }
    }

    /// Calculate derivative of surface tension with respect to radius
    #[must_use]
    pub fn surface_tension_derivative(&self, radius: f64) -> f64 {
        let r_b = self.shell.r_buckling;
        let r_r = self.shell.r_rupture;

        if radius <= r_b || radius > r_r {
            0.0
        } else {
            // Elastic regime
            2.0 * self.chi * r_b.powi(2) / radius.powi(3)
        }
    }

    /// Calculate bubble wall acceleration with Marmottant shell model
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
        let p_eq = self.params.p0 + 2.0 * self.shell.sigma_initial / r0;
        let p_gas = p_eq * (r0 / r).powf(3.0 * gamma);

        // Variable surface tension (Marmottant model)
        let sigma = self.surface_tension(r);
        let dsigma_dr = self.surface_tension_derivative(r);

        // Surface tension terms
        let surface_term = 2.0 * sigma / r;
        let surface_rate_term = r * dsigma_dr * v; // Time derivative contribution

        // Viscous damping (liquid + shell)
        let viscous_liquid = 4.0 * self.params.mu_liquid * v / r;
        let d = self.shell.thickness;
        let mu_s = self.shell.shear_viscosity;
        let viscous_shell = 12.0 * mu_s * (d / r) * v / r;

        // Net pressure difference
        let net_pressure =
            p_gas - p_inf - surface_term - surface_rate_term - viscous_liquid - viscous_shell;

        // Solve for R̈ from modified Rayleigh-Plesset equation
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

    /// Check if shell is buckled
    #[must_use]
    pub fn is_buckled(&self, radius: f64) -> bool {
        radius <= self.shell.r_buckling
    }

    /// Check if shell is ruptured
    #[must_use]
    pub fn is_ruptured(&self, radius: f64) -> bool {
        radius > self.shell.r_rupture
    }

    /// Get current shell state as string
    #[must_use]
    pub fn shell_state(&self, radius: f64) -> &'static str {
        if self.is_buckled(radius) {
            "buckled"
        } else if self.is_ruptured(radius) {
            "ruptured"
        } else {
            "elastic"
        }
    }
}
