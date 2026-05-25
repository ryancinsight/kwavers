use super::super::shell::ShellProperties;
use crate::core::constants::cavitation::SURFACE_TENSION_WATER;
use crate::core::error::KwaversResult;
use crate::physics::acoustics::bubble_dynamics::bubble_state::{
    young_laplace_pressure, BubbleParameters, BubbleState,
};

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
            // Elastic regime: σ(R) = χ(R²/R_b² − 1)  [Marmottant 2005, Eq. 1]
            self.chi * r_b.mul_add(-r_b, radius.powi(2)) / r_b.powi(2)
        } else {
            // Ruptured state: bare water-vapour interface (no shell contribution)
            SURFACE_TENSION_WATER
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
            // Elastic regime: dσ/dR = 2χR/R_b²  [derivative of σ = χ(R²/R_b² − 1)]
            2.0 * self.chi * radius / r_b.powi(2)
        }
    }

    /// Calculate bubble wall acceleration with Marmottant shell model
    /// # Errors
    /// - Returns [`Err`] if an internal constraint is violated.
    ///
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

        // Variable surface tension (Marmottant 2005, eq. 1 — σ(R) is R-dependent
        // but enters the pressure balance as just 2σ(R)/R; no separate dσ/dR rate term).
        let sigma = self.surface_tension(r);

        // Surface tension term: 2σ(R)/R — variable σ from Marmottant state, free-function form.
        let surface_term = young_laplace_pressure(sigma, r);

        // Viscous damping (liquid + shell). Shell term: 4·κ_s·Ṙ/R² with
        // κ_s = 3·μ_s·d for a thin shell, giving 12·μ_s·d·Ṙ/R².
        let viscous_liquid = self.params.viscous_wall_stress(v, r);
        let d = self.shell.thickness;
        let mu_s = self.shell.shear_viscosity;
        let viscous_shell = 12.0 * mu_s * (d / r) * v / r;

        // Net pressure difference (Marmottant et al. 2005, JASA 118(6), eq. 3)
        let net_pressure = p_gas - p_inf - surface_term - viscous_liquid - viscous_shell;

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
