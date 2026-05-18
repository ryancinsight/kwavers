//! Cavitation and bubble-scattering PDE residuals for PINN training.

use super::domain::CavitationCoupledDomain;
use super::mie_scattering::mie_backscatter_form_function;
use crate::solver::inverse::pinn::ml::physics::PinnDomainPhysicsParameters;
use burn::prelude::ElementConversion;
use burn::tensor::{backend::AutodiffBackend, Tensor};

impl<B: AutodiffBackend> CavitationCoupledDomain<B> {
    /// Compute the Keller–Miksis cavitation residual.
    ///
    /// ## Theorem (Keller–Miksis equation)
    /// For a spherical bubble driven by an acoustic pressure field `P_ac(t)`, the
    /// radial acceleration of the bubble wall satisfies (Keller & Miksis 1980,
    /// J. Acoust. Soc. Am. 68(2), eq. 1–4):
    ///
    /// ```text
    /// R̈ = [P_gas + P_vapor − P_0 − P_ac − (2σ/R) − (4μ Ṙ/R²)]
    ///        / (ρ_l R)  −  (3/2) Ṙ²/R
    /// ```
    ///
    /// where `P_gas = P_g0 (R_0/R)^{3γ}` (polytropic gas law, γ = 1.4 adiabatic).
    ///
    /// For PINN training the radial velocity `Ṙ` is taken as zero (coupling-
    /// residual steady-state assumption); the full time-dependent derivative can be
    /// recovered from neural-network output once trained.
    ///
    /// ## References
    /// - Keller, J. B. & Miksis, M. (1980). *J. Acoust. Soc. Am.* 68, 628–633.
    /// - Prosperetti, A. & Lezzi, A. (1986). *J. Fluid Mech.* 168, 457–478.
    /// - Brennen, C. E. (1995). *Cavitation and Bubble Dynamics*. Oxford, §2.4.
    pub(super) fn cavitation_residual(
        &self,
        acoustic_pressure: &Tensor<B, 2>,
        _bubble_positions: &Tensor<B, 2>,
        physics_params: &PinnDomainPhysicsParameters,
    ) -> Tensor<B, 2> {
        let ambient_pressure = physics_params
            .domain_params
            .get("ambient_pressure")
            .copied()
            .unwrap_or(101_325.0) as f32;

        let viscosity = physics_params
            .domain_params
            .get("liquid_viscosity")
            .copied()
            .unwrap_or(0.001) as f32;

        let pressure_forcing = acoustic_pressure.clone() - ambient_pressure;

        let r_eq = self.config.bubble_params.r0 as f32;
        let gamma = 1.4_f32; // adiabatic index for air
        let rho_l = 1000.0_f32; // water [kg/m³]
        let sigma = 0.072_f32; // surface tension water-air [N/m]
        let p_vapor = 2330.0_f32; // water vapour pressure at 20 °C [Pa]

        // Polytropic gas law: P_gas = P_g0 · (R_0/R_eq)^{3γ}
        let p_gas = self.config.bubble_params.initial_gas_pressure as f32
            * (self.config.bubble_params.r0 as f32 / r_eq).powf(3.0 * gamma);

        // Young–Laplace surface tension term: 2σ/R
        let p_surface = 2.0_f32 * sigma / r_eq;

        // Ṙ = 0 (steady coupling residual)
        let rdot = 0.0_f32;
        let p_viscous = 4.0 * viscosity * rdot / (r_eq * r_eq);

        let p_internal = p_gas + p_vapor;
        let p_external = ambient_pressure + pressure_forcing + p_surface + p_viscous;

        // Pressure-driven acceleration term: (P_int − P_ext) / (ρ_l · R)
        let accel = (p_internal - p_external) / (rho_l * r_eq);
        // Inertial term: −(3/2) Ṙ²/R — zero since Ṙ = 0
        let inertial = -1.5_f32 * rdot * rdot / r_eq;

        (accel + inertial) * self.config.coupling_strength as f32
    }

    /// Compute the acoustic scattering residual from bubble cloud.
    ///
    /// ## Theorem (Green's function acoustic scattering)
    ///
    /// For a spherical scatterer (bubble j) at `r_j` insonified by incident field
    /// `u_inc(r_j)`, the scattered pressure at receiver `r_i` is
    /// (Morse & Ingard 1968 §8.3):
    ///
    /// ```text
    /// u_scat(r_i) = u_inc(r_j) · f_bs(θ) · cos(k_l r) / r
    /// ```
    ///
    /// where `r = |r_i − r_j|` and `f_bs` is the Mie backscattering form function
    /// (Anderson 1950, eq. 14).  The incident field is evaluated at the *bubble*
    /// position (causality): the scatterer is driven by the field it is immersed
    /// in, not the field it radiates into.
    ///
    /// ## References
    /// - Anderson, V. C. (1950). *J. Acoust. Soc. Am.* 22, 426–431.
    /// - Morse, P. M. & Ingard, K. U. (1968). *Theoretical Acoustics* §8.3.
    pub(super) fn bubble_scattering_residual(
        &self,
        acoustic_field: &Tensor<B, 2>,
        bubble_positions: &Tensor<B, 2>,
    ) -> Tensor<B, 2> {
        if !self.config.nonlinear_acoustic {
            return Tensor::zeros_like(acoustic_field);
        }

        // ─── Mie constants (computed once per call, constant across all bubble pairs) ───
        let omega = 2.0 * std::f32::consts::PI * self.config.center_frequency as f32;
        let k_l = omega / self.config.sound_speed as f32;

        // Interior sound speed: c_b = √(γ R_gas T / M)  — adiabatic ideal gas
        let gamma_b = self.config.bubble_params.gamma as f32;
        let m_gas = self.config.bubble_params.gas_species.molecular_weight() as f32;
        let t_amb = self.config.bubble_params.t0 as f32;
        let r_gas = crate::core::constants::GAS_CONSTANT as f32;
        let c_b = (gamma_b * r_gas * t_amb / m_gas).sqrt();
        let k_b = omega / c_b;

        // Ideal-gas density: ρ_b = P_0 M / (R_gas T)
        let rho_b = self.config.bubble_params.p0 as f32 * m_gas / (r_gas * t_amb);
        let rho_l = self.config.bubble_params.rho_liquid as f32;
        let r_bubble = self.config.bubble_params.r0 as f32;

        let n_points = acoustic_field.shape().dims[0];
        let mut scattered_rows = Vec::with_capacity(n_points);

        for i in 0..n_points {
            let pos_i = bubble_positions.clone().slice([i..i + 1, 0..2]);
            let mut total = Tensor::zeros([1, 1], &acoustic_field.device());

            for j in 0..n_points {
                if i == j {
                    continue;
                }

                let pos_j = bubble_positions.clone().slice([j..j + 1, 0..2]);
                let dist_val = (pos_i.clone() - pos_j)
                    .powf_scalar(2.0)
                    .sum_dim(1)
                    .sqrt()
                    .clamp(1e-6_f32, f32::MAX)
                    .into_scalar()
                    .elem::<f32>();

                // Mie backscattering form function (Anderson 1950, eq. 14).
                let f_bs = mie_backscatter_form_function(k_l, k_b, rho_l, rho_b, r_bubble);
                let phase = k_l * dist_val;
                let scatter_coeff =
                    (f_bs.re * phase.cos() - f_bs.im * phase.sin()) / dist_val.max(1e-6_f32);

                // Incident field at bubble j drives the scattered field at receiver i.
                let contribution = acoustic_field.clone().slice([j..j + 1, 0..1]) * scatter_coeff;
                total = total + contribution;
            }
            scattered_rows.push(total);
        }

        Tensor::cat(scattered_rows, 0)
    }
}
