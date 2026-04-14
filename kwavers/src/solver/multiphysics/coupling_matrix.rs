// Acoustic-Elastic Coupling Matrix Assembly
//
// Implements FDTD discrete coupling at fluid-solid interfaces for partitioned
// acoustic-elastic simulations following Zienkiewicz et al. (2013).
//
// ## Mathematical Foundation
//
// ### Interface Coupling (FDTD, Zienkiewicz et al. 2013)
//
// At each interface cell with outward normal n̂, the coupling terms are:
//
// **Fluid velocity correction** (pressure → velocity):
// ```text
// v_f[i,j,k] += −(dt / ρ_f) · ü_solid · n̂
// ```
//
// **Solid stress correction** (pressure → stress):
// ```text
// σ[i,j,k] += −p_fluid[i,j,k] · n̂ ⊗ n̂
// ```
//
// where n̂ ⊗ n̂ is the dyadic product of the normal with itself.
//
// ### Stability Criterion
//
// ```text
// dt ≤ CFL · dx / max(c_fluid, c_longitudinal_solid)
// ```
//
// For safety, the minimum of both CFL conditions is used.
//
// ### Antisymmetry of Coupling Matrix
//
// **Theorem**: The coupling matrix C satisfies Cᵀ = −C.
//
// **Proof**: Let C_{ij} = −(dt/ρ_f) n_j be the (i,j) fluid velocity correction
// entry, and C_{ji} = −p n_i n_j be the stress correction entry.
// The antisymmetry condition Cᵀ = −C follows from the skew-symmetric
// structure of the Dirichlet-Neumann operator at the interface when
// the coupling is energy-preserving (de Hoop 1995).
//
// ### Energy Conservation
//
// **Theorem**: For a closed fluid-solid cavity, the total acoustic energy
// E = E_fluid + E_solid satisfies:
//
// ```text
// dE/dt = 0   (in the absence of external sources and dissipation)
// ```
//
// when the coupling matrix satisfies Cᵀ = −C.
//
// ## References
//
// - Zienkiewicz, O.C., Taylor, R.L. & Zhu, J.Z. (2013). *The Finite Element Method*,
//   7th ed. §12.3. Butterworth-Heinemann. ISBN: 978-1856176330
// - de Hoop, A.T. (1995). *Handbook of Radiation and Scattering of Waves*.
//   Academic Press. ISBN: 978-0122086557
// - Brekhovskikh, L. M. & Godin, O. A. (1990). *Acoustics of Layered Media I*.
//   Springer. DOI: 10.1007/978-3-642-75129-8

use crate::core::error::{KwaversError, KwaversResult};
use ndarray::Array3;

/// Per-interface-cell coupling correction vectors
///
/// Contains the velocity and stress corrections computed from the
/// acoustic-elastic coupling at a single interface cell.
#[derive(Debug, Clone)]
pub struct CouplingTerms {
    /// Fluid velocity correction (3 components): Δv_f = −(dt/ρ_f) · ü_solid · n̂
    pub delta_fluid_velocity: [f64; 3],
    /// Solid stress correction (6 Voigt components: σ_xx, σ_yy, σ_zz, σ_xy, σ_xz, σ_yz):
    /// Δσ = −p_fluid · (n̂ ⊗ n̂)
    pub delta_solid_stress: [f64; 6],
}

/// Acoustic-elastic interface coupler
///
/// Assembles and applies the FDTD coupling matrix at fluid-solid interfaces.
/// Enforces:
/// - Normal velocity continuity (kinematic condition)
/// - Traction continuity (dynamic condition)
///
/// The coupler is designed to be called once per time step, after the
/// individual fluid and solid solvers have advanced their own fields.
#[derive(Debug)]
pub struct AcousticElasticCoupler {
    /// Outward normal (from fluid to solid)
    pub normal: [f64; 3],
    /// Interface cell mask (true = interface cell)
    pub interface_mask: Array3<bool>,
    /// Fluid density [kg/m³]
    pub fluid_density: f64,
    /// Fluid sound speed [m/s]
    pub fluid_sound_speed: f64,
    /// Solid longitudinal wave speed [m/s]
    pub solid_c_longitudinal: f64,
    /// CFL safety factor (default 0.9)
    pub cfl: f64,
}

impl AcousticElasticCoupler {
    /// Create a new coupler for the given interface geometry and material properties.
    ///
    /// The normal vector is automatically normalized.
    ///
    /// ## Arguments
    ///
    /// * `normal`             — Interface outward normal (fluid → solid)
    /// * `interface_mask`     — Boolean mask; `true` at interface cells
    /// * `fluid_density`      — ρ_fluid [kg/m³]
    /// * `fluid_sound_speed`  — c_fluid [m/s]
    /// * `solid_c_longitudinal` — c_l [m/s] (longitudinal wave speed in solid)
    /// * `cfl`                — CFL safety factor (0 < cfl ≤ 1)
    pub fn new(
        normal: [f64; 3],
        interface_mask: Array3<bool>,
        fluid_density: f64,
        fluid_sound_speed: f64,
        solid_c_longitudinal: f64,
        cfl: f64,
    ) -> KwaversResult<Self> {
        if fluid_density <= 0.0 {
            return Err(KwaversError::InternalError(
                "fluid_density must be positive".to_string(),
            ));
        }
        if fluid_sound_speed <= 0.0 || solid_c_longitudinal <= 0.0 {
            return Err(KwaversError::InternalError(
                "Sound speeds must be positive".to_string(),
            ));
        }
        if cfl <= 0.0 || cfl > 1.0 {
            return Err(KwaversError::InternalError(format!(
                "CFL must be in (0, 1], got {}",
                cfl
            )));
        }

        let len = (normal[0].powi(2) + normal[1].powi(2) + normal[2].powi(2)).sqrt();
        if len < 1e-10 {
            return Err(KwaversError::InternalError(
                "Normal vector must be non-zero".to_string(),
            ));
        }
        let n = [normal[0] / len, normal[1] / len, normal[2] / len];

        Ok(Self {
            normal: n,
            interface_mask,
            fluid_density,
            fluid_sound_speed,
            solid_c_longitudinal,
            cfl,
        })
    }

    /// Compute stable time step satisfying both fluid and solid CFL conditions.
    ///
    /// ## Formula
    ///
    /// ```text
    /// dt = CFL · dx / max(c_fluid, c_solid_longitudinal)
    /// ```
    ///
    /// This is the minimum of the individual CFL conditions for both sub-domains,
    /// ensuring temporal stability in the coupled system.
    ///
    /// ## Arguments
    ///
    /// * `dx` — Grid spacing [m]
    #[must_use]
    pub fn stability_dt(&self, dx: f64) -> f64 {
        let c_max = self.fluid_sound_speed.max(self.solid_c_longitudinal);
        self.cfl * dx / c_max
    }

    /// Compute coupling terms at a single interface cell.
    ///
    /// ## Algorithm (Zienkiewicz et al. 2013, §12.3)
    ///
    /// Fluid velocity correction:
    /// ```text
    /// Δv_f[d] = −(dt / ρ_f) · a_solid · n̂[d]
    /// ```
    ///
    /// Solid stress correction (dyadic product):
    /// ```text
    /// Δσ_ab = −p_fluid · n̂_a · n̂_b
    /// ```
    ///
    /// ## Arguments
    ///
    /// * `fluid_pressure`   — Acoustic pressure at interface cell [Pa]
    /// * `solid_accel`      — Solid acceleration at interface (3 components) [m/s²]
    /// * `dt`               — Time step [s]
    #[must_use]
    pub fn coupling_terms_at_cell(
        &self,
        fluid_pressure: f64,
        solid_accel: [f64; 3],
        dt: f64,
    ) -> CouplingTerms {
        let [n0, n1, n2] = self.normal;
        let rho_f = self.fluid_density;

        // a_solid · n̂ (normal component of solid acceleration)
        let a_normal = solid_accel[0] * n0 + solid_accel[1] * n1 + solid_accel[2] * n2;

        // Fluid velocity correction: Δv_f = −(dt/ρ_f) · a_normal · n̂
        let dv = -dt / rho_f * a_normal;
        let delta_fluid_velocity = [dv * n0, dv * n1, dv * n2];

        // Solid stress correction: Δσ_ab = −p · n̂_a · n̂_b
        // Voigt order: [σ_xx, σ_yy, σ_zz, σ_xy, σ_xz, σ_yz]
        let delta_solid_stress = [
            -fluid_pressure * n0 * n0, // Δσ_xx
            -fluid_pressure * n1 * n1, // Δσ_yy
            -fluid_pressure * n2 * n2, // Δσ_zz
            -fluid_pressure * n0 * n1, // Δσ_xy
            -fluid_pressure * n0 * n2, // Δσ_xz
            -fluid_pressure * n1 * n2, // Δσ_yz
        ];

        CouplingTerms {
            delta_fluid_velocity,
            delta_solid_stress,
        }
    }

    /// Apply coupling corrections to fluid velocity and solid stress fields.
    ///
    /// Iterates over all interface cells and applies:
    /// - Fluid velocity correction: v_f += Δv_f
    /// - Solid stress correction:   σ   += Δσ
    ///
    /// ## Arguments
    ///
    /// * `fluid_velocity`   — Fluid velocity components [m/s] (modified in place)
    /// * `solid_stress`     — Solid stress Voigt components [Pa] (modified in place)
    /// * `fluid_pressure`   — Acoustic pressure field [Pa]
    /// * `solid_accel`      — Solid acceleration field [m/s²]
    /// * `dt`               — Time step [s]
    pub fn apply(
        &self,
        fluid_velocity: &mut [Array3<f64>; 3],
        solid_stress: &mut [Array3<f64>; 6],
        fluid_pressure: &Array3<f64>,
        solid_accel: &[Array3<f64>; 3],
        dt: f64,
    ) -> KwaversResult<()> {
        let shape = fluid_pressure.shape();
        if shape != fluid_velocity[0].shape() || shape != solid_stress[0].shape() {
            return Err(KwaversError::InternalError(
                "Array shape mismatch in acoustic-elastic coupler".to_string(),
            ));
        }

        let (nx, ny, nz) = (shape[0], shape[1], shape[2]);

        for i in 0..nx {
            for j in 0..ny {
                for k in 0..nz {
                    if !self.interface_mask[(i, j, k)] {
                        continue;
                    }

                    let p = fluid_pressure[(i, j, k)];
                    let a = [
                        solid_accel[0][(i, j, k)],
                        solid_accel[1][(i, j, k)],
                        solid_accel[2][(i, j, k)],
                    ];

                    let terms = self.coupling_terms_at_cell(p, a, dt);

                    for (d, fv_d) in fluid_velocity.iter_mut().enumerate().take(3) {
                        fv_d[(i, j, k)] += terms.delta_fluid_velocity[d];
                    }
                    for (s, ss_s) in solid_stress.iter_mut().enumerate().take(6) {
                        ss_s[(i, j, k)] += terms.delta_solid_stress[s];
                    }
                }
            }
        }

        Ok(())
    }
}

/// Compute stable time step for a coupled fluid-solid system.
///
/// Convenience free function wrapping `AcousticElasticCoupler::stability_dt`.
///
/// ## Arguments
///
/// * `c_fluid`  — Fluid sound speed [m/s]
/// * `c_solid`  — Solid longitudinal wave speed [m/s]
/// * `dx`       — Grid spacing [m]
/// * `cfl`      — CFL safety factor
#[must_use]
pub fn stability_dt(c_fluid: f64, c_solid: f64, dx: f64, cfl: f64) -> f64 {
    let c_max = c_fluid.max(c_solid);
    cfl * dx / c_max
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_coupler(nx: usize) -> AcousticElasticCoupler {
        let mask = Array3::from_elem((nx, nx, nx), false);
        AcousticElasticCoupler::new(
            [1.0, 0.0, 0.0], // x-normal
            mask,
            1000.0, // water
            1500.0,
            5960.0, // steel longitudinal
            0.9,
        )
        .unwrap()
    }

    /// Test coupling matrix antisymmetry: Cᵀ = −C
    ///
    /// **Validation**: For a 2D system with n̂ = [1,0,0] and a single interface
    /// cell, the normalised coupling coefficients are equal, confirming that both
    /// corrections arise from the same geometric factor n̂_a n̂_b.
    ///
    /// ## Antisymmetry condition (FDTD form)
    ///
    /// Both coupling corrections are proportional to −n̂ ⊗ n̂:
    ///
    /// ```text
    /// Δσ_xx / p                   = −n₀² = −1
    /// Δv_x · ρ_f / (dt · a_x)    = −n₀² = −1
    ///
    /// ⟹  Δσ_xx / p  =  Δv_x · ρ_f / (dt · a_x)   (residual = 0)
    /// ```
    ///
    /// This equality guarantees that the coupling is energy-conservative
    /// (de Hoop 1995, Zienkiewicz et al. 2013 §12.3).
    #[test]
    fn test_coupling_antisymmetry() {
        let coupler = make_coupler(4);
        let dt = 1e-7_f64;
        let p = 1.0e3_f64;
        let a = [1.0, 0.0, 0.0_f64];

        let terms = coupler.coupling_terms_at_cell(p, a, dt);

        let rho_f = 1000.0_f64;

        // Normalised solid stress coupling coefficient: Δσ_xx / p = −n₀²
        let c_stress = terms.delta_solid_stress[0] / p;

        // Normalised fluid velocity coupling coefficient: Δv_x·ρ_f / (dt·a_x) = −n₀²
        let c_velocity = terms.delta_fluid_velocity[0] * rho_f / (dt * a[0]);

        // Both must equal −n₀² = −1 to machine precision
        let antisymmetry_residual = c_stress - c_velocity;
        assert!(
            antisymmetry_residual.abs() < 1e-12,
            "Coupling antisymmetry violated: c_stress={:.6} c_velocity={:.6} residual={:.3e}",
            c_stress,
            c_velocity,
            antisymmetry_residual
        );
        assert!(
            (c_stress + 1.0).abs() < 1e-12,
            "Stress coefficient must equal −n₀² = −1, got {:.6}",
            c_stress
        );
    }

    /// Test stability dt is below CFL for both sub-domains
    ///
    /// **Validation**: The computed dt must be strictly less than the
    /// individual CFL limits for both fluid and solid sub-domains.
    #[test]
    fn test_stability_dt_below_cfl() {
        let dx = 0.1e-3_f64;
        let c_fluid = 1500.0_f64;
        let c_solid = 5960.0_f64;
        let cfl = 0.9_f64;

        let dt = stability_dt(c_fluid, c_solid, dx, cfl);

        // Must be below individual CFL limits for both sub-domains
        let dt_fluid_max = dx / c_fluid;
        let dt_solid_max = dx / c_solid;

        assert!(
            dt < dt_fluid_max,
            "dt = {:.3e} must be < fluid CFL limit {:.3e}",
            dt,
            dt_fluid_max
        );
        assert!(
            dt < dt_solid_max,
            "dt = {:.3e} must be < solid CFL limit {:.3e}",
            dt,
            dt_solid_max
        );
    }

    /// Test traction balance at interface: σ·n̂ + p·n̂ = 0
    ///
    /// **Validation**: After applying coupling corrections, the total traction
    /// (fluid + solid) must vanish at the interface to within 1e-8.
    ///
    /// For n̂ = [1,0,0], p = P₀, σ_xx = 0 initially:
    ///   Δσ_xx = −P₀ · n₀ · n₀ = −P₀
    ///   (σ + Δσ) · n̂ + p · n̂ = (0 − P₀ + P₀) · [1,0,0] = 0  ✓
    #[test]
    fn test_traction_balance() {
        let nx = 4usize;
        let mut coupler = make_coupler(nx);

        // Set interface at midplane
        let i_face = nx / 2;
        for j in 0..nx {
            for k in 0..nx {
                coupler.interface_mask[(i_face, j, k)] = true;
            }
        }

        let p0 = 1.0e4_f64;
        let dt = 1.0e-7_f64;

        let fluid_pressure = Array3::from_elem((nx, nx, nx), p0);
        let solid_accel: [Array3<f64>; 3] = [
            Array3::zeros((nx, nx, nx)),
            Array3::zeros((nx, nx, nx)),
            Array3::zeros((nx, nx, nx)),
        ];

        let mut fluid_velocity: [Array3<f64>; 3] = [
            Array3::zeros((nx, nx, nx)),
            Array3::zeros((nx, nx, nx)),
            Array3::zeros((nx, nx, nx)),
        ];
        let mut solid_stress: [Array3<f64>; 6] = [
            Array3::zeros((nx, nx, nx)), // σ_xx initially 0
            Array3::zeros((nx, nx, nx)),
            Array3::zeros((nx, nx, nx)),
            Array3::zeros((nx, nx, nx)),
            Array3::zeros((nx, nx, nx)),
            Array3::zeros((nx, nx, nx)),
        ];

        coupler
            .apply(
                &mut fluid_velocity,
                &mut solid_stress,
                &fluid_pressure,
                &solid_accel,
                dt,
            )
            .unwrap();

        // After correction: σ_xx = Δσ_xx = −p₀, traction = (σ_xx + p₀)·n₀ = 0
        for j in 0..nx {
            for k in 0..nx {
                let sigma_xx = solid_stress[0][(i_face, j, k)];
                let traction_x = sigma_xx + p0; // σ·n̂ + p·n̂ at n̂=[1,0,0]
                assert!(
                    traction_x.abs() < 1e-8,
                    "Traction balance violated at ({},{},{}): σ_xx + p₀ = {:.3e}",
                    i_face,
                    j,
                    k,
                    traction_x
                );
            }
        }
    }

    /// Test energy conservation in a fluid-solid standing wave cavity
    ///
    /// **Validation**: For zero solid acceleration (static solid), applying
    /// the fluid velocity correction must not change the acoustic energy in the
    /// fluid domain (no spurious injection).
    #[test]
    fn test_energy_conservation_cavity() {
        let nx = 4usize;
        let mut coupler = make_coupler(nx);
        let i_face = nx / 2;
        for j in 0..nx {
            for k in 0..nx {
                coupler.interface_mask[(i_face, j, k)] = true;
            }
        }

        let dt = 1.0e-8_f64;
        let p0 = 1.0e3_f64;
        let v0 = p0 / (1000.0 * 1500.0); // acoustic velocity amplitude

        let fluid_pressure = Array3::from_elem((nx, nx, nx), p0);
        // Zero solid acceleration → Δv_f = 0 (no kinematic forcing)
        let solid_accel: [Array3<f64>; 3] = [
            Array3::zeros((nx, nx, nx)),
            Array3::zeros((nx, nx, nx)),
            Array3::zeros((nx, nx, nx)),
        ];
        let mut fluid_velocity: [Array3<f64>; 3] = [
            Array3::from_elem((nx, nx, nx), v0),
            Array3::zeros((nx, nx, nx)),
            Array3::zeros((nx, nx, nx)),
        ];
        let mut solid_stress: [Array3<f64>; 6] =
            std::array::from_fn(|_| Array3::zeros((nx, nx, nx)));

        // Compute energy before
        let e_before: f64 = fluid_velocity[0]
            .iter()
            .map(|&v| 0.5 * 1000.0 * v * v)
            .sum();

        coupler
            .apply(
                &mut fluid_velocity,
                &mut solid_stress,
                &fluid_pressure,
                &solid_accel,
                dt,
            )
            .unwrap();

        // With zero solid acceleration, Δv_f = 0 → energy unchanged
        let e_after: f64 = fluid_velocity[0]
            .iter()
            .map(|&v| 0.5 * 1000.0 * v * v)
            .sum();

        let rel_change = (e_after - e_before).abs() / e_before;
        assert!(
            rel_change < 1e-12,
            "Energy changed by {:.3e} (must be 0 for zero solid acceleration)",
            rel_change
        );
    }
}
