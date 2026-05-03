//! Acoustic-elastic interface coupler and stability utilities.
//!
//! ## References
//!
//! - Zienkiewicz, O.C., Taylor, R.L. & Zhu, J.Z. (2013). *The Finite Element Method*, 7th ed. §12.3.
//! - de Hoop, A.T. (1995). *Handbook of Radiation and Scattering of Waves*.

use super::terms::CouplingTerms;
use crate::core::error::{KwaversError, KwaversResult};
use ndarray::Array3;

/// Acoustic-elastic interface coupler.
///
/// Assembles and applies the FDTD coupling matrix at fluid-solid interfaces,
/// enforcing normal velocity continuity (kinematic) and traction continuity (dynamic).
#[derive(Debug)]
pub struct AcousticElasticCoupler {
    /// Outward normal (from fluid to solid, unit vector)
    pub normal: [f64; 3],
    /// Interface cell mask (true = interface cell)
    pub interface_mask: Array3<bool>,
    /// Fluid density [kg/m³]
    pub fluid_density: f64,
    /// Fluid sound speed [m/s]
    pub fluid_sound_speed: f64,
    /// Solid longitudinal wave speed [m/s]
    pub solid_c_longitudinal: f64,
    /// CFL safety factor
    pub cfl: f64,
}

impl AcousticElasticCoupler {
    /// Create a new coupler. The normal vector is automatically normalized.
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

    /// Stable time step: dt = CFL · dx / max(c_fluid, c_solid).
    #[must_use]
    pub fn stability_dt(&self, dx: f64) -> f64 {
        let c_max = self.fluid_sound_speed.max(self.solid_c_longitudinal);
        self.cfl * dx / c_max
    }

    /// Compute coupling corrections at a single interface cell.
    ///
    /// - Fluid velocity correction: Δv_f[d] = −(dt/ρ_f) · a_solid · n̂[d]
    /// - Solid stress correction: Δσ_ab = −p_fluid · n̂_a · n̂_b
    #[must_use]
    pub fn coupling_terms_at_cell(
        &self,
        fluid_pressure: f64,
        solid_accel: [f64; 3],
        dt: f64,
    ) -> CouplingTerms {
        let [n0, n1, n2] = self.normal;
        let rho_f = self.fluid_density;

        let a_normal = solid_accel[0] * n0 + solid_accel[1] * n1 + solid_accel[2] * n2;
        let dv = -dt / rho_f * a_normal;
        let delta_fluid_velocity = [dv * n0, dv * n1, dv * n2];

        // Voigt: [σ_xx, σ_yy, σ_zz, σ_xy, σ_xz, σ_yz]
        let delta_solid_stress = [
            -fluid_pressure * n0 * n0,
            -fluid_pressure * n1 * n1,
            -fluid_pressure * n2 * n2,
            -fluid_pressure * n0 * n1,
            -fluid_pressure * n0 * n2,
            -fluid_pressure * n1 * n2,
        ];

        CouplingTerms {
            delta_fluid_velocity,
            delta_solid_stress,
        }
    }

    /// Apply coupling corrections over all interface cells.
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

/// Stable time step for a coupled fluid-solid system.
///
/// dt = CFL · dx / max(c_fluid, c_solid)
#[must_use]
pub fn stability_dt(c_fluid: f64, c_solid: f64, dx: f64, cfl: f64) -> f64 {
    let c_max = c_fluid.max(c_solid);
    cfl * dx / c_max
}
