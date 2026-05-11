use super::ThermalAcousticCoupler;
use crate::core::constants::RHO_C_SOFT_TISSUE;
use crate::core::error::{KwaversError, KwaversResult};

impl ThermalAcousticCoupler {
    /// Update material properties based on current temperature
    ///
    /// Uses linear temperature dependence:
    /// - c(T) = c_ref + dc/dT * (T - T_ref)
    /// - ρ(T) = ρ_ref + dρ/dT * (T - T_ref)
    pub(super) fn update_material_properties(&mut self) {
        for idx in self.temperature.indexed_iter() {
            let (i, j, k) = idx.0;
            let t = self.temperature[[i, j, k]];
            let d_t = t - self.config.t_ref;

            self.sound_speed[[i, j, k]] = self.config.dc_d_t.mul_add(d_t, self.config.c_ref);
            self.density[[i, j, k]] = self.config.drho_d_t.mul_add(d_t, self.config.rho_ref);
        }
    }

    /// Compute acoustic heating source from pressure field
    ///
    /// Uses formula: Q = α|p|² / (ρc)
    /// where α is attenuation coefficient, p is pressure amplitude
    pub(super) fn compute_acoustic_heating(&mut self) {
        for k in 0..self.config.nz {
            for j in 0..self.config.ny {
                for i in 0..self.config.nx {
                    let p = self.pressure_prev[[i, j, k]].abs();
                    let rho = self.density[[i, j, k]];
                    let c = self.sound_speed[[i, j, k]];

                    if rho > 0.0 && c > 0.0 {
                        let intensity = (p * p) / (rho * c);
                        self.acoustic_heating[[i, j, k]] = self.config.alpha_ac * intensity;
                    } else {
                        self.acoustic_heating[[i, j, k]] = 0.0;
                    }
                }
            }
        }
    }

    /// Advance thermal equation (Pennes bioheat) one time step
    ///
    /// ρc ∂T/∂t = ∇·(k∇T) + w_b c_b (T_a - T) + Q_met + Q_ac
    pub(super) fn step_thermal(&mut self) {
        let dt = self.config.dt;
        let dx = self.config.dx;
        let dy = self.config.dy;
        let dz = self.config.dz;

        let k_thermal = self.config.alpha_thermal;
        // Volumetric heat capacity: ρ·c_p = 1050 kg/m³ × 3600 J/(kg·°C) = 3 780 000 J/(m³·°C).
        // Using SPECIFIC_HEAT_TISSUE alone (3600) would be a 1050× dimensional error.
        let rho_c = RHO_C_SOFT_TISSUE;

        for k in 1..self.config.nz - 1 {
            for j in 1..self.config.ny - 1 {
                for i in 1..self.config.nx - 1 {
                    let t = self.temperature_prev[[i, j, k]];

                    let d2_t_dx2 = (2.0f64.mul_add(-t, self.temperature_prev[[i + 1, j, k]])
                        + self.temperature_prev[[i - 1, j, k]])
                        / (dx * dx);
                    let d2_t_dy2 = (2.0f64.mul_add(-t, self.temperature_prev[[i, j + 1, k]])
                        + self.temperature_prev[[i, j - 1, k]])
                        / (dy * dy);
                    let d2_t_dz2 = (2.0f64.mul_add(-t, self.temperature_prev[[i, j, k + 1]])
                        + self.temperature_prev[[i, j, k - 1]])
                        / (dz * dz);

                    let laplacian_t = d2_t_dx2 + d2_t_dy2 + d2_t_dz2;

                    let perfusion_term = self.config.w_b * (self.config.t_arterial - t);
                    let metabolic_term = self.config.q_met / rho_c;
                    let acoustic_term = self.acoustic_heating[[i, j, k]] / rho_c;

                    let d_t_dt =
                        k_thermal.mul_add(laplacian_t, perfusion_term) + metabolic_term + acoustic_term;

                    self.temperature[[i, j, k]] = dt.mul_add(d_t_dt, t);
                }
            }
        }

        self.apply_boundary_conditions();
    }

    /// Apply boundary conditions
    ///
    /// Zero-gradient (Neumann) for acoustic fields
    /// Constant temperature (Dirichlet) for thermal field
    fn apply_boundary_conditions(&mut self) {
        for k in 0..self.config.nz {
            for j in 0..self.config.ny {
                self.pressure[[0, j, k]] = self.pressure[[1, j, k]];
                self.pressure[[self.config.nx - 1, j, k]] =
                    self.pressure[[self.config.nx - 2, j, k]];
                self.velocity_x[[0, j, k]] = 0.0;
                self.velocity_x[[self.config.nx - 1, j, k]] = 0.0;
            }
        }

        for k in 0..self.config.nz {
            for i in 0..self.config.nx {
                self.pressure[[i, 0, k]] = self.pressure[[i, 1, k]];
                self.pressure[[i, self.config.ny - 1, k]] =
                    self.pressure[[i, self.config.ny - 2, k]];
                self.velocity_y[[i, 0, k]] = 0.0;
                self.velocity_y[[i, self.config.ny - 1, k]] = 0.0;
            }
        }

        for j in 0..self.config.ny {
            for i in 0..self.config.nx {
                self.pressure[[i, j, 0]] = self.pressure[[i, j, 1]];
                self.pressure[[i, j, self.config.nz - 1]] =
                    self.pressure[[i, j, self.config.nz - 2]];
                self.velocity_z[[i, j, 0]] = 0.0;
                self.velocity_z[[i, j, self.config.nz - 1]] = 0.0;
            }
        }

        for k in 0..self.config.nz {
            for j in 0..self.config.ny {
                self.temperature[[0, j, k]] = self.config.t_ref;
                self.temperature[[self.config.nx - 1, j, k]] = self.config.t_ref;
            }
        }

        for k in 0..self.config.nz {
            for i in 0..self.config.nx {
                self.temperature[[i, 0, k]] = self.config.t_ref;
                self.temperature[[i, self.config.ny - 1, k]] = self.config.t_ref;
            }
        }

        for j in 0..self.config.ny {
            for i in 0..self.config.nx {
                self.temperature[[i, j, 0]] = self.config.t_ref;
                self.temperature[[i, j, self.config.nz - 1]] = self.config.t_ref;
            }
        }
    }

    /// Check for numerical divergence
    ///
    /// Returns error if pressure or temperature exceed physical bounds
    /// # Errors
    /// - Returns [`KwaversError::Numerical`] if the precondition for a Numerical-class constraint is violated.
    ///
    pub(super) fn check_stability(&self) -> KwaversResult<()> {
        let max_pressure = self
            .pressure
            .iter()
            .map(|p| p.abs())
            .fold(f64::NEG_INFINITY, f64::max);
        let max_temp = self
            .temperature
            .iter()
            .copied()
            .fold(f64::NEG_INFINITY, f64::max);
        let min_temp = self
            .temperature
            .iter()
            .copied()
            .fold(f64::INFINITY, f64::min);

        if !max_pressure.is_finite() {
            return Err(KwaversError::Numerical(
                crate::core::error::NumericalError::Instability {
                    operation: "thermal_acoustic_coupling".to_owned(),
                    condition: max_pressure,
                },
            ));
        }

        if max_temp > 100.0 || min_temp < 0.0 {
            return Err(KwaversError::Numerical(
                crate::core::error::NumericalError::InvalidOperation(format!(
                    "Temperature out of bounds: [{}, {}]°C",
                    min_temp, max_temp
                )),
            ));
        }

        Ok(())
    }
}
