use super::ThermalAcousticCoupler;
use kwavers_core::constants::RHO_C_SOFT_TISSUE;
use kwavers_core::error::{KwaversError, KwaversResult};
use moirai_parallel::{enumerate_mut_with, for_each_chunk_pair_mut_enumerated_with, Adaptive};
impl ThermalAcousticCoupler {
    #[inline]
    pub(super) fn cell_indices(index: usize, ny: usize, nz: usize) -> (usize, usize, usize) {
        let slab_len = ny * nz;
        let i = index / slab_len;
        let offset = index % slab_len;
        (i, offset / nz, offset % nz)
    }

    /// Update material properties based on current temperature.
    ///
    /// Linear temperature dependence:
    /// - `c(T) = c_ref + dc/dT · (T − T_ref)`
    /// - `ρ(T) = ρ_ref + dρ/dT · (T − T_ref)`
    pub(super) fn update_material_properties(&mut self) {
        let d_t_offset = self.config.t_ref;
        let dc_dt = self.config.dc_d_t;
        let c_ref = self.config.c_ref;
        let drho_dt = self.config.drho_d_t;
        let rho_ref = self.config.rho_ref;
        // Split borrows: temperature (read), sound_speed + density (write).
        let temperature = &self.temperature;
        let sound_speed = &mut self.sound_speed;
        let density = &mut self.density;

        if let (Some(sound_speed), Some(density), Some(temperature)) = (
            sound_speed.as_slice_mut(),
            density.as_slice_mut(),
            temperature.as_slice(),
        ) {
            for_each_chunk_pair_mut_enumerated_with::<Adaptive, _, _, _>(
                sound_speed,
                density,
                self.config.ny * self.config.nz,
                |slab_index, sound_speed, density| {
                    let base = slab_index * self.config.ny * self.config.nz;
                    for (offset, (c, rho)) in
                        sound_speed.iter_mut().zip(density.iter_mut()).enumerate()
                    {
                        let t = temperature[base + offset];
                        let d_t = t - d_t_offset;
                        *c = dc_dt.mul_add(d_t, c_ref);
                        *rho = drho_dt.mul_add(d_t, rho_ref);
                    }
                },
            );
            return;
        }

        Zip::from(sound_speed.view_mut())
            .and(density.view_mut())
            .and(temperature.view())
            .for_each(|c, rho, &t| {
                let d_t = t - d_t_offset;
                *c = dc_dt.mul_add(d_t, c_ref);
                *rho = drho_dt.mul_add(d_t, rho_ref);
            });
    }

    /// Compute acoustic heating source from pressure field.
    ///
    /// For a plane wave, the volumetric power deposition is:
    ///   Q = 2α · I = 2α · p² / (ρ·c)
    ///
    /// The factor of 2 arises because α is the **amplitude** attenuation
    /// coefficient (pressure ∝ exp(−αz)), so intensity ∝ exp(−2αz) and
    /// the absorbed power per unit volume is −dI/dz = 2α·I.
    /// (Pierce 1989 §10.2; Szabo 2014 §1.6)
    pub(super) fn compute_acoustic_heating(&mut self) {
        let alpha_ac = self.config.alpha_ac;
        let pressure_prev = &self.pressure_prev;
        let density = &self.density;
        let sound_speed = &self.sound_speed;
        let acoustic_heating = &mut self.acoustic_heating;

        if let (Some(acoustic_heating), Some(pressure_prev), Some(density), Some(sound_speed)) = (
            acoustic_heating.as_slice_mut(),
            pressure_prev.as_slice(),
            density.as_slice(),
            sound_speed.as_slice(),
        ) {
            enumerate_mut_with::<Adaptive, _, _>(acoustic_heating, |index, q| {
                let p = pressure_prev[index];
                let rho = density[index];
                let c = sound_speed[index];
                if rho > 0.0 && c > 0.0 {
                    *q = 2.0 * alpha_ac * (p * p) / (rho * c);
                } else {
                    *q = 0.0;
                }
            });
            return;
        }

        Zip::from(acoustic_heating.view_mut())
            .and(pressure_prev.view())
            .and(density.view())
            .and(sound_speed.view())
            .for_each(|q, &p, &rho, &c| {
                if rho > 0.0 && c > 0.0 {
                    *q = 2.0 * alpha_ac * (p * p) / (rho * c);
                } else {
                    *q = 0.0;
                }
            });
    }

    /// Advance thermal equation (Pennes bioheat) one time step.
    ///
    /// Solved in temperature-rate form (canonical Pennes divided by `ρc`):
    /// `∂T/∂t = α∇²T + w_b·(T_a − T) + (Q_met + Q_ac)/(ρc)`, where `α = k/(ρc)` and
    /// `w_b` is the lumped volumetric perfusion rate `[1/s]` (= ω_b·ρ_b·c_b/(ρc)),
    /// so no separate `c_b`/`ρc` factor is applied to the perfusion term.
    ///
    /// Interior uses explicit FD Laplacian with reads from `temperature_prev`
    /// (no dependency between elements) → fully data-parallel.
    /// Boundary cells are set by `apply_boundary_conditions` after the update.
    pub(super) fn step_thermal(&mut self) {
        let dt = self.config.dt;
        let dx = self.config.dx;
        let dy = self.config.dy;
        let dz = self.config.dz;
        let k_thermal = self.config.alpha_thermal;
        // Volumetric heat capacity: ρ·c_p = 1050 kg/m³ × 3600 J/(kg·°C) = 3 780 000 J/(m³·°C).
        // Using SPECIFIC_HEAT_TISSUE alone (3600) would be a 1050× dimensional error.
        let rho_c = RHO_C_SOFT_TISSUE;
        let nx = self.config.nx;
        let ny = self.config.ny;
        let nz = self.config.nz;
        let w_b = self.config.w_b;
        let t_arterial = self.config.t_arterial;
        let q_met = self.config.q_met;

        // Split borrows: temperature_prev + acoustic_heating (read), temperature (write).
        let temperature_prev = &self.temperature_prev;
        let acoustic_heating = &self.acoustic_heating;
        let temperature = &mut self.temperature;

        if let (Some(temperature), Some(temperature_prev), Some(acoustic_heating)) = (
            temperature.as_slice_mut(),
            temperature_prev.as_slice(),
            acoustic_heating.as_slice(),
        ) {
            enumerate_mut_with::<Adaptive, _, _>(temperature, |index, t_new| {
                let (i, j, k) = Self::cell_indices(index, ny, nz);
                if i == 0 || i >= nx - 1 || j == 0 || j >= ny - 1 || k == 0 || k >= nz - 1 {
                    return;
                }
                let t = temperature_prev[index];
                let d2_t_dx2 = (2.0f64.mul_add(-t, temperature_prev[index + ny * nz])
                    + temperature_prev[index - ny * nz])
                    / (dx * dx);
                let d2_t_dy2 = (2.0f64.mul_add(-t, temperature_prev[index + nz])
                    + temperature_prev[index - nz])
                    / (dy * dy);
                let d2_t_dz2 = (2.0f64.mul_add(-t, temperature_prev[index + 1])
                    + temperature_prev[index - 1])
                    / (dz * dz);
                let laplacian_t = d2_t_dx2 + d2_t_dy2 + d2_t_dz2;
                let perfusion_term = w_b * (t_arterial - t);
                let metabolic_term = q_met / rho_c;
                let acoustic_term = acoustic_heating[index] / rho_c;
                let d_t_dt =
                    k_thermal.mul_add(laplacian_t, perfusion_term) + metabolic_term + acoustic_term;
                *t_new = dt.mul_add(d_t_dt, t);
            });
        } else {
            Zip::indexed(temperature.view_mut()).for_each(|(i, j, k), t_new| {
                if i == 0 || i >= nx - 1 || j == 0 || j >= ny - 1 || k == 0 || k >= nz - 1 {
                    return;
                }
                let t = temperature_prev[[i, j, k]];
                let d2_t_dx2 = (2.0f64.mul_add(-t, temperature_prev[[i + 1, j, k]])
                    + temperature_prev[[i - 1, j, k]])
                    / (dx * dx);
                let d2_t_dy2 = (2.0f64.mul_add(-t, temperature_prev[[i, j + 1, k]])
                    + temperature_prev[[i, j - 1, k]])
                    / (dy * dy);
                let d2_t_dz2 = (2.0f64.mul_add(-t, temperature_prev[[i, j, k + 1]])
                    + temperature_prev[[i, j, k - 1]])
                    / (dz * dz);
                let laplacian_t = d2_t_dx2 + d2_t_dy2 + d2_t_dz2;
                let perfusion_term = w_b * (t_arterial - t);
                let metabolic_term = q_met / rho_c;
                let acoustic_term = acoustic_heating[[i, j, k]] / rho_c;
                let d_t_dt =
                    k_thermal.mul_add(laplacian_t, perfusion_term) + metabolic_term + acoustic_term;
                *t_new = dt.mul_add(d_t_dt, t);
            });
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
                kwavers_core::error::NumericalError::Instability {
                    operation: "thermal_acoustic_coupling".to_owned(),
                    condition: max_pressure,
                },
            ));
        }

        if max_temp > 100.0 || min_temp < 0.0 {
            return Err(KwaversError::Numerical(
                kwavers_core::error::NumericalError::InvalidOperation(format!(
                    "Temperature out of bounds: [{}, {}]°C",
                    min_temp, max_temp
                )),
            ));
        }

        Ok(())
    }
}
