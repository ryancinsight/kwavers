//! Right-hand side computation for the Kuznetsov equation.

use super::wave::KuznetsovWave;
use crate::domain::medium::Medium;
use crate::domain::source::Source;
use crate::solver::forward::nonlinear::kuznetsov::config::AcousticEquationMode;
use crate::solver::forward::nonlinear::kuznetsov::diffusion::compute_diffusive_term_workspace;
use crate::solver::forward::nonlinear::kuznetsov::nonlinear::compute_nonlinear_term_workspace;
use ndarray::Zip;

impl KuznetsovWave {
    /// Compute the right-hand side of the Kuznetsov equation.
    ///
    /// For heterogeneous media, nonlinear and diffusive terms are computed using
    /// local material properties at each grid point. For homogeneous media,
    /// properties are computed once for efficiency.
    pub(super) fn compute_rhs(
        &mut self,
        source: &dyn Source,
        medium: &dyn Medium,
        t: f64,
        dt: f64,
    ) {
        let pressure = &self.pressure_current;
        let is_heterogeneous = !medium.is_homogeneous();

        let (uniform_density, uniform_sound_speed, _uniform_nonlinearity, uniform_diffusivity) =
            if !is_heterogeneous {
                let center_x = self.grid.dx * (self.grid.nx as f64) / 2.0;
                let center_y = self.grid.dy * (self.grid.ny as f64) / 2.0;
                let center_z = self.grid.dz * (self.grid.nz as f64) / 2.0;
                (
                    crate::domain::medium::density_at(
                        medium, center_x, center_y, center_z, &self.grid,
                    ),
                    crate::domain::medium::sound_speed_at(
                        medium, center_x, center_y, center_z, &self.grid,
                    ),
                    crate::domain::medium::nonlinearity_at(
                        medium, center_x, center_y, center_z, &self.grid,
                    ),
                    self.config.acoustic_diffusivity,
                )
            } else {
                (0.0, 0.0, 0.0, 0.0)
            };

        // 1. Compute linear term: c₀²∇²p using spectral methods
        self.workspace.spectral_op.compute_laplacian_workspace(
            pressure,
            &mut self.workspace.laplacian,
            &self.grid,
        );

        let rhs = &mut self.workspace.k1;

        let include_nonlinearity = matches!(
            self.config.equation_mode,
            AcousticEquationMode::FullKuznetsov
                | AcousticEquationMode::KZK
                | AcousticEquationMode::Westervelt
        );

        if include_nonlinearity && !is_heterogeneous {
            compute_nonlinear_term_workspace(
                pressure,
                &self.workspace.pressure_prev,
                &self.workspace.pressure_prev2,
                dt,
                uniform_density,
                uniform_sound_speed,
                self.config.nonlinearity_coefficient * self.nonlinearity_scaling,
                &mut self.workspace.nonlinear_term,
            );
        }

        let include_diffusion = matches!(
            self.config.equation_mode,
            AcousticEquationMode::FullKuznetsov | AcousticEquationMode::KZK
        ) && self.config.acoustic_diffusivity > 0.0;

        if include_diffusion && !is_heterogeneous {
            compute_diffusive_term_workspace(
                pressure,
                &self.workspace.pressure_prev,
                &self.workspace.pressure_prev2,
                &self.workspace.pressure_prev3,
                dt,
                uniform_sound_speed,
                uniform_diffusivity,
                &mut self.workspace.diffusive_term,
            );
        }

        if is_heterogeneous {
            // Heterogeneous: compute properties per grid point
            for k in 0..self.grid.nz {
                for j in 0..self.grid.ny {
                    for i in 0..self.grid.nx {
                        let (x, y, z) = self.grid.indices_to_coordinates(i, j, k);
                        let local_density =
                            crate::domain::medium::density_at(medium, x, y, z, &self.grid);
                        let local_sound_speed =
                            crate::domain::medium::sound_speed_at(medium, x, y, z, &self.grid);
                        let local_nonlinearity =
                            crate::domain::medium::nonlinearity_at(medium, x, y, z, &self.grid);
                        let c0_squared = local_sound_speed * local_sound_speed;

                        rhs[[i, j, k]] = c0_squared * self.workspace.laplacian[[i, j, k]];

                        if include_nonlinearity {
                            let beta = crate::physics::constants::NONLINEARITY_COEFFICIENT_OFFSET
                                + local_nonlinearity / crate::physics::constants::B_OVER_A_DIVISOR;
                            let coeff = beta / (local_density * local_sound_speed.powi(4));
                            let p2 = pressure[[i, j, k]] * pressure[[i, j, k]];
                            let p2_prev = self.workspace.pressure_prev[[i, j, k]]
                                * self.workspace.pressure_prev[[i, j, k]];
                            let p2_prev2 = self.workspace.pressure_prev2[[i, j, k]]
                                * self.workspace.pressure_prev2[[i, j, k]];
                            let d2p2_dt2 = (p2 - 2.0 * p2_prev + p2_prev2) / (dt * dt);
                            rhs[[i, j, k]] += -coeff * d2p2_dt2;
                        }

                        if include_diffusion {
                            let d3p_dt3 = (pressure[[i, j, k]]
                                - 3.0 * self.workspace.pressure_prev[[i, j, k]]
                                + 3.0 * self.workspace.pressure_prev2[[i, j, k]]
                                - self.workspace.pressure_prev3[[i, j, k]])
                                / dt.powi(3);
                            rhs[[i, j, k]] += self.config.acoustic_diffusivity * d3p_dt3;
                        }

                        rhs[[i, j, k]] += source.get_source_term(t, x, y, z, &self.grid);
                    }
                }
            }
        } else {
            // Homogeneous: use pre-computed uniform properties
            let c0_squared = uniform_sound_speed * uniform_sound_speed;
            Zip::from(&mut *rhs)
                .and(&self.workspace.laplacian)
                .par_for_each(|r, &lap| *r = c0_squared * lap);

            for k in 0..self.grid.nz {
                for j in 0..self.grid.ny {
                    for i in 0..self.grid.nx {
                        let (x, y, z) = self.grid.indices_to_coordinates(i, j, k);
                        rhs[[i, j, k]] += source.get_source_term(t, x, y, z, &self.grid);
                    }
                }
            }

            if include_nonlinearity {
                Zip::from(&mut *rhs)
                    .and(&self.workspace.nonlinear_term)
                    .par_for_each(|r, &nl| *r += nl);
            }

            if include_diffusion {
                Zip::from(&mut *rhs)
                    .and(&self.workspace.diffusive_term)
                    .par_for_each(|r, &diff| *r += diff);
            }
        }
    }
}
