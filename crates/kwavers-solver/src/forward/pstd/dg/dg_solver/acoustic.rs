//! One-dimensional first-order acoustic DG stepping.
//!
//! ## System
//!
//! The homogeneous 1-D first-order acoustic equations are
//! ```text
//! p_t + K u_x = 0
//! u_t + (1/rho) p_x = 0
//! K = rho c^2
//! ```
//! with conservative state `q = [p, u]^T` and flux
//! `F(q) = [K u, p/rho]^T`. The flux Jacobian has eigenvalues `-c` and `+c`,
//! so the Lax-Friedrichs penalty speed is the acoustic sound speed `c`.
//!
//! ## Conservation Contract
//!
//! On periodic line elements, the strong-form residual applies component-wise
//! with face-normal orientation:
//! ```text
//! L(q)_e = -D F(q_e)
//!        + LIFT_left  (F*_{e-1/2} - F(q_e^-))
//!        + LIFT_right (F(q_e^+) - F*_{e+1/2})
//! ```
//! Therefore the quadrature-weighted integrals of pressure and velocity are
//! conserved because all interface fluxes telescope.
//!
//! References: Hesthaven & Warburton (2008) §6; Cockburn & Shu (2001) §4;
//! Pierce (1989) Ch. 1.

use super::core::DGSolver;
use kwavers_core::error::{KwaversError, KwaversResult};
use ndarray::{Array3, Zip};

mod tensor;
pub use tensor::{
    AcousticDgTensorWorkspace, ACOUSTIC_PRESSURE_VAR, ACOUSTIC_VARIABLES, ACOUSTIC_VELOCITY_X_VAR,
    ACOUSTIC_VELOCITY_Y_VAR, ACOUSTIC_VELOCITY_Z_VAR,
};

#[derive(Debug, Clone)]
pub struct AcousticDg1DWorkspace {
    p_original: Array3<f64>,
    u_original: Array3<f64>,
    p_stage: Array3<f64>,
    u_stage: Array3<f64>,
    p_rhs: Array3<f64>,
    u_rhs: Array3<f64>,
}

impl AcousticDg1DWorkspace {
    #[must_use]
    pub fn new(dim: (usize, usize, usize)) -> Self {
        Self {
            p_original: Array3::zeros(dim),
            u_original: Array3::zeros(dim),
            p_stage: Array3::zeros(dim),
            u_stage: Array3::zeros(dim),
            p_rhs: Array3::zeros(dim),
            u_rhs: Array3::zeros(dim),
        }
    }

    fn ensure_dim(&mut self, dim: (usize, usize, usize)) {
        if self.p_original.dim() != dim {
            *self = Self::new(dim);
        }
    }
}

impl DGSolver {
    /// Advance a pressure/velocity coefficient pair by one SSP-RK3 acoustic step.
    ///
    /// The coefficient layout is the legacy 1-D line layout:
    /// `(n_elements, p + 1, 1)`, with each element occupying the reference span
    /// `[-1, 1]`. This is the native coupled acoustic RHS needed before full
    /// pressure-field comparison with FDTD/PSTD.
    ///
    /// # Errors
    /// Returns an error for dimension mismatches or non-positive density.
    pub fn step_acoustic_1d_ssp_rk3(
        &self,
        pressure: &mut Array3<f64>,
        velocity: &mut Array3<f64>,
        density: f64,
        dt: f64,
        workspace: &mut AcousticDg1DWorkspace,
    ) -> KwaversResult<()> {
        validate_acoustic_inputs(self.n_nodes, pressure, velocity, density)?;
        workspace.ensure_dim(pressure.dim());
        workspace.p_original.assign(pressure);
        workspace.u_original.assign(velocity);

        self.compute_acoustic_rhs_1d_into(
            &workspace.p_original,
            &workspace.u_original,
            density,
            &mut workspace.p_rhs,
            &mut workspace.u_rhs,
        )?;
        Zip::from(&mut workspace.p_stage)
            .and(&workspace.p_original)
            .and(&workspace.p_rhs)
            .for_each(|stage, &p0, &rhs| *stage = p0 + dt * rhs);
        Zip::from(&mut workspace.u_stage)
            .and(&workspace.u_original)
            .and(&workspace.u_rhs)
            .for_each(|stage, &u0, &rhs| *stage = u0 + dt * rhs);

        self.compute_acoustic_rhs_1d_into(
            &workspace.p_stage,
            &workspace.u_stage,
            density,
            &mut workspace.p_rhs,
            &mut workspace.u_rhs,
        )?;
        Zip::from(&mut workspace.p_stage)
            .and(&workspace.p_original)
            .and(&workspace.p_rhs)
            .for_each(|stage, &p0, &rhs| {
                let p1 = *stage;
                *stage = 0.75 * p0 + 0.25 * (p1 + dt * rhs);
            });
        Zip::from(&mut workspace.u_stage)
            .and(&workspace.u_original)
            .and(&workspace.u_rhs)
            .for_each(|stage, &u0, &rhs| {
                let u1 = *stage;
                *stage = 0.75 * u0 + 0.25 * (u1 + dt * rhs);
            });

        self.compute_acoustic_rhs_1d_into(
            &workspace.p_stage,
            &workspace.u_stage,
            density,
            &mut workspace.p_rhs,
            &mut workspace.u_rhs,
        )?;
        Zip::from(pressure)
            .and(&workspace.p_original)
            .and(&workspace.p_stage)
            .and(&workspace.p_rhs)
            .for_each(|p_new, &p0, &p2, &rhs| {
                *p_new = (1.0 / 3.0) * p0 + (2.0 / 3.0) * (p2 + dt * rhs);
            });
        Zip::from(velocity)
            .and(&workspace.u_original)
            .and(&workspace.u_stage)
            .and(&workspace.u_rhs)
            .for_each(|u_new, &u0, &u2, &rhs| {
                *u_new = (1.0 / 3.0) * u0 + (2.0 / 3.0) * (u2 + dt * rhs);
            });
        Ok(())
    }

    /// Compute the 1-D first-order acoustic DG residual into caller-owned arrays.
    ///
    /// # Errors
    /// Returns an error for dimension mismatches or non-positive density.
    pub fn compute_acoustic_rhs_1d_into(
        &self,
        pressure: &Array3<f64>,
        velocity: &Array3<f64>,
        density: f64,
        pressure_rhs: &mut Array3<f64>,
        velocity_rhs: &mut Array3<f64>,
    ) -> KwaversResult<()> {
        validate_acoustic_inputs(self.n_nodes, pressure, velocity, density)?;
        if pressure_rhs.dim() != pressure.dim() || velocity_rhs.dim() != pressure.dim() {
            return Err(KwaversError::InvalidInput(format!(
                "acoustic RHS output dimension mismatch: pressure={:?}, p_rhs={:?}, u_rhs={:?}",
                pressure.dim(),
                pressure_rhs.dim(),
                velocity_rhs.dim()
            )));
        }

        pressure_rhs.fill(0.0);
        velocity_rhs.fill(0.0);
        let n_elements = pressure.shape()[0];
        let n_face = self.lift_matrix.shape()[1];
        let bulk = density * self.config.sound_speed * self.config.sound_speed;
        let inv_density = density.recip();

        for elem in 0..n_elements {
            for i in 0..self.n_nodes {
                let mut du = 0.0;
                let mut dp = 0.0;
                for j in 0..self.n_nodes {
                    du += self.diff_matrix[[i, j]] * velocity[(elem, j, 0)];
                    dp += self.diff_matrix[[i, j]] * pressure[(elem, j, 0)];
                }
                pressure_rhs[(elem, i, 0)] -= bulk * du;
                velocity_rhs[(elem, i, 0)] -= inv_density * dp;
            }
        }

        for elem in 0..n_elements {
            let left_elem = if elem == 0 { n_elements - 1 } else { elem - 1 };
            let right_elem = (elem + 1) % n_elements;

            let left_ext = AcousticState {
                pressure: pressure[(left_elem, self.n_nodes - 1, 0)],
                velocity: velocity[(left_elem, self.n_nodes - 1, 0)],
            };
            let left_int = AcousticState {
                pressure: pressure[(elem, 0, 0)],
                velocity: velocity[(elem, 0, 0)],
            };
            let right_int = AcousticState {
                pressure: pressure[(elem, self.n_nodes - 1, 0)],
                velocity: velocity[(elem, self.n_nodes - 1, 0)],
            };
            let right_ext = AcousticState {
                pressure: pressure[(right_elem, 0, 0)],
                velocity: velocity[(right_elem, 0, 0)],
            };

            let flux_left = lax_friedrichs_acoustic_flux(
                left_ext,
                left_int,
                bulk,
                inv_density,
                self.config.sound_speed,
            );
            let flux_right = lax_friedrichs_acoustic_flux(
                right_int,
                right_ext,
                bulk,
                inv_density,
                self.config.sound_speed,
            );
            let left_inner_flux = acoustic_flux(left_int, bulk, inv_density);
            let right_inner_flux = acoustic_flux(right_int, bulk, inv_density);

            let p_res_left = flux_left.pressure - left_inner_flux.pressure;
            let u_res_left = flux_left.velocity - left_inner_flux.velocity;
            let p_res_right = right_inner_flux.pressure - flux_right.pressure;
            let u_res_right = right_inner_flux.velocity - flux_right.velocity;

            for i in 0..self.n_nodes {
                if n_face > 0 {
                    pressure_rhs[(elem, i, 0)] += self.lift_matrix[[i, 0]] * p_res_left;
                    velocity_rhs[(elem, i, 0)] += self.lift_matrix[[i, 0]] * u_res_left;
                }
                if n_face > 1 {
                    pressure_rhs[(elem, i, 0)] += self.lift_matrix[[i, 1]] * p_res_right;
                    velocity_rhs[(elem, i, 0)] += self.lift_matrix[[i, 1]] * u_res_right;
                }
            }
        }
        Ok(())
    }
}

#[derive(Debug, Clone, Copy)]
struct AcousticState {
    pressure: f64,
    velocity: f64,
}

fn validate_acoustic_inputs(
    n_nodes: usize,
    pressure: &Array3<f64>,
    velocity: &Array3<f64>,
    density: f64,
) -> KwaversResult<()> {
    if !density.is_finite() || density <= 0.0 {
        return Err(KwaversError::InvalidInput(format!(
            "acoustic DG density must be finite and positive, got {density}"
        )));
    }
    if pressure.dim() != velocity.dim() {
        return Err(KwaversError::InvalidInput(format!(
            "acoustic DG pressure/velocity dimension mismatch: {:?} vs {:?}",
            pressure.dim(),
            velocity.dim()
        )));
    }
    if pressure.shape()[1] != n_nodes || pressure.shape()[2] != 1 {
        return Err(KwaversError::InvalidInput(format!(
            "acoustic DG expects line coefficients (n_elements, {n_nodes}, 1), got {:?}",
            pressure.dim()
        )));
    }
    Ok(())
}

#[inline]
fn acoustic_flux(state: AcousticState, bulk: f64, inv_density: f64) -> AcousticState {
    AcousticState {
        pressure: bulk * state.velocity,
        velocity: inv_density * state.pressure,
    }
}

#[inline]
fn lax_friedrichs_acoustic_flux(
    left: AcousticState,
    right: AcousticState,
    bulk: f64,
    inv_density: f64,
    wave_speed: f64,
) -> AcousticState {
    let left_flux = acoustic_flux(left, bulk, inv_density);
    let right_flux = acoustic_flux(right, bulk, inv_density);
    AcousticState {
        pressure: 0.5 * (left_flux.pressure + right_flux.pressure)
            - 0.5 * wave_speed * (right.pressure - left.pressure),
        velocity: 0.5 * (left_flux.velocity + right_flux.velocity)
            - 0.5 * wave_speed * (right.velocity - left.velocity),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::forward::pstd::dg::DGConfig;
    use kwavers_grid::Grid;
    use std::sync::Arc;

    fn make_solver() -> DGSolver {
        let grid = Arc::new(Grid::new(12, 1, 1, 1.0, 1.0, 1.0).unwrap());
        let config = DGConfig {
            polynomial_order: 2,
            sound_speed: 1.0,
            ..DGConfig::default()
        };
        DGSolver::new(config, grid).unwrap()
    }

    #[test]
    fn constant_acoustic_state_has_zero_rhs() {
        let solver = make_solver();
        let pressure = Array3::from_elem((4, 3, 1), 2.0);
        let velocity = Array3::from_elem((4, 3, 1), -0.25);
        let mut pressure_rhs = Array3::zeros((4, 3, 1));
        let mut velocity_rhs = Array3::zeros((4, 3, 1));

        solver
            .compute_acoustic_rhs_1d_into(
                &pressure,
                &velocity,
                1.0,
                &mut pressure_rhs,
                &mut velocity_rhs,
            )
            .unwrap();

        for &value in pressure_rhs.iter().chain(velocity_rhs.iter()) {
            assert!(value.abs() < 1.0e-12);
        }
    }

    #[test]
    fn periodic_acoustic_rhs_preserves_component_masses() {
        let solver = make_solver();
        let pressure = Array3::from_shape_fn((4, 3, 1), |(elem, node, _)| {
            (elem as f64 + node as f64).sin()
        });
        let velocity = Array3::from_shape_fn((4, 3, 1), |(elem, node, _)| {
            (0.3 * elem as f64 + 0.7 * node as f64).cos()
        });
        let mut pressure_rhs = Array3::zeros((4, 3, 1));
        let mut velocity_rhs = Array3::zeros((4, 3, 1));

        solver
            .compute_acoustic_rhs_1d_into(
                &pressure,
                &velocity,
                1.0,
                &mut pressure_rhs,
                &mut velocity_rhs,
            )
            .unwrap();

        let pressure_rate = weighted_sum(&pressure_rhs, &solver.weights);
        let velocity_rate = weighted_sum(&velocity_rhs, &solver.weights);
        assert!(
            pressure_rate.abs() < 1.0e-12,
            "periodic acoustic DG must conserve weighted pressure mass; rate={pressure_rate:e}"
        );
        assert!(
            velocity_rate.abs() < 1.0e-12,
            "periodic acoustic DG must conserve weighted velocity mass; rate={velocity_rate:e}"
        );
    }

    fn weighted_sum(values: &Array3<f64>, weights: &ndarray::Array1<f64>) -> f64 {
        let mut sum = 0.0;
        for elem in 0..values.dim().0 {
            for node in 0..values.dim().1 {
                sum += weights[node] * values[(elem, node, 0)];
            }
        }
        sum
    }
}
