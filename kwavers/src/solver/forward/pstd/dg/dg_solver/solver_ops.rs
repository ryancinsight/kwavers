//! DG time-stepping: SSP-RK3 and Forward Euler
//!
//! ## Algorithm: SSP-RK3 (Shu & Osher 1988)
//!
//! Given the semi-discrete ODE `du/dt = L(u)` where `L` is the DG spatial operator
//! (volume integrals + surface flux), the three-stage update is:
//!
//! ```text
//!   u⁽¹⁾ = u^n + dt · L(u^n)
//!   u⁽²⁾ = (3/4) u^n + (1/4) [u⁽¹⁾ + dt · L(u⁽¹⁾)]
//!   u^{n+1} = (1/3) u^n + (2/3) [u⁽²⁾ + dt · L(u⁽²⁾)]
//! ```
//!
//! **Theorem (Shu & Osher 1988, Thm. 2.1):** SSP-RK3 is a convex combination of
//! forward Euler steps. Provided each Euler sub-step satisfies the TVD condition
//! `‖u⁽¹⁾‖_TV ≤ ‖u^n‖_TV`, the full scheme satisfies `‖u^{n+1}‖_TV ≤ ‖u^n‖_TV`.
//!
//! **CFL condition for DG(p):** `dt ≤ h / [c · (2p+1)]`  (Cockburn & Shu 2001 §4).
//!
//! ## References
//!
//! - Shu & Osher (1988). J. Comput. Phys. 77(2):439–471.
//! - Cockburn & Shu (2001). J. Sci. Comput. 16(3):173–261.

use super::super::config::DgTimeIntegrator;
use super::super::matrices::matrix_inverse;
use super::core::DGSolver;
use crate::core::error::KwaversResult;
use crate::core::error::{KwaversError, ValidationError};
use ndarray::{Array2, Array3};

impl DGSolver {
    /// Advance the field by one time step `dt` using the configured integrator.
    ///
    /// Projects `field` to the DG basis on the first call.  Subsequent calls
    /// operate on the stored modal coefficients, then back-project to `field`.
    /// # Errors
    /// - Propagates any [`KwaversError`] returned by called functions.
    ///
    pub fn solve_step(&mut self, field: &mut Array3<f64>, dt: f64) -> KwaversResult<()> {
        if self.modal_coefficients.is_none() {
            self.project_to_dg(field)?;
        }

        let wave_speed = self.config.sound_speed;
        let mass_inv = matrix_inverse(&self.mass_matrix)?;

        match self.config.time_integrator {
            DgTimeIntegrator::SspRk3 => self.step_ssp_rk3(dt, wave_speed, &mass_inv),
            DgTimeIntegrator::ForwardEuler => self.step_forward_euler(dt, wave_speed, &mass_inv),
        }
    }

    // ── SSP-RK3 (Shu & Osher 1988) ─────────────────────────────────────────

    /// Three-stage SSP-RK3 time step.
    ///
    /// Butcher coefficients:
    /// ```text
    ///   Stage 1: u⁽¹⁾ = u^n + dt · L(u^n)
    ///   Stage 2: u⁽²⁾ = (3/4) u^n + (1/4) [u⁽¹⁾ + dt · L(u⁽¹⁾)]
    ///   Stage 3: u^{n+1} = (1/3) u^n + (2/3) [u⁽²⁾ + dt · L(u⁽²⁾)]
    /// ```
    /// # Errors
    /// - Propagates any [`KwaversError`] returned by called functions.
    ///
    fn step_ssp_rk3(
        &mut self,
        dt: f64,
        wave_speed: f64,
        mass_inv: &Array2<f64>,
    ) -> KwaversResult<()> {
        let u_n = self
            .modal_coefficients
            .as_ref()
            .ok_or_else(|| {
                KwaversError::Validation(ValidationError::MissingField {
                    field: "modal_coefficients".to_owned(),
                })
            })?
            .clone();

        // Stage 1
        let rhs_n = self.compute_rhs_from_coeffs(&u_n, mass_inv, wave_speed)?;
        let u1 = &u_n + &(dt * &rhs_n);

        // Stage 2
        let rhs_1 = self.compute_rhs_from_coeffs(&u1, mass_inv, wave_speed)?;
        let u2 = 0.75 * &u_n + 0.25 * (&u1 + &(dt * &rhs_1));

        // Stage 3
        let rhs_2 = self.compute_rhs_from_coeffs(&u2, mass_inv, wave_speed)?;
        let u_new = (1.0 / 3.0) * &u_n + (2.0 / 3.0) * (&u2 + &(dt * &rhs_2));

        self.modal_coefficients = Some(u_new);
        Ok(())
    }

    // ── Forward Euler (legacy, p=0 only) ───────────────────────────────────

    /// First-order Forward Euler step.
    ///
    /// **Warning**: conditionally stable for p=0, unconditionally unstable for p ≥ 1
    /// under the DG operator (Cockburn & Shu 2001 §4). Use only for baseline p=0 tests.
    /// # Errors
    /// - Propagates any [`KwaversError`] returned by called functions.
    ///
    fn step_forward_euler(
        &mut self,
        dt: f64,
        wave_speed: f64,
        mass_inv: &Array2<f64>,
    ) -> KwaversResult<()> {
        let u_n = self
            .modal_coefficients
            .as_ref()
            .ok_or_else(|| {
                KwaversError::Validation(ValidationError::MissingField {
                    field: "modal_coefficients".to_owned(),
                })
            })?
            .clone();

        let rhs = self.compute_rhs_from_coeffs(&u_n, mass_inv, wave_speed)?;
        self.modal_coefficients = Some(&u_n + &(dt * &rhs));
        Ok(())
    }

    // ── RHS operator L(u) ──────────────────────────────────────────────────

    /// Compute the DG spatial operator `L(u) = −M⁻¹ S f(u) + surface_flux_lift`.
    ///
    /// **Volume term** (strong form for linear advection `f(u) = c·u`):
    /// ```text
    ///   R_e^vol = −c · D · u_e
    /// ```
    ///
    /// **Surface term** (Lax–Friedrichs upwind flux with periodic BCs):
    /// ```text
    ///   f*(u⁻, u⁺) = (1/2) c (u⁻ + u⁺) − (c/2)(u⁺ − u⁻)
    /// ```
    /// The LIFT matrix maps face residuals (f* − f_interior) to volume DOFs.
    ///
    /// ## Reference
    /// Hesthaven & Warburton (2008). *Nodal Discontinuous Galerkin Methods*. §3, §6.3.
    /// # Errors
    /// - Returns [`Err`] if an internal constraint is violated.
    ///
    fn compute_rhs_from_coeffs(
        &self,
        coeffs: &Array3<f64>,
        _mass_inv: &Array2<f64>,
        wave_speed: f64,
    ) -> KwaversResult<Array3<f64>> {
        let n_elements = coeffs.shape()[0];
        let n_vars = coeffs.shape()[2];
        let mut rhs = Array3::zeros(coeffs.raw_dim());

        let d_matrix = &self.diff_matrix;
        let lift = &self.lift_matrix;
        let n_face = lift.shape()[1];

        // Volume integrals: strong form u_t = −c · D · u
        for var in 0..n_vars {
            for elem in 0..n_elements {
                for i in 0..self.n_nodes {
                    let mut du = 0.0;
                    for j in 0..self.n_nodes {
                        du += d_matrix[[i, j]] * coeffs[(elem, j, var)];
                    }
                    rhs[(elem, i, var)] -= wave_speed * du;
                }
            }
        }

        // Surface flux with periodic boundary conditions
        for var in 0..n_vars {
            for elem in 0..n_elements {
                let left_elem = if elem == 0 { n_elements - 1 } else { elem - 1 };
                let right_elem = (elem + 1) % n_elements;

                let u_minus_left = coeffs[(left_elem, self.n_nodes - 1, var)];
                let u_plus_left = coeffs[(elem, 0, var)];
                let u_minus_right = coeffs[(elem, self.n_nodes - 1, var)];
                let u_plus_right = coeffs[(right_elem, 0, var)];

                // Lax–Friedrichs: f*(u⁻,u⁺) = 0.5 c(u⁻+u⁺) − 0.5 c(u⁺−u⁻)
                let flux_left = (0.5 * wave_speed).mul_add(u_minus_left + u_plus_left, -(0.5 * wave_speed * (u_plus_left - u_minus_left)));
                let flux_right = (0.5 * wave_speed).mul_add(u_minus_right + u_plus_right, -(0.5 * wave_speed * (u_plus_right - u_minus_right)));

                let f_int_left = wave_speed * u_plus_left;
                let f_int_right = wave_speed * u_minus_right;

                let mut face_res = vec![0.0f64; n_face];
                if n_face >= 2 {
                    face_res[0] = -(flux_left - f_int_left);
                    face_res[1] = flux_right - f_int_right;
                }

                for i in 0..self.n_nodes {
                    for f in 0..n_face {
                        rhs[(elem, i, var)] += lift[[i, f]] * face_res[f];
                    }
                }
            }
        }

        Ok(rhs)
    }
}
