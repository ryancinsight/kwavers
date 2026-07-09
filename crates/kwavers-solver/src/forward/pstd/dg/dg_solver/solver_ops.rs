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
//! The nodal DG matrices are assembled so the differentiation matrix `D = M⁻¹S`
//! and lift matrix `LIFT = M⁻¹E` already contain the inverse mass action.
//! Time stepping must therefore not recompute `M⁻¹` per step; doing so is both
//! redundant and an avoidable dense allocation.
//!
//! ## Troubled-cell limiting
//!
//! When shock capture is enabled, each SSP-RK stage is followed by a conservative
//! TVD projection on troubled elements. The indicator is the maximum of
//! left/right element-mean jumps and intra-element variation, normalised by
//! local amplitude. Flagged elements keep their quadrature-weighted mean and
//! receive the limited linear reconstruction
//! `u(xi)=mean + 0.5*(xi - xi_bar)*limited_slope`, where `xi_bar` is the
//! quadrature-weighted node mean. Thus the same mass functional represented by
//! the diagonal DG mass matrix is preserved exactly.
//!
//! ## References
//!
//! - Shu & Osher (1988). J. Comput. Phys. 77(2):439–471.
//! - Cockburn & Shu (2001). J. Sci. Comput. 16(3):173–261.

use super::super::config::DgTimeIntegrator;
use super::core::DGSolver;
use super::limiting::{apply_shock_capture_to_coeffs, apply_shock_capture_to_tensor_coeffs};
use super::rhs::{compute_rhs_from_coeffs_into, RhsOperator};
use super::rk_update::{update_euler, update_forward_euler, update_ssp_final, update_ssp_second};
use super::topology::CoefficientLayout;
use kwavers_core::error::KwaversResult;
use kwavers_core::error::{KwaversError, ValidationError};
use leto::Array3;

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
        let dim = self
            .modal_coefficients
            .as_ref()
            .ok_or_else(|| {
                KwaversError::Validation(ValidationError::MissingField {
                    field: "modal_coefficients".to_owned(),
                })
            })?
            .dim();
        self.ensure_rk_workspace(dim);

        match self.config.time_integrator {
            DgTimeIntegrator::SspRk3 => self.step_ssp_rk3(dt, self.config.sound_speed),
            DgTimeIntegrator::ForwardEuler => self.step_forward_euler(dt, self.config.sound_speed),
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
    fn step_ssp_rk3(&mut self, dt: f64, wave_speed: f64) -> KwaversResult<()> {
        let diff_matrix = self.diff_matrix.clone();
        let lift_matrix = self.lift_matrix.clone();
        let n_nodes = self.n_nodes;

        self.rk_original
            .assign(self.modal_coefficients.as_ref().ok_or_else(|| {
                KwaversError::Validation(ValidationError::MissingField {
                    field: "modal_coefficients".to_owned(),
                })
            })?);

        // Stage 1
        compute_rhs_from_coeffs_into(
            RhsOperator {
                n_nodes,
                d_matrix: &diff_matrix,
                lift: &lift_matrix,
                layout: self.coefficient_layout,
                wave_speed,
                axis_scales: self.axis_scales(),
            },
            &self.rk_original,
            &mut self.rk_rhs,
        );
        update_euler(&mut self.rk_stage, &self.rk_original, &self.rk_rhs, dt);
        if self.config.shock_capture.apply_per_stage {
            apply_shock_capture_for_layout(
                self.config,
                self.coefficient_layout,
                &self.xi_nodes,
                &self.weights,
                &mut self.rk_stage,
                &mut self.rk_rhs,
            )?;
        }

        // Stage 2
        compute_rhs_from_coeffs_into(
            RhsOperator {
                n_nodes,
                d_matrix: &diff_matrix,
                lift: &lift_matrix,
                layout: self.coefficient_layout,
                wave_speed,
                axis_scales: self.axis_scales(),
            },
            &self.rk_stage,
            &mut self.rk_rhs,
        );
        update_ssp_second(&mut self.rk_stage, &self.rk_original, &self.rk_rhs, dt);
        if self.config.shock_capture.apply_per_stage {
            apply_shock_capture_for_layout(
                self.config,
                self.coefficient_layout,
                &self.xi_nodes,
                &self.weights,
                &mut self.rk_stage,
                &mut self.rk_rhs,
            )?;
        }

        // Stage 3
        compute_rhs_from_coeffs_into(
            RhsOperator {
                n_nodes,
                d_matrix: &diff_matrix,
                lift: &lift_matrix,
                layout: self.coefficient_layout,
                wave_speed,
                axis_scales: self.axis_scales(),
            },
            &self.rk_stage,
            &mut self.rk_rhs,
        );
        let coeffs = self.modal_coefficients.as_mut().ok_or_else(|| {
            KwaversError::Validation(ValidationError::MissingField {
                field: "modal_coefficients".to_owned(),
            })
        })?;
        update_ssp_final(coeffs, &self.rk_original, &self.rk_stage, &self.rk_rhs, dt);
        let coeffs = self.modal_coefficients.as_mut().ok_or_else(|| {
            KwaversError::Validation(ValidationError::MissingField {
                field: "modal_coefficients".to_owned(),
            })
        })?;
        apply_shock_capture_for_layout(
            self.config,
            self.coefficient_layout,
            &self.xi_nodes,
            &self.weights,
            coeffs,
            &mut self.rk_rhs,
        )?;

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
    fn step_forward_euler(&mut self, dt: f64, wave_speed: f64) -> KwaversResult<()> {
        let diff_matrix = self.diff_matrix.clone();
        let lift_matrix = self.lift_matrix.clone();
        let n_nodes = self.n_nodes;
        compute_rhs_from_coeffs_into(
            RhsOperator {
                n_nodes,
                d_matrix: &diff_matrix,
                lift: &lift_matrix,
                layout: self.coefficient_layout,
                wave_speed,
                axis_scales: self.axis_scales(),
            },
            self.modal_coefficients.as_ref().ok_or_else(|| {
                KwaversError::Validation(ValidationError::MissingField {
                    field: "modal_coefficients".to_owned(),
                })
            })?,
            &mut self.rk_rhs,
        );
        let coeffs = self.modal_coefficients.as_mut().ok_or_else(|| {
            KwaversError::Validation(ValidationError::MissingField {
                field: "modal_coefficients".to_owned(),
            })
        })?;
        update_forward_euler(coeffs, &self.rk_rhs, dt);
        let coeffs = self.modal_coefficients.as_mut().ok_or_else(|| {
            KwaversError::Validation(ValidationError::MissingField {
                field: "modal_coefficients".to_owned(),
            })
        })?;
        apply_shock_capture_for_layout(
            self.config,
            self.coefficient_layout,
            &self.xi_nodes,
            &self.weights,
            coeffs,
            &mut self.rk_rhs,
        )?;
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
    #[cfg(test)]
    fn compute_rhs_from_coeffs(
        &self,
        coeffs: &Array3<f64>,
        wave_speed: f64,
    ) -> KwaversResult<Array3<f64>> {
        let mut rhs = Array3::zeros(coeffs.raw_dim());
        compute_rhs_from_coeffs_into(
            RhsOperator {
                n_nodes: self.n_nodes,
                d_matrix: &self.diff_matrix,
                lift: &self.lift_matrix,
                layout: self.coefficient_layout,
                wave_speed,
                axis_scales: self.axis_scales(),
            },
            coeffs,
            &mut rhs,
        );
        Ok(rhs)
    }
}

#[cfg(test)]
mod tests;

impl DGSolver {
    fn axis_scales(&self) -> [f64; 3] {
        let element_spans = [
            self.n_nodes as f64 * self.grid.dx,
            self.n_nodes as f64 * self.grid.dy,
            self.n_nodes as f64 * self.grid.dz,
        ];
        [
            2.0 / element_spans[0],
            2.0 / element_spans[1],
            2.0 / element_spans[2],
        ]
    }
}

fn apply_shock_capture_for_layout(
    config: super::super::config::DGConfig,
    layout: CoefficientLayout,
    xi_nodes: &leto::Array1<f64>,
    weights: &leto::Array1<f64>,
    coeffs: &mut Array3<f64>,
    scratch: &mut Array3<f64>,
) -> KwaversResult<()> {
    match layout {
        CoefficientLayout::TensorProduct(topology)
            if coeffs.shape()[1] == topology.nodes_per_element =>
        {
            apply_shock_capture_to_tensor_coeffs(config, topology, weights, coeffs, scratch)
        }
        _ => apply_shock_capture_to_coeffs(config, xi_nodes, weights, coeffs, scratch),
    }
}
