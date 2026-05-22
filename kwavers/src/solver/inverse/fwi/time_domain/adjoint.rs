//! Adjoint simulation run, adjoint-source construction, L2 residual and objective.

use super::{geometry::FwiGeometry, FwiProcessor};
use crate::core::error::{KwaversError, KwaversResult, ValidationError};
use crate::domain::grid::Grid;
use crate::domain::source::{GridSource, SourceMode};
use crate::solver::inverse::fwi::time_domain::{
    accumulate_signed_correlation, l2_objective, l2_residual, reverse_time_axis,
};
use ndarray::{Array2, Array3, Array4, ArrayView3, ArrayViewMut3, Axis, Zip};

/// Apply the Plessix (2006) eq. (12) per-voxel scaling
/// `g_c(x) ← -(2 / (ρ(x) · c(x)³)) · g_correlation(x)` in place.
///
/// ## Theorem
/// Given an accumulated correlation `I(x) = ∫₀ᵀ p̈_fwd(x,t) · λ(x,t) dt` and a
/// strictly positive density / velocity field, the velocity-model gradient
/// of the acoustic L2 misfit is `g_c(x) = -(2 / (ρ(x) · c(x)³)) · I(x)`.
/// This routine is the post-correlation scaling step; it is independent of
/// the forward/adjoint wavefield computation and therefore directly
/// unit-testable for the local ρ-dependence.
///
/// # Errors
/// Returns [`KwaversError::Validation`] if any sound-speed or density entry
/// is non-finite or non-positive (avoids silent NaN production from a 1/0
/// or 1/NaN multiplication).
pub(super) fn apply_velocity_gradient_scaling(
    mut gradient: ArrayViewMut3<'_, f64>,
    model: ArrayView3<'_, f64>,
    density: ArrayView3<'_, f64>,
) -> KwaversResult<()> {
    if gradient.dim() != model.dim() || gradient.dim() != density.dim() {
        return Err(KwaversError::Validation(
            ValidationError::ConstraintViolation {
                message: format!(
                    "FWI gradient scaling shape mismatch: gradient {:?}, model {:?}, density {:?}",
                    gradient.dim(),
                    model.dim(),
                    density.dim()
                ),
            },
        ));
    }
    for (&c, &rho) in model.iter().zip(density.iter()) {
        if !c.is_finite() || c <= 0.0 || !rho.is_finite() || rho <= 0.0 {
            return Err(KwaversError::Validation(
                ValidationError::ConstraintViolation {
                    message: format!(
                        "FWI gradient scaling requires positive finite c and ρ; got c={c}, ρ={rho}"
                    ),
                },
            ));
        }
    }
    Zip::from(&mut gradient)
        .and(model)
        .and(density)
        .par_for_each(|g, &c, &rho| {
            *g *= -2.0 / (rho * c.powi(3));
        });
    Ok(())
}

impl FwiProcessor {
    /// Compute the acoustic L2 objective between observed and synthetic data.
    ///
    /// ## Theorem
    /// `J = (dt / 2) Σ_{r,t} (d_syn - d_obs)²` is non-negative and vanishes
    /// iff the synthetic and observed traces match pointwise.
    /// # Errors
    /// - Returns [`Err`] if an internal constraint is violated.
    ///
    pub(super) fn compute_l2_objective(
        &self,
        observed: &Array2<f64>,
        synthetic: &Array2<f64>,
    ) -> KwaversResult<f64> {
        l2_objective(self.parameters.dt, observed, synthetic)
    }

    /// Compute the discrete adjoint source for L2 misfit.
    ///
    /// Returns `d_syn - d_obs`; time reversal is applied when constructing
    /// the pressure source signal.
    /// # Errors
    /// - Returns [`Err`] if an internal constraint is violated.
    ///
    pub(super) fn compute_adjoint_source(
        &self,
        observed: &Array2<f64>,
        synthetic: &Array2<f64>,
    ) -> KwaversResult<Array2<f64>> {
        l2_residual(observed, synthetic)
    }

    /// Build the time-reversed pressure source used in the adjoint run.
    ///
    /// ## Theorem
    /// Reversing the residual in time and injecting it through the receiver mask
    /// produces the discrete time-reversal adjoint for the linear acoustic
    /// operator when the forward and adjoint solvers share the same stencil and
    /// boundary model.
    ///
    /// ## Proof sketch
    /// The discrete adjoint of the wave operator is its time reverse under the
    /// same inner product.
    ///
    /// ## Adjoint source injection mode
    ///
    /// The adjoint equation `L†λ = −δ_r · (d_syn − d_obs)(T−t)` is a
    /// body-force forcing, not a Dirichlet BC.  Injecting as Dirichlet pins `λ`
    /// at receiver positions, suppressing back-propagation and producing an
    /// incorrect gradient.
    ///
    /// Reference: Plessix (2006), GFJI 167(2), 495–503, eq. (2)–(6).
    /// # Errors
    /// - Returns [`KwaversError::Validation`] if the precondition for a Validation-class constraint is violated.
    ///
    pub(super) fn build_adjoint_source(
        &self,
        residual: &Array2<f64>,
        geometry: &FwiGeometry,
    ) -> KwaversResult<GridSource> {
        let expected_rows = geometry.receiver_count();
        if residual.nrows() != expected_rows {
            return Err(KwaversError::Validation(
                ValidationError::ConstraintViolation {
                    message: format!(
                        "Residual receiver count mismatch: expected {}, got {}",
                        expected_rows,
                        residual.nrows()
                    ),
                },
            ));
        }
        if residual.ncols() != self.parameters.nt {
            return Err(KwaversError::Validation(
                ValidationError::ConstraintViolation {
                    message: format!(
                        "Residual time length mismatch: expected {}, got {}",
                        self.parameters.nt,
                        residual.ncols()
                    ),
                },
            ));
        }

        let reversed_residual = reverse_time_axis(residual);
        let mut p_signal = Array2::zeros((expected_rows, self.parameters.nt));
        for source_row in 0..expected_rows {
            let sensor_row = geometry.receiver_row_to_sensor_row[source_row];
            for t in 0..self.parameters.nt {
                p_signal[[source_row, t]] = reversed_residual[[sensor_row, t]];
            }
        }

        let p_mask = geometry
            .sensor_mask
            .mapv(|active| if active { 1.0 } else { 0.0 });

        Ok(GridSource {
            p0: None,
            u0: None,
            p_mask: Some(p_mask),
            p_signal: Some(p_signal),
            p_mode: SourceMode::Additive,
            u_mask: None,
            u_signal: None,
            u_mode: SourceMode::default(),
        })
    }

    /// Adjoint modeling using time-reversed simulation.
    ///
    /// Computes the adjoint wavefield by running the solver (same type as the
    /// forward pass, selected by `FwiParameters::solver_type`) in reverse time
    /// with the adjoint source (data residual) as input, then accumulates the
    /// discrete velocity-model gradient
    ///
    /// ```text
    /// g_c(x) = (-2 / c(x)^3) · Σ_t  (-dt) · p̈_fwd(x, T-t) · λ(x, t)
    ///        ≈ (-2 / c(x)^3) · ∫₀ᵀ p̈_fwd · λ  dt
    /// ```
    ///
    /// where the `−dt` factor inside [`accumulate_signed_correlation`] is the
    /// time-integration measure (Riemann sum approximating the continuous
    /// adjoint-state integral; the discrete L2 objective `J = (dt/2) Σ_t r²` is
    /// the discretisation of `½∫r² dt`, so the residual itself carries no
    /// extra `dt` weight in the adjoint source — see
    /// [`build_adjoint_source`] / [`l2_residual`]).
    ///
    /// ## Solver-type reciprocity
    ///
    /// The adjoint solver is constructed through the same `build_fdtd_boxed` /
    /// `build_pstd_boxed` helpers as the forward pass (dispatching on the same
    /// `FwiParameters::solver_type`).  This guarantees that the discrete adjoint
    /// operator is the exact time-reversal of the forward operator — the
    /// time-reversal theorem holds if and only if forward and adjoint share the
    /// same stencil and boundary treatment.
    ///
    /// ## Density handling
    ///
    /// The full acoustic gradient (Plessix 2006, eq. 12) is
    /// `g_c = -(2 / (ρ c³)) ∫ p̈ · λ dt` (after one integration by parts).
    /// The local density is taken from
    /// [`FwiProcessor::resolved_density`]: if the caller supplied a
    /// heterogeneous field via [`FwiProcessor::with_density`], `ρ(x)` is
    /// used both in the forward / adjoint medium and in the per-voxel
    /// scaling below; otherwise the constant
    /// [`RHO_SEISMIC_REF`](super::RHO_SEISMIC_REF) (2000 kg/m³) is used
    /// uniformly. Because the forward and adjoint media share the same
    /// resolved density, the discrete adjoint operator is exactly the
    /// time-reverse of the forward operator in both cases.
    ///
    /// # References
    /// * Tromp et al. (2005): "Seismic tomography, adjoint methods"
    /// * Plessix (2006), GFJI 167(2), eq. (5)–(6) and (12)
    /// # Errors
    /// - Returns [`KwaversError::Validation`] if the precondition for a Validation-class constraint is violated.
    /// - Propagates any [`KwaversError`] returned by called functions.
    ///
    pub(super) fn adjoint_model(
        &self,
        adjoint_source: &GridSource,
        model: &Array3<f64>,
        grid: &Grid,
        forward_history: &Array4<f64>,
        source_mask: Option<&Array3<f64>>,
    ) -> KwaversResult<Array3<f64>> {
        use crate::solver::config::SolverType;

        if forward_history.len_of(Axis(0)) != self.parameters.nt {
            return Err(KwaversError::Validation(
                ValidationError::ConstraintViolation {
                    message: format!(
                        "Forward history length mismatch: expected {}, got {}",
                        self.parameters.nt,
                        forward_history.len_of(Axis(0))
                    ),
                },
            ));
        }

        let (nx, ny, nz) = grid.dimensions();
        let dt = self.validate_time_step(model, grid)?;

        // Resolved once here so both the medium construction inside the solver
        // builder and the per-voxel Plessix (2006) eq. (12) scaling below use
        // exactly the same density field.
        let density_adj = self.resolved_density(grid)?;

        // Adjoint solver uses the same SolverType as the forward pass to preserve
        // the discrete time-reversal symmetry (time-reversal theorem).
        // No sensor mask on the adjoint solver — receiver data is injected through
        // the adjoint_source pressure signal, not recorded.
        let mut solver: Box<dyn crate::solver::interface::Solver> =
            match self.parameters.solver_type {
                SolverType::FDTD => self.build_fdtd_boxed(
                    model,
                    None, // no sensor recording on adjoint pass
                    grid,
                    dt,
                    adjoint_source.clone(),
                )?,
                SolverType::PSTD => self.build_pstd_boxed(
                    model,
                    None, // no sensor recording on adjoint pass
                    grid,
                    dt,
                    adjoint_source.clone(),
                )?,
                other => {
                    return Err(KwaversError::InvalidInput(format!(
                        "FWI adjoint builder: SolverType::{other:?} is not yet supported; \
                         use FDTD or PSTD"
                    )))
                }
            };

        let mut gradient_m = Array3::zeros((nx, ny, nz));
        let mut p_tt = Array3::zeros((nx, ny, nz));

        for t in 0..self.parameters.nt {
            // `solver` is `Box<dyn Solver>` — dynamic dispatch through the
            // unified Solver trait (T19a/T19b/T10).
            solver.step_forward()?;
            let fwd_idx = self.parameters.nt - 1 - t;
            self.pressure_second_derivative_into(forward_history, fwd_idx, dt, &mut p_tt)?;

            // Exclude Dirichlet-source voxels from the gradient kernel.
            // Reference: Sun & Symes (1991), SEG Expanded Abstracts.
            if let Some(mask) = source_mask {
                Zip::from(&mut p_tt).and(mask).par_for_each(|pt, &m| {
                    if m > 0.5 {
                        *pt = 0.0;
                    }
                });
            }

            // pressure_field() via Box<dyn Solver> dynamic dispatch — replaces
            // the previous UFCS `<FdtdSolver as Solver>::pressure_field(&solver)`.
            accumulate_signed_correlation(
                &mut gradient_m,
                p_tt.view(),
                solver.pressure_field().view(),
                -dt,
            )?;
        }

        // Apply the local ρ(x)⁻¹ · c(x)⁻³ scaling from Plessix (2006) eq. (12).
        // The free function is value-semantically tested in isolation; this
        // call is its sole production caller.
        apply_velocity_gradient_scaling(gradient_m.view_mut(), model.view(), density_adj.view())?;

        let (gmax, gmax_idx) = gradient_m.indexed_iter().fold(
            (0.0_f64, (0usize, 0usize, 0usize)),
            |(best, bi), (idx, &v)| {
                if v.abs() > best {
                    (v.abs(), idx)
                } else {
                    (best, bi)
                }
            },
        );
        log::info!(
            "adjoint gradient peak {:.4e} at ({},{},{})",
            gmax,
            gmax_idx.0,
            gmax_idx.1,
            gmax_idx.2
        );

        Ok(gradient_m)
    }
}
