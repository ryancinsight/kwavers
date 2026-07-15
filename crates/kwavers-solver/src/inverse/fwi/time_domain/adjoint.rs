//! Adjoint simulation run, adjoint-source construction, L2 residual and objective.

use super::{geometry::FwiGeometry, FwiProcessor};
use crate::inverse::fwi::time_domain::field_ops::{
    scale_velocity_gradient, zero_masked_by_threshold,
};
use crate::inverse::fwi::time_domain::{
    accumulate_signed_correlation, l2_objective, l2_residual, reverse_time_axis,
};
use crate::inverse::reconstruction::seismic::{
    trace_weights, weighted_l2_objective, weighted_l2_residual, DataWeighting, MisfitFunction,
    MisfitType,
};
use kwavers_core::error::{KwaversError, KwaversResult, ValidationError};
use kwavers_grid::Grid;
use kwavers_source::{GridSource, SourceMode};
use leto::{Array2 as LetoArray2, Array3 as LetoArray3};
use leto::{Array2, Array3, Array4, ArrayView3, ArrayViewMut3};

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
/// Returns [`crate::KwaversError::Validation`] if any sound-speed or density entry
/// is non-finite or non-positive (avoids silent NaN production from a 1/0
/// or 1/NaN multiplication).
pub(super) fn apply_velocity_gradient_scaling(
    gradient: ArrayViewMut3<'_, f64>,
    model: ArrayView3<'_, f64>,
    density: ArrayView3<'_, f64>,
) -> KwaversResult<()> {
    if gradient.shape() != model.shape() || gradient.shape() != density.shape() {
        return Err(KwaversError::Validation(
            ValidationError::ConstraintViolation {
                message: format!(
                    "FWI gradient scaling shape mismatch: gradient {:?}, model {:?}, density {:?}",
                    gradient.shape(),
                    model.shape(),
                    density.shape()
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
    scale_velocity_gradient(gradient, model, density);
    Ok(())
}

impl FwiProcessor {
    /// Compute the acoustic L2 objective between observed and synthetic data.
    ///
    /// ## Theorem
    /// `J = (dt / 2) Σ_{r,t} w_{r,t} (d_syn - d_obs)²` is non-negative and vanishes
    /// iff the synthetic and observed traces match pointwise. For the default
    /// [`DataWeighting::Uniform`] (`w ≡ 1`) this is the classical L2 objective; for
    /// [`DataWeighting::InverseNoiseVariance`] it is the PWLS / MBIR objective
    /// (the low-dose-CT lesson), with per-trace weights derived from `observed`.
    /// # Errors
    /// - Returns [`Err`] if an internal constraint is violated.
    ///
    pub(super) fn compute_l2_objective(
        &self,
        observed: &Array2<f64>,
        synthetic: &Array2<f64>,
    ) -> KwaversResult<f64> {
        match self.data_weighting {
            DataWeighting::Uniform => l2_objective(self.parameters.dt, observed, synthetic),
            weighting => {
                let w = trace_weights(observed, weighting);
                weighted_l2_objective(self.parameters.dt, observed, synthetic, &w)
            }
        }
    }

    /// Compute the data misfit for the configured [`MisfitType`].
    ///
    /// ## Theorem
    /// For `MisfitType::L2Norm` this returns the discrete least-squares objective
    /// `J = (dt/2)‖d_syn − d_obs‖²` (the `dt` measure keeps the objective a
    /// Riemann discretisation of `½∫‖r‖² dt`, consistent with the adjoint-source
    /// scaling). For the envelope, phase, Wasserstein, correlation, and L1
    /// alternatives the value is delegated to the canonical [`MisfitFunction`]
    /// dispatcher, which is the single source of truth for every misfit and its
    /// matching adjoint source.
    ///
    /// The same functional is used for the convergence test and the Armijo line
    /// search, so the objective the driver minimises always matches the gradient
    /// produced by [`Self::compute_adjoint_source`].
    /// # Errors
    /// - Propagates any [`crate::KwaversError`] returned by the misfit evaluation.
    ///
    pub(super) fn compute_misfit_objective(
        &self,
        observed: &Array2<f64>,
        synthetic: &Array2<f64>,
    ) -> KwaversResult<f64> {
        match self.band_limit_hz {
            None => self.misfit_objective_on(observed, synthetic),
            Some(corner_hz) => {
                let obs = self.band_limit(observed, corner_hz);
                let syn = self.band_limit(synthetic, corner_hz);
                self.misfit_objective_on(&obs, &syn)
            }
        }
    }

    /// Evaluate the selected misfit on already-prepared traces (no band limit).
    fn misfit_objective_on(
        &self,
        observed: &Array2<f64>,
        synthetic: &Array2<f64>,
    ) -> KwaversResult<f64> {
        match self.misfit_type {
            MisfitType::L2Norm => self.compute_l2_objective(observed, synthetic),
            other => MisfitFunction::new(other).compute(observed, synthetic),
        }
    }

    /// Compute the discrete adjoint source `∂J/∂d_syn` for the configured misfit.
    ///
    /// For `MisfitType::L2Norm` this is the data residual `d_syn − d_obs`; for the
    /// other functionals it is the misfit-specific Fréchet derivative (e.g. the
    /// Hilbert-transform envelope/phase adjoint of Bozdağ et al. 2011 and
    /// Fichtner et al. 2008, or the CDF-difference optimal-transport adjoint of
    /// Métivier et al. 2016). The returned trace matrix is subsequently reversed
    /// in time and injected through the receiver mask by
    /// [`Self::build_adjoint_source`]; the time-reversal adjoint theorem holds for
    /// an arbitrary receiver-side forcing, so the same injection path is valid
    /// for every misfit type.
    /// # Errors
    /// - Propagates any [`crate::KwaversError`] returned by the adjoint-source evaluation.
    ///
    pub(super) fn compute_adjoint_source(
        &self,
        observed: &Array2<f64>,
        synthetic: &Array2<f64>,
    ) -> KwaversResult<Array2<f64>> {
        match self.band_limit_hz {
            None => self.adjoint_source_on(observed, synthetic),
            Some(corner_hz) => {
                // Multiscale data-domain filtering. With a zero-phase low-pass F
                // (Fᵀ = F), the misfit becomes J(F·d_syn, F·d_obs); its adjoint
                // source is Fᵀ·∂J/∂(F·d_syn) = F·g̃, i.e. apply the same filter to
                // the band-limited adjoint source. (Bunks et al. 1995, §"Multiscale".)
                let obs = self.band_limit(observed, corner_hz);
                let syn = self.band_limit(synthetic, corner_hz);
                let band_limited_adjoint = self.adjoint_source_on(&obs, &syn)?;
                Ok(self.band_limit(&band_limited_adjoint, corner_hz))
            }
        }
    }

    /// Adjoint source for the selected misfit on already-prepared traces.
    fn adjoint_source_on(
        &self,
        observed: &Array2<f64>,
        synthetic: &Array2<f64>,
    ) -> KwaversResult<Array2<f64>> {
        match self.misfit_type {
            MisfitType::L2Norm => match self.data_weighting {
                DataWeighting::Uniform => l2_residual(observed, synthetic),
                weighting => {
                    // PWLS adjoint source w ⊙ (d_syn − d_obs): the gradient of the
                    // weighted L2 objective, matching compute_l2_objective (SSOT).
                    let w = trace_weights(observed, weighting);
                    weighted_l2_residual(observed, synthetic, &w)
                }
            },
            other => MisfitFunction::new(other).compute_adjoint_source(observed, synthetic),
        }
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
    /// - Returns [`crate::KwaversError::Validation`] if the precondition for a Validation-class constraint is violated.
    ///
    pub(super) fn build_adjoint_source(
        &self,
        residual: &Array2<f64>,
        geometry: &FwiGeometry,
    ) -> KwaversResult<GridSource> {
        let expected_rows = geometry.receiver_count();
        if residual.shape()[0] != expected_rows {
            return Err(KwaversError::Validation(
                ValidationError::ConstraintViolation {
                    message: format!(
                        "Residual receiver count mismatch: expected {}, got {}",
                        expected_rows,
                        residual.shape()[0]
                    ),
                },
            ));
        }
        if residual.shape()[1] != self.parameters.nt {
            return Err(KwaversError::Validation(
                ValidationError::ConstraintViolation {
                    message: format!(
                        "Residual time length mismatch: expected {}, got {}",
                        self.parameters.nt,
                        residual.shape()[1]
                    ),
                },
            ));
        }

        let reversed_residual = reverse_time_axis(residual);
        let mut p_signal = LetoArray2::zeros([expected_rows, self.parameters.nt]);
        for source_row in 0..expected_rows {
            let sensor_row = geometry.receiver_row_to_sensor_row[source_row];
            for t in 0..self.parameters.nt {
                p_signal[[source_row, t]] = reversed_residual[[sensor_row, t]];
            }
        }

        let p_mask = {
            let [nx, ny, nz] = geometry.sensor_mask.shape();
            LetoArray3::from_shape_vec(
                [nx, ny, nz],
                geometry
                    .sensor_mask
                    .iter()
                    .map(|&active| if active { 1.0 } else { 0.0 })
                    .collect(),
            )
            .expect("adjoint source mask shape must match its Leto storage")
        };

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
    /// `FwiParameters::solver_type`), so forward and adjoint share the same
    /// stencil and boundary treatment.  This makes the adjoint a faithful
    /// *approximation* of the discrete transpose — sufficient for a correct
    /// descent direction — but NOT the exact discrete adjoint: the additive
    /// source injection (scaled by `2·dt·c₀/(N·dx)`) is not the transpose of the
    /// direct-pressure receiver sampling, and the PML/leapfrog stepping is not
    /// self-adjoint. The gradient is consequently off by a global constant
    /// (~200×) plus a ~20% direction-dependent shape error, both absorbed by the
    /// Armijo line search. Verified by
    /// `tests::gradient::test_fwi_adjoint_gradient_is_valid_descent_direction`.
    /// For the exact gradient (`κ ≈ 1`) use the self-adjoint engine
    /// (`super::FwiEngine::SecondOrderSelfAdjoint`, ADR 016) instead of this path.
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
    /// [`RHO_SEISMIC_REF`] (2000 kg/m³) is used
    /// uniformly. Because the forward and adjoint media share the same
    /// resolved density, the discrete adjoint operator is exactly the
    /// time-reverse of the forward operator in both cases.
    ///
    /// # References
    /// * Tromp et al. (2005): "Seismic tomography, adjoint methods"
    /// * Plessix (2006), GFJI 167(2), eq. (5)–(6) and (12)
    /// # Errors
    /// - Returns [`crate::KwaversError::Validation`] if the precondition for a Validation-class constraint is violated.
    /// - Propagates any [`crate::KwaversError`] returned by called functions.
    ///
    pub(super) fn adjoint_model(
        &self,
        adjoint_source: &GridSource,
        model: &Array3<f64>,
        grid: &Grid,
        forward_history: &Array4<f64>,
        source_mask: Option<&Array3<f64>>,
    ) -> KwaversResult<Array3<f64>> {
        use crate::config::SolverType;

        if forward_history.shape()[0] != self.parameters.nt {
            return Err(KwaversError::Validation(
                ValidationError::ConstraintViolation {
                    message: format!(
                        "Forward history length mismatch: expected {}, got {}",
                        self.parameters.nt,
                        forward_history.shape()[0]
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
        let mut solver: Box<dyn crate::interface::Solver> = match self.parameters.solver_type {
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
                zero_masked_by_threshold(&mut p_tt, mask, 0.5);
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
            (0.0_f64, [0usize, 0usize, 0usize]),
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
            gmax_idx[0],
            gmax_idx[1],
            gmax_idx[2]
        );

        Ok(gradient_m)
    }
}
