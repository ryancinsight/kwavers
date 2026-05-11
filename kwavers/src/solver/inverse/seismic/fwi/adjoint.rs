//! Adjoint FDTD run, adjoint-source construction, L2 residual and objective.

use super::{geometry::FwiGeometry, FwiProcessor, RHO_SEISMIC_REF};
use crate::core::error::{KwaversError, KwaversResult, ValidationError};
use crate::domain::boundary::cpml::{CPMLConfig, PerDimensionPML};
use crate::domain::grid::Grid;
use crate::domain::medium::heterogeneous::HeterogeneousFactory;
use crate::domain::source::{GridSource, SourceMode};
use crate::solver::inverse::acoustic_fwi::{
    accumulate_signed_correlation, l2_objective, l2_residual, reverse_time_axis,
};
use ndarray::{Array2, Array3, Array4, Axis, Zip};

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

    /// Adjoint modeling using time-reversed FDTD.
    ///
    /// Computes the adjoint wavefield by running the FDTD solver in reverse time
    /// with the adjoint source (data residual) as input, then accumulates
    /// `∂J/∂m = −dt · p̈(x,t) · λ(x,t)` over all timesteps.
    ///
    /// # References
    /// * Tromp et al. (2005): "Seismic tomography, adjoint methods"
    /// * Plessix (2006), GFJI 167(2), eq. (5)–(6)
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
        use crate::solver::fdtd::{FdtdConfig, FdtdSolver, KSpaceCorrectionMode};

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
        let num_steps = self.parameters.nt;

        let config = FdtdConfig {
            spatial_order: 2,
            staggered_grid: true,
            cfl_factor: 0.3,
            subgridding: false,
            subgrid_factor: 2,
            enable_gpu_acceleration: false,
            enable_nonlinear: false,
            kspace_correction: KSpaceCorrectionMode::None,
            nt: num_steps,
            dt,
            sensor_mask: None,
            geometry: Default::default(),
        };

        let density_adj = Array3::from_elem((nx, ny, nz), RHO_SEISMIC_REF);
        let medium_adj = HeterogeneousFactory::from_arrays(
            model.clone(),
            density_adj,
            None,
            None, // alpha_power: default 1.0
            None,
            self.parameters.frequency,
        )
        .map_err(crate::core::error::KwaversError::InvalidInput)?;

        let mut solver = FdtdSolver::new(config, grid, &medium_adj, adjoint_source.clone())?;

        // CPML must match the forward solver's boundary treatment (time-reversal theorem).
        let c_max_adj = model.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        let y_pml_adj = if ny > 20 { 10usize } else { 0usize };
        let cpml_adj = CPMLConfig {
            thickness: 10,
            per_dimension: PerDimensionPML::new(10, y_pml_adj, 10),
            ..CPMLConfig::default()
        };
        solver.enable_cpml(cpml_adj, dt, c_max_adj)?;

        let mut gradient_m = Array3::zeros((nx, ny, nz));
        let mut p_tt = Array3::zeros((nx, ny, nz));

        for t in 0..self.parameters.nt {
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

            accumulate_signed_correlation(
                &mut gradient_m,
                p_tt.view(),
                solver.fields.p.view(),
                -dt,
            )?;
        }

        Zip::from(&mut gradient_m).and(model).par_for_each(|g, &c| {
            *g *= -2.0 / c.powi(3);
        });

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
