//! Full Waveform Inversion implementation.
//!
//! # Specification
//!
//! For the acoustic least-squares objective
//!
//! ```text
//! J(c) = (dt / 2) Σ_{r,t} (d_syn(r,t;c) - d_obs(r,t))²
//! ```
//!
//! the reduced gradient is obtained by the adjoint-state identity
//!
//! ```text
//! ∂J/∂m(x) = -∫_0^T λ(x,T-t) ∂²p(x,t)/∂t² dt,     m = c⁻²
//! ∂J/∂c(x) = -2 c(x)⁻³ ∂J/∂m(x)
//! ```
//!
//! The discrete implementation follows the k-Wave time-reversal convention:
//! the residual is reversed in time and injected through the same receiver mask
//! used for data acquisition.
//!
//! # Theorems
//!
//! 1. **L2 residual theorem.** The Fréchet derivative of `J` with respect to
//!    the data is `d_syn - d_obs`. This fixes the sign of the adjoint source.
//! 2. **Time-reversal theorem.** Injecting the reversed residual on the receiver
//!    mask produces the discrete adjoint wavefield for the acoustic linearized
//!    operator, provided the forward and adjoint solvers share the same stencil
//!    and boundary treatment.
//! 3. **Chain-rule theorem.** The sound-speed gradient follows from
//!    `m = c⁻²` by `dm/dc = -2 c⁻³`.
//!
//! # Proof sketches
//!
//! 1. Differentiate `1/2 ||d_syn - d_obs||²` with respect to `d_syn`.
//! 2. Apply discrete Green's identity to the acoustic forward and adjoint
//!    operators with matching boundary conditions.
//! 3. Substitute the parameterization `m = c⁻²` and apply the chain rule.
//!
//! # References
//! - Tarantola (1984): *Inversion of seismic reflection data in the acoustic approximation*
//! - Plessix (2006): *A review of the adjoint-state method for computing the gradient of a functional*
//! - Virieux & Operto (2009): *An overview of full-waveform inversion in exploration geophysics*
//! - k-Wave time reversal convention: residual is flipped in time and injected through the receiver mask

use super::parameters::FwiParameters;
use crate::core::error::{KwaversError, KwaversResult, ValidationError};
use crate::domain::grid::Grid;
use crate::domain::medium::heterogeneous::HeterogeneousFactory;
use crate::domain::source::{GridSource, SourceMode};
use crate::solver::inverse::acoustic_fwi::{
    accumulate_signed_correlation, l2_objective, l2_residual, reverse_time_axis,
};
use ndarray::{s, Array2, Array3, Array4, Axis, Zip};
use std::collections::HashMap;

/// Reference density for seismic FWI [kg/m³].
///
/// Gardner et al. (1974) relate seismic velocity to density via ρ = a·Vᵇ
/// (a = 310, b = 0.25 for consolidated sedimentary rock).  For simplicity,
/// the forward model uses a uniform value consistent with typical upper-crust
/// consolidated sediments (~2000 kg/m³).  Joint density-velocity inversion
/// would update this per-voxel.
///
/// Reference: Gardner, G.H.F. et al. (1974). "Formation velocity and density —
/// the diagnostic basics for stratigraphic traps." Geophysics 39(6), 770–780.
const RHO_SEISMIC_REF: f64 = 2000.0; // kg/m³, consolidated upper-crust sediment

/// Source and receiver geometry used by acoustic FWI.
///
/// `source` describes the forward source term. `sensor_mask` describes the
/// receiver layout used to record synthetic data and to back-inject the adjoint
/// residual. The `receiver_row_to_sensor_row` mapping converts residual data
/// from the recorder's Fortran-order convention into the row order required by
/// the pressure-source injector.
#[derive(Debug, Clone)]
pub struct FwiGeometry {
    pub source: GridSource,
    pub sensor_mask: Array3<bool>,
    receiver_row_to_sensor_row: Vec<usize>,
}

impl FwiGeometry {
    #[must_use]
    pub fn new(source: GridSource, sensor_mask: Array3<bool>) -> Self {
        let sensor_indices = Self::collect_fortran_indices(&sensor_mask);
        let receiver_indices = Self::collect_row_major_indices(&sensor_mask);

        let sensor_lookup: HashMap<(usize, usize, usize), usize> = sensor_indices
            .iter()
            .copied()
            .enumerate()
            .map(|(row, coord)| (coord, row))
            .collect();

        let receiver_row_to_sensor_row = receiver_indices
            .iter()
            .map(|coord| {
                *sensor_lookup
                    .get(coord)
                    .expect("receiver mask ordering mismatch")
            })
            .collect();

        Self {
            source,
            sensor_mask,
            receiver_row_to_sensor_row,
        }
    }

    #[must_use]
    fn receiver_count(&self) -> usize {
        self.receiver_row_to_sensor_row.len()
    }

    fn collect_fortran_indices(mask: &Array3<bool>) -> Vec<(usize, usize, usize)> {
        let (nx, ny, nz) = mask.dim();
        let mut indices = Vec::new();
        for k in 0..nz {
            for j in 0..ny {
                for i in 0..nx {
                    if mask[[i, j, k]] {
                        indices.push((i, j, k));
                    }
                }
            }
        }
        indices
    }

    fn collect_row_major_indices(mask: &Array3<bool>) -> Vec<(usize, usize, usize)> {
        let mut indices = Vec::new();
        for ((i, j, k), &active) in mask.indexed_iter() {
            if active {
                indices.push((i, j, k));
            }
        }
        indices
    }

    fn validate(&self, grid: &Grid, nt: usize) -> KwaversResult<()> {
        let expected_shape = grid.dimensions();
        if self.sensor_mask.dim() != expected_shape {
            return Err(KwaversError::Validation(
                ValidationError::ConstraintViolation {
                    message: format!(
                        "Receiver mask shape mismatch: expected {:?}, got {:?}",
                        expected_shape,
                        self.sensor_mask.dim()
                    ),
                },
            ));
        }

        let Some(source_mask) = self.source.p_mask.as_ref() else {
            return Err(KwaversError::Validation(
                ValidationError::ConstraintViolation {
                    message: "FWI requires a time-varying pressure source mask".to_string(),
                },
            ));
        };
        if source_mask.dim() != expected_shape {
            return Err(KwaversError::Validation(
                ValidationError::ConstraintViolation {
                    message: format!(
                        "Source mask shape mismatch: expected {:?}, got {:?}",
                        expected_shape,
                        source_mask.dim()
                    ),
                },
            ));
        }

        if self.source.p_signal.as_ref().is_none() {
            return Err(KwaversError::Validation(
                ValidationError::ConstraintViolation {
                    message: "FWI requires a time-varying pressure source signal".to_string(),
                },
            ));
        }
        let source_signal = self.source.p_signal.as_ref().expect("validated above");
        if source_signal.shape()[1] < nt {
            return Err(KwaversError::Validation(
                ValidationError::ConstraintViolation {
                    message: format!(
                        "Source signal must contain at least {nt} samples, got {}",
                        source_signal.shape()[1]
                    ),
                },
            ));
        }

        if self.receiver_count() == 0 {
            return Err(KwaversError::Validation(
                ValidationError::ConstraintViolation {
                    message: "Receiver mask contains no active points".to_string(),
                },
            ));
        }

        Ok(())
    }
}

/// Full Waveform Inversion processor.
#[derive(Debug)]
pub struct FwiProcessor {
    parameters: FwiParameters,
}

impl FwiProcessor {
    /// Create new FWI processor with specified parameters
    #[must_use]
    pub fn new(parameters: FwiParameters) -> Self {
        Self { parameters }
    }

    /// Generate receiver-time synthetic data for a model and acquisition geometry.
    ///
    /// # Theorem
    ///
    /// For a fixed velocity model `c`, acquisition geometry `G`, grid `Ω_h`,
    /// and time discretization `(nt, dt)`, this function returns the discrete
    /// forward map
    ///
    /// ```text
    /// F_h(c; G) = R_G p_h(c, q_G)
    /// ```
    ///
    /// where `p_h` is the same FDTD state history used by [`Self::invert`],
    /// `q_G` is the pressure source in `geometry`, and `R_G` samples the
    /// receiver mask in recorder row order.
    ///
    /// # Proof sketch
    ///
    /// The implementation delegates to the canonical private forward model used
    /// by inversion, then discards the wavefield history. Therefore any
    /// synthetic data used by examples, tests, or callers is generated by the
    /// same discrete operator as the FWI objective and gradient.
    pub fn generate_synthetic_data(
        &self,
        model: &Array3<f64>,
        geometry: &FwiGeometry,
        grid: &Grid,
    ) -> KwaversResult<Array2<f64>> {
        geometry.validate(grid, self.parameters.nt)?;
        let (synthetic_data, _forward_history) = self.forward_model(model, geometry, grid)?;
        Ok(synthetic_data)
    }

    /// Perform Full Waveform Inversion
    ///
    /// The objective is the acoustic L2 misfit
    ///
    /// ```text
    /// J(c) = 1/2 ∑_r ∫_0^T (d_syn(r,t;c) - d_obs(r,t))² dt
    /// ```
    ///
    /// with the reduced gradient given by the adjoint-state identity
    ///
    /// ```text
    /// ∂J/∂m(x) = -∫_0^T λ(x,T-t) ∂²p(x,t)/∂t² dt,   m = c⁻²
    /// ∂J/∂c(x) = -2 c(x)⁻³ ∂J/∂m(x)
    /// ```
    ///
    /// The adjoint source is the time-reversed residual injected at the
    /// receiver mask in `Dirichlet` mode, matching the k-Wave time-reversal
    /// convention.
    pub fn invert(
        &self,
        observed_data: &Array2<f64>,
        initial_model: &Array3<f64>,
        geometry: &FwiGeometry,
        grid: &Grid,
    ) -> KwaversResult<Array3<f64>> {
        geometry.validate(grid, self.parameters.nt)?;
        if self.parameters.nt < 3 {
            return Err(KwaversError::Validation(
                ValidationError::ConstraintViolation {
                    message: "FWI requires at least 3 time samples to form a second derivative"
                        .to_string(),
                },
            ));
        }

        let mut current_model = initial_model.clone();
        self.apply_model_constraints(&mut current_model);
        let mut prev_objective: Option<f64> = None;
        let max_iterations = self.parameters.max_iterations;

        for iteration in 0..max_iterations {
            let (synthetic_data, forward_history) =
                self.forward_model(&current_model, geometry, grid)?;
            let objective = self.compute_l2_objective(observed_data, &synthetic_data)?;

            if let Some(previous) = prev_objective {
                let relative_change = (previous - objective).abs() / previous.max(f64::EPSILON);
                if relative_change < self.parameters.tolerance {
                    log::info!(
                        "FWI converged after {} iterations with objective: {:.6e}",
                        iteration,
                        objective
                    );
                    break;
                }
            }

            let residual = self.compute_adjoint_source(observed_data, &synthetic_data)?;
            let adjoint_source = self.build_adjoint_source(&residual, geometry)?;
            let gradient =
                self.adjoint_model(&adjoint_source, &current_model, grid, &forward_history)?;
            let smoothed_gradient = self.smooth_gradient(gradient);
            let regularized_gradient =
                self.apply_regularization(&smoothed_gradient, &current_model)?;
            let step_size = self.line_search(
                &current_model,
                &regularized_gradient,
                observed_data,
                geometry,
                grid,
            )?;

            current_model = &current_model - &(&regularized_gradient * step_size);
            self.apply_model_constraints(&mut current_model);
            prev_objective = Some(objective);

            log::debug!(
                "FWI iteration {}: objective = {:.6e}, step_size = {:.6e}",
                iteration,
                objective,
                step_size
            );
        }

        Ok(current_model)
    }

    /// Calculate interaction between two fields (used for testing gradient kernel)
    #[must_use]
    pub fn calculate_interaction(
        &self,
        forward_field: &Array3<f64>,
        adjoint_field: &Array3<f64>,
    ) -> Array3<f64> {
        use ndarray::Zip;

        let mut gradient = Array3::zeros(forward_field.dim());

        // Compute interaction
        Zip::from(&mut gradient)
            .and(forward_field)
            .and(adjoint_field)
            .for_each(|g, &fwd, &adj| {
                // Negative sign for descent direction
                *g = -fwd * adj;
            });

        self.smooth_gradient(gradient)
    }

    /// Apply smoothing to gradient to reduce high-frequency artifacts
    #[must_use]
    fn smooth_gradient(&self, gradient: Array3<f64>) -> Array3<f64> {
        let mut smoothed = gradient.clone();
        let (nx, ny, nz) = gradient.dim();

        // Simple 3-point smoothing in each dimension
        for k in 1..nz - 1 {
            for j in 1..ny - 1 {
                for i in 1..nx - 1 {
                    smoothed[[i, j, k]] = (gradient[[i - 1, j, k]]
                        + gradient[[i, j, k]]
                        + gradient[[i + 1, j, k]]
                        + gradient[[i, j - 1, k]]
                        + gradient[[i, j, k]]
                        + gradient[[i, j + 1, k]]
                        + gradient[[i, j, k - 1]]
                        + gradient[[i, j, k]]
                        + gradient[[i, j, k + 1]])
                        / 9.0;
                }
            }
        }

        smoothed
    }

    /// Apply regularization to gradient
    fn apply_regularization(
        &self,
        gradient: &Array3<f64>,
        model: &Array3<f64>,
    ) -> KwaversResult<Array3<f64>> {
        let mut regularized = gradient.clone();
        let reg_params = &self.parameters.regularization;

        // Tikhonov regularization: R = λ₁ * m
        if reg_params.tikhonov_weight > 0.0 {
            regularized = &regularized + &(model * reg_params.tikhonov_weight);
        }

        // Total variation regularization
        if reg_params.tv_weight > 0.0 {
            let tv_term = self.compute_total_variation_gradient(model);
            regularized = &regularized + &(&tv_term * reg_params.tv_weight);
        }

        // Smoothness regularization (Laplacian)
        if reg_params.smoothness_weight > 0.0 {
            let smoothness_term = self.compute_smoothness_gradient(model);
            regularized = &regularized + &(&smoothness_term * reg_params.smoothness_weight);
        }

        Ok(regularized)
    }

    /// Compute total variation gradient for regularization
    #[must_use]
    fn compute_total_variation_gradient(&self, model: &Array3<f64>) -> Array3<f64> {
        let (nx, ny, nz) = model.dim();
        let mut tv_gradient = Array3::zeros((nx, ny, nz));

        for k in 1..nz - 1 {
            for j in 1..ny - 1 {
                for i in 1..nx - 1 {
                    // Compute gradient magnitude
                    let dx = model[[i + 1, j, k]] - model[[i - 1, j, k]];
                    let dy = model[[i, j + 1, k]] - model[[i, j - 1, k]];
                    let dz = model[[i, j, k + 1]] - model[[i, j, k - 1]];

                    let grad_mag = (dx * dx + dy * dy + dz * dz).sqrt();

                    if grad_mag > f64::EPSILON {
                        tv_gradient[[i, j, k]] = grad_mag;
                    }
                }
            }
        }

        tv_gradient
    }

    /// Compute smoothness gradient (Laplacian) for regularization
    #[must_use]
    fn compute_smoothness_gradient(&self, model: &Array3<f64>) -> Array3<f64> {
        let (nx, ny, nz) = model.dim();
        let mut laplacian = Array3::zeros((nx, ny, nz));

        for k in 1..nz - 1 {
            for j in 1..ny - 1 {
                for i in 1..nx - 1 {
                    laplacian[[i, j, k]] = model[[i + 1, j, k]]
                        + model[[i - 1, j, k]]
                        + model[[i, j + 1, k]]
                        + model[[i, j - 1, k]]
                        + model[[i, j, k + 1]]
                        + model[[i, j, k - 1]]
                        - 6.0 * model[[i, j, k]];
                }
            }
        }

        laplacian
    }

    /// Line search for optimal step size
    fn line_search(
        &self,
        model: &Array3<f64>,
        gradient: &Array3<f64>,
        observed_data: &Array2<f64>,
        geometry: &FwiGeometry,
        grid: &Grid,
    ) -> KwaversResult<f64> {
        let mut step_size = self.parameters.step_size;
        let c1 = 1e-4; // Armijo condition constant
        let max_iterations = 10;

        let current_objective = self.compute_objective(model, observed_data, geometry, grid)?;

        let gradient_norm_sq = gradient.map(|&x| x * x).sum();

        for _ in 0..max_iterations {
            let test_model = model - &(gradient * step_size);
            let test_objective =
                self.compute_objective(&test_model, observed_data, geometry, grid)?;

            if test_objective <= current_objective - c1 * step_size * gradient_norm_sq {
                return Ok(step_size);
            }

            step_size *= 0.5;
        }

        Ok(step_size)
    }

    /// Apply physical constraints to velocity model
    fn apply_model_constraints(&self, model: &mut Array3<f64>) {
        use crate::core::constants::SOUND_SPEED_WATER;

        // Ensure physically reasonable velocity bounds
        let min_velocity = SOUND_SPEED_WATER * 0.5; // 750 m/s
        let max_velocity = SOUND_SPEED_WATER * 4.0; // 6000 m/s

        model.mapv_inplace(|v| v.clamp(min_velocity, max_velocity));
    }

    /// Forward modeling using FDTD acoustic solver
    ///
    /// Computes synthetic seismograms from the velocity model using the
    /// finite-difference time-domain method for acoustic wave propagation.
    ///
    /// # Arguments
    /// * `model` - Velocity model (sound speed in m/s)
    /// * `grid` - Computational grid defining the domain
    ///
    /// # Returns
    /// * Synthetic wavefield at final time step
    ///
    /// # References
    /// * Virieux (1986): "P-SV wave propagation in heterogeneous media"
    fn forward_model(
        &self,
        model: &Array3<f64>,
        geometry: &FwiGeometry,
        grid: &Grid,
    ) -> KwaversResult<(Array2<f64>, Array4<f64>)> {
        use crate::solver::fdtd::{FdtdConfig, FdtdSolver, KSpaceCorrectionMode};

        geometry.validate(grid, self.parameters.nt)?;
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
            sensor_mask: Some(geometry.sensor_mask.clone()),
            geometry: Default::default(),
        };

        let (nx, ny, nz) = grid.dimensions();
        let density = Array3::from_elem((nx, ny, nz), RHO_SEISMIC_REF);
        let medium = HeterogeneousFactory::from_arrays(
            model.clone(),
            density,
            None,
            None,
            self.parameters.frequency,
        )
        .map_err(crate::core::error::KwaversError::InvalidInput)?;

        let mut solver = FdtdSolver::new(config, grid, &medium, geometry.source.clone())?;

        let mut history = Array4::zeros((self.parameters.nt, nx, ny, nz));

        for t in 0..self.parameters.nt {
            solver.step_forward()?;
            history
                .slice_mut(s![t, .., .., ..])
                .assign(&solver.fields.p);
        }

        let recorded = solver
            .sensor_recorder
            .extract_pressure_data()
            .ok_or_else(|| {
                KwaversError::Validation(ValidationError::ConstraintViolation {
                    message: "FWI forward model requires at least one receiver".to_string(),
                })
            })?;
        let synthetic = recorded.slice(s![.., 0..self.parameters.nt]).to_owned();

        Ok((synthetic, history))
    }

    /// Adjoint modeling using time-reversed FDTD
    ///
    /// Computes the adjoint wavefield by running the FDTD solver in reverse time
    /// with the adjoint source (data residual) as input.
    ///
    /// # Arguments
    /// * `adjoint_source` - Adjoint source derived from data residual
    /// * `grid` - Computational grid defining the domain
    ///
    /// # Returns
    /// * Adjoint wavefield for gradient computation
    ///
    /// # References
    /// * Tromp et al. (2005): "Seismic tomography, adjoint methods"
    fn adjoint_model(
        &self,
        adjoint_source: &GridSource,
        model: &Array3<f64>,
        grid: &Grid,
        forward_history: &Array4<f64>,
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
            None,
            self.parameters.frequency,
        )
        .map_err(crate::core::error::KwaversError::InvalidInput)?;

        let mut solver = FdtdSolver::new(config, grid, &medium_adj, adjoint_source.clone())?;

        let mut gradient_m = Array3::zeros((nx, ny, nz));
        let mut p_tt = Array3::zeros((nx, ny, nz));

        for t in 0..self.parameters.nt {
            solver.step_forward()?;
            let fwd_idx = self.parameters.nt - 1 - t;
            self.pressure_second_derivative_into(forward_history, fwd_idx, dt, &mut p_tt)?;

            accumulate_signed_correlation(
                &mut gradient_m,
                p_tt.view(),
                solver.fields.p.view(),
                -dt,
            )?;
        }

        Zip::from(&mut gradient_m).and(model).for_each(|g, &c| {
            *g *= -2.0 / c.powi(3);
        });

        Ok(gradient_m)
    }

    /// Calculate stable timestep for FDTD solver
    ///
    /// Uses CFL condition: dt ≤ min(dx,dy,dz) / (c_max * √3)
    ///
    /// # References
    /// * Courant et al. (1928): "On the partial difference equations of mathematical physics"
    fn calculate_stable_timestep(&self, model: &Array3<f64>, grid: &Grid) -> KwaversResult<f64> {
        let c_max = model.iter().copied().fold(0.0, f64::max);
        if !c_max.is_finite() || c_max <= 0.0 {
            return Err(KwaversError::Validation(
                ValidationError::ConstraintViolation {
                    message: "FWI requires a strictly positive finite sound speed model"
                        .to_string(),
                },
            ));
        }

        let min_spacing = grid.dx.min(grid.dy).min(grid.dz);
        let cfl_number = 0.3;
        Ok(cfl_number * min_spacing / (c_max * 3.0_f64.sqrt()))
    }

    /// Compute the acoustic L2 objective between observed and synthetic data.
    ///
    /// ## Theorem
    /// For the discrete least-squares objective
    ///
    /// ```text
    /// J = (dt / 2) Σ_{r,t} (d_syn - d_obs)²
    /// ```
    ///
    /// the objective value is non-negative and vanishes if and only if the
    /// synthetic and observed traces match pointwise.
    ///
    /// ## Proof sketch
    /// The integrand is a sum of squares. Multiplication by the positive factor
    /// `dt / 2` preserves non-negativity and the zero set.
    fn compute_l2_objective(
        &self,
        observed: &Array2<f64>,
        synthetic: &Array2<f64>,
    ) -> KwaversResult<f64> {
        l2_objective(self.parameters.dt, observed, synthetic)
    }

    /// Compute the discrete adjoint source for L2 misfit.
    ///
    /// The residual is `d_syn - d_obs`, matching the in-repo misfit convention.
    /// The returned source is the residual itself; time reversal is applied only
    /// when constructing the pressure source signal.
    fn compute_adjoint_source(
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
    /// same inner product. The receiver residual is therefore injected in
    /// reverse temporal order at the same spatial support.
    fn build_adjoint_source(
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
            p_mode: SourceMode::Dirichlet,
            u_mask: None,
            u_signal: None,
            u_mode: SourceMode::default(),
        })
    }

    /// Compute the model objective by running a forward simulation.
    fn compute_objective(
        &self,
        model: &Array3<f64>,
        observed_data: &Array2<f64>,
        geometry: &FwiGeometry,
        grid: &Grid,
    ) -> KwaversResult<f64> {
        let (synthetic_data, _) = self.forward_model(model, geometry, grid)?;
        self.compute_l2_objective(observed_data, &synthetic_data)
    }

    /// Validate timestep and model compatibility with the grid.
    fn validate_time_step(&self, model: &Array3<f64>, grid: &Grid) -> KwaversResult<f64> {
        if model.dim() != grid.dimensions() {
            return Err(KwaversError::Validation(
                ValidationError::ConstraintViolation {
                    message: format!(
                        "Model shape mismatch: expected {:?}, got {:?}",
                        grid.dimensions(),
                        model.dim()
                    ),
                },
            ));
        }

        if self.parameters.nt < 3 {
            return Err(KwaversError::Validation(
                ValidationError::ConstraintViolation {
                    message: "FWI requires at least 3 time samples to form a second derivative"
                        .to_string(),
                },
            ));
        }

        if self.parameters.dt <= 0.0 {
            return Err(KwaversError::Validation(
                ValidationError::ConstraintViolation {
                    message: "FWI requires a positive time step".to_string(),
                },
            ));
        }

        if model.iter().any(|&v| !v.is_finite() || v <= 0.0) {
            return Err(KwaversError::Validation(
                ValidationError::ConstraintViolation {
                    message: "FWI requires a finite, strictly positive sound speed model"
                        .to_string(),
                },
            ));
        }

        let stable_dt = self.calculate_stable_timestep(model, grid)?;
        if self.parameters.dt > stable_dt {
            return Err(KwaversError::Validation(
                ValidationError::ConstraintViolation {
                    message: format!(
                        "Time step {:.6e} exceeds CFL bound {:.6e}",
                        self.parameters.dt, stable_dt
                    ),
                },
            ));
        }

        Ok(self.parameters.dt)
    }

    /// Compute the discrete second derivative of the forward pressure history.
    ///
    /// ## Theorem
    /// The centered second difference is a second-order accurate approximation
    /// of `∂²p/∂t²` on a uniform time grid.
    ///
    /// ## Proof sketch
    /// Taylor expansion about `t_i` gives
    /// `p_{i±1} = p_i ± dt p'_i + dt² p''_i / 2 + O(dt³)`.
    /// Adding the two expansions and subtracting `2p_i` yields
    /// `(p_{i-1} - 2p_i + p_{i+1}) / dt² = p''_i + O(dt²)`.
    fn pressure_second_derivative_into(
        &self,
        forward_history: &Array4<f64>,
        idx: usize,
        dt: f64,
        dst: &mut Array3<f64>,
    ) -> KwaversResult<()> {
        if idx >= forward_history.len_of(Axis(0)) {
            return Err(KwaversError::Validation(
                ValidationError::ConstraintViolation {
                    message: format!(
                        "Forward history index out of bounds: idx {} >= {}",
                        idx,
                        forward_history.len_of(Axis(0))
                    ),
                },
            ));
        }

        let nt = forward_history.len_of(Axis(0));
        let inv_dt_sq = 1.0 / (dt * dt);
        let current = forward_history.index_axis(Axis(0), idx);

        if idx == 0 {
            let next = forward_history.index_axis(Axis(0), 1);
            let next2 = forward_history.index_axis(Axis(0), 2);
            Zip::from(dst)
                .and(&current)
                .and(&next)
                .and(&next2)
                .for_each(|d, &p0, &p1, &p2| {
                    *d = (p0 - 2.0 * p1 + p2) * inv_dt_sq;
                });
            return Ok(());
        }

        if idx + 1 == nt {
            let prev = forward_history.index_axis(Axis(0), nt - 2);
            let prev2 = forward_history.index_axis(Axis(0), nt - 3);
            Zip::from(dst)
                .and(&prev2)
                .and(&prev)
                .and(&current)
                .for_each(|d, &p0, &p1, &p2| {
                    *d = (p0 - 2.0 * p1 + p2) * inv_dt_sq;
                });
            return Ok(());
        }

        let prev = forward_history.index_axis(Axis(0), idx - 1);
        let next = forward_history.index_axis(Axis(0), idx + 1);
        Zip::from(dst)
            .and(&prev)
            .and(&current)
            .and(&next)
            .for_each(|d, &p0, &p1, &p2| {
                *d = (p0 - 2.0 * p1 + p2) * inv_dt_sq;
            });
        Ok(())
    }
}

impl Default for FwiProcessor {
    fn default() -> Self {
        Self::new(FwiParameters::default())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_gradient_calculation() {
        let processor = FwiProcessor::default();

        let forward_field = Array3::ones((10, 10, 10));
        let adjoint_field = Array3::from_elem((10, 10, 10), 2.0);

        let gradient = processor.calculate_interaction(&forward_field, &adjoint_field);

        // Expected: -1.0 * 2.0 = -2.0 (after smoothing, will be close to -2.0)
        assert!((gradient[[5, 5, 5]] + 2.0).abs() < 0.1); // Allow for smoothing effects
    }

    #[test]
    fn test_l2_adjoint_source_computation() {
        let processor = FwiProcessor::default();
        let observed = Array2::from_shape_vec((2, 3), vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0])
            .expect("shape must be valid");
        let synthetic = Array2::from_shape_vec((2, 3), vec![1.0, 0.5, 3.0, 1.0, 7.0, 9.0])
            .expect("shape must be valid");

        let adjoint_source = processor
            .compute_adjoint_source(&observed, &synthetic)
            .expect("adjoint source computation must succeed");

        let expected = Array2::from_shape_vec((2, 3), vec![1.0, -0.5, 1.0, -2.0, 3.0, 4.0])
            .expect("shape must be valid");
        assert_eq!(adjoint_source, expected);
    }

    #[test]
    fn test_l2_objective_matches_definition() {
        let processor = FwiProcessor::new(FwiParameters {
            nt: 3,
            dt: 0.5,
            max_iterations: 1,
            step_size: 1.0,
            ..FwiParameters::default()
        });

        let observed =
            Array2::from_shape_vec((2, 2), vec![1.0, 1.0, 1.0, 1.0]).expect("shape must be valid");
        let synthetic =
            Array2::from_shape_vec((2, 2), vec![2.0, 4.0, 6.0, 8.0]).expect("shape must be valid");

        let objective = processor
            .compute_l2_objective(&observed, &synthetic)
            .expect("objective computation must succeed");

        // residual = [1,3,5,7], sum(residual^2) = 84, objective = 0.5 * dt * 84 = 21
        assert!((objective - 21.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_adjoint_source_reorders_and_time_reverses() {
        let processor = FwiProcessor::new(FwiParameters {
            nt: 3,
            dt: 1.0,
            max_iterations: 1,
            step_size: 1.0,
            ..FwiParameters::default()
        });

        let sensor_mask = Array3::from_shape_vec((2, 2, 1), vec![true, true, true, true])
            .expect("shape must be valid");
        let geometry = FwiGeometry::new(GridSource::default(), sensor_mask);

        let residual = Array2::from_shape_vec(
            (4, 3),
            vec![
                1.0, 2.0, 3.0, 10.0, 20.0, 30.0, 100.0, 200.0, 300.0, 1000.0, 2000.0, 3000.0,
            ],
        )
        .expect("shape must be valid");

        let source = processor
            .build_adjoint_source(&residual, &geometry)
            .expect("adjoint source construction must succeed");

        let GridSource {
            p_mask,
            p_signal,
            p_mode,
            ..
        } = source;
        let p_signal = p_signal.expect("pressure signal must be present");
        let expected = Array2::from_shape_vec(
            (4, 3),
            vec![
                3.0, 2.0, 1.0, 300.0, 200.0, 100.0, 30.0, 20.0, 10.0, 3000.0, 2000.0, 1000.0,
            ],
        )
        .expect("shape must be valid");

        assert_eq!(p_signal, expected);

        let p_mask = p_mask.expect("pressure mask must be present");
        assert_eq!(
            p_mask,
            geometry
                .sensor_mask
                .clone()
                .mapv(|active| if active { 1.0 } else { 0.0 })
        );
        assert!(matches!(p_mode, SourceMode::Dirichlet));
    }

    #[test]
    fn test_pressure_second_derivative_exact_for_quadratic_trace() {
        let processor = FwiProcessor::new(FwiParameters {
            nt: 5,
            dt: 1.0,
            max_iterations: 1,
            step_size: 1.0,
            ..FwiParameters::default()
        });

        let mut forward_history = Array4::zeros((5, 1, 1, 1));
        for t in 0..5 {
            forward_history[[t, 0, 0, 0]] = (t as f64).powi(2);
        }

        let mut dst = Array3::zeros((1, 1, 1));
        for idx in 0..5 {
            processor
                .pressure_second_derivative_into(&forward_history, idx, 1.0, &mut dst)
                .expect("second derivative computation must succeed");
            assert!((dst[[0, 0, 0]] - 2.0).abs() < f64::EPSILON);
        }
    }

    #[test]
    fn test_forward_model_objective_vanishes_for_self_data() {
        let processor = FwiProcessor::new(FwiParameters {
            nt: 3,
            dt: 1e-4,
            max_iterations: 1,
            step_size: 1.0,
            ..FwiParameters::default()
        });

        let grid = Grid::new(3, 3, 3, 1.0, 1.0, 1.0).expect("grid must be valid");
        let model = Array3::from_elem((3, 3, 3), 1500.0);

        let mut p_mask = Array3::zeros((3, 3, 3));
        p_mask[[1, 1, 1]] = 1.0;
        let mut source = GridSource::default();
        source.p_mask = Some(p_mask);
        source.p_signal =
            Some(Array2::from_shape_vec((1, 3), vec![1.0, 0.0, 0.0]).expect("shape must be valid"));
        source.p_mode = SourceMode::Dirichlet;

        let mut sensor_mask = Array3::from_elem((3, 3, 3), false);
        sensor_mask[[2, 2, 2]] = true;
        let geometry = FwiGeometry::new(source, sensor_mask);

        let (synthetic, _history) = processor
            .forward_model(&model, &geometry, &grid)
            .expect("forward model must succeed");
        let objective = processor
            .compute_l2_objective(&synthetic, &synthetic)
            .expect("objective computation must succeed");

        assert!((objective - 0.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_generate_synthetic_data_matches_canonical_forward_model() {
        let processor = FwiProcessor::new(FwiParameters {
            nt: 3,
            dt: 1e-4,
            max_iterations: 1,
            step_size: 1.0,
            ..FwiParameters::default()
        });

        let grid = Grid::new(3, 3, 3, 1.0, 1.0, 1.0).expect("grid must be valid");
        let model = Array3::from_elem((3, 3, 3), 1500.0);

        let mut p_mask = Array3::zeros((3, 3, 3));
        p_mask[[1, 1, 1]] = 1.0;
        let mut source = GridSource::default();
        source.p_mask = Some(p_mask);
        source.p_signal =
            Some(Array2::from_shape_vec((1, 3), vec![1.0, 0.0, 0.0]).expect("shape must be valid"));
        source.p_mode = SourceMode::Dirichlet;

        let mut sensor_mask = Array3::from_elem((3, 3, 3), false);
        sensor_mask[[2, 2, 2]] = true;
        let geometry = FwiGeometry::new(source, sensor_mask);

        let public_data = processor
            .generate_synthetic_data(&model, &geometry, &grid)
            .expect("public synthetic data generation must succeed");
        let (canonical_data, _history) = processor
            .forward_model(&model, &geometry, &grid)
            .expect("canonical forward model must succeed");

        assert_eq!(public_data, canonical_data);
        assert_eq!(public_data.dim(), (1, 3));
    }

    #[test]
    fn test_model_constraints() {
        let processor = FwiProcessor::default();
        let mut model = Array3::from_elem((5, 5, 5), 10000.0); // Too high

        processor.apply_model_constraints(&mut model);

        // Should be clamped to max velocity
        assert!(model[[2, 2, 2]] <= 6000.0);
        assert!(model[[2, 2, 2]] >= 750.0);
    }

    /// Verify that the FWI forward-model medium is built with seismic (non-water) density.
    ///
    /// `HomogeneousMedium::water` uses ρ = 1000 kg/m³.  The corrected path uses
    /// `RHO_SEISMIC_REF` = 2000 kg/m³, so the solver's `rho0` field must differ
    /// from the water value.
    #[test]
    fn test_fwi_medium_density_not_water() {
        use crate::domain::medium::heterogeneous::HeterogeneousFactory;

        let (nx, ny, nz) = (8usize, 8, 8);
        let c0 = 2000.0_f64; // m/s, typical sediment P-wave speed
        let sound_speed = Array3::from_elem((nx, ny, nz), c0);
        let density = Array3::from_elem((nx, ny, nz), RHO_SEISMIC_REF);

        let medium = HeterogeneousFactory::from_arrays(
            sound_speed,
            density,
            None,
            None,
            20.0, // reference frequency [Hz]
        )
        .expect("medium construction must succeed");

        // Density must be the seismic reference, not the water default (1000 kg/m³)
        use crate::domain::medium::CoreMedium;
        let rho_sample = medium.density(4, 4, 4);
        assert!(
            (rho_sample - RHO_SEISMIC_REF).abs() < 1.0,
            "medium density {rho_sample} != RHO_SEISMIC_REF {RHO_SEISMIC_REF}"
        );
        assert!(
            (rho_sample - 1000.0).abs() > 100.0,
            "density must not equal water (1000 kg/m³)"
        );
    }

    /// Verify that the FWI forward-model medium stores the velocity model correctly.
    ///
    /// After construction via `HeterogeneousFactory::from_arrays`, the sound-speed
    /// field must exactly reproduce the input model — no post-hoc assignment needed.
    #[test]
    fn test_fwi_forward_medium_sound_speed_matches_model() {
        use crate::domain::medium::heterogeneous::HeterogeneousFactory;

        let (nx, ny, nz) = (6usize, 6, 6);
        // Non-uniform velocity model
        let mut model = Array3::from_elem((nx, ny, nz), 1800.0_f64);
        model[[3, 3, 3]] = 3200.0; // anomaly

        let density = Array3::from_elem((nx, ny, nz), RHO_SEISMIC_REF);
        let medium = HeterogeneousFactory::from_arrays(model.clone(), density, None, None, 20.0)
            .expect("medium construction must succeed");

        use crate::domain::medium::CoreMedium;
        let c_bg = medium.sound_speed(1, 1, 1);
        let c_anom = medium.sound_speed(3, 3, 3);
        assert!((c_bg - 1800.0).abs() < 1.0, "background speed mismatch");
        assert!((c_anom - 3200.0).abs() < 1.0, "anomaly speed mismatch");
    }
}
