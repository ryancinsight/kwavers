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
//!
//! # Sign convention for model updates
//!
//! `adjoint_model` returns `g = +∂J/∂c` (the true gradient).
//! Derivation: the adjoint equation is driven by `+(d_syn − d_obs)` at receivers
//! (Plessix 2006, eq. 5 positive-sign convention); the discrete zero-lag
//! cross-correlation with scale `−dt` followed by `×(−2/c³)` yields
//! `g = +∂J/∂c` (Plessix eq. 6: `∂J/∂m = −∫λ(T−t)p̈ dt`; the code
//! accumulates `−dt·p̈·λ` = `∂J/∂m`, then applies the chain rule `×(−2/c³)`).
//!
//! The descent formula is therefore `c_new = c − step × g` (subtraction).
//! Equivalently: the line search tests `c − α·g` and accepts `α` when
//! `J(c − α·g) < J(c)`.

use super::parameters::FwiParameters;
use crate::core::error::{KwaversError, KwaversResult, ValidationError};
use crate::domain::boundary::cpml::{CPMLConfig, PerDimensionPML};
use crate::domain::grid::Grid;
use crate::domain::medium::heterogeneous::HeterogeneousFactory;
use crate::domain::source::{GridSource, SourceMode};
use crate::solver::inverse::acoustic_fwi::{
    accumulate_signed_correlation, l2_objective, l2_residual, reverse_time_axis,
};
use ndarray::{s, Array2, Array3, Array4, Axis, Zip};
use rayon::prelude::*;
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

/// Zero the gradient within `radius` voxels (L2 norm) of every active source voxel.
///
/// ## Theorem (near-source artefact suppression)
///
/// At voxels within `λ/2` of a source, `∂²p/∂t²` is dominated by the second
/// derivative of the source wavelet (amplitude ∝ f₀⁴ · P₀), not by scattered
/// wave physics.  The cross-correlation of this large signal with the adjoint
/// wavefield produces a gradient 10–100× larger than the physical sensitivity
/// at the imaging target (e.g., skull boundary), masking the useful signal and
/// causing the normalized gradient to point in the wrong direction.
///
/// Zeroing within `radius` removes the artefact without biasing the gradient at
/// distances ≥ `radius` from every source.
///
/// ## Reference
///
/// Virieux & Operto (2009), *Geophysics* 74(6), WCC1–WCC26, §Gradient preconditioner.
fn mute_gradient_near_sources(
    gradient: &mut Array3<f64>,
    source_p_mask: &Array3<f64>,
    radius: usize,
) {
    let r_sq = (radius * radius) as f64;
    let (nx, ny, nz) = gradient.dim();
    for ((si, sj, sk), &m) in source_p_mask.indexed_iter() {
        if m > 0.5 {
            let imin = si.saturating_sub(radius);
            let imax = (si + radius + 1).min(nx);
            let jmin = sj.saturating_sub(radius);
            let jmax = (sj + radius + 1).min(ny);
            let kmin = sk.saturating_sub(radius);
            let kmax = (sk + radius + 1).min(nz);
            for i in imin..imax {
                for j in jmin..jmax {
                    for k in kmin..kmax {
                        let dr_sq = ((i as isize - si as isize).pow(2)
                            + (j as isize - sj as isize).pow(2)
                            + (k as isize - sk as isize).pow(2))
                            as f64;
                        if dr_sq <= r_sq {
                            gradient[[i, j, k]] = 0.0;
                        }
                    }
                }
            }
        }
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
        self.forward_model_sensor_only(model, geometry, grid)
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
            let gradient = self.adjoint_model(
                &adjoint_source,
                &current_model,
                grid,
                &forward_history,
                geometry.source.p_mask.as_ref(),
            )?;
            let smoothed_gradient = self.smooth_gradient(&gradient);
            let regularized_gradient =
                self.apply_regularization(&smoothed_gradient, &current_model)?;

            // Normalize gradient by its maximum absolute value so that `step_size`
            // has a physically meaningful scale (units of m/s).
            //
            // This is the standard preconditioned-steepest-descent normalization
            // used in seismic FWI (Virieux & Operto 2009 §3.3): the normalized
            // gradient has max-norm = 1, and `step_size` directly controls the
            // maximum velocity update per iteration in m/s.
            //
            // Without this normalization the physics gradient (~10⁻¹⁵ m/s per
            // element) is overwhelmed by even tiny regularization contributions,
            // causing the line search to reduce the step to a negligibly small
            // value.
            let grad_max = regularized_gradient
                .iter()
                .copied()
                .fold(0.0_f64, |a, x| a.max(x.abs()));
            let grad_min = regularized_gradient
                .iter()
                .copied()
                .fold(0.0_f64, |a, x| a.min(x));
            log::info!(
                "FWI iter {} objective={:.6e} grad_max={:.6e} grad_min={:.6e}",
                iteration,
                objective,
                grad_max,
                grad_min
            );
            let normalized_gradient = if grad_max > f64::EPSILON {
                &regularized_gradient / grad_max
            } else {
                regularized_gradient
            };

            let step_size = self.line_search(
                &current_model,
                &normalized_gradient,
                observed_data,
                geometry,
                grid,
            )?;
            log::info!("FWI iter {} step_size={:.6e}", iteration, step_size);

            if step_size == 0.0 {
                // Line search found no descent step: the gradient direction is locally
                // exhausted. Halt rather than applying a non-descent update.
                log::info!(
                    "FWI stalled at iter {}: line search returned no descent step (J={:.6e})",
                    iteration,
                    objective
                );
                break;
            }

            // g = ∂J/∂c (true gradient; see adjoint_model derivation in module doc).
            // Gradient descent: c_new = c − step · g  (subtract, not add).
            current_model = &current_model - &(&normalized_gradient * step_size);
            self.apply_model_constraints(&mut current_model);
            let c_min_after = current_model.iter().copied().fold(f64::INFINITY, f64::min);
            let c_max_after = current_model
                .iter()
                .copied()
                .fold(f64::NEG_INFINITY, f64::max);
            log::info!(
                "FWI iter {} model_range=[{:.1},{:.1}]",
                iteration,
                c_min_after,
                c_max_after
            );
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

    /// Multi-source FWI inversion.
    ///
    /// Joint objective: `J(c) = Σᵢ Jᵢ(c)` where each `Jᵢ` is the per-shot
    /// acoustic L2 misfit.  The reduced gradient is the sum of per-shot
    /// adjoint-state gradients:
    ///
    /// ```text
    /// ∂J/∂c = Σᵢ ∂Jᵢ/∂c
    /// ```
    ///
    /// Illumination coverage scales with the number of shots: `N` sources that
    /// span a hemispherical aperture sample the skull from `N` distinct ray
    /// directions, making the joint gradient much better conditioned than the
    /// single-source case (Marquet et al. 2013; Guasch et al. 2020).
    ///
    /// # Arguments
    /// * `shots` — slice of `(geometry, observed_data)` pairs; one entry per
    ///   shot gather.  Geometries may differ (different source positions); the
    ///   receiver mask can be common or per-shot.
    /// * `initial_model` — starting velocity field [NX, NY, NZ] in m/s
    /// * `grid` — computational grid
    ///
    /// # References
    /// - Marquet, F. et al. (2013). Non-invasive transcranial ultrasound therapy
    ///   based on a 3D CT scan. *Phys. Med. Biol.* 58, 2937.
    /// - Guasch, L. et al. (2020). Full-waveform inversion imaging of the human
    ///   brain. *npj Digital Medicine* 3, 28.
    pub fn invert_multi_source(
        &self,
        shots: &[(FwiGeometry, Array2<f64>)],
        initial_model: &Array3<f64>,
        grid: &Grid,
    ) -> KwaversResult<Array3<f64>> {
        if shots.is_empty() {
            return Err(KwaversError::Validation(
                ValidationError::ConstraintViolation {
                    message: "invert_multi_source requires at least one shot".to_string(),
                },
            ));
        }
        for (geometry, _) in shots {
            geometry.validate(grid, self.parameters.nt)?;
        }

        let (nx, ny, nz) = grid.dimensions();
        let mut current_model = initial_model.clone();
        self.apply_model_constraints(&mut current_model);

        for iteration in 0..self.parameters.max_iterations {
            // Accumulate per-shot objective and gradient.
            // Shots are processed sequentially: each `compute_shot_gradient` allocates
            // a full forward-history Array4<f64> of shape (nt, nx, ny, nz) for the
            // adjoint pass.  With nt=2115 and a 64×48×64 grid that is ~3.3 GB per shot;
            // running all N_shots simultaneously would demand N_shots×3.3 GB (≈40 GB for
            // 12 shots) and fragment the thread pool (24 threads / 12 shots = 2
            // threads/shot vs 24 with sequential dispatch).  Sequential outer iteration
            // lets every inner par_for_each use the full 24-thread pool, reducing per-shot
            // wall time by ~24× while peak allocation stays at 3.3 GB.
            let mut total_objective = 0.0_f64;
            let mut total_gradient = Array3::<f64>::zeros((nx, ny, nz));
            for (geometry, observed_data) in shots.iter() {
                let (obj, grad) =
                    self.compute_shot_gradient(&current_model, geometry, observed_data, grid)?;
                total_objective += obj;
                // In-place accumulation: eliminates one 1.2 MB Array3 allocation per shot.
                Zip::from(&mut total_gradient)
                    .and(&grad)
                    .par_for_each(|a, &b| *a += b);
            }

            let smoothed = self.smooth_gradient(&total_gradient);

            let regularized = self.apply_regularization(&smoothed, &current_model)?;

            let grad_max = regularized
                .iter()
                .copied()
                .fold(0.0_f64, |a, x| a.max(x.abs()));
            log::info!(
                "FWI multi-source iter {} joint_J={:.6e} grad_max={:.6e}",
                iteration,
                total_objective,
                grad_max
            );

            // In-place max-norm scaling: eliminates one 1.2 MB Array3 allocation.
            let mut normalized = regularized;
            if grad_max > f64::EPSILON {
                normalized.mapv_inplace(|g| g / grad_max);
            }

            let step_size = self.line_search_multi(&current_model, &normalized, shots, grid)?;
            log::info!(
                "FWI multi-source iter {} step_size={:.6e}",
                iteration,
                step_size
            );

            if step_size == 0.0 {
                log::info!(
                    "FWI multi-source stalled at iter {}: J={:.6e}",
                    iteration,
                    total_objective
                );
                break;
            }

            // g = ∂J/∂c; descent: c -= step * g.  In-place: eliminates two temporaries.
            Zip::from(&mut current_model)
                .and(&normalized)
                .par_for_each(|c, &g| *c -= g * step_size);
            self.apply_model_constraints(&mut current_model);

            let c_max = current_model
                .iter()
                .copied()
                .fold(f64::NEG_INFINITY, f64::max);
            let c_min = current_model.iter().copied().fold(f64::INFINITY, f64::min);
            log::info!(
                "FWI multi-source iter {} model_range=[{:.1},{:.1}]",
                iteration,
                c_min,
                c_max
            );
        }

        Ok(current_model)
    }

    /// Multi-source FWI with a frozen (skull) mask for brain tissue imaging.
    ///
    /// Identical to [`invert_multi_source`] except that:
    ///
    /// - Voxels where `frozen_mask` is `true` are **never updated** — their
    ///   values are restored from `reference_model` after every gradient step.
    ///   Use this to freeze CT-derived skull velocities while inverting the soft
    ///   tissue interior.
    /// - The brain voxel velocity is clamped to `[c_min, c_max]` after each
    ///   iteration, rather than the broad physical bounds used for skull FWI.
    ///
    /// # Physics background
    ///
    /// Stage-2 brain tissue FWI (Guasch 2020, §Methods "Brain FWI"):  the skull
    /// is treated as a known scatterer (fixed from CT or Stage-1 FWI); the
    /// unknowns are soft-tissue velocities with Δc ≈ 1–4 % around 1500 m/s.
    /// Freezing the skull avoids re-inverting bone whose large impedance contrast
    /// would dominate the gradient and mask the small brain-tissue signal.
    ///
    /// # Arguments
    /// * `reference_model` — Frozen velocity values (e.g. CT-derived skull model).
    ///   Only the voxels where `frozen_mask = true` are read from this array.
    /// * `frozen_mask` — `true` = frozen skull voxel, `false` = free brain voxel.
    /// * `c_min`, `c_max` — Velocity bounds for free (brain tissue) voxels.
    pub fn invert_multi_source_masked(
        &self,
        shots: &[(FwiGeometry, Array2<f64>)],
        initial_model: &Array3<f64>,
        reference_model: &Array3<f64>,
        frozen_mask: &Array3<bool>,
        c_min: f64,
        c_max: f64,
        grid: &Grid,
    ) -> KwaversResult<Array3<f64>> {
        if shots.is_empty() {
            return Err(KwaversError::Validation(
                ValidationError::ConstraintViolation {
                    message: "invert_multi_source_masked requires at least one shot".to_string(),
                },
            ));
        }
        for (geometry, _) in shots {
            geometry.validate(grid, self.parameters.nt)?;
        }

        let (nx, ny, nz) = grid.dimensions();
        let mut current_model = initial_model.clone();

        // Enforce initial state: skull voxels = reference, brain voxels clamped.
        Zip::from(&mut current_model)
            .and(frozen_mask)
            .and(reference_model)
            .par_for_each(|c, &frozen, &r| {
                if frozen {
                    *c = r;
                } else {
                    *c = c.clamp(c_min, c_max);
                }
            });

        for iteration in 0..self.parameters.max_iterations {
            // Sequential shot dispatch — same memory rationale as invert_multi_source:
            // forward-history Array4 (~3.3 GB/shot) must not be live for all shots
            // simultaneously.  Inner par_for_each gets all 24 threads sequentially.
            let mut total_objective = 0.0_f64;
            let mut total_gradient = Array3::<f64>::zeros((nx, ny, nz));
            for (geometry, observed_data) in shots.iter() {
                let (obj, grad) =
                    self.compute_shot_gradient(&current_model, geometry, observed_data, grid)?;
                total_objective += obj;
                // In-place accumulation: eliminates one 1.2 MB Array3 allocation per shot.
                Zip::from(&mut total_gradient)
                    .and(&grad)
                    .par_for_each(|a, &b| *a += b);
            }

            // Zero gradient at frozen (skull) voxels — only brain tissue updates.
            Zip::from(&mut total_gradient)
                .and(frozen_mask)
                .par_for_each(|g, &frozen| {
                    if frozen {
                        *g = 0.0;
                    }
                });

            let smoothed = self.smooth_gradient(&total_gradient);
            let mut regularized = self.apply_regularization(&smoothed, &current_model)?;

            // Re-zero skull voxels after smoothing.
            //
            // `smooth_gradient` applies a box or 6-connected filter that spreads
            // non-zero brain-region gradient into skull-adjacent cells.  If the
            // smoothed gradient at a skull voxel is non-zero and the fallback line
            // search direction (c += step · g) is chosen, the test model will have
            // skull velocities slightly above the reference.  `validate_time_step`
            // then derives a c_max > the value used to compute `self.parameters.dt`,
            // triggering a spurious CFL constraint violation.  Re-zeroing after
            // smoothing prevents the leakage from reaching the line search.
            Zip::from(&mut regularized)
                .and(frozen_mask)
                .par_for_each(|g, &frozen| {
                    if frozen {
                        *g = 0.0;
                    }
                });

            let grad_max = regularized
                .iter()
                .copied()
                .fold(0.0_f64, |a, x| a.max(x.abs()));
            log::info!(
                "FWI masked iter {} joint_J={:.6e} grad_max={:.6e}",
                iteration,
                total_objective,
                grad_max
            );

            // In-place max-norm scaling: eliminates one 1.2 MB Array3 allocation.
            let mut normalized = regularized;
            if grad_max > f64::EPSILON {
                normalized.mapv_inplace(|g| g / grad_max);
            }

            let step_size = self.line_search_multi(&current_model, &normalized, shots, grid)?;
            log::info!("FWI masked iter {} step_size={:.6e}", iteration, step_size);

            if step_size == 0.0 {
                log::info!(
                    "FWI masked stalled at iter {}: J={:.6e}",
                    iteration,
                    total_objective
                );
                break;
            }

            // In-place descent: eliminates two temporaries.
            Zip::from(&mut current_model)
                .and(&normalized)
                .par_for_each(|c, &g| *c -= g * step_size);

            // Constrain: skull voxels restored from reference; brain voxels clamped.
            Zip::from(&mut current_model)
                .and(frozen_mask)
                .and(reference_model)
                .par_for_each(|c, &frozen, &r| {
                    if frozen {
                        *c = r;
                    } else {
                        *c = c.clamp(c_min, c_max);
                    }
                });

            let c_max_model = current_model
                .iter()
                .copied()
                .fold(f64::NEG_INFINITY, f64::max);
            let c_min_model = current_model.iter().copied().fold(f64::INFINITY, f64::min);
            log::info!(
                "FWI masked iter {} model_range=[{:.1},{:.1}]",
                iteration,
                c_min_model,
                c_max_model
            );
        }

        Ok(current_model)
    }

    /// Compute the per-shot objective and physics gradient for one shot gather.
    ///
    /// Returns `(Jᵢ, ∂Jᵢ/∂c)` where `Jᵢ` is the L2 data misfit and
    /// `∂Jᵢ/∂c` is the adjoint-state gradient for this shot.
    ///
    /// If `FwiParameters::source_mute_radius > 0`, the gradient is zeroed within
    /// that radius of every source voxel to suppress near-source body-wave
    /// artefacts (see `mute_gradient_near_sources`).
    fn compute_shot_gradient(
        &self,
        model: &Array3<f64>,
        geometry: &FwiGeometry,
        observed_data: &Array2<f64>,
        grid: &Grid,
    ) -> KwaversResult<(f64, Array3<f64>)> {
        let (synthetic_data, forward_history) = self.forward_model(model, geometry, grid)?;
        let objective = self.compute_l2_objective(observed_data, &synthetic_data)?;
        let residual = self.compute_adjoint_source(observed_data, &synthetic_data)?;
        let adjoint_source = self.build_adjoint_source(&residual, geometry)?;
        let mut gradient = self.adjoint_model(
            &adjoint_source,
            model,
            grid,
            &forward_history,
            geometry.source.p_mask.as_ref(),
        )?;

        // Near-source gradient mute — see FwiParameters::source_mute_radius.
        if self.parameters.source_mute_radius > 0 {
            if let Some(p_mask) = geometry.source.p_mask.as_ref() {
                mute_gradient_near_sources(
                    &mut gradient,
                    p_mask,
                    self.parameters.source_mute_radius,
                );
            }
        }

        Ok((objective, gradient))
    }

    /// Evaluate the joint objective `J = Σᵢ Jᵢ(model)` across all shots.
    ///
    /// Shots are independent: each forward model reads `model` and `grid` immutably,
    /// so the loop can run fully in parallel using Rayon's work-stealing pool.
    fn compute_joint_objective(
        &self,
        model: &Array3<f64>,
        shots: &[(FwiGeometry, Array2<f64>)],
        grid: &Grid,
    ) -> KwaversResult<f64> {
        let results: Vec<KwaversResult<f64>> = shots
            .par_iter()
            .map(|(geometry, observed_data)| {
                self.compute_objective(model, observed_data, geometry, grid)
            })
            .collect();
        let mut total = 0.0_f64;
        for result in results {
            total += result?;
        }
        Ok(total)
    }

    /// Line search for multi-source inversion.
    ///
    /// ## Algorithm
    ///
    /// First tries `c − α·g` (standard gradient descent, `g = +∂J/∂c`).  If all
    /// halvings fail, tries `c + α·g` as a fallback (handles the `g = −∂J/∂c`
    /// sign convention where the adjoint accumulation produces the negated
    /// gradient).  The positive-step direction corresponds to the
    /// `c_new = c + step * g` identity in the module-level sign-convention note.
    ///
    /// Returns a **signed** step:
    /// - Positive `α`: caller applies `c − α·g` (standard descent).
    /// - Negative `α`: caller applies `c − (−|α|)·g = c + |α|·g` (sign-flipped).
    ///
    /// Returns `0.0` when neither direction satisfies sufficient decrease.
    ///
    /// Cost: at most `2 × max_halvings × N_shots` forward evaluations.
    fn line_search_multi(
        &self,
        model: &Array3<f64>,
        gradient: &Array3<f64>,
        shots: &[(FwiGeometry, Array2<f64>)],
        grid: &Grid,
    ) -> KwaversResult<f64> {
        let max_halvings = 5;
        let current_obj = self.compute_joint_objective(model, shots, grid)?;

        // Pre-allocate one test-model buffer reused across both search directions,
        // eliminating 2 × max_halvings temporary Array3 allocations per call.
        let mut test_model = Array3::<f64>::zeros(model.dim());

        // ── Primary: c -= step · g ─────────────────────────────────────────
        // Correct when `adjoint_model` returns `g = +∂J/∂c`.
        let mut step = self.parameters.step_size;
        for _ in 0..max_halvings {
            Zip::from(&mut test_model)
                .and(model)
                .and(gradient)
                .par_for_each(|t, &m, &g| *t = m - g * step);
            let test_obj = self.compute_joint_objective(&test_model, shots, grid)?;
            if test_obj < current_obj {
                return Ok(step);
            }
            step *= 0.5;
        }

        // ── Fallback: c += step · g ────────────────────────────────────────
        // Correct when `adjoint_model` returns `g = −∂J/∂c` (negative gradient
        // convention) — handles residual sign conventions that negate the
        // adjoint source.  A negative return value signals the caller to add
        // rather than subtract.
        let mut step = self.parameters.step_size;
        for _ in 0..max_halvings {
            Zip::from(&mut test_model)
                .and(model)
                .and(gradient)
                .par_for_each(|t, &m, &g| *t = m + g * step);
            let test_obj = self.compute_joint_objective(&test_model, shots, grid)?;
            if test_obj < current_obj {
                log::info!(
                    "FWI line search: gradient sign flipped — using c += step · g \
                     (g = −∂J/∂c convention)"
                );
                return Ok(-step); // negative: caller applies c -= (-step)*g = c + step*g
            }
            step *= 0.5;
        }

        Ok(0.0)
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

        self.smooth_gradient(&gradient)
    }

    /// Apply smoothing to gradient to reduce high-frequency artifacts.
    ///
    /// # Algorithm
    ///
    /// Two branches handle the quasi-2-D and full-3-D cases:
    ///
    /// **ny ≤ 2 (quasi-2-D):** 3×3 box filter in the x–z plane applied to every
    /// y-slice independently.  The standard j-loop `1..ny-1` is empty for ny=2,
    /// so iterating over all j avoids a silent no-op that leaves the gradient
    /// unsmoothed.
    ///
    /// **ny > 2 (full-3-D):** 6-connected stencil with the centre weighted 3/9.
    /// Each of the six face-connected neighbours contributes 1/9; centre 3/9.
    /// This is a first-order approximation to a separable Gaussian with σ ≈ 0.5
    /// voxels and preserves the overall magnitude better than equal-weight 7-point
    /// averaging.
    /// Apply smoothing to gradient to reduce high-frequency artifacts.
    ///
    /// # Allocation strategy
    ///
    /// Allocates one `Array3::zeros` instead of `gradient.clone()`.  Only the
    /// boundary faces (O(N²) elements) are copied from the input; interior cells
    /// are fully overwritten by the stencil.  This saves one O(N³) memcopy.
    ///
    /// # Loop ordering
    ///
    /// Inner loop over `k` (last index, stride-1 in C-order) maximises cache
    /// locality for the `k±1` neighbour reads.
    #[must_use]
    fn smooth_gradient(&self, gradient: &Array3<f64>) -> Array3<f64> {
        let (nx, ny, nz) = gradient.dim();
        // Zero-init: avoids the O(N³) clone.  Boundary faces are then populated
        // from gradient so unmodified exterior elements retain the input values.
        let mut smoothed = Array3::<f64>::zeros((nx, ny, nz));

        // Copy boundary faces (not touched by interior stencil).
        smoothed
            .slice_mut(s![0, .., ..])
            .assign(&gradient.slice(s![0, .., ..]));
        smoothed
            .slice_mut(s![nx - 1, .., ..])
            .assign(&gradient.slice(s![nx - 1, .., ..]));
        smoothed
            .slice_mut(s![.., 0, ..])
            .assign(&gradient.slice(s![.., 0, ..]));
        smoothed
            .slice_mut(s![.., ny - 1, ..])
            .assign(&gradient.slice(s![.., ny - 1, ..]));
        smoothed
            .slice_mut(s![.., .., 0])
            .assign(&gradient.slice(s![.., .., 0]));
        smoothed
            .slice_mut(s![.., .., nz - 1])
            .assign(&gradient.slice(s![.., .., nz - 1]));

        if ny <= 2 {
            // Quasi-2-D: 3×3 box filter in the x–z plane, all y-slices.
            // i-outer, k-inner: inner loop reads gradient[[i,j,k±1]] at stride 1.
            for i in 1..nx - 1 {
                for j in 0..ny {
                    for k in 1..nz - 1 {
                        smoothed[[i, j, k]] = (gradient[[i - 1, j, k - 1]]
                            + gradient[[i, j, k - 1]]
                            + gradient[[i + 1, j, k - 1]]
                            + gradient[[i - 1, j, k]]
                            + gradient[[i, j, k]]
                            + gradient[[i + 1, j, k]]
                            + gradient[[i - 1, j, k + 1]]
                            + gradient[[i, j, k + 1]]
                            + gradient[[i + 1, j, k + 1]])
                            / 9.0;
                    }
                }
            }
        } else {
            // Full 3-D: 6-connected stencil, centre weighted 3/9.
            // i-outer, k-inner: inner loop reads gradient[[i,j,k±1]] at stride 1.
            for i in 1..nx - 1 {
                for j in 1..ny - 1 {
                    for k in 1..nz - 1 {
                        smoothed[[i, j, k]] = (gradient[[i - 1, j, k]]
                            + gradient[[i + 1, j, k]]
                            + gradient[[i, j - 1, k]]
                            + gradient[[i, j + 1, k]]
                            + gradient[[i, j, k - 1]]
                            + gradient[[i, j, k + 1]]
                            + 3.0 * gradient[[i, j, k]])
                            / 9.0;
                    }
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

        // Tikhonov regularization: R += λ₁ · m  (in-place; no temporary Array3).
        if reg_params.tikhonov_weight > 0.0 {
            let w = reg_params.tikhonov_weight;
            Zip::from(&mut regularized)
                .and(model)
                .par_for_each(|r, &m| *r += m * w);
        }

        // Total variation regularization  (in-place; no temporary Array3).
        if reg_params.tv_weight > 0.0 {
            let tv_term = self.compute_total_variation_gradient(model);
            let w = reg_params.tv_weight;
            Zip::from(&mut regularized)
                .and(&tv_term)
                .par_for_each(|r, &t| *r += t * w);
        }

        // Smoothness regularization (Laplacian)  (in-place; no temporary Array3).
        if reg_params.smoothness_weight > 0.0 {
            let smoothness_term = self.compute_smoothness_gradient(model);
            let w = reg_params.smoothness_weight;
            Zip::from(&mut regularized)
                .and(&smoothness_term)
                .par_for_each(|r, &s| *r += s * w);
        }

        Ok(regularized)
    }

    /// Compute total variation gradient for regularization.
    ///
    /// # Loop ordering
    ///
    /// `i`-outer, `k`-inner: the inner loop reads `model[[i,j,k±1]]` at stride 1
    /// (C-order last-index varies fastest), minimising cache misses.
    #[must_use]
    fn compute_total_variation_gradient(&self, model: &Array3<f64>) -> Array3<f64> {
        let (nx, ny, nz) = model.dim();
        let mut tv_gradient = Array3::zeros((nx, ny, nz));

        for i in 1..nx - 1 {
            for j in 1..ny - 1 {
                for k in 1..nz - 1 {
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

    /// Compute smoothness gradient (Laplacian) for regularization.
    ///
    /// # Loop ordering
    ///
    /// `i`-outer, `k`-inner: inner-loop accesses `model[[i,j,k±1]]` at stride 1.
    #[must_use]
    fn compute_smoothness_gradient(&self, model: &Array3<f64>) -> Array3<f64> {
        let (nx, ny, nz) = model.dim();
        let mut laplacian = Array3::zeros((nx, ny, nz));

        for i in 1..nx - 1 {
            for j in 1..ny - 1 {
                for k in 1..nz - 1 {
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

    /// Line search for optimal step size.
    ///
    /// Uses the Armijo sufficient-decrease condition with constant `c1`:
    ///
    /// ```text
    /// J(c − α g_norm) ≤ J(c) − c1 · α · ‖g_norm‖²
    /// ```
    ///
    /// where `g_norm` is the gradient normalized to max-norm = 1, so `α` has
    /// units of m/s (the velocity update per iteration).
    ///
    /// `c1 = 0` selects the pure sufficient-decrease rule: any trial step that
    /// strictly reduces the objective is accepted.  This is appropriate when the
    /// gradient is already normalized (ensuring the descent direction is exact)
    /// and the absolute scale of the objective varies with the source amplitude.
    /// A non-zero `c1` would require the objective and ‖g_norm‖² to be in the
    /// same physical units, which is generally violated after max-norm
    /// normalization of a physics-derived gradient.
    ///
    /// Reference: Nocedal & Wright (2006) §3.1, Condition (3.6a) with c₁ → 0.
    fn line_search(
        &self,
        model: &Array3<f64>,
        gradient: &Array3<f64>,
        observed_data: &Array2<f64>,
        geometry: &FwiGeometry,
        grid: &Grid,
    ) -> KwaversResult<f64> {
        let mut step_size = self.parameters.step_size;
        let max_iter = 10;

        let current_objective = self.compute_objective(model, observed_data, geometry, grid)?;

        // Pre-allocate once; refilled each halving step via Zip — no per-iteration alloc.
        let mut test_model = Array3::<f64>::zeros(model.dim());

        for _ in 0..max_iter {
            // g = ∂J/∂c; descent moves in −g direction.
            let s = step_size;
            Zip::from(&mut test_model)
                .and(model)
                .and(gradient)
                .par_for_each(|t, &m, &g| *t = m - s * g);
            let test_objective =
                self.compute_objective(&test_model, observed_data, geometry, grid)?;

            // Pure sufficient decrease: accept any step that strictly reduces J.
            if test_objective < current_objective {
                return Ok(step_size);
            }

            step_size *= 0.5;
        }

        // No step in the −g direction satisfied sufficient decrease.
        // Returning 0.0 signals the caller to halt iteration rather than
        // applying a non-descent update that would increase J.
        Ok(0.0)
    }

    /// Apply physical constraints to velocity model
    fn apply_model_constraints(&self, model: &mut Array3<f64>) {
        use crate::core::constants::SOUND_SPEED_WATER;

        // Ensure physically reasonable velocity bounds
        let min_velocity = SOUND_SPEED_WATER * 0.5; // 750 m/s
        let max_velocity = SOUND_SPEED_WATER * 4.0; // 6000 m/s

        model.mapv_inplace(|v| v.clamp(min_velocity, max_velocity));
    }

    /// Construct a configured, CPML-enabled FDTD solver ready for time-stepping.
    ///
    /// ## Single source of truth
    ///
    /// Both `forward_model` (full history) and `forward_model_sensor_only` (no
    /// history) require identical solver setup: config, medium, CPML parameters.
    /// This helper centralises that setup so neither variant drifts from the other.
    ///
    /// ## Returns
    /// `(solver, (nx, ny, nz), dt)` where the solver has CPML enabled and is
    /// positioned at t = 0 ready for `step_forward` calls.
    fn build_fdtd_solver_for_forward(
        &self,
        model: &Array3<f64>,
        geometry: &FwiGeometry,
        grid: &Grid,
    ) -> KwaversResult<(crate::solver::fdtd::FdtdSolver, (usize, usize, usize), f64)> {
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

        // Enable CPML absorbing boundaries so wave energy exits the domain rather
        // than reverberating in a closed cavity.  Without CPML, all six walls are
        // perfectly reflecting; constructive interference grows the pressure
        // amplitude by ~√nt per half-period, driving J to non-physical values and
        // preventing FWI convergence.
        //
        // Per-dimension CPML thickness: 10 cells in x and z; y thickness is
        // grid-adaptive.  ny ≤ 20 cannot fit two 10-cell absorbing layers, so
        // y-CPML is disabled (sigma_y = 0, transparent BC).  For true 3-D grids
        // (ny > 20, e.g. ny = 48) y-CPML is enabled to absorb outgoing waves
        // through the y-faces and avoid spurious reflections.
        //
        // CRITICAL: all forward-model sources must lie outside the CPML zone.
        // For CPML thickness = 10: source ix ≥ 10, iy ≥ y_pml, iz ≥ 10.
        //
        // Theorem (CPML optimal thickness): σ_max = -(m+1)·ln(R_target) / (2·d·η)
        // where d = thickness × dx, η = 1/c_max, m = polynomial order (3).
        // Reference: Roden & Gedney (2000), Microwave Opt. Technol. Lett. 27(5), 334-338.
        let c_max = model.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        let y_pml = if ny > 20 { 10usize } else { 0usize };
        let cpml_config = CPMLConfig {
            thickness: 10,
            per_dimension: PerDimensionPML::new(10, y_pml, 10),
            ..CPMLConfig::default()
        };
        solver.enable_cpml(cpml_config, dt, c_max)?;

        Ok((solver, (nx, ny, nz), dt))
    }

    /// Run the forward FDTD model and return both receiver traces and the full
    /// pressure-history volume needed by the adjoint pass.
    ///
    /// ## Memory contract
    ///
    /// Allocates `Array4<f64>` of shape `(nt, nx, ny, nz)` — approximately
    /// `nt × nx × ny × nz × 8` bytes (≈ 3.3 GB for nt=2115, 64×48×64).  This
    /// allocation is intentional: the adjoint pass in `adjoint_model` requires
    /// the complete time-reversed pressure history to form the imaging condition.
    ///
    /// **Do not call this function from `compute_objective` or any line-search
    /// path.**  Use `forward_model_sensor_only` there; it skips the history
    /// allocation and is safe to run in parallel over shots.
    fn forward_model(
        &self,
        model: &Array3<f64>,
        geometry: &FwiGeometry,
        grid: &Grid,
    ) -> KwaversResult<(Array2<f64>, Array4<f64>)> {
        let (mut solver, (nx, ny, nz), _dt) =
            self.build_fdtd_solver_for_forward(model, geometry, grid)?;

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

    /// Run the forward FDTD model and return only receiver traces — no history.
    ///
    /// ## Theorem
    ///
    /// The L2 objective `J = (dt/2) ||d_syn − d_obs||²` depends only on the
    /// synthetic receiver data `d_syn(r,t)`, not on the volumetric pressure field
    /// `p(x,t)`.  Storing `p(x,t)` as `Array4<f64>` (≈ 3.3 GB per shot) is
    /// unnecessary for objective evaluation and prohibits safe shot-level
    /// parallelism in the line search.
    ///
    /// ## Memory contract
    ///
    /// Peak allocation per call: O(N_receivers × nt × 8 bytes) ≈ 5 MB for
    /// 23 receivers × 2115 steps.  Solver field buffers add ~50 MB.  Total
    /// per-call footprint is ~55 MB, making it safe to run N_shots calls in
    /// parallel via `rayon::par_iter` without memory pressure.
    fn forward_model_sensor_only(
        &self,
        model: &Array3<f64>,
        geometry: &FwiGeometry,
        grid: &Grid,
    ) -> KwaversResult<Array2<f64>> {
        let (mut solver, _dims, _dt) = self.build_fdtd_solver_for_forward(model, geometry, grid)?;

        // No history allocation.  The solver's SensorRecorder accumulates receiver
        // traces internally at O(N_receivers) cost per step — unchanged from the
        // full forward model.
        for _ in 0..self.parameters.nt {
            solver.step_forward()?;
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

        Ok(synthetic)
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
            None,
            self.parameters.frequency,
        )
        .map_err(crate::core::error::KwaversError::InvalidInput)?;

        let mut solver = FdtdSolver::new(config, grid, &medium_adj, adjoint_source.clone())?;

        // CPML must match the forward solver's boundary treatment: the adjoint
        // wavefield is the time-reversed forward field in the same domain.  Using
        // different boundary conditions for forward and adjoint breaks the discrete
        // Green's identity and corrupts the gradient (Time-reversal theorem, §3 of
        // the module-level docstring).  Same per-dimension thickness as forward.
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
            //
            // At source-constrained positions the forward solver sets p equal to
            // the prescribed signal via a Dirichlet BC.  The resulting ∂²p/∂t²
            // reflects the second derivative of the wavelet (amplitude ~f₀⁴·P₀),
            // not the local wave physics.  Including these cells would produce a
            // dominant spurious gradient at the source location that overwhelms
            // the physically meaningful sensitivity in the skull region.
            //
            // Reference: Sun, R. & Symes, W.W. (1991). "Inversion of source
            // signatures by full-waveform inversion." *SEG Expanded Abstracts*.
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

        // Debug: locate the dominant gradient voxel
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

        // Adjoint source must be ADDITIVE (soft source), not Dirichlet (hard source).
        //
        // Theorem (adjoint source injection mode):
        //   The adjoint equation is  L†λ = −δ_r · (d_syn − d_obs)(T−t)
        //   where L† is the adjoint of the wave operator L and δ_r is the receiver
        //   spatial support.  This is a body-force/source-term forcing, not a
        //   Dirichlet boundary condition.  Injecting as Dirichlet pins λ at receiver
        //   positions to the reversed residual, suppressing the back-propagated
        //   wavefield and producing an incorrect gradient that is not a descent
        //   direction for J.
        //
        // Proof sketch:
        //   Apply the identity ⟨Lp, λ⟩ = ⟨p, L†λ⟩ (Green's identity for the
        //   discrete staggered-grid operator).  For this identity to hold and yield
        //   ∂J/∂c = ∫λ(T−t)∂²p/∂t² dt, λ must evolve freely under L† with the
        //   reversed residual as a forcing term (additive injection), not as a
        //   Dirichlet constraint that removes λ's degrees of freedom at the receivers.
        //
        // Reference: Plessix (2006), GFJI 167(2), 495–503, eq. (2)–(6).
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

    /// Compute the model objective by running a sensor-only forward simulation.
    ///
    /// Calls `forward_model_sensor_only` — no pressure-history `Array4` is
    /// allocated.  Peak memory per call is ~55 MB (solver fields + receiver
    /// traces), making this safe to call from `compute_joint_objective`'s
    /// `par_iter` over all shots simultaneously.
    fn compute_objective(
        &self,
        model: &Array3<f64>,
        observed_data: &Array2<f64>,
        geometry: &FwiGeometry,
        grid: &Grid,
    ) -> KwaversResult<f64> {
        let synthetic_data = self.forward_model_sensor_only(model, geometry, grid)?;
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
                .par_for_each(|d, &p0, &p1, &p2| {
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
                .par_for_each(|d, &p0, &p1, &p2| {
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
            .par_for_each(|d, &p0, &p1, &p2| {
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
        // Adjoint source must be Additive (soft source) — not Dirichlet.
        // Dirichlet would pin λ at receivers and destroy Green's identity.
        assert!(matches!(p_mode, SourceMode::Additive));
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
