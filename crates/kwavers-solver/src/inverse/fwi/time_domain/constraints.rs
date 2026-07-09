//! Physical constraints: CFL validation, model clamping, pressure second-derivative.

use super::FwiProcessor;
use kwavers_core::error::{KwaversError, KwaversResult, ValidationError};
use kwavers_grid::Grid;
use moirai_parallel::{for_each_chunk_mut_enumerated_with, for_each_chunk_mut_with, Adaptive};
use leto::{
    Array3,
    Array4,
    ArrayView3,
};

fn pressure_second_derivative_views_into(
    dst: &mut Array3<f64>,
    p0: ArrayView3<'_, f64>,
    p1: ArrayView3<'_, f64>,
    p2: ArrayView3<'_, f64>,
    inv_dt_sq: f64,
) {
    if dst.is_standard_layout()
        && p0.is_standard_layout()
        && p1.is_standard_layout()
        && p2.is_standard_layout()
    {
        let p0 = p0
            .as_slice_memory_order()
            .expect("invariant: standard-layout p0 view exposes memory-order slice");
        let p1 = p1
            .as_slice_memory_order()
            .expect("invariant: standard-layout p1 view exposes memory-order slice");
        let p2 = p2
            .as_slice_memory_order()
            .expect("invariant: standard-layout p2 view exposes memory-order slice");
        let dst_slice = dst
            .as_slice_memory_order_mut()
            .expect("invariant: standard-layout destination exposes memory-order slice");
        let chunk_size = super::FWI_FIELD_CHUNK;
        for_each_chunk_mut_enumerated_with::<Adaptive, _, _>(
            dst_slice,
            chunk_size,
            |chunk_index, chunk| {
                let start = chunk_index * chunk_size;
                for (offset, d) in chunk.iter_mut().enumerate() {
                    let idx = start + offset;
                    *d = (2.0f64.mul_add(-p1[idx], p0[idx]) + p2[idx]) * inv_dt_sq;
                }
            },
        );
    } else {
        Zip::from(dst)
            .and(&p0)
            .and(&p1)
            .and(&p2)
            .for_each(|d, &v0, &v1, &v2| {
                *d = (2.0f64.mul_add(-v1, v0) + v2) * inv_dt_sq;
            });
    }
}

impl FwiProcessor {
    /// Apply physical constraints to velocity model.
    /// # Errors
    /// - Returns [`Err`] if an internal constraint is violated.
    ///
    pub(super) fn apply_model_constraints(&self, model: &mut Array3<f64>) {
        use kwavers_core::constants::SOUND_SPEED_WATER;
        let min_velocity = SOUND_SPEED_WATER * 0.5; // 750 m/s
        let max_velocity = SOUND_SPEED_WATER * 4.0; // 6000 m/s
        if model.is_standard_layout() {
            let values = model
                .as_slice_memory_order_mut()
                .expect("invariant: standard-layout model exposes memory-order slice");
            for_each_chunk_mut_with::<Adaptive, _, _>(values, super::FWI_FIELD_CHUNK, |chunk| {
                for value in chunk {
                    *value = value.clamp(min_velocity, max_velocity);
                }
            });
        } else {
            model
                .iter_mut()
                .for_each(|value| *value = value.clamp(min_velocity, max_velocity));
        }
    }

    /// Validate timestep and model compatibility with the grid.
    /// # Errors
    /// - Returns [`KwaversError::Validation`] if the precondition for a Validation-class constraint is violated.
    /// - Propagates any [`KwaversError`] returned by called functions.
    ///
    pub(super) fn validate_time_step(
        &self,
        model: &Array3<f64>,
        grid: &Grid,
    ) -> KwaversResult<f64> {
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
                        .to_owned(),
                },
            ));
        }

        if self.parameters.dt <= 0.0 {
            return Err(KwaversError::Validation(
                ValidationError::ConstraintViolation {
                    message: "FWI requires a positive time step".to_owned(),
                },
            ));
        }

        if model.iter().any(|&v| !v.is_finite() || v <= 0.0) {
            return Err(KwaversError::Validation(
                ValidationError::ConstraintViolation {
                    message: "FWI requires a finite, strictly positive sound speed model"
                        .to_owned(),
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

    /// Calculate stable timestep for FDTD solver.
    ///
    /// Uses CFL condition: `dt ≤ min(dx,dy,dz) / (c_max × √3)`.
    ///
    /// Reference: Courant et al. (1928). *Math. Ann.* 100(1), 32–74.
    /// # Errors
    /// - Returns [`KwaversError::Validation`] if the precondition for a Validation-class constraint is violated.
    ///
    pub(super) fn calculate_stable_timestep(
        &self,
        model: &Array3<f64>,
        grid: &Grid,
    ) -> KwaversResult<f64> {
        let c_max = model.iter().copied().fold(0.0, f64::max);
        if !c_max.is_finite() || c_max <= 0.0 {
            return Err(KwaversError::Validation(
                ValidationError::ConstraintViolation {
                    message: "FWI requires a strictly positive finite sound speed model".to_owned(),
                },
            ));
        }

        let min_spacing = grid.dx.min(grid.dy).min(grid.dz);
        let cfl_number = 0.3;
        Ok(cfl_number * min_spacing / (c_max * 3.0_f64.sqrt()))
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
    /// # Errors
    /// - Returns [`KwaversError::Validation`] if the precondition for a Validation-class constraint is violated.
    ///
    pub(super) fn pressure_second_derivative_into(
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
        if dst.dim() != current.dim() {
            return Err(KwaversError::Validation(
                ValidationError::ConstraintViolation {
                    message: format!(
                        "Second-derivative destination shape mismatch: dst {:?}, source {:?}",
                        dst.dim(),
                        current.dim()
                    ),
                },
            ));
        }

        if idx == 0 {
            let next = forward_history.index_axis(Axis(0), 1);
            let next2 = forward_history.index_axis(Axis(0), 2);
            pressure_second_derivative_views_into(dst, current, next, next2, inv_dt_sq);
            return Ok(());
        }

        if idx + 1 == nt {
            let prev = forward_history.index_axis(Axis(0), nt - 2);
            let prev2 = forward_history.index_axis(Axis(0), nt - 3);
            pressure_second_derivative_views_into(dst, prev2, prev, current, inv_dt_sq);
            return Ok(());
        }

        let prev = forward_history.index_axis(Axis(0), idx - 1);
        let next = forward_history.index_axis(Axis(0), idx + 1);
        pressure_second_derivative_views_into(dst, prev, current, next, inv_dt_sq);
        Ok(())
    }
}
