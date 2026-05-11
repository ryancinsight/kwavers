//! Physical constraints: CFL validation, model clamping, pressure second-derivative.

use super::FwiProcessor;
use crate::core::error::{KwaversError, KwaversResult, ValidationError};
use crate::domain::grid::Grid;
use ndarray::{Array3, Array4, Axis, Zip};

impl FwiProcessor {
    /// Apply physical constraints to velocity model.
    /// # Errors
    /// - Returns [`Err`] if an internal constraint is violated.
    ///
    pub(super) fn apply_model_constraints(&self, model: &mut Array3<f64>) {
        use crate::core::constants::SOUND_SPEED_WATER;
        let min_velocity = SOUND_SPEED_WATER * 0.5; // 750 m/s
        let max_velocity = SOUND_SPEED_WATER * 4.0; // 6000 m/s
        model.par_mapv_inplace(|v| v.clamp(min_velocity, max_velocity));
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
                    message: "FWI requires at least 3 time samples to form a second derivative".to_owned(),
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
                    message: "FWI requires a finite, strictly positive sound speed model".to_owned(),
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

        if idx == 0 {
            let next = forward_history.index_axis(Axis(0), 1);
            let next2 = forward_history.index_axis(Axis(0), 2);
            Zip::from(dst)
                .and(&current)
                .and(&next)
                .and(&next2)
                .par_for_each(|d, &p0, &p1, &p2| {
                    *d = (2.0f64.mul_add(-p1, p0) + p2) * inv_dt_sq;
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
                    *d = (2.0f64.mul_add(-p1, p0) + p2) * inv_dt_sq;
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
                *d = (2.0f64.mul_add(-p1, p0) + p2) * inv_dt_sq;
            });
        Ok(())
    }
}
