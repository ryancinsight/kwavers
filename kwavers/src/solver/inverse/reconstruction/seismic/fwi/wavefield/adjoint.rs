use crate::core::error::{KwaversError, KwaversResult, PhysicsError, ValidationError};
use crate::solver::inverse::acoustic_fwi::accumulate_signed_correlation;
use ndarray::{Array3, Array4, Axis};

use super::WavefieldModeler;

impl WavefieldModeler {
    /// Adjoint-state gradient accumulation.
    ///
    /// ## Theorem
    /// Let the discrete forward recurrence be deterministic and let checkpoints
    /// store `(u_{k-1}, u_k)` at times `k = q m` for stride `m`.
    /// Replaying each segment from its checkpoint with the same source term,
    /// grid spacing, PML profile, and velocity model reconstructs the exact
    /// forward snapshots pointwise. The adjoint imaging condition computed from
    /// these replayed snapshots is therefore identical to the full-history
    /// formulation.
    ///
    /// ## Memory bound
    /// Peak storage ∝ `m + nt/m`, minimized at `m = ceil(sqrt(nt))`.
    ///
    /// ## Proof sketch
    /// The explicit finite-difference update is a pure recurrence with no hidden
    /// state. By induction, replay from a checkpoint uses the same initial
    /// conditions and update operator, so every reconstructed slice equals the
    /// corresponding full-history slice.
    pub fn adjoint_model(
        &mut self,
        velocity_model: &Array3<f64>,
        adjoint_source: &Array2<f64>,
    ) -> KwaversResult<Array3<f64>> {
        let dt = self.config.dt;
        if dt <= 0.0 || !dt.is_finite() {
            return Err(KwaversError::Validation(
                ValidationError::ConstraintViolation {
                    message: "WavefieldConfig.dt must be positive and finite".to_string(),
                },
            ));
        }
        let nt = adjoint_source.shape()[1];
        if nt == 0 {
            return Err(KwaversError::Validation(
                ValidationError::ConstraintViolation {
                    message: "Adjoint source must contain at least one timestep".to_string(),
                },
            ));
        }

        if adjoint_source.shape()[0] != self.config.receivers.len() {
            return Err(KwaversError::Validation(
                ValidationError::ConstraintViolation {
                    message: format!(
                        "Adjoint source receiver count mismatch: expected {}, got {}",
                        self.config.receivers.len(),
                        adjoint_source.shape()[0]
                    ),
                },
            ));
        }

        let replay_cache = self.forward_replay.as_ref().ok_or_else(|| {
            KwaversError::Physics(PhysicsError::InvalidState {
                field: "forward_replay".to_string(),
                value: "None".to_string(),
                reason: "Forward replay cache must be computed before adjoint modeling".to_string(),
            })
        })?;

        let (nx, ny, nz) = velocity_model.dim();
        self.validate_geometry((nx, ny, nz))?;

        if replay_cache.nt != nt {
            return Err(KwaversError::Validation(
                ValidationError::ConstraintViolation {
                    message: format!(
                        "Adjoint source time length mismatch: expected {}, got {}",
                        replay_cache.nt, nt
                    ),
                },
            ));
        }

        let expected_checkpoints = nt.div_ceil(replay_cache.stride);
        if replay_cache.checkpoints.len() != expected_checkpoints {
            return Err(KwaversError::Validation(
                ValidationError::ConstraintViolation {
                    message: "Forward replay cache is incomplete".to_string(),
                },
            ));
        }

        let replay_cache = self.forward_replay.take().ok_or_else(|| {
            KwaversError::Physics(PhysicsError::InvalidState {
                field: "forward_replay".to_string(),
                value: "None".to_string(),
                reason: "Forward replay cache must be computed before adjoint modeling".to_string(),
            })
        })?;

        let v2 = velocity_model.mapv(|v| v * v);
        let dx2 = self.config.dx * self.config.dx;
        let pml_thickness = self.pml_width;
        let pml_damping = self.compute_pml_profile(pml_thickness, nx.max(ny).max(nz));

        let mut adj_curr = Array3::zeros((nx, ny, nz));
        let mut adj_prev = Array3::zeros((nx, ny, nz));
        let mut adj_next = Array3::zeros((nx, ny, nz));
        let mut gradient = Array3::zeros((nx, ny, nz));
        let mut segment_history = Array4::zeros((replay_cache.stride, nx, ny, nz));

        for checkpoint_idx in (0..replay_cache.checkpoints.len()).rev() {
            let start_t = checkpoint_idx * replay_cache.stride;
            let end_t = (start_t + replay_cache.stride).min(nt);
            let segment_len = end_t - start_t;
            let checkpoint = &replay_cache.checkpoints[checkpoint_idx];

            self.replay_forward_segment(
                &v2,
                checkpoint,
                start_t,
                end_t,
                &mut segment_history,
                &pml_damping,
            );

            for local_idx in (0..segment_len).rev() {
                let t_idx = start_t + local_idx;

                for (r_idx, &(rx, ry, rz)) in self.config.receivers.iter().enumerate() {
                    adj_curr[[rx, ry, rz]] += adjoint_source[[r_idx, t_idx]];
                }

                self.advance_forward_state(
                    &adj_curr,
                    &adj_prev,
                    &mut adj_next,
                    &v2,
                    dt,
                    dx2,
                    &pml_damping,
                );

                let current_forward = segment_history.index_axis(Axis(0), local_idx);
                accumulate_signed_correlation(&mut gradient, current_forward, adj_curr.view(), dt)?;

                std::mem::swap(&mut adj_prev, &mut adj_curr);
                std::mem::swap(&mut adj_curr, &mut adj_next);
            }
        }

        Ok(gradient)
    }
}

// Re-export ndarray::Array2 in scope for this file only
use ndarray::Array2;
