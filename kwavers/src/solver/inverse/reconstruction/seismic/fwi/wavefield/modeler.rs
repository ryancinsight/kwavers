use crate::core::error::{KwaversError, KwaversResult, ValidationError};
use ndarray::{s, Array1, Array3, Array4};

use super::{ForwardCheckpoint, WavefieldConfig, WavefieldModeler};

impl WavefieldModeler {
    #[must_use]
    pub fn new() -> Self {
        Self {
            config: WavefieldConfig::default(),
            forward_replay: None,
            last_forward_wavefield: None,
            pml_width: 20,
        }
    }

    #[must_use]
    pub fn with_config(config: WavefieldConfig) -> Self {
        Self {
            config,
            forward_replay: None,
            last_forward_wavefield: None,
            pml_width: 20,
        }
    }

    /// Compute checkpoint stride minimizing `m + nt/m` at `m = sqrt(nt)`.
    #[inline]
    pub(super) fn checkpoint_stride(nt: usize) -> usize {
        ((nt as f64).sqrt().ceil() as usize).max(1)
    }

    #[allow(clippy::too_many_arguments)]
    pub(super) fn advance_forward_state(
        &self,
        current: &Array3<f64>,
        previous: &Array3<f64>,
        next: &mut Array3<f64>,
        v2: &Array3<f64>,
        dt: f64,
        dx2: f64,
        pml_damping: &Array1<f64>,
    ) {
        let (nx, ny, nz) = current.dim();

        for i in 2..nx.saturating_sub(2) {
            for j in 2..ny.saturating_sub(2) {
                for k in 2..nz.saturating_sub(2) {
                    let laplacian = self.compute_laplacian_stencil_7pt(current, i, j, k, dx2);
                    let u_c = current[[i, j, k]];
                    let u_p = previous[[i, j, k]];
                    let v2_local = v2[[i, j, k]];

                    next[[i, j, k]] =
                        (dt * dt * v2_local).mul_add(laplacian, 2.0f64.mul_add(u_c, -u_p));
                }
            }
        }

        self.apply_pml_boundaries(next, pml_damping, nx, ny, nz, self.pml_width);
    }

    pub(super) fn replay_forward_segment(
        &self,
        v2: &Array3<f64>,
        checkpoint: &ForwardCheckpoint,
        start_t: usize,
        end_t: usize,
        segment_history: &mut Array4<f64>,
        pml_damping: &Array1<f64>,
    ) {
        let dt = self.config.dt;
        let dx2 = self.config.dx * self.config.dx;
        let mut u_prev = checkpoint.previous.clone();
        let mut u_curr = checkpoint.current.clone();
        let mut u_next = Array3::zeros(v2.dim());

        for (local_idx, t_idx) in (start_t..end_t).enumerate() {
            if let Some(src_pos) = self.config.source_position {
                let wavelet = self.ricker_wavelet(t_idx as f64 * dt, self.config.peak_frequency);
                u_curr[[src_pos.0, src_pos.1, src_pos.2]] += wavelet;
            }

            segment_history
                .slice_mut(s![local_idx, .., .., ..])
                .assign(&u_curr);

            self.advance_forward_state(&u_curr, &u_prev, &mut u_next, v2, dt, dx2, pml_damping);

            std::mem::swap(&mut u_prev, &mut u_curr);
            std::mem::swap(&mut u_curr, &mut u_next);
        }
    }
    /// Validate geometry.
    /// # Errors
    /// - Returns [`KwaversError::Validation`] if the precondition for a Validation-class constraint is violated.
    /// - Propagates any [`KwaversError`] returned by called functions.
    ///
    pub(super) fn validate_geometry(&self, shape: (usize, usize, usize)) -> KwaversResult<()> {
        let (nx, ny, nz) = shape;

        if let Some((i, j, k)) = self.config.source_position {
            if i >= nx || j >= ny || k >= nz {
                return Err(KwaversError::Validation(
                    ValidationError::ConstraintViolation {
                        message: format!(
                            "Source position out of bounds: ({i}, {j}, {k}) for shape {shape:?}"
                        ),
                    },
                ));
            }
        }

        if self
            .config
            .receivers
            .iter()
            .any(|&(i, j, k)| i >= nx || j >= ny || k >= nz)
        {
            return Err(KwaversError::Validation(
                ValidationError::ConstraintViolation {
                    message: format!("Receiver position out of bounds for shape {shape:?}"),
                },
            ));
        }

        Ok(())
    }
    /// Validate timestep.
    /// # Errors
    /// - Returns [`KwaversError::Validation`] if the precondition for a Validation-class constraint is violated.
    /// - Propagates any [`KwaversError`] returned by called functions.
    ///
    pub(super) fn validate_timestep(&self, velocity_model: &Array3<f64>) -> KwaversResult<f64> {
        if self.config.dt <= 0.0 || !self.config.dt.is_finite() {
            return Err(KwaversError::Validation(
                ValidationError::ConstraintViolation {
                    message: "WavefieldConfig.dt must be positive and finite".to_owned(),
                },
            ));
        }

        let stable_dt = self.calculate_stable_timestep(velocity_model)?;
        if self.config.dt > stable_dt {
            return Err(KwaversError::Validation(
                ValidationError::ConstraintViolation {
                    message: format!(
                        "Configured dt {:.6e} exceeds CFL bound {:.6e}",
                        self.config.dt, stable_dt
                    ),
                },
            ));
        }

        Ok(self.config.dt)
    }
}
