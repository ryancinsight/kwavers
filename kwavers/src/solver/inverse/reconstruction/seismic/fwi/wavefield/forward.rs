use crate::core::error::{KwaversError, KwaversResult, ValidationError};
use ndarray::{Array2, Array3};

use super::{ForwardCheckpoint, ForwardReplayCache, WavefieldModeler};

impl WavefieldModeler {
    /// Forward wavefield modeling — solves (1/v²)∂²u/∂t² − ∇²u = f.
    /// # Errors
    /// - Returns [`KwaversError::Validation`] if the precondition for a Validation-class constraint is violated.
    /// - Propagates any [`KwaversError`] returned by called functions.
    ///
    pub fn forward_model(&mut self, velocity_model: &Array3<f64>) -> KwaversResult<Array2<f64>> {
        let (nx, ny, nz) = velocity_model.dim();
        self.validate_geometry((nx, ny, nz))?;
        let dt = self.validate_timestep(velocity_model)?;
        let nt = (self.config.max_time / dt) as usize;
        if nt == 0 {
            return Err(KwaversError::Validation(
                ValidationError::ConstraintViolation {
                    message: "WavefieldConfig.max_time must span at least one timestep".to_owned(),
                },
            ));
        }

        let mut u_curr = Array3::zeros((nx, ny, nz));
        let mut u_prev = Array3::zeros((nx, ny, nz));
        let mut u_next = Array3::zeros((nx, ny, nz));

        let pml_thickness = self.pml_width;
        let pml_damping = self.compute_pml_profile(pml_thickness, nx.max(ny).max(nz));

        let v2 = velocity_model.mapv(|v| v * v);
        let dx2 = self.config.dx * self.config.dx;
        let checkpoint_stride = Self::checkpoint_stride(nt);
        let mut checkpoints = Vec::with_capacity(nt.div_ceil(checkpoint_stride));

        let mut seismogram = Array2::zeros((self.config.receivers.len(), nt));

        for t_idx in 0..nt {
            if t_idx % checkpoint_stride == 0 {
                checkpoints.push(ForwardCheckpoint {
                    previous: u_prev.clone(),
                    current: u_curr.clone(),
                });
            }

            let t = t_idx as f64 * dt;

            if let Some(src_pos) = self.config.source_position {
                let wavelet = self.ricker_wavelet(t, self.config.peak_frequency);
                u_curr[[src_pos.0, src_pos.1, src_pos.2]] += wavelet;
            }

            for (r_idx, &(rx, ry, rz)) in self.config.receivers.iter().enumerate() {
                seismogram[[r_idx, t_idx]] = u_curr[[rx, ry, rz]];
            }

            self.advance_forward_state(&u_curr, &u_prev, &mut u_next, &v2, dt, dx2, &pml_damping);

            std::mem::swap(&mut u_prev, &mut u_curr);
            std::mem::swap(&mut u_curr, &mut u_next);
        }

        self.forward_replay = Some(ForwardReplayCache {
            nt,
            stride: checkpoint_stride,
            checkpoints,
        });
        self.last_forward_wavefield = Some(u_curr);

        Ok(seismogram)
    }

    /// Get the final forward wavefield snapshot.
    /// # Errors
    /// - Returns [`Err`] if an internal constraint is violated.
    ///
    pub fn get_forward_wavefield(&self) -> KwaversResult<Array3<f64>> {
        self.last_forward_wavefield
            .as_ref()
            .cloned()
            .ok_or_else(|| {
                crate::core::error::KwaversError::InvalidInput(
                    "Forward wavefield not computed".to_owned(),
                )
            })
    }
}
