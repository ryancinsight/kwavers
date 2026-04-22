//! Wavefield modeling for FWI
//! Based on Virieux (1986): "P-SV wave propagation in heterogeneous media"

use crate::core::error::{KwaversError, KwaversResult, PhysicsError, ValidationError};
use crate::solver::inverse::acoustic_fwi::accumulate_signed_correlation;
use ndarray::{s, Array1, Array2, Array3, Array4, Axis};

/// Configuration for wavefield modeling
#[derive(Debug, Clone)]
pub struct WavefieldConfig {
    /// Grid spacing \[m\]
    pub dx: f64,
    /// Time step \[s\]
    pub dt: f64,
    /// Maximum simulation time \[s\]
    pub max_time: f64,
    /// Peak frequency for source wavelet \[Hz\]
    pub peak_frequency: f64,
    /// Source position (i, j, k)
    pub source_position: Option<(usize, usize, usize)>,
    /// Receiver positions
    pub receivers: Vec<(usize, usize, usize)>,
}

#[derive(Debug)]
struct ForwardCheckpoint {
    previous: Array3<f64>,
    current: Array3<f64>,
}

#[derive(Debug)]
struct ForwardReplayCache {
    nt: usize,
    stride: usize,
    checkpoints: Vec<ForwardCheckpoint>,
}

impl Default for WavefieldConfig {
    fn default() -> Self {
        Self {
            dx: 0.001,           // 1mm
            dt: 1e-6,            // 1 microsecond
            max_time: 0.01,      // 10ms
            peak_frequency: 1e6, // 1 MHz
            source_position: None,
            receivers: Vec::new(),
        }
    }
}

/// Wavefield modeling for forward and adjoint problems
#[derive(Debug)]
pub struct WavefieldModeler {
    /// Configuration
    config: WavefieldConfig,
    /// Sparse forward checkpoints for exact replay-based adjoint accumulation
    forward_replay: Option<ForwardReplayCache>,
    /// Final forward wavefield snapshot for diagnostics
    last_forward_wavefield: Option<Array3<f64>>,
    /// PML boundary width
    pml_width: usize,
}

impl Default for WavefieldModeler {
    fn default() -> Self {
        Self::new()
    }
}

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

    /// Compute the checkpoint stride that minimizes the peak memory bound.
    ///
    /// If `m` is the checkpoint spacing, the replay cache stores
    /// `ceil(nt / m)` checkpoint pairs while the replay buffer stores at most
    /// `m` forward snapshots at a time. The peak bound is proportional to
    /// `m + nt / m`, which is minimized at `m = sqrt(nt)`.
    #[inline]
    fn checkpoint_stride(nt: usize) -> usize {
        ((nt as f64).sqrt().ceil() as usize).max(1)
    }

    fn advance_forward_state(
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

                    next[[i, j, k]] = 2.0 * u_c - u_p + dt * dt * v2_local * laplacian;
                }
            }
        }

        self.apply_pml_boundaries(next, pml_damping, nx, ny, nz, self.pml_width);
    }

    fn replay_forward_segment(
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

            self.advance_forward_state(&u_curr, &u_prev, &mut u_next, &v2, dt, dx2, pml_damping);

            std::mem::swap(&mut u_prev, &mut u_curr);
            std::mem::swap(&mut u_curr, &mut u_next);
        }
    }

    fn validate_geometry(&self, shape: (usize, usize, usize)) -> KwaversResult<()> {
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

    fn validate_timestep(&self, velocity_model: &Array3<f64>) -> KwaversResult<f64> {
        if self.config.dt <= 0.0 || !self.config.dt.is_finite() {
            return Err(KwaversError::Validation(
                ValidationError::ConstraintViolation {
                    message: "WavefieldConfig.dt must be positive and finite".to_string(),
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

    /// Forward wavefield modeling
    /// Solves: (1/v²)∂²u/∂t² - ∇²u = f
    pub fn forward_model(&mut self, velocity_model: &Array3<f64>) -> KwaversResult<Array2<f64>> {
        let (nx, ny, nz) = velocity_model.dim();
        self.validate_geometry((nx, ny, nz))?;
        let dt = self.validate_timestep(velocity_model)?;
        let nt = (self.config.max_time / dt) as usize;
        if nt == 0 {
            return Err(KwaversError::Validation(
                ValidationError::ConstraintViolation {
                    message: "WavefieldConfig.max_time must span at least one timestep".to_string(),
                },
            ));
        }

        // Initialize wavefield arrays (pressure and particle velocity)
        let mut u_curr = Array3::zeros((nx, ny, nz));
        let mut u_prev = Array3::zeros((nx, ny, nz));
        let mut u_next = Array3::zeros((nx, ny, nz));

        // Initialize PML absorbing boundaries
        let pml_thickness = self.pml_width;
        let pml_damping = self.compute_pml_profile(pml_thickness, nx.max(ny).max(nz));

        // Precompute velocity squared for efficiency
        let v2 = velocity_model.mapv(|v| v * v);
        let dx2 = self.config.dx * self.config.dx;
        let checkpoint_stride = Self::checkpoint_stride(nt);
        let mut checkpoints = Vec::with_capacity((nt + checkpoint_stride - 1) / checkpoint_stride);

        // Storage for receiver data
        let mut seismogram = Array2::zeros((self.config.receivers.len(), nt));

        // Time stepping loop with second-order finite difference
        for t_idx in 0..nt {
            if t_idx % checkpoint_stride == 0 {
                checkpoints.push(ForwardCheckpoint {
                    previous: u_prev.clone(),
                    current: u_curr.clone(),
                });
            }

            let t = t_idx as f64 * dt;

            // Apply source wavelet at source position
            if let Some(src_pos) = self.config.source_position {
                let wavelet = self.ricker_wavelet(t, self.config.peak_frequency);
                u_curr[[src_pos.0, src_pos.1, src_pos.2]] += wavelet;
            }

            // Record at receiver locations
            for (r_idx, &(rx, ry, rz)) in self.config.receivers.iter().enumerate() {
                seismogram[[r_idx, t_idx]] = u_curr[[rx, ry, rz]];
            }

            self.advance_forward_state(&u_curr, &u_prev, &mut u_next, &v2, dt, dx2, &pml_damping);

            // Rotate arrays for next timestep
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
    /// If the checkpoint stride is `m`, the replay cache stores
    /// `ceil(nt / m)` checkpoint pairs and the replay buffer stores at most
    /// `m` forward snapshots at a time. The peak bound is proportional to
    /// `m + nt / m`, which is minimized at `m = ceil(sqrt(nt))`.
    ///
    /// ## Proof sketch
    /// The explicit finite-difference update is a pure recurrence with no hidden
    /// state. By induction on the segment index, replay from a checkpoint uses
    /// the same initial conditions and the same update operator as the original
    /// forward solve, so every reconstructed slice equals the corresponding
    /// full-history slice. Since the gradient is a discrete sum over matched
    /// forward and adjoint slices, the replayed gradient is identical.
    ///
    /// Solves backward in time with residual as source.
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

        let expected_checkpoints = (nt + replay_cache.stride - 1) / replay_cache.stride;
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

        // Initialize adjoint wavefield
        let mut adj_curr = Array3::zeros((nx, ny, nz));
        let mut adj_prev = Array3::zeros((nx, ny, nz));
        let mut adj_next = Array3::zeros((nx, ny, nz));

        // Accumulate gradient
        let mut gradient = Array3::zeros((nx, ny, nz));
        let mut segment_history = Array4::zeros((replay_cache.stride, nx, ny, nz));

        // Time stepping backward over replayed segments.
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

                // Inject adjoint source at receiver locations
                for (r_idx, &(rx, ry, rz)) in self.config.receivers.iter().enumerate() {
                    adj_curr[[rx, ry, rz]] += adjoint_source[[r_idx, t_idx]];
                }

                // Update adjoint wavefield (same wave equation, backward in time)
                self.advance_forward_state(
                    &adj_curr,
                    &adj_prev,
                    &mut adj_next,
                    &v2,
                    dt,
                    dx2,
                    &pml_damping,
                );

                // Accumulate gradient: correlation of the matched forward snapshot.
                let current_forward = segment_history.index_axis(Axis(0), local_idx);
                accumulate_signed_correlation(&mut gradient, current_forward, adj_curr.view(), dt)?;

                // Rotate arrays
                std::mem::swap(&mut adj_prev, &mut adj_curr);
                std::mem::swap(&mut adj_curr, &mut adj_next);
            }
        }

        Ok(gradient)
    }

    /// Get the final forward wavefield snapshot.
    pub fn get_forward_wavefield(&self) -> KwaversResult<Array3<f64>> {
        self.last_forward_wavefield
            .as_ref()
            .cloned()
            .ok_or_else(|| {
                crate::core::error::KwaversError::InvalidInput(
                    "Forward wavefield not computed".to_string(),
                )
            })
    }

    /// Apply PML boundary conditions
    /// Based on Berenger (1994): "A perfectly matched layer for the absorption of electromagnetic waves"
    /// Journal of Computational Physics, 114(2), 185-200
    #[allow(dead_code)]
    fn apply_pml(&self, wavefield: &mut Array3<f64>) {
        let (nx, ny, nz) = wavefield.dim();
        let width = self.pml_width;
        if width == 0 {
            return;
        }
        let limit = width.min(nx).min(ny).min(nz);

        // PML parameters following Collino & Tsogka (2001)
        let reflection_coeff: f64 = 1e-6; // Target reflection coefficient
        let pml_order = 2.0; // Polynomial order for damping profile
        let max_velocity = 4000.0; // Maximum velocity in model (m/s)

        // Maximum damping coefficient
        let max_damping = -(pml_order + 1.0) * max_velocity * reflection_coeff.ln()
            / (2.0 * width as f64 * 0.001); // Assuming 1mm grid spacing

        // Apply damping in boundary regions with polynomial profile
        for i in 0..limit {
            // Damping profile: d(x) = d_max * (x/L)^n
            let xi = (width - i) as f64 / width as f64;
            let damping = max_damping * xi.powf(pml_order);

            // X boundaries
            for j in 0..ny {
                for k in 0..nz {
                    wavefield[[i, j, k]] *= (-damping).exp();
                    wavefield[[nx - 1 - i, j, k]] *= (-damping).exp();
                }
            }

            // Y boundaries
            for ii in 0..nx {
                for k in 0..nz {
                    wavefield[[ii, i, k]] *= (-damping).exp();
                    wavefield[[ii, ny - 1 - i, k]] *= (-damping).exp();
                }
            }

            // Z boundaries
            for ii in 0..nx {
                for j in 0..ny {
                    wavefield[[ii, j, i]] *= (-damping).exp();
                    wavefield[[ii, j, nz - 1 - i]] *= (-damping).exp();
                }
            }
        }
    }

    /// Apply finite difference stencil for wave equation
    /// 4th order accurate in space, 2nd order in time
    #[allow(dead_code)]
    fn apply_fd_stencil(
        &self,
        current: &Array3<f64>,
        previous: &Array3<f64>,
        velocity: &Array3<f64>,
        dt: f64,
        dx: f64,
    ) -> Array3<f64> {
        let (nx, ny, nz) = current.dim();
        let mut next = Array3::zeros((nx, ny, nz));

        // Finite difference coefficients for 4th order
        const C0: f64 = -5.0 / 2.0;
        const C1: f64 = 4.0 / 3.0;
        const C2: f64 = -1.0 / 12.0;

        // Apply stencil avoiding boundaries (handled by PML)
        for i in 2..nx.saturating_sub(2) {
            for j in 2..ny.saturating_sub(2) {
                for k in 2..nz.saturating_sub(2) {
                    let laplacian = C0 * current[[i, j, k]]
                        + C1 * (current[[i + 1, j, k]]
                            + current[[i - 1, j, k]]
                            + current[[i, j + 1, k]]
                            + current[[i, j - 1, k]]
                            + current[[i, j, k + 1]]
                            + current[[i, j, k - 1]])
                        + C2 * (current[[i + 2, j, k]]
                            + current[[i - 2, j, k]]
                            + current[[i, j + 2, k]]
                            + current[[i, j - 2, k]]
                            + current[[i, j, k + 2]]
                            + current[[i, j, k - 2]]);

                    let v2 = velocity[[i, j, k]] * velocity[[i, j, k]];
                    next[[i, j, k]] = 2.0 * current[[i, j, k]] - previous[[i, j, k]]
                        + dt * dt * v2 * laplacian / (dx * dx);
                }
            }
        }

        next
    }

    /// Calculate stable timestep using CFL condition
    fn calculate_stable_timestep(&self, velocity_model: &Array3<f64>) -> KwaversResult<f64> {
        let v_max = velocity_model.iter().copied().fold(0.0f64, f64::max);
        if !v_max.is_finite() || v_max <= 0.0 {
            return Err(KwaversError::Validation(
                ValidationError::ConstraintViolation {
                    message: "Velocity model must contain a finite, strictly positive maximum"
                        .to_string(),
                },
            ));
        }
        let cfl = 0.5; // CFL number for stability
        Ok(cfl * self.config.dx / v_max)
    }

    /// Compute PML damping profile
    fn compute_pml_profile(&self, thickness: usize, _max_dim: usize) -> Array1<f64> {
        if thickness == 0 {
            return Array1::zeros(0);
        }
        let mut profile = Array1::zeros(thickness);
        let reflection_coeff: f64 = 1e-6;
        let pml_order = 2.0;
        let max_velocity = 4000.0;

        let max_damping = -(pml_order + 1.0) * max_velocity * reflection_coeff.ln()
            / (2.0 * thickness as f64 * self.config.dx);

        for i in 0..thickness {
            let xi = (thickness - i) as f64 / thickness as f64;
            profile[i] = max_damping * xi.powf(pml_order);
        }

        profile
    }

    /// Apply PML boundaries to wavefield
    fn apply_pml_boundaries(
        &self,
        wavefield: &mut Array3<f64>,
        damping: &Array1<f64>,
        nx: usize,
        ny: usize,
        nz: usize,
        thickness: usize,
    ) {
        let limit = thickness.min(damping.len()).min(nx).min(ny).min(nz);

        for i in 0..limit {
            let d = (-damping[i]).exp();

            // X boundaries
            for j in 0..ny {
                for k in 0..nz {
                    wavefield[[i, j, k]] *= d;
                    if nx > i {
                        wavefield[[nx - 1 - i, j, k]] *= d;
                    }
                }
            }

            // Y boundaries
            for ii in 0..nx {
                for k in 0..nz {
                    wavefield[[ii, i, k]] *= d;
                    if ny > i {
                        wavefield[[ii, ny - 1 - i, k]] *= d;
                    }
                }
            }

            // Z boundaries
            for ii in 0..nx {
                for j in 0..ny {
                    wavefield[[ii, j, i]] *= d;
                    if nz > i {
                        wavefield[[ii, j, nz - 1 - i]] *= d;
                    }
                }
            }
        }
    }

    /// Compute 7-point stencil Laplacian (2nd order accurate)
    /// ∇²u ≈ (u_{i+1} + u_{i-1} + u_{j+1} + u_{j-1} + u_{k+1} + u_{k-1} - 6u_{i,j,k}) / dx²
    ///
    /// References:
    /// - LeVeque (2007): "Finite Difference Methods for Ordinary and Partial Differential Equations"
    fn compute_laplacian_stencil_7pt(
        &self,
        field: &Array3<f64>,
        i: usize,
        j: usize,
        k: usize,
        dx2: f64,
    ) -> f64 {
        let center = field[[i, j, k]];
        let neighbors_sum = field[[i + 1, j, k]]
            + field[[i - 1, j, k]]
            + field[[i, j + 1, k]]
            + field[[i, j - 1, k]]
            + field[[i, j, k + 1]]
            + field[[i, j, k - 1]];

        (neighbors_sum - 6.0 * center) / dx2
    }

    /// Generate Ricker wavelet source
    fn ricker_wavelet(&self, t: f64, f_peak: f64) -> f64 {
        let t0 = 1.0 / f_peak;
        let a = std::f64::consts::PI * f_peak * (t - t0);
        (1.0 - 2.0 * a * a) * (-a * a).exp()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array3;

    #[test]
    fn test_forward_model_rejects_invalid_velocity() {
        let mut modeler = WavefieldModeler::new();
        let velocity_model = Array3::zeros((4, 4, 4));

        let err = modeler
            .forward_model(&velocity_model)
            .expect_err("zero velocity must fail");

        assert!(format!("{err:?}").contains("strictly positive maximum"));
    }

    #[test]
    fn test_forward_model_rejects_out_of_bounds_geometry() {
        let mut modeler = WavefieldModeler::with_config(WavefieldConfig {
            source_position: Some((4, 0, 0)),
            receivers: vec![(0, 0, 0)],
            ..WavefieldConfig::default()
        });
        let velocity_model = Array3::from_elem((4, 4, 4), 1500.0);

        let err = modeler
            .forward_model(&velocity_model)
            .expect_err("out-of-bounds source must fail");

        assert!(format!("{err:?}").contains("Source position out of bounds"));
    }

    #[test]
    fn test_adjoint_model_uses_checkpointed_replay() {
        let mut modeler = WavefieldModeler::with_config(WavefieldConfig {
            dx: 2.0,
            dt: 1.0,
            max_time: 3.0,
            peak_frequency: 1.0,
            source_position: Some((0, 0, 0)),
            receivers: vec![(0, 0, 0)],
        });
        modeler.pml_width = 0;

        let velocity_model = Array3::from_elem((4, 4, 4), 1.0);
        let synthetic = modeler
            .forward_model(&velocity_model)
            .expect("forward model must succeed");
        assert_eq!(synthetic.shape(), &[1, 3]);
        assert!((synthetic[[0, 1]] - 1.0).abs() < f64::EPSILON);
        let replay_cache = modeler
            .forward_replay
            .as_ref()
            .expect("forward replay cache must exist");
        assert_eq!(replay_cache.stride, 2);
        assert_eq!(replay_cache.checkpoints.len(), 2);
        assert!(replay_cache.checkpoints[0]
            .current
            .iter()
            .all(|&v| v.abs() < f64::EPSILON));

        let residual = synthetic.clone();
        let gradient = modeler
            .adjoint_model(&velocity_model, &residual)
            .expect("adjoint model must succeed");

        let expected = (0..3)
            .map(|t| {
                let tau = t as f64 - 1.0;
                let a = std::f64::consts::PI * tau;
                let value = (1.0 - 2.0 * a * a) * (-a * a).exp();
                value * value
            })
            .sum::<f64>();

        assert!((gradient[[0, 0, 0]] - expected).abs() < 1e-12);
        assert_eq!(gradient.sum(), expected);
        assert!(modeler.forward_replay.is_none());
    }
}
