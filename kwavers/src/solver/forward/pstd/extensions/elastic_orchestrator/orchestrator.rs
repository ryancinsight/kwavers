//! `ElasticPstdOrchestrator` — leapfrog stress-velocity time loop.
//!
//! # Algorithm (leapfrog stress-velocity)
//!
//! Per step:
//!
//! 1. **Velocity-source injection** — inject the per-cell signal into
//!    `vx[mask] += s_n` (additive) or `vx[mask] = s_n` (Dirichlet),
//!    matching KWave.jl `pstd_elastic_2d::Additive` / `Dirichlet` semantics.
//! 2. **Forward FFT** of `(vx, vy, vz)` → spectral velocity field.
//! 3. **Stress update** via [`crate::solver::forward::pstd::extensions::PstdElasticPlugin::apply_stress_update_in_place`]:
//!    `σ̃(t+dt) = σ̃(t) + dt · C : ε̃(t+dt/2)`.
//! 4. **Velocity update** via [`crate::solver::forward::pstd::extensions::PstdElasticPlugin::apply_velocity_update_in_place`]:
//!    `ṽ(t+dt) = ṽ(t) + (dt/ρ) · ∇·σ̃(t+dt)`.
//! 5. **Inverse FFT** of the new spectral velocity → real-space `(vx, vy, vz)`.
//! 6. **Sensor recording** — copy the per-component values at masked grid
//!    points into the recorder matrices.

use super::super::PstdElasticPlugin;
use super::types::{ElasticPstdMedium, ElasticPstdSensorData, ElasticPstdVelocitySource};
use crate::core::error::KwaversResult;
use crate::domain::grid::Grid;
use crate::physics::acoustics::mechanics::elastic_wave::{
    fields::VelocityFields,
    parameters::{StressUpdateParams, VelocityUpdateParams},
    spectral_fields::{SpectralStressFields, SpectralVelocityFields},
};
use ndarray::{Array2, Array3};
use num_complex::Complex;

/// Pre-built complex spectral derivative operators for the staggered-grid
/// PSTD scheme used by KWave.jl `pstd_elastic_2d` (Treeby & Cox 2010, eq.
/// 16). For each axis α, the operator stored here is
///
/// ```text
///   D_α^±[k_α] = i · k_α · exp(± i · k_α · Δα / 2)
/// ```
///
/// so the plugin's spectral derivative collapses to `D_α^± · F_α` instead
/// of the collocated `i · k_α · F_α`. The `+` set is used for `∇·σ`
/// (velocity update — sampled at the velocity grid), the `−` set for
/// `∇v` (stress update — sampled at the stress grid). Without the shift
/// the orchestrator runs a collocated-grid scheme, which numerically
/// disagrees with KWave.jl at non-trivial wavenumbers (matched-mode
/// peak_ratio sat at 0.13–0.23 instead of ≈ 1.0 prior to this change).
#[derive(Debug)]
struct StaggeredDerivativeOps {
    dkx_pos: Array3<Complex<f64>>,
    dky_pos: Array3<Complex<f64>>,
    dkz_pos: Array3<Complex<f64>>,
    dkx_neg: Array3<Complex<f64>>,
    dky_neg: Array3<Complex<f64>>,
    dkz_neg: Array3<Complex<f64>>,
}

impl StaggeredDerivativeOps {
    fn build(
        kx: &Array3<f64>,
        ky: &Array3<f64>,
        kz: &Array3<f64>,
        dx: f64,
        dy: f64,
        dz: f64,
    ) -> Self {
        let make = |k: &Array3<f64>, d: f64, sign: f64| -> Array3<Complex<f64>> {
            k.mapv(|kv| {
                let shift = Complex::new(0.0, sign * kv * d * 0.5).exp();
                Complex::new(0.0, kv) * shift
            })
        };
        Self {
            dkx_pos: make(kx, dx, 1.0),
            dky_pos: make(ky, dy, 1.0),
            dkz_pos: make(kz, dz, 1.0),
            dkx_neg: make(kx, dx, -1.0),
            dky_neg: make(ky, dy, -1.0),
            dkz_neg: make(kz, dz, -1.0),
        }
    }
}

/// Top-level orchestrator state.
///
/// Owns persistent `(stress, velocity)` spectral fields plus wavenumber
/// arrays. One instance per simulation; `propagate` runs `n_steps`
/// consecutive leapfrog steps while recording sensors.
#[derive(Debug)]
pub struct ElasticPstdOrchestrator {
    pub(super) plugin: PstdElasticPlugin,
    pub(super) medium: ElasticPstdMedium,
    /// Pre-built complex derivative operators with the half-cell k-shift
    /// baked in (matches KWave.jl `pstd_elastic_2d`'s `ddx_k_shift_pos/neg`).
    derivative_ops: StaggeredDerivativeOps,
    pub(super) velocity: VelocityFields,
    pub(super) spectral_stress: SpectralStressFields,
    spectral_stress_next: SpectralStressFields,
    spectral_velocity_next: SpectralVelocityFields,
    pub(super) dt: f64,
    grid_shape: (usize, usize, usize),
    step_index: usize,
}

impl ElasticPstdOrchestrator {
    /// Create a new orchestrator for the given grid + medium + time step.
    ///
    /// The medium's `lame_lambda`, `lame_mu`, `density` arrays must match
    /// the grid's `(nx, ny, nz)` dimensions; this constructor returns
    /// [`Err`] otherwise.
    ///
    /// # Errors
    /// - Returns [`Err`] if any medium array has the wrong shape.
    pub fn new(grid: &Grid, medium: ElasticPstdMedium, dt: f64) -> KwaversResult<Self> {
        let (nx, ny, nz) = grid.dimensions();
        let shape = (nx, ny, nz);

        let validate = |name: &str, arr: &Array3<f64>| -> KwaversResult<()> {
            if arr.dim() != shape {
                return Err(crate::core::error::KwaversError::InvalidInput(format!(
                    "ElasticPstdMedium.{name} shape {:?} must equal grid {:?}",
                    arr.dim(),
                    shape
                )));
            }
            Ok(())
        };
        validate("lame_lambda", &medium.lame_lambda)?;
        validate("lame_mu", &medium.lame_mu)?;
        validate("density", &medium.density)?;

        let kx = wavenumber_axis(nx, grid.dx);
        let ky = wavenumber_axis(ny, grid.dy);
        let kz = wavenumber_axis(nz, grid.dz);
        let derivative_ops =
            StaggeredDerivativeOps::build(&kx, &ky, &kz, grid.dx, grid.dy, grid.dz);

        Ok(Self {
            plugin: PstdElasticPlugin::default(),
            medium,
            derivative_ops,
            velocity: VelocityFields::new(nx, ny, nz),
            spectral_stress: SpectralStressFields::new(nx, ny, nz),
            spectral_stress_next: SpectralStressFields::new(nx, ny, nz),
            spectral_velocity_next: SpectralVelocityFields::new(nx, ny, nz),
            dt,
            grid_shape: shape,
            step_index: 0,
        })
    }

    /// Run `n_steps` time steps, optionally injecting a velocity source and
    /// recording per-component velocity traces at masked sensor positions.
    ///
    /// # Errors
    /// - Returns [`Err`] if source signal length disagrees with `n_steps`,
    ///   or if any provided mask has the wrong shape.
    pub fn propagate(
        &mut self,
        n_steps: usize,
        source: Option<&ElasticPstdVelocitySource>,
        sensor_mask: Option<&Array3<bool>>,
    ) -> KwaversResult<ElasticPstdSensorData> {
        let shape = self.grid_shape;
        if let Some(src) = source {
            validate_source(src, shape, n_steps)?;
        }

        let sensor_indices: Vec<(usize, usize, usize)> = sensor_mask
            .map(|m| {
                m.indexed_iter()
                    .filter_map(|(idx, &b)| b.then_some(idx))
                    .collect()
            })
            .unwrap_or_default();
        let n_sensors = sensor_indices.len();
        let mut sensor_vx = (n_sensors > 0).then(|| Array2::<f64>::zeros((n_sensors, n_steps)));
        let mut sensor_vy = (n_sensors > 0).then(|| Array2::<f64>::zeros((n_sensors, n_steps)));
        let mut sensor_vz = (n_sensors > 0).then(|| Array2::<f64>::zeros((n_sensors, n_steps)));

        for step in 0..n_steps {
            if let Some(src) = source {
                inject_velocity_source(&mut self.velocity, src, step);
            }

            let spectral_v_in = SpectralVelocityFields::from_real(&self.velocity);

            let stress_params = StressUpdateParams {
                vx_fft: &spectral_v_in.vx,
                vy_fft: &spectral_v_in.vy,
                vz_fft: &spectral_v_in.vz,
                txx_fft: &self.spectral_stress.txx,
                tyy_fft: &self.spectral_stress.tyy,
                tzz_fft: &self.spectral_stress.tzz,
                txy_fft: &self.spectral_stress.txy,
                txz_fft: &self.spectral_stress.txz,
                tyz_fft: &self.spectral_stress.tyz,
                // Stress update consumes ∇v sampled at the staggered stress
                // grid → use the negative-shift derivative operator
                // (KWave.jl ddx_k_shift_neg).
                dkx_op: &self.derivative_ops.dkx_neg,
                dky_op: &self.derivative_ops.dky_neg,
                dkz_op: &self.derivative_ops.dkz_neg,
                lame_lambda: &self.medium.lame_lambda,
                lame_mu: &self.medium.lame_mu,
                density: self.medium.density.view(),
                dt: self.dt,
            };
            self.plugin
                .apply_stress_update_in_place(&stress_params, &mut self.spectral_stress_next);

            let velocity_params = VelocityUpdateParams {
                vx_fft: &spectral_v_in.vx,
                vy_fft: &spectral_v_in.vy,
                vz_fft: &spectral_v_in.vz,
                txx_fft: &self.spectral_stress_next.txx,
                tyy_fft: &self.spectral_stress_next.tyy,
                tzz_fft: &self.spectral_stress_next.tzz,
                txy_fft: &self.spectral_stress_next.txy,
                txz_fft: &self.spectral_stress_next.txz,
                tyz_fft: &self.spectral_stress_next.tyz,
                // Velocity update consumes ∇·σ sampled at the staggered
                // velocity grid → use the positive-shift derivative
                // operator (KWave.jl ddx_k_shift_pos).
                dkx_op: &self.derivative_ops.dkx_pos,
                dky_op: &self.derivative_ops.dky_pos,
                dkz_op: &self.derivative_ops.dkz_pos,
                density: self.medium.density.view(),
                dt: self.dt,
            };
            self.plugin
                .apply_velocity_update_in_place(&velocity_params, &mut self.spectral_velocity_next);

            std::mem::swap(&mut self.spectral_stress, &mut self.spectral_stress_next);
            self.velocity = self.spectral_velocity_next.to_real();

            if n_sensors > 0 {
                record_sensors(
                    &self.velocity,
                    &sensor_indices,
                    step,
                    &mut sensor_vx,
                    &mut sensor_vy,
                    &mut sensor_vz,
                );
            }

            self.step_index += 1;
        }

        Ok(ElasticPstdSensorData {
            vx: sensor_vx,
            vy: sensor_vy,
            vz: sensor_vz,
        })
    }

    /// Borrow the current real-space velocity field.
    #[must_use]
    pub fn velocity(&self) -> &VelocityFields {
        &self.velocity
    }

    /// Borrow the current persistent spectral stress tensor.
    ///
    /// Exposed for invariant tests (e.g. `μ ≡ 0 ⇒ shear stress = 0`).
    /// Not part of the simulation API surface — the orchestrator manages
    /// stress internally.
    #[must_use]
    pub fn spectral_stress(&self) -> &SpectralStressFields {
        &self.spectral_stress
    }

    /// Number of steps executed so far.
    #[must_use]
    pub fn step_index(&self) -> usize {
        self.step_index
    }
}

// ─── Private helpers ─────────────────────────────────────────────────────────

fn wavenumber_axis(n: usize, dx: f64) -> Array3<f64> {
    let mut k = Array3::<f64>::zeros((n, 1, 1));
    let dk = 2.0 * std::f64::consts::PI / (n as f64 * dx);
    for i in 0..n / 2 {
        k[[i, 0, 0]] = i as f64 * dk;
    }
    for i in n / 2..n {
        k[[i, 0, 0]] = (i as f64 - n as f64) * dk;
    }
    k
}

fn validate_source(
    src: &ElasticPstdVelocitySource,
    shape: (usize, usize, usize),
    n_steps: usize,
) -> KwaversResult<()> {
    if src.mask.dim() != shape {
        return Err(crate::core::error::KwaversError::InvalidInput(format!(
            "ElasticPstdVelocitySource.mask shape {:?} must equal grid {:?}",
            src.mask.dim(),
            shape
        )));
    }
    for (axis, sig) in [("ux", &src.ux), ("uy", &src.uy), ("uz", &src.uz)] {
        if let Some(s) = sig {
            if s.len() != n_steps {
                return Err(crate::core::error::KwaversError::InvalidInput(format!(
                    "{axis} signal length {} must equal n_steps {n_steps}",
                    s.len()
                )));
            }
        }
    }
    Ok(())
}

fn inject_velocity_source(
    velocity: &mut VelocityFields,
    src: &ElasticPstdVelocitySource,
    step: usize,
) {
    use super::types::ElasticPstdSourceMode;
    let active: Vec<(usize, usize, usize)> = src
        .mask
        .indexed_iter()
        .filter_map(|(idx, &b)| b.then_some(idx))
        .collect();
    let inject = |field: &mut Array3<f64>, sig: &Option<ndarray::Array1<f64>>| {
        if let Some(s) = sig {
            if let Some(&val) = s.as_slice().and_then(|sl| sl.get(step)) {
                for &(i, j, k) in &active {
                    match src.mode {
                        ElasticPstdSourceMode::Additive => field[[i, j, k]] += val,
                        ElasticPstdSourceMode::Dirichlet => field[[i, j, k]] = val,
                    }
                }
            }
        }
    };
    inject(&mut velocity.vx, &src.ux);
    inject(&mut velocity.vy, &src.uy);
    inject(&mut velocity.vz, &src.uz);
}

fn record_sensors(
    velocity: &VelocityFields,
    sensor_indices: &[(usize, usize, usize)],
    step: usize,
    sensor_vx: &mut Option<Array2<f64>>,
    sensor_vy: &mut Option<Array2<f64>>,
    sensor_vz: &mut Option<Array2<f64>>,
) {
    if let Some(ref mut buf) = sensor_vx {
        for (row, &(i, j, k)) in sensor_indices.iter().enumerate() {
            buf[[row, step]] = velocity.vx[[i, j, k]];
        }
    }
    if let Some(ref mut buf) = sensor_vy {
        for (row, &(i, j, k)) in sensor_indices.iter().enumerate() {
            buf[[row, step]] = velocity.vy[[i, j, k]];
        }
    }
    if let Some(ref mut buf) = sensor_vz {
        for (row, &(i, j, k)) in sensor_indices.iter().enumerate() {
            buf[[row, step]] = velocity.vz[[i, j, k]];
        }
    }
}
