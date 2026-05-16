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
use super::kspace::{build_kappa, grid_spacing_from_wavenumber, max_p_wave_speed, wavenumber_axis};
use super::pml::{ElasticPml, ElasticPmlSpec};
use super::source_sensor::{
    inject_velocity_source, inject_velocity_source_subfields, record_sensors, validate_source,
};
use super::split_field_pml::{ElasticSplitFieldPml, SplitFieldState};
use super::split_field_step::{propagate_split_field_step, SpectralOperators, SpectralScratch};
use super::staggered_ops::StaggeredDerivativeOps;
use super::types::{ElasticPstdMedium, ElasticPstdSensorData, ElasticPstdVelocitySource};
use crate::core::error::KwaversResult;
use crate::domain::grid::Grid;
use crate::math::fft::{fft_3d_array_into, ifft_3d_array_into};
use crate::physics::acoustics::mechanics::elastic_wave::{
    fields::VelocityFields,
    parameters::{StressUpdateParams, VelocityUpdateParams},
    spectral_fields::{SpectralStressFields, SpectralVelocityFields},
};
use ndarray::{Array2, Array3};

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
    /// k-space correction factor `sinc(c_ref·dt·|k|/2)` (Tabei et al. 2002).
    /// Pre-computed at construction from the medium's maximum P-wave speed.
    /// Applied to every spectral derivative during stress and velocity updates,
    /// eliminating temporal dispersion and allowing CFL → 1.0 without
    /// first-order phase error (Treeby & Cox 2010, Eq. 18).
    pub(super) kappa: Array3<f64>,
    /// Optional real-space exponential PML applied each step to the
    /// post-IFFT velocity field. `None` ⇒ periodic boundary (pristine
    /// FFT, suitable for short propagation where wraparound is benign);
    /// `Some(_)` ⇒ exponentially-attenuating boundary cells per the
    /// theorem in [`super::pml`].
    pml: Option<ElasticPml>,
    /// Optional Bérenger split-field PML (zero theoretical reflection at
    /// all incidence angles). When `Some`, `propagate` routes each step
    /// through [`propagate_split_field_step`] instead of the standard
    /// leapfrog. Takes precedence over the scalar `pml` field.
    split_pml: Option<ElasticSplitFieldPml>,
    /// Persistent sub-field state for the split-field PML (24 arrays).
    /// `Box`-allocated to avoid stack overflow at large grid sizes.
    split_pml_state: Option<Box<SplitFieldState>>,
    /// Scratch real-space buffer reused across split-field-step derivative
    /// IFFTs. One allocation at construction; zero allocations per step.
    scratch_r: ndarray::Array3<f64>,
    pub(super) velocity: VelocityFields,
    pub(super) spectral_stress: SpectralStressFields,
    spectral_stress_next: SpectralStressFields,
    /// Persistent scratch for `fft(velocity)` — reused every step via
    /// `fft_3d_array_into` to avoid allocating 3 fresh `Array3<Complex<f64>>`
    /// per step inside `SpectralVelocityFields::from_real`.
    spectral_velocity_in: SpectralVelocityFields,
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

        let c_ref = max_p_wave_speed(&medium);
        let kappa = build_kappa(&kx, &ky, &kz, (nx, ny, nz), c_ref, dt);

        Ok(Self {
            plugin: PstdElasticPlugin::default(),
            medium,
            derivative_ops,
            kappa,
            pml: None,
            split_pml: None,
            split_pml_state: None,
            scratch_r: ndarray::Array3::<f64>::zeros(shape),
            velocity: VelocityFields::new(nx, ny, nz),
            spectral_stress: SpectralStressFields::new(nx, ny, nz),
            spectral_stress_next: SpectralStressFields::new(nx, ny, nz),
            spectral_velocity_in: SpectralVelocityFields::new(nx, ny, nz),
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

            if let (Some(pml), Some(state)) =
                (self.split_pml.as_ref(), self.split_pml_state.as_mut())
            {
                // ── Bérenger split-field PML path ─────────────────────────
                // Phase 4 of the split-field step overwrites `velocity` with
                // the sum of velocity sub-fields.  To ensure source-injected
                // velocity carries through to Phase 4, the source is also
                // injected into the x-directional velocity sub-fields before
                // the step.  The invariant `vx = vxx + vxy + vxz` is then
                // maintained: vxx absorbs the source, vxy/vxz are unchanged.
                if let Some(src) = source {
                    inject_velocity_source_subfields(state.as_mut(), src, step);
                }
                propagate_split_field_step(
                    &mut self.velocity,
                    SpectralScratch {
                        stress: &mut self.spectral_stress,
                        stress_next: &mut self.spectral_stress_next,
                        velocity_in: &mut self.spectral_velocity_in,
                        velocity_next: &mut self.spectral_velocity_next,
                    },
                    pml,
                    state.as_mut(),
                    &self.medium,
                    SpectralOperators {
                        dkx_neg: &self.derivative_ops.dkx_neg,
                        dky_neg: &self.derivative_ops.dky_neg,
                        dkz_neg: &self.derivative_ops.dkz_neg,
                        dkx_pos: &self.derivative_ops.dkx_pos,
                        dky_pos: &self.derivative_ops.dky_pos,
                        dkz_pos: &self.derivative_ops.dkz_pos,
                        kappa: &self.kappa,
                    },
                    &mut self.scratch_r,
                );
            } else if self.split_pml.is_some() {
                return Err(crate::core::error::KwaversError::InvalidInput(
                    "Split-field PML requires persistent split-field state".to_owned(),
                ));
            } else {
                // ── Standard leapfrog path (no split-field PML) ───────────
                fft_3d_array_into(&self.velocity.vx, &mut self.spectral_velocity_in.vx);
                fft_3d_array_into(&self.velocity.vy, &mut self.spectral_velocity_in.vy);
                fft_3d_array_into(&self.velocity.vz, &mut self.spectral_velocity_in.vz);

                let stress_params = StressUpdateParams {
                    vx_fft: &self.spectral_velocity_in.vx,
                    vy_fft: &self.spectral_velocity_in.vy,
                    vz_fft: &self.spectral_velocity_in.vz,
                    txx_fft: &self.spectral_stress.txx,
                    tyy_fft: &self.spectral_stress.tyy,
                    tzz_fft: &self.spectral_stress.tzz,
                    txy_fft: &self.spectral_stress.txy,
                    txz_fft: &self.spectral_stress.txz,
                    tyz_fft: &self.spectral_stress.tyz,
                    dkx_op: &self.derivative_ops.dkx_neg,
                    dky_op: &self.derivative_ops.dky_neg,
                    dkz_op: &self.derivative_ops.dkz_neg,
                    lame_lambda: &self.medium.lame_lambda,
                    lame_mu: &self.medium.lame_mu,
                    density: self.medium.density.view(),
                    dt: self.dt,
                    kappa: &self.kappa,
                };
                self.plugin
                    .apply_stress_update_in_place(&stress_params, &mut self.spectral_stress_next);

                let velocity_params = VelocityUpdateParams {
                    vx_fft: &self.spectral_velocity_in.vx,
                    vy_fft: &self.spectral_velocity_in.vy,
                    vz_fft: &self.spectral_velocity_in.vz,
                    txx_fft: &self.spectral_stress_next.txx,
                    tyy_fft: &self.spectral_stress_next.tyy,
                    tzz_fft: &self.spectral_stress_next.tzz,
                    txy_fft: &self.spectral_stress_next.txy,
                    txz_fft: &self.spectral_stress_next.txz,
                    tyz_fft: &self.spectral_stress_next.tyz,
                    dkx_op: &self.derivative_ops.dkx_pos,
                    dky_op: &self.derivative_ops.dky_pos,
                    dkz_op: &self.derivative_ops.dkz_pos,
                    density: self.medium.density.view(),
                    dt: self.dt,
                    kappa: &self.kappa,
                };
                self.plugin.apply_velocity_update_in_place(
                    &velocity_params,
                    &mut self.spectral_velocity_next,
                );

                std::mem::swap(&mut self.spectral_stress, &mut self.spectral_stress_next);
                ifft_3d_array_into(&mut self.spectral_velocity_next.vx, &mut self.velocity.vx);
                ifft_3d_array_into(&mut self.spectral_velocity_next.vy, &mut self.velocity.vy);
                ifft_3d_array_into(&mut self.spectral_velocity_next.vz, &mut self.velocity.vz);

                if let Some(pml) = self.pml.as_ref() {
                    pml.apply_to_field(&mut self.velocity.vx);
                    pml.apply_to_field(&mut self.velocity.vy);
                    pml.apply_to_field(&mut self.velocity.vz);
                }
            }

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

    /// Attach a real-space exponential PML to the orchestrator.
    ///
    /// `thickness_cells` specifies the absorbing-layer thickness on each
    /// side along each axis (e.g. `(10, 10, 0)` for 10-cell PML on x and
    /// y, none on z). `c_max` is the maximum sound speed in the medium
    /// (used to set σ_max). `r0` is the target theoretical reflection
    /// coefficient (Roden & Gedney 2000); `1e-4` is a standard choice.
    ///
    /// See [`super::pml`] for the full PML theorem and σ_max derivation.
    /// Calling this method REPLACES any previously-attached PML.
    pub fn set_pml(&mut self, thickness_cells: (usize, usize, usize), c_max: f64, r0: f64) {
        let (nx, ny, nz) = self.grid_shape;
        // Recover dx/dy/dz from the precomputed wavenumber arrays — the
        // orchestrator doesn't carry the Grid handle past construction.
        // dk = 2π / (n · dx) ⇒ dx = 2π / (n · dk).
        let dx = grid_spacing_from_wavenumber(&self.derivative_ops.dkx_pos, nx);
        let dy = grid_spacing_from_wavenumber(&self.derivative_ops.dky_pos, ny);
        let dz = grid_spacing_from_wavenumber(&self.derivative_ops.dkz_pos, nz);
        self.pml = Some(ElasticPml::new(ElasticPmlSpec {
            shape: (nx, ny, nz),
            thickness_cells,
            spacing: (dx, dy, dz),
            c_max,
            dt: self.dt,
            r0,
        }));
    }

    /// Disable any attached PML and revert to periodic boundaries.
    pub fn clear_pml(&mut self) {
        self.pml = None;
    }

    /// Attach a Bérenger split-field PML (zero theoretical reflection at all
    /// incidence angles and frequencies). Replaces any previously-attached
    /// `pml` or `split_pml`. The split-field path takes precedence over the
    /// scalar PML in `propagate`.
    ///
    /// See [`super::split_field_pml`] for the exact discrete integrator and
    /// memory layout of the 24 persistent sub-fields.
    pub fn set_split_field_pml(
        &mut self,
        thickness_cells: (usize, usize, usize),
        c_max: f64,
        r0: f64,
    ) {
        let (nx, ny, nz) = self.grid_shape;
        let dx = grid_spacing_from_wavenumber(&self.derivative_ops.dkx_pos, nx);
        let dy = grid_spacing_from_wavenumber(&self.derivative_ops.dky_pos, ny);
        let dz = grid_spacing_from_wavenumber(&self.derivative_ops.dkz_pos, nz);
        self.split_pml = Some(ElasticSplitFieldPml::new(ElasticPmlSpec {
            shape: (nx, ny, nz),
            thickness_cells,
            spacing: (dx, dy, dz),
            c_max,
            dt: self.dt,
            r0,
        }));
        self.split_pml_state = Some(Box::new(SplitFieldState::new(nx, ny, nz)));
    }

    /// Disable the split-field PML and revert to periodic boundaries.
    pub fn clear_split_field_pml(&mut self) {
        self.split_pml = None;
        self.split_pml_state = None;
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
