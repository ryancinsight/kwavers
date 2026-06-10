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
//! 3. **Stress update** via [`crate::forward::pstd::extensions::PstdElasticPlugin::apply_stress_update_in_place`]:
//!    `σ̃(t+dt) = σ̃(t) + dt · C : ε̃(t+dt/2)`.
//! 4. **Velocity update** via [`crate::forward::pstd::extensions::PstdElasticPlugin::apply_velocity_update_in_place`]:
//!    `ṽ(t+dt) = ṽ(t) + (dt/ρ) · ∇·σ̃(t+dt)`.
//! 5. **Inverse FFT** of the new spectral velocity → real-space `(vx, vy, vz)`.
//! 6. **Sensor recording** — copy the per-component values at masked grid
//!    points into the recorder matrices.

use super::kspace::{build_kappa, grid_spacing_from_wavenumber, max_p_wave_speed, wavenumber_axis};
use super::leapfrog_step::{propagate_leapfrog_step, seed_stress_from_displacement};
use super::pml::{ElasticPml, ElasticPmlSpec};
use super::source_sensor::{
    inject_velocity_source, inject_velocity_source_subfields, record_sensors, validate_source,
};
use super::split_field_pml::{ElasticSplitFieldPml, SplitFieldState};
use super::split_field_step::{propagate_split_field_step, SpectralOperators, SpectralScratch};
use super::staggered_ops::StaggeredDerivativeOps;
use super::types::{ElasticPstdMedium, ElasticPstdSensorData, ElasticPstdVelocitySource};
use kwavers_core::error::KwaversResult;
use kwavers_grid::Grid;
use kwavers_physics::acoustics::mechanics::elastic_wave::{
    fields::{StressFields, VelocityFields},
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
    /// Real-space stress state for the standard (non-split-field) leapfrog
    /// path. Persisted between steps; updated with real-space Lamé coefficients
    /// (see [`super::leapfrog_step`]) so heterogeneous media are correct.
    stress: StressFields,
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
                return Err(kwavers_core::error::KwaversError::InvalidInput(format!(
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
        let kappa = build_kappa(
            &kx,
            &ky,
            &kz,
            (nx, ny, nz),
            c_ref,
            dt,
            grid.dx,
            grid.dy,
            grid.dz,
        );

        Ok(Self {
            medium,
            derivative_ops,
            kappa,
            pml: None,
            split_pml: None,
            split_pml_state: None,
            scratch_r: ndarray::Array3::<f64>::zeros(shape),
            velocity: VelocityFields::new(nx, ny, nz),
            stress: StressFields::new(nx, ny, nz),
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
                // ── Bérenger split-field PML source coupling ──────────────
                // Phase 4 of the split-field step overwrites `velocity` with
                // the sum of velocity sub-fields.  To ensure source-injected
                // velocity carries through to Phase 4, the source is also
                // injected into the x-directional velocity sub-fields before
                // the step.  The invariant `vx = vxx + vxy + vxz` is then
                // maintained: vxx absorbs the source, vxy/vxz are unchanged.
                // Injected here (rather than inside `step`) because `step` is
                // the source-agnostic single-step primitive.
                if self.split_pml.is_some() {
                    if let Some(state) = self.split_pml_state.as_mut() {
                        inject_velocity_source_subfields(state.as_mut(), src, step);
                    }
                }
            }

            self.step()?;

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
        }

        Ok(ElasticPstdSensorData {
            vx: sensor_vx,
            vy: sensor_vy,
            vz: sensor_vz,
        })
    }

    /// Advance the elastic stress–velocity state by **exactly one** leapfrog
    /// step. This is the single-step SSOT primitive that [`propagate`] iterates.
    ///
    /// Source injection (into [`velocity_mut`]) and sensor recording are the
    /// caller's responsibility; for the split-field-PML path the caller must
    /// also inject the source into the PML sub-fields *before* calling `step`
    /// (see [`propagate`]). Selects the split-field PML path when one is
    /// attached, else the standard real-space-coefficient leapfrog with an
    /// optional scalar exponential PML.
    ///
    /// [`propagate`]: Self::propagate
    /// [`velocity_mut`]: Self::velocity_mut
    ///
    /// # Errors
    /// - Returns [`Err`] if a split-field PML is configured without its
    ///   persistent sub-field state.
    pub fn step(&mut self) -> KwaversResult<()> {
        if let (Some(pml), Some(state)) = (self.split_pml.as_ref(), self.split_pml_state.as_mut()) {
            // ── Bérenger split-field PML path ─────────────────────────────
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
            return Err(kwavers_core::error::KwaversError::InvalidInput(
                "Split-field PML requires persistent split-field state".to_owned(),
            ));
        } else {
            // ── Standard leapfrog path (no split-field PML) ───────────────
            // Coefficients (λ, μ, 1/ρ) are applied in REAL space after each
            // k-space derivative IFFT, so heterogeneous media propagate at
            // the correct local speed (see `super::leapfrog_step`).
            let ops = SpectralOperators {
                dkx_neg: &self.derivative_ops.dkx_neg,
                dky_neg: &self.derivative_ops.dky_neg,
                dkz_neg: &self.derivative_ops.dkz_neg,
                dkx_pos: &self.derivative_ops.dkx_pos,
                dky_pos: &self.derivative_ops.dky_pos,
                dkz_pos: &self.derivative_ops.dkz_pos,
                kappa: &self.kappa,
            };
            propagate_leapfrog_step(
                &mut self.velocity,
                &mut self.stress,
                &self.medium,
                &ops,
                &mut self.spectral_velocity_in,
                &mut self.spectral_stress,
                &mut self.spectral_stress_next,
                &mut self.scratch_r,
                self.dt,
            );

            if let Some(pml) = self.pml.as_ref() {
                pml.apply_to_field(&mut self.velocity.vx);
                pml.apply_to_field(&mut self.velocity.vy);
                pml.apply_to_field(&mut self.velocity.vz);
            }
        }

        self.step_index += 1;
        Ok(())
    }

    /// Isotropic acoustic pressure `p = -⅓ tr(σ) = -⅓(σxx + σyy + σzz)`
    /// computed from the current **real-space** stress state.
    ///
    /// Bridges the elastic stress tensor to the scalar-acoustic pipeline: with
    /// shear modulus `μ ≡ 0` the three normal stresses are equal and collapse
    /// to `-p` exactly, so this reproduces the acoustic pressure; with `μ > 0`
    /// it is the mechanical pressure (negative mean normal stress). Valid on
    /// the standard (non-split-field) leapfrog path, which maintains `stress`
    /// in real space.
    #[must_use]
    pub fn pressure_field(&self) -> Array3<f64> {
        let mut p = self.stress.txx.clone();
        p += &self.stress.tyy;
        p += &self.stress.tzz;
        p.mapv_inplace(|v| -v / 3.0);
        p
    }

    /// Mutable access to the real-space velocity field, for source injection
    /// or initial-condition seeding prior to [`step`](Self::step).
    pub fn velocity_mut(&mut self) -> &mut VelocityFields {
        &mut self.velocity
    }

    /// Read access to the real-space stress field (standard leapfrog path).
    #[must_use]
    pub fn stress(&self) -> &StressFields {
        &self.stress
    }

    /// Mutable access to the real-space stress field, for initial-condition
    /// seeding (e.g. an isotropic compressional perturbation `σxx=σyy=σzz=-p₀`)
    /// prior to [`step`](Self::step).
    pub fn stress_mut(&mut self) -> &mut StressFields {
        &mut self.stress
    }

    /// Seed the initial stress from an initial-value-problem displacement.
    ///
    /// `u0` is the displacement field along `axis` (0=x, 1=y, 2=z); the other
    /// components and the initial velocity are zero. Sets the initial stress
    /// `σ = λ(∇·u)I + μ(∇u + ∇uᵀ)` so a standing-start displacement IC (e.g.
    /// an SH plane-wave packet) propagates correctly. Call once after
    /// construction, before `propagate`.
    pub fn seed_initial_displacement(&mut self, u0: &Array3<f64>, axis: usize) {
        let ops = SpectralOperators {
            dkx_neg: &self.derivative_ops.dkx_neg,
            dky_neg: &self.derivative_ops.dky_neg,
            dkz_neg: &self.derivative_ops.dkz_neg,
            dkx_pos: &self.derivative_ops.dkx_pos,
            dky_pos: &self.derivative_ops.dky_pos,
            dkz_pos: &self.derivative_ops.dkz_pos,
            kappa: &self.kappa,
        };
        seed_stress_from_displacement(
            &mut self.stress,
            u0,
            axis,
            &self.medium,
            &ops,
            &mut self.spectral_velocity_in,
            &mut self.spectral_stress_next.txx,
            &mut self.scratch_r,
        );
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

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array3;

    fn small_grid() -> Grid {
        Grid::new(16, 16, 1, 1e-3, 1e-3, 1e-3).expect("grid")
    }

    fn solid_medium(shape: (usize, usize, usize)) -> ElasticPstdMedium {
        // c_p = 1500, c_s = 80, ρ = 1000 ⇒ μ = ρc_s², λ = ρ(c_p² − 2c_s²).
        let rho = 1000.0;
        let mu = rho * 80.0 * 80.0;
        let lambda = rho * 1500.0f64.mul_add(1500.0, -(2.0 * 80.0 * 80.0));
        ElasticPstdMedium {
            lame_lambda: Array3::from_elem(shape, lambda),
            lame_mu: Array3::from_elem(shape, mu),
            density: Array3::from_elem(shape, rho),
        }
    }

    fn seed_centre_compression(orch: &mut ElasticPstdOrchestrator, p0: f64) {
        let (nx, ny, _) = orch.grid_shape;
        let s = orch.stress_mut();
        s.txx[[nx / 2, ny / 2, 0]] = -p0;
        s.tyy[[nx / 2, ny / 2, 0]] = -p0;
        s.tzz[[nx / 2, ny / 2, 0]] = -p0;
    }

    /// `propagate(n, None, None)` must equal `step()` iterated `n` times from
    /// the same seeded state — pins the plugin's single-step usage to the
    /// validated batch path and guards the loop-body extraction against drift.
    #[test]
    fn step_iterated_matches_propagate_bit_for_bit() {
        let grid = small_grid();
        let shape = grid.dimensions();
        let dt = 5e-8;
        let n = 5;

        let mut batch =
            ElasticPstdOrchestrator::new(&grid, solid_medium(shape), dt).expect("batch");
        let mut manual =
            ElasticPstdOrchestrator::new(&grid, solid_medium(shape), dt).expect("manual");
        seed_centre_compression(&mut batch, 1.0e5);
        seed_centre_compression(&mut manual, 1.0e5);

        batch.propagate(n, None, None).expect("propagate");
        for _ in 0..n {
            manual.step().expect("step");
        }

        let pb = batch.pressure_field();
        let pm = manual.pressure_field();
        for (b, m) in pb.iter().zip(pm.iter()) {
            assert_eq!(*b, *m, "step-loop must reproduce propagate exactly");
        }
        assert_eq!(batch.step_index(), manual.step_index());
    }

    /// `pressure_field` is exactly `−⅓(σxx+σyy+σzz)` element-wise.
    #[test]
    fn pressure_field_is_negative_mean_normal_stress() {
        let grid = small_grid();
        let shape = grid.dimensions();
        let mut orch =
            ElasticPstdOrchestrator::new(&grid, solid_medium(shape), 5e-8).expect("orch");
        {
            let s = orch.stress_mut();
            s.txx.fill(2.0);
            s.tyy.fill(4.0);
            s.tzz.fill(6.0);
        }
        let p = orch.pressure_field();
        // −(2+4+6)/3 = −4.
        for v in p.iter() {
            assert!((v - (-4.0)).abs() < 1e-12, "expected −4, got {v}");
        }
    }

    /// A transverse (shear) velocity perturbation generates shear stress under
    /// `μ > 0` — behaviour an acoustic (μ = 0) stepper cannot produce, proving
    /// the elastic path is genuinely elastic and not an acoustic alias.
    #[test]
    fn shear_velocity_generates_shear_stress_when_mu_positive() {
        let grid = small_grid();
        let shape = grid.dimensions();
        let mut orch =
            ElasticPstdOrchestrator::new(&grid, solid_medium(shape), 5e-8).expect("orch");
        // vy varying along x ⇒ ∂vy/∂x ≠ 0 ⇒ σxy develops.
        {
            let v = orch.velocity_mut();
            for i in 0..shape.0 {
                let x = i as f64 / shape.0 as f64;
                let amp = (std::f64::consts::TAU * x).sin();
                for j in 0..shape.1 {
                    v.vy[[i, j, 0]] = amp;
                }
            }
        }
        orch.step().expect("step");
        let max_txy = orch.stress().txy.iter().fold(0.0_f64, |m, &v| m.max(v.abs()));
        assert!(
            max_txy > 1e-6,
            "shear velocity gradient must induce non-zero σxy (got {max_txy})"
        );
    }
}
