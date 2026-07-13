//! PSTD time-stepping kernel: `step_forward` and k-space variant.

use super::super::orchestrator::PSTDSolver;
use super::ops::{add_gradient_source_term, add_masked_source_term};
use crate::forward::pstd::config::KSpaceMethod;
use crate::forward::pstd::implementation::k_space::PSTDKSOperators;
use kwavers_core::error::{KwaversError, KwaversResult};
use kwavers_source::{SourceField, SourceInjectionMode};
use leto::Array3;
use tracing::{enabled, trace, warn, Level};

impl PSTDSolver {
    /// Perform a single time step using k-space pseudospectral method
    /// # Errors
    /// - Propagates any [`crate::KwaversError`] returned by called functions.
    ///
    #[inline]
    pub fn step_forward(&mut self) -> KwaversResult<()> {
        let dt = self.config.dt;
        let time_index = self.time_step_index;

        // FullKSpace → exact second-order k-space pressure propagator. StandardPSTD
        // and Hybrid both fall through to the split-field first-order kernel below
        // (Hybrid is not yet a true band-split; see KSpaceMethod::Hybrid docs).
        if self.config.kspace_method == KSpaceMethod::FullKSpace {
            return self.step_forward_kspace(dt, time_index);
        }

        // Time loop order matching C++ k-wave binary (KSpaceFirstOrderSolver.cpp):
        //   1. computePressureGradient + computeVelocity  (update_velocity)
        //   2. addVelocitySource
        //   3. computeVelocityGradient + computeDensity   (update_density)
        //   4. addPressureSource
        //   5. computePressure (including absorption)      (update_pressure)

        self.update_velocity(dt)?;

        self.apply_dynamic_velocity_sources(dt);
        if self.source_handler.has_velocity_source() {
            self.source_handler.inject_force_source(
                time_index,
                &mut self.fields.ux,
                &mut self.fields.uy,
                &mut self.fields.uz,
            );
        }

        self.update_density(dt)?;

        self.apply_pressure_sources(time_index, dt)?;

        self.update_pressure(dt)?;

        // Dirichlet sources: override p[sensor] = data[t] directly post-EOS,
        // mirroring KWave.jl's time_reversal_boundary_data which sets p after
        // the normal density→pressure update rather than pre-setting density.
        // Density evolves naturally at sensor locations; apply_pressure_sources
        // is a no-op for Dirichlet mode.
        if self.source_handler.pressure_mode() == kwavers_source::SourceMode::Dirichlet {
            self.source_handler
                .enforce_pressure_dirichlet(self.time_step_index, &mut self.fields.p);
        }

        if enabled!(Level::TRACE)
            && (self.time_step_index < 5 || self.time_step_index.is_multiple_of(10))
        {
            let max_p = self.fields.p.iter().fold(0.0f64, |m, &v| m.max(v.abs()));
            trace!(
                time_step = self.time_step_index,
                max_p,
                "After update_pressure"
            );
        }

        if self.filter.is_some() {
            self.apply_anti_aliasing_filter()?;
        }

        // NOTE: PML is already applied inside update_velocity() and
        // update_density() to the split velocity/density fields,
        // matching K-Wave's convention. Pressure (p = c²·Σρ) is computed
        // from the already-damped density, so no additional boundary
        // application is needed here.

        self.sensor_recorder.record_step(&self.fields.p)?;
        if self.sensor_recorder.needs_velocity() {
            self.sensor_recorder.record_velocity_step(
                &self.fields.ux,
                &self.fields.uy,
                &self.fields.uz,
            )?;
        }

        self.time_step_index += 1;

        Ok(())
    }

    /// Time step using full k-space pseudospectral method (dispersion-free)
    /// # Errors
    /// - Propagates any [`crate::KwaversError`] returned by called functions.
    ///
    fn step_forward_kspace(&mut self, dt: f64, time_index: usize) -> KwaversResult<()> {
        self.dpx.fill(0.0);

        if self.source_handler.has_pressure_source() {
            self.source_handler
                .add_pressure_source_into_density(time_index, &mut self.dpx);
        }

        let t = time_index as f64 * dt;
        for (idx, (source, mask)) in self.dynamic_sources.iter().enumerate() {
            let amp = source.amplitude(t);
            if amp.abs() < 1e-12 {
                continue;
            }

            match source.source_type() {
                SourceField::Pressure => {
                    let mode = self.source_injection_modes[idx];

                    match mode {
                        SourceInjectionMode::Boundary => {
                            add_masked_source_term(&mut self.dpx, mask, amp);
                        }
                        SourceInjectionMode::Additive { scale } => {
                            add_masked_source_term(&mut self.dpx, mask, amp * scale);
                        }
                    }
                }
                SourceField::VelocityX | SourceField::VelocityY | SourceField::VelocityZ => {
                    // Pressure-equivalent injection for velocity sources in the FullKSpace
                    // pressure-only scheme.
                    //
                    // From the linearised acoustic equations (eliminating velocity):
                    //   ∂²p/∂t² = c²∇²p − c²∇·f_u
                    // For a velocity source f_u = amp(t)·mask(x)·ê_α:
                    //   S_p = −c²·amp·∂mask/∂α
                    let c_sq = self.c_ref * self.c_ref;
                    match self.velocity_source_grad_masks.get(idx) {
                        Some(Some(grad_mask)) => {
                            add_gradient_source_term(&mut self.dpx, grad_mask, -c_sq * amp);
                        }
                        Some(None) => {}
                        None => {
                            warn!(
                                "velocity_source_grad_masks[{}] missing — source index \
                                 out of sync with dynamic_sources; source dropped",
                                idx
                            );
                        }
                    }
                }
            }
        }

        let mut ops = self.kspace_operators.take().ok_or_else(|| {
            KwaversError::Config(kwavers_core::error::ConfigError::InvalidValue {
                parameter: "kspace_operators".to_owned(),
                value: "None".to_owned(),
                constraint: "k-space operators must be initialized for FullKSpace method"
                    .to_owned(),
            })
        })?;

        let source_term = std::mem::replace(&mut self.dpx, Array3::zeros([0, 0, 0]));
        self.propagate_kspace(dt, &source_term, &mut ops)?;
        self.dpx = source_term;
        self.kspace_operators = Some(ops);

        self.apply_boundary(time_index)?;
        self.sensor_recorder.record_step(&self.fields.p)?;
        if self.sensor_recorder.needs_velocity() {
            self.sensor_recorder.record_velocity_step(
                &self.fields.ux,
                &self.fields.uy,
                &self.fields.uz,
            )?;
        }
        self.time_step_index += 1;

        Ok(())
    }

    /// Advance the pressure field one step with the EXACT second-order
    /// dispersion-free k-space wave-equation propagator (homogeneous `c_ref`).
    ///
    /// ## Scheme
    ///
    /// For the lossless wave equation `∂²p/∂t² = c²∇²p`, every spectral mode is a
    /// harmonic oscillator `d²p̂/dt² = −(c|k|)² p̂` whose exact discrete recurrence is
    /// ```text
    /// p̂ⁿ⁺¹ = 2cos(c·|k|·Δt)·p̂ⁿ − p̂ⁿ⁻¹
    /// ```
    /// This is EXACT for any Δt (no CFL limit, no numerical dispersion) when `c` is
    /// constant — the defining property of the k-space method (Mast et al. 2001;
    /// Tabei, Mast & Waag 2002). The previous implementation applied a first-order
    /// forward Euler of `∂p/∂t = c²∇²p` (a diffusion equation, not the wave
    /// equation), which is the wrong PDE and unconditionally unstable.
    ///
    /// **First step (zero-velocity IVP).** With `∂p/∂t(0)=0` the solution is even in
    /// time, so `p̂⁻¹ = p̂¹` and the recurrence collapses to `p̂¹ = cos(c·|k|·Δt)·p̂⁰`
    /// (half the leapfrog coefficient). `kspace_ops.p_prev == None` selects this
    /// branch; thereafter the full `2cos` coefficient with explicit `pⁿ⁻¹`.
    ///
    /// **Sources.** `source_term` (assembled in `step_forward_kspace`) is added as an
    /// additive per-step pressure increment. Only the source-FREE propagation is
    /// reference-validated (vs. the analytical 3-D Gaussian IVP in
    /// `pstd_fullkspace_gaussian_ivp_matches_analytical`); the additive source
    /// scaling for FullKSpace has not been validated against an external oracle.
    ///
    /// **Homogeneity.** `c_ref = medium.max_sound_speed()` is used for every mode,
    /// so this propagator is exact only for homogeneous media. Heterogeneous media
    /// must use `StandardPSTD`, whose first-order split-field system carries `c(x)`
    /// in the equation of state.
    /// # Errors
    /// - Propagates any [`crate::KwaversError`] returned by called functions.
    ///
    fn propagate_kspace(
        &mut self,
        dt: f64,
        source_term: &Array3<f64>,
        kspace_ops: &mut PSTDKSOperators,
    ) -> KwaversResult<()> {
        let c_ref = self.c_ref;
        kspace_ops.ensure_wave_coeff(c_ref, dt);
        let p_prev_old = kspace_ops.p_prev.take(); // pⁿ⁻¹ (None on the first step)
        let is_first = p_prev_old.is_none();

        let mut p_hat = kspace_ops.forward_fft_3d(&self.fields.p)?;
        {
            let coeff = kspace_ops
                .wave_coeff
                .as_ref()
                .expect("invariant: ensure_wave_coeff populated wave_coeff");
            let factor = if is_first { 0.5 } else { 1.0 };
            for (value, &coef) in p_hat.iter_mut().zip(coeff.iter()) {
                *value *= factor * coef;
            }
        }
        let mut new_p = kspace_ops.inverse_fft_3d(&p_hat)?;

        if let Some(prev) = &p_prev_old {
            for ((dst, &old), &source) in new_p.iter_mut().zip(prev.iter()).zip(source_term.iter())
            {
                *dst += source - old;
            }
        } else {
            for (dst, &source) in new_p.iter_mut().zip(source_term.iter()) {
                *dst += source;
            }
        }

        // pⁿ becomes pⁿ⁻¹ for the next step; new_p becomes the current pressure.
        let shape = self.fields.p.shape();
        let mut old_p = leto::Array3::zeros((shape[0], shape[1], shape[2]));
        for (dst, src) in old_p.iter_mut().zip(self.fields.p.iter()) {
            *dst = *src;
        }
        for (dst, src) in self.fields.p.iter_mut().zip(new_p.iter()) {
            *dst = *src;
        }
        kspace_ops.p_prev = Some(old_p);
        Ok(())
    }
}
