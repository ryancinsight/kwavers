//! PSTD time-stepping kernel: `step_forward` and k-space variant.

use super::super::orchestrator::PSTDSolver;
use kwavers_core::error::{KwaversError, KwaversResult};
use kwavers_source::{SourceField, SourceInjectionMode};
use crate::forward::pstd::config::KSpaceMethod;
use crate::forward::pstd::implementation::k_space::PSTDKSOperators;
use ndarray::{Array3, Zip};
use tracing::{enabled, trace, warn, Level};

impl PSTDSolver {
    /// Perform a single time step using k-space pseudospectral method
    /// # Errors
    /// - Propagates any [`KwaversError`] returned by called functions.
    ///
    #[inline]
    pub fn step_forward(&mut self) -> KwaversResult<()> {
        let dt = self.config.dt;
        let time_index = self.time_step_index;

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
    /// - Propagates any [`KwaversError`] returned by called functions.
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
                            Zip::from(&mut self.dpx).and(mask).par_for_each(|s, &m| {
                                if m.abs() > 1e-12 {
                                    *s += m * amp;
                                }
                            });
                        }
                        SourceInjectionMode::Additive { scale } => {
                            Zip::from(&mut self.dpx).and(mask).par_for_each(|s, &m| {
                                if m.abs() > 1e-12 {
                                    *s += m * amp * scale;
                                }
                            });
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
                            Zip::from(&mut self.dpx)
                                .and(grad_mask)
                                .par_for_each(|s, &gm| {
                                    *s -= c_sq * amp * gm;
                                });
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

        let ops = self.kspace_operators.take().ok_or_else(|| {
            KwaversError::Config(kwavers_core::error::ConfigError::InvalidValue {
                parameter: "kspace_operators".to_owned(),
                value: "None".to_owned(),
                constraint: "k-space operators must be initialized for FullKSpace method"
                    .to_owned(),
            })
        })?;

        let source_term = std::mem::replace(&mut self.dpx, Array3::zeros((0, 0, 0)));
        self.propagate_kspace(dt, &source_term, &ops)?;
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

    /// Propagate wave using the k-space spectral Laplacian.
    ///
    /// ## Theorem: spectral Laplacian identity
    ///
    /// For a field `p` on an `N`-point uniform grid with spacing `Δx`:
    /// ```text
    /// ∇²p  ↔  −|k|² p̂    (k = wavenumber vector, p̂ = FFT(p))
    /// ```
    ///
    /// `apply_helmholtz(p, 0.0)` evaluates `IFFT[(-|k|² + 0²) · FFT(p)] = ∇²p`.
    /// # Errors
    /// - Propagates any [`KwaversError`] returned by called functions.
    ///
    fn propagate_kspace(
        &mut self,
        dt: f64,
        source_term: &Array3<f64>,
        kspace_ops: &PSTDKSOperators,
    ) -> KwaversResult<()> {
        let laplacian = kspace_ops.apply_helmholtz(&self.fields.p, 0.0)?;
        let c_sq = self.c_ref * self.c_ref;
        Zip::from(&mut self.fields.p)
            .and(&laplacian)
            .and(source_term)
            .par_for_each(|p, &lap, &s| *p += dt * c_sq.mul_add(lap, s));
        Ok(())
    }
}
