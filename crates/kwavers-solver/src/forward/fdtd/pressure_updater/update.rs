//! Pressure field update dispatch and CPU/GPU implementations.

use kwavers_core::error::{KwaversError, KwaversResult};
use leto::Array3 as LetoArray3;
use leto::{Array3, ArrayView3};

use super::super::solver::{FdtdGpuAccelerator, FdtdSolver};

use kwavers_math::numerics::operators::Axis;

impl FdtdSolver {
    /// Dispatch pressure update to GPU or CPU; apply nonlinear correction if enabled.
    /// # Errors
    /// - Propagates any [`crate::KwaversError`] returned by called functions.
    ///
    #[inline]
    pub fn update_pressure(&mut self, dt: f64) -> KwaversResult<()> {
        if self.config.enable_gpu_acceleration {
            // The GPU accelerator trait carries no relaxation state, so running
            // it with absorption configured would silently propagate a lossless
            // medium -- a quiet accuracy loss rather than a failure. Refuse.
            if self.absorption.is_some() {
                return Err(KwaversError::Config(
                    kwavers_core::error::ConfigError::InvalidValue {
                        parameter: "enable_gpu_acceleration".to_owned(),
                        value: "true".to_owned(),
                        constraint: "GPU acceleration does not implement relaxation absorption; \
                                     use FdtdAbsorption::Lossless or the CPU path"
                            .to_owned(),
                    },
                ));
            }
            let accelerator = self.gpu_accelerator.as_ref().ok_or_else(|| {
                KwaversError::Config(kwavers_core::error::ConfigError::InvalidValue {
                    parameter: "enable_gpu_acceleration".to_owned(),
                    value: "true".to_owned(),
                    constraint: "GPU accelerator must be configured".to_owned(),
                })
            })?;
            let new_pressure = self.update_pressure_gpu(accelerator.as_ref(), dt)?;
            self.fields.p = new_pressure;
            if self.config.enable_nonlinear {
                self.apply_westervelt_nonlinear_correction(dt);
                self.rotate_pressure_history();
            }
            return Ok(());
        }
        self.update_pressure_cpu(dt)?;
        if self.config.enable_nonlinear {
            self.apply_westervelt_nonlinear_correction(dt);
            self.rotate_pressure_history();
        }
        Ok(())
    }

    /// CPU pressure update: p^{n+1} = p^n − dt · ρc² · div(v).
    ///
    /// Dispatch order:
    /// 1. K-space spectral divergence when `kspace_ops` is Some.
    /// 2. Staggered backward-difference when `staggered_grid = true`.
    /// 3. Central-difference otherwise.
    /// # Errors
    /// - Propagates any [`crate::KwaversError`] returned by called functions.
    ///
    pub(crate) fn update_pressure_cpu(&mut self, dt: f64) -> KwaversResult<()> {
        if let Some(kops) = self.kspace_ops.as_mut() {
            kops.compute_divergence_neg(&self.fields.ux, &self.fields.uy, &self.fields.uz);
            let divergence = kops.divergence.view();
            if let Some(absorption) = self.absorption.as_mut() {
                let (modulus, relaxation) = absorption.accumulate(divergence, dt);
                super::apply_absorbing_pressure_update(
                    &mut self.fields.p,
                    divergence,
                    modulus,
                    relaxation,
                    dt,
                );
            } else {
                Self::update_pressure_simd(&mut self.fields.p, divergence, &self.rho_c_squared, dt);
            }
            return Ok(());
        }

        if self.config.staggered_grid {
            self.compute_divergence_staggered()?;
            self.apply_pressure_from_divergence(dt);
        } else {
            // The same operator the velocity update uses. On a collocated grid
            // that is the point: one operator means the energy behaviour is
            // governed entirely by its summation-by-parts property.
            self.conservative_operator.apply_into(
                Axis::X,
                self.fields.ux.view(),
                &mut self.dvx_scratch,
            );
            self.conservative_operator.apply_into(
                Axis::Y,
                self.fields.uy.view(),
                &mut self.dvy_scratch,
            );
            self.conservative_operator.apply_into(
                Axis::Z,
                self.fields.uz.view(),
                &mut self.divergence_scratch,
            );

            if let Some(ref mut cpml) = self.cpml_boundary {
                cpml.update_and_apply_v_gradient_correction(&mut self.dvx_scratch, 0);
                cpml.update_and_apply_v_gradient_correction(&mut self.dvy_scratch, 1);
                cpml.update_and_apply_v_gradient_correction(&mut self.divergence_scratch, 2);
            }

            super::accumulate_two_fields(
                &mut self.divergence_scratch,
                &self.dvx_scratch,
                &self.dvy_scratch,
            );

            self.apply_pressure_from_divergence(dt);
        }
        Ok(())
    }

    /// Apply the pressure update from `divergence_scratch`, with the relaxation
    /// term when absorption is configured.
    ///
    /// Both finite-difference branches converge here **after** the CPML
    /// gradient correction and the per-axis accumulation, so the memory
    /// variables integrate exactly the divergence the pressure does. Driving
    /// them from an uncorrected divergence would let the two drift apart inside
    /// the PML, where the correction is largest.
    fn apply_pressure_from_divergence(&mut self, dt: f64) {
        if let Some(absorption) = self.absorption.as_mut() {
            let divergence = self.divergence_scratch.view();
            let (modulus, relaxation) = absorption.accumulate(divergence, dt);
            super::apply_absorbing_pressure_update(
                &mut self.fields.p,
                divergence,
                modulus,
                relaxation,
                dt,
            );
        } else {
            Self::update_pressure_simd(
                &mut self.fields.p,
                self.divergence_scratch.view(),
                &self.rho_c_squared,
                dt,
            );
        }
    }

    /// Element-wise pressure update: p -= dt · ρc² · div(v).
    ///
    /// Accepts `ArrayView3<f64>` to avoid O(N³) `.to_owned()` at call sites.
    /// Dense standard-layout arrays dispatch through Moirai; non-standard
    /// ndarray views keep sequential Zip semantics.
    pub(crate) fn update_pressure_simd(
        pressure: &mut LetoArray3<f64>,
        divergence: ArrayView3<f64>,
        rho_c_squared: &Array3<f64>,
        dt: f64,
    ) {
        super::apply_pressure_update(pressure, divergence, rho_c_squared, dt);
    }

    /// GPU-accelerated pressure update via external accelerator trait.
    /// # Errors
    /// - Returns [`Err`] if an internal constraint is violated.
    ///
    pub(crate) fn update_pressure_gpu(
        &self,
        accelerator: &dyn FdtdGpuAccelerator,
        dt: f64,
    ) -> KwaversResult<LetoArray3<f64>> {
        accelerator.propagate_acoustic_wave(
            &self.fields.p,
            &self.fields.ux,
            &self.fields.uy,
            &self.fields.uz,
            &self.materials.rho0,
            &self.materials.c0,
            dt,
            self.grid.dx,
            self.grid.dy,
            self.grid.dz,
        )
    }
}
