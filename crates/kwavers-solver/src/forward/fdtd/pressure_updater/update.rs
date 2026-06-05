//! Pressure field update dispatch and CPU/GPU implementations.

use kwavers_core::error::{KwaversError, KwaversResult};
use ndarray::{Array3, ArrayView3, Zip};

use super::super::solver::{FdtdGpuAccelerator, FdtdSolver};

impl FdtdSolver {
    /// Dispatch pressure update to GPU or CPU; apply nonlinear correction if enabled.
    /// # Errors
    /// - Propagates any [`KwaversError`] returned by called functions.
    ///
    #[inline]
    pub fn update_pressure(&mut self, dt: f64) -> KwaversResult<()> {
        if self.config.enable_gpu_acceleration {
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
    /// - Propagates any [`KwaversError`] returned by called functions.
    ///
    pub(crate) fn update_pressure_cpu(&mut self, dt: f64) -> KwaversResult<()> {
        if let Some(kops) = self.kspace_ops.as_mut() {
            kops.compute_divergence_neg(&self.fields.ux, &self.fields.uy, &self.fields.uz);
            let divergence = kops.divergence.view();
            Self::update_pressure_simd(&mut self.fields.p, divergence, &self.rho_c_squared, dt);
            return Ok(());
        }

        if self.config.staggered_grid {
            self.compute_divergence_staggered()?;
            Self::update_pressure_simd(
                &mut self.fields.p,
                self.divergence_scratch.view(),
                &self.rho_c_squared,
                dt,
            );
        } else {
            self.central_operator
                .apply_x_into(self.fields.ux.view(), &mut self.dvx_scratch)?;
            self.central_operator
                .apply_y_into(self.fields.uy.view(), &mut self.dvy_scratch)?;
            self.central_operator
                .apply_z_into(self.fields.uz.view(), &mut self.divergence_scratch)?;

            if let Some(ref mut cpml) = self.cpml_boundary {
                cpml.update_and_apply_v_gradient_correction(&mut self.dvx_scratch, 0);
                cpml.update_and_apply_v_gradient_correction(&mut self.dvy_scratch, 1);
                cpml.update_and_apply_v_gradient_correction(&mut self.divergence_scratch, 2);
            }

            Zip::from(&mut self.divergence_scratch)
                .and(&self.dvx_scratch)
                .and(&self.dvy_scratch)
                .par_for_each(|d, &dx, &dy| *d += dx + dy);

            Self::update_pressure_simd(
                &mut self.fields.p,
                self.divergence_scratch.view(),
                &self.rho_c_squared,
                dt,
            );
        }
        Ok(())
    }

    /// Element-wise pressure update: p -= dt · ρc² · div(v).
    ///
    /// Accepts `ArrayView3<f64>` to avoid O(N³) `.to_owned()` at call sites.
    /// Rayon parallel Zip; LLVM auto-vectorizes each worker lane.
    pub(crate) fn update_pressure_simd(
        pressure: &mut Array3<f64>,
        divergence: ArrayView3<f64>,
        rho_c_squared: &Array3<f64>,
        dt: f64,
    ) {
        Zip::from(pressure)
            .and(divergence)
            .and(rho_c_squared)
            .par_for_each(|p, &div, &rc2| *p -= dt * rc2 * div);
    }

    /// GPU-accelerated pressure update via external accelerator trait.
    /// # Errors
    /// - Returns [`Err`] if an internal constraint is violated.
    ///
    pub(crate) fn update_pressure_gpu(
        &self,
        accelerator: &dyn FdtdGpuAccelerator,
        dt: f64,
    ) -> KwaversResult<Array3<f64>> {
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
