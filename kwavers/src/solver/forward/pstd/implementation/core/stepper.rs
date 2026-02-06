//! PSTD Time-stepping and Propagation Logic

use super::orchestrator::PSTDSolver;
use crate::core::error::{KwaversError, KwaversResult};
use crate::domain::source::{SourceField, SourceInjectionMode};
use crate::math::fft::Complex64;

use crate::solver::forward::pstd::config::KSpaceMethod;
use crate::solver::forward::pstd::implementation::k_space::PSTDKSOperators;
use ndarray::{Array3, Zip};
use tracing::trace;

impl PSTDSolver {
    /// Perform a single time step using k-space pseudospectral method
    pub fn step_forward(&mut self) -> KwaversResult<()> {
        let dt = self.config.dt;
        let time_index = self.time_step_index;

        if self.config.kspace_method == KSpaceMethod::FullKSpace {
            return self.step_forward_kspace(dt, time_index);
        }

        // Standard PSTD method (k-Wave style ordering)
        // 1. Update velocity
        self.update_velocity(dt)?;

        // 2. Apply velocity sources (both source_handler and dynamic)
        self.apply_dynamic_velocity_sources(dt);
        if self.source_handler.has_velocity_source() {
            self.source_handler.inject_force_source(
                time_index,
                &mut self.fields.ux,
                &mut self.fields.uy,
                &mut self.fields.uz,
            );
        }

        // 3. Update density
        self.update_density(dt)?;

        // 4. Apply pressure sources as mass sources (grid + dynamic), with k-space correction
        self.apply_pressure_sources(time_index, dt)?;

        // 5. Update pressure from density
        self.update_pressure();

        if self.time_step_index < 5 || self.time_step_index.is_multiple_of(10) {
            let max_p = self.fields.p.iter().fold(0.0f64, |m, &v| m.max(v.abs()));
            trace!(
                time_step = self.time_step_index,
                max_p,
                "After update_pressure"
            );
        }

        // 6. Apply anti-aliasing filter
        if self.filter.is_some() {
            self.apply_anti_aliasing_filter()?;
        }

        // 7. Apply boundary conditions
        self.apply_boundary(time_index)?;

        // 8. Record sensor data
        self.sensor_recorder.record_step(&self.fields.p)?;

        self.time_step_index += 1;

        Ok(())
    }

    fn apply_pressure_sources(&mut self, time_index: usize, dt: f64) -> KwaversResult<()> {
        let p_mode = self.source_handler.pressure_mode();
        let mut has_sources = false;

        // Reset source term buffer
        self.dpx.fill(0.0);

        if self.source_handler.has_pressure_source() {
            self.source_handler
                .add_pressure_source_into_density(time_index, &mut self.dpx);
            has_sources = true;
        }

        // Dynamic sources (always additive in PSTD)
        let t = time_index as f64 * dt;
        for (idx, (source, mask)) in self.dynamic_sources.iter().enumerate() {
            let amp = source.amplitude(t);
            if amp.abs() < 1e-12 {
                continue;
            }

            if source.source_type() == SourceField::Pressure {
                let mode = self.source_injection_modes[idx];
                let scale = match mode {
                    SourceInjectionMode::Additive { scale } => scale,
                    SourceInjectionMode::Boundary => 1.0,
                };

                Zip::from(&mut self.dpx).and(mask).for_each(|p, &m| {
                    if m.abs() > 1e-12 {
                        *p += m * amp * scale;
                    }
                });
                has_sources = true;
            }
        }

        if !has_sources {
            return Ok(());
        }

        match p_mode {
            crate::domain::source::SourceMode::Dirichlet => {
                // Dirichlet: enforce source values directly into density
                Zip::from(&mut self.rhox)
                    .and(&mut self.rhoy)
                    .and(&mut self.rhoz)
                    .and(&self.dpx)
                    .for_each(|rx, ry, rz, &s| {
                        if s.abs() > 0.0 {
                            *rx = s;
                            *ry = s;
                            *rz = s;
                        }
                    });
            }
            crate::domain::source::SourceMode::Additive => {
                // Apply k-space source correction (k-Wave additive source_kappa)
                self.fft.forward_into(&self.dpx, &mut self.p_k);
                Zip::from(&mut self.p_k)
                    .and(&self.source_kappa)
                    .for_each(|val, &k| {
                        *val *= Complex64::new(k, 0.0);
                    });
                self.fft
                    .inverse_into(&self.p_k, &mut self.dpx, &mut self.ux_k);

                Zip::from(&mut self.rhox)
                    .and(&mut self.rhoy)
                    .and(&mut self.rhoz)
                    .and(&self.dpx)
                    .for_each(|rx, ry, rz, &s| {
                        *rx += s;
                        *ry += s;
                        *rz += s;
                    });
            }
            crate::domain::source::SourceMode::AdditiveNoCorrection => {
                Zip::from(&mut self.rhox)
                    .and(&mut self.rhoy)
                    .and(&mut self.rhoz)
                    .and(&self.dpx)
                    .for_each(|rx, ry, rz, &s| {
                        *rx += s;
                        *ry += s;
                        *rz += s;
                    });
            }
        }

        Ok(())
    }

    /// Time step using full k-space pseudospectral method (dispersion-free)
    fn step_forward_kspace(&mut self, dt: f64, time_index: usize) -> KwaversResult<()> {
        let ops = self
            .kspace_operators
            .as_ref()
            .ok_or_else(|| {
                KwaversError::Config(crate::core::error::ConfigError::InvalidValue {
                    parameter: "kspace_operators".to_string(),
                    value: "None".to_string(),
                    constraint: "k-space operators must be initialized for FullKSpace method"
                        .to_string(),
                })
            })?
            .clone();

        let mut source_term = Array3::<f64>::zeros(self.fields.p.dim());

        // Apply source_handler sources
        if self.source_handler.has_pressure_source() {
            let mut temp_rho = Array3::<f64>::zeros(self.fields.p.dim());
            self.source_handler
                .inject_mass_source(time_index, &mut temp_rho, &self.materials.c0);

            Zip::from(&mut source_term)
                .and(&temp_rho)
                .and(&self.materials.c0)
                .for_each(|s, &rho, &c| *s = rho * c * c);
        }

        // Apply dynamic sources (from add_source_arc)
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
                            // Should not happen in PSTD, but handle gracefully
                            // Treat as additive without normalization
                            Zip::from(&mut source_term).and(mask).for_each(|s, &m| {
                                if m.abs() > 1e-12 {
                                    *s += m * amp;
                                }
                            });
                        }
                        SourceInjectionMode::Additive { scale } => {
                            // Additive sources with normalization
                            Zip::from(&mut source_term).and(mask).for_each(|s, &m| {
                                if m.abs() > 1e-12 {
                                    *s += m * amp * scale;
                                }
                            });
                        }
                    }
                }
                _ => {
                    // Velocity sources not yet supported in k-space mode
                }
            }
        }

        self.propagate_kspace(dt, &source_term, &ops)?;
        self.apply_boundary(time_index)?;
        self.sensor_recorder.record_step(&self.fields.p)?;
        self.time_step_index += 1;

        Ok(())
    }

    /// Propagate wave using k-space pseudospectral method
    fn propagate_kspace(
        &mut self,
        dt: f64,
        source_term: &Array3<f64>,
        kspace_ops: &PSTDKSOperators,
    ) -> KwaversResult<()> {
        let wavenumber = 2.0 * std::f64::consts::PI * 1e6 / self.c_ref;
        let helmholtz_term = kspace_ops.apply_helmholtz(&self.fields.p, wavenumber)?;

        Zip::from(&mut self.fields.p)
            .and(&helmholtz_term)
            .and(source_term)
            .for_each(|p, h_term, s| {
                let c_squared = self.c_ref * self.c_ref;
                *p += dt * (c_squared * h_term + s);
            });

        Ok(())
    }

    /// Apply dynamic velocity sources
    pub(crate) fn apply_dynamic_velocity_sources(&mut self, dt: f64) {
        let t = self.time_step_index as f64 * dt;
        for (source, mask) in &self.dynamic_sources {
            let amp = source.amplitude(t);
            if amp.abs() < 1e-12 {
                continue;
            }

            match source.source_type() {
                SourceField::VelocityX => {
                    Zip::from(&mut self.fields.ux).and(mask).for_each(|u, &m| {
                        if m.abs() > 1e-12 {
                            *u += m * amp;
                        }
                    });
                }
                SourceField::VelocityY => {
                    Zip::from(&mut self.fields.uy).and(mask).for_each(|u, &m| {
                        if m.abs() > 1e-12 {
                            *u += m * amp;
                        }
                    });
                }
                SourceField::VelocityZ => {
                    Zip::from(&mut self.fields.uz).and(mask).for_each(|u, &m| {
                        if m.abs() > 1e-12 {
                            *u += m * amp;
                        }
                    });
                }
                _ => {}
            }
        }
    }

    /// Apply anti-aliasing filter to field variables
    ///
    /// This removes high-frequency spatial components that can cause
    /// instability or aliasing when using PSTD with nonlinearities.
    pub(crate) fn apply_anti_aliasing_filter(&mut self) -> KwaversResult<()> {
        if let Some(filter) = &self.filter {
            // Apply filter to pressure
            // Use p_k as transform buffer and ux_k as scratch for inverse
            self.fft.forward_into(&self.fields.p, &mut self.p_k);
            Zip::from(&mut self.p_k)
                .and(filter)
                .for_each(|val, &f| *val *= Complex64::new(f, 0.0));
            self.fft
                .inverse_into(&self.p_k, &mut self.fields.p, &mut self.ux_k);

            // Apply filter to split density components
            self.fft.forward_into(&self.rhox, &mut self.p_k);
            Zip::from(&mut self.p_k)
                .and(filter)
                .for_each(|val, &f| *val *= Complex64::new(f, 0.0));
            self.fft
                .inverse_into(&self.p_k, &mut self.rhox, &mut self.uy_k);

            self.fft.forward_into(&self.rhoy, &mut self.p_k);
            Zip::from(&mut self.p_k)
                .and(filter)
                .for_each(|val, &f| *val *= Complex64::new(f, 0.0));
            self.fft
                .inverse_into(&self.p_k, &mut self.rhoy, &mut self.uy_k);

            self.fft.forward_into(&self.rhoz, &mut self.p_k);
            Zip::from(&mut self.p_k)
                .and(filter)
                .for_each(|val, &f| *val *= Complex64::new(f, 0.0));
            self.fft
                .inverse_into(&self.p_k, &mut self.rhoz, &mut self.uy_k);

            // Apply filter to Ux
            // Use ux_k as transform buffer and p_k as scratch
            self.fft.forward_into(&self.fields.ux, &mut self.ux_k);
            Zip::from(&mut self.ux_k)
                .and(filter)
                .for_each(|val, &f| *val *= Complex64::new(f, 0.0));
            self.fft
                .inverse_into(&self.ux_k, &mut self.fields.ux, &mut self.p_k);

            // Apply filter to Uy
            // Use uy_k as transform buffer and p_k as scratch
            self.fft.forward_into(&self.fields.uy, &mut self.uy_k);
            Zip::from(&mut self.uy_k)
                .and(filter)
                .for_each(|val, &f| *val *= Complex64::new(f, 0.0));
            self.fft
                .inverse_into(&self.uy_k, &mut self.fields.uy, &mut self.p_k);

            // Apply filter to Uz
            // Use uz_k as transform buffer and p_k as scratch
            self.fft.forward_into(&self.fields.uz, &mut self.uz_k);
            Zip::from(&mut self.uz_k)
                .and(filter)
                .for_each(|val, &f| *val *= Complex64::new(f, 0.0));
            self.fft
                .inverse_into(&self.uz_k, &mut self.fields.uz, &mut self.p_k);
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::domain::grid::Grid;
    use crate::domain::medium::HomogeneousMedium;
    use crate::domain::source::GridSource;
    use crate::solver::forward::pstd::config::{AntiAliasingConfig, PSTDConfig};

    #[test]
    fn test_anti_aliasing_runs() {
        // Setup configuration
        let config = PSTDConfig {
            anti_aliasing: AntiAliasingConfig {
                enabled: true,
                cutoff: 0.8,
                order: 4,
            },
            dt: 1e-8,
            nt: 10,
            ..Default::default()
        };

        // Create Grid
        let grid = Grid::new(64, 64, 64, 0.001, 0.001, 0.001).unwrap();

        // Create Medium
        let medium = HomogeneousMedium::new(1000.0, 1500.0, 0.0, 0.0, &grid);

        // Create Source (empty)
        let source = GridSource::new_empty();

        // Create Solver
        let mut solver = PSTDSolver::new(config, grid, &medium, source).unwrap();

        // Run one step
        let result = solver.step_forward();
        assert!(
            result.is_ok(),
            "Step forward failed with anti-aliasing enabled: {:?}",
            result.err()
        );
    }
}
