//! PSTD Time-stepping and Propagation Logic

use super::orchestrator::PSTDSolver;
use crate::core::error::{KwaversError, KwaversResult};
use crate::domain::source::SourceField;

use crate::solver::forward::pstd::config::KSpaceMethod;
use crate::solver::forward::pstd::implementation::k_space::PSTDKSOperators;
use ndarray::{Array3, Zip};

impl PSTDSolver {
    /// Perform a single time step using k-space pseudospectral method
    pub fn step_forward(&mut self) -> KwaversResult<()> {
        let dt = self.config.dt;
        let time_index = self.time_step_index;

        if self.config.kspace_method == KSpaceMethod::FullKSpace {
            return self.step_forward_kspace(dt, time_index);
        }

        // Standard PSTD method
        if self.source_handler.has_pressure_source() {
            self.source_handler
                .inject_mass_source(time_index, &mut self.rho, &self.materials.c0);
        }

        self.update_pressure();
        self.update_velocity(dt)?;

        if self.source_handler.has_velocity_source() {
            self.source_handler.inject_force_source(
                time_index,
                &mut self.fields.ux,
                &mut self.fields.uy,
                &mut self.fields.uz,
            );
        }

        self.update_density(dt)?;
        self.apply_absorption(dt)?;

        if self.source_handler.pressure_mode() == crate::domain::source::SourceMode::Dirichlet {
            self.source_handler.inject_mass_source(
                time_index + 1,
                &mut self.rho,
                &self.materials.c0,
            );
        }

        self.update_pressure();

        // TODO: Implement apply_anti_aliasing_filter method
        // if self.filter.is_some() {
        //     self.apply_anti_aliasing_filter()?;
        // }

        self.apply_boundary(time_index)?;
        self.sensor_recorder.record_step(&self.fields.p)?;
        self.apply_dynamic_sources(dt);
        self.time_step_index += 1;

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
        if self.source_handler.has_pressure_source() {
            let mut temp_rho = Array3::<f64>::zeros(self.fields.p.dim());
            self.source_handler
                .inject_mass_source(time_index, &mut temp_rho, &self.materials.c0);

            Zip::from(&mut source_term)
                .and(&temp_rho)
                .and(&self.materials.c0)
                .for_each(|s, &rho, &c| *s = rho * c * c);
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

    pub(crate) fn apply_dynamic_sources(&mut self, dt: f64) {
        let t = self.time_step_index as f64 * dt;
        for (source, mask) in &self.dynamic_sources {
            let amp = source.amplitude(t);
            match source.source_type() {
                SourceField::Pressure => {
                    Zip::from(&mut self.rho)
                        .and(mask)
                        .and(&self.materials.c0)
                        .for_each(|rho, &m, &c| *rho += (m * amp) / (c * c));
                }
                SourceField::VelocityX => {
                    Zip::from(&mut self.fields.ux)
                        .and(mask)
                        .for_each(|u, &m| *u += m * amp);
                }
                SourceField::VelocityY => {
                    Zip::from(&mut self.fields.uy)
                        .and(mask)
                        .for_each(|u, &m| *u += m * amp);
                }
                SourceField::VelocityZ => {
                    Zip::from(&mut self.fields.uz)
                        .and(mask)
                        .for_each(|u, &m| *u += m * amp);
                }
            }
        }
    }
}
