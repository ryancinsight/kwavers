//! PSTD Time-stepping and Propagation Logic

use super::orchestrator::PSTDSolver;
use crate::core::error::{KwaversError, KwaversResult};
use crate::domain::source::SourceField;
use crate::math::fft::Complex64;

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

        if self.filter.is_some() {
            self.apply_anti_aliasing_filter()?;
        }

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

            // Apply filter to density
            // Use p_k as transform buffer and ux_k as scratch
            self.fft.forward_into(&self.rho, &mut self.p_k);
            Zip::from(&mut self.p_k)
                .and(filter)
                .for_each(|val, &f| *val *= Complex64::new(f, 0.0));
            self.fft
                .inverse_into(&self.p_k, &mut self.rho, &mut self.ux_k);

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
        let mut config = PSTDConfig::default();
        config.anti_aliasing = AntiAliasingConfig {
            enabled: true,
            cutoff: 0.8,
            order: 4,
        };
        config.dt = 1e-8;
        config.nt = 10;

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
        assert!(result.is_ok(), "Step forward failed with anti-aliasing enabled: {:?}", result.err());
    }
}
