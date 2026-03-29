//! PSTD Time-stepping and Propagation Logic

use super::orchestrator::PSTDSolver;
use crate::core::error::{KwaversError, KwaversResult};
use crate::domain::source::{SourceField, SourceInjectionMode};
use crate::math::fft::Complex64;

use crate::solver::forward::pstd::config::KSpaceMethod;
use crate::solver::forward::pstd::implementation::k_space::PSTDKSOperators;
use ndarray::{Array3, Zip};
use std::env;
use tracing::trace;

fn pstd_source_time_shift_samples() -> isize {
    match env::var("KWAVERS_PSTD_SOURCE_TIME_SHIFT") {
        Ok(value) => value.trim().parse::<isize>().unwrap_or(0),
        Err(_) => 0,
    }
}

fn pstd_source_gain() -> f64 {
    match env::var("KWAVERS_PSTD_SOURCE_GAIN") {
        Ok(value) => value.trim().parse::<f64>().unwrap_or(1.0),
        Err(_) => 1.0,
    }
}


impl PSTDSolver {
    /// Perform a single time step using k-space pseudospectral method
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

        // Step 1: Velocity update from pressure gradient
        self.update_velocity(dt)?;

        // Step 2: Velocity source injection
        self.apply_dynamic_velocity_sources(dt);
        if self.source_handler.has_velocity_source() {
            self.source_handler.inject_force_source(
                time_index,
                &mut self.fields.ux,
                &mut self.fields.uy,
                &mut self.fields.uz,
            );
        }

        // Step 3: Density update from velocity divergence
        self.update_density(dt)?;

        // Step 4: Pressure source injection (into split density)
        self.apply_pressure_sources(time_index, dt)?;

        // Step 5: Pressure computation from density (with absorption)
        self.update_pressure(dt)?;


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

        // NOTE: PML is already applied inside update_velocity() and
        // update_density() to the split velocity/density fields,
        // matching K-Wave's convention. Pressure (p = c²·Σρ) is computed
        // from the already-damped density, so no additional boundary
        // application is needed here.

        // 7. Record sensor data
        self.sensor_recorder.record_step(&self.fields.p)?;

        self.time_step_index += 1;

        Ok(())
    }

    fn apply_pressure_sources(&mut self, time_index: usize, dt: f64) -> KwaversResult<()> {
        let p_mode = self.source_handler.pressure_mode();
        let mut has_sources = false;
        let source_gain = pstd_source_gain();
        let shifted_time_index = {
            let shift = pstd_source_time_shift_samples();
            if shift >= 0 {
                time_index.saturating_add(shift as usize)
            } else {
                time_index.saturating_sub((-shift) as usize)
            }
        };

        // Reset source term buffer
        self.dpx.fill(0.0);

        if self.source_handler.has_pressure_source() {
            self.source_handler
                .add_pressure_source_into_density(shifted_time_index, &mut self.dpx);
            if source_gain != 1.0 {
                self.dpx.iter_mut().for_each(|v| *v *= source_gain);
            }
            has_sources = true;
        }

        // Dynamic sources (always additive in PSTD)
        // mass_source_scale = 2·Δt / (N·c₀·Δx_min) converts Pa → density source rate.
        // Precomputed at construction (grid constants never change during simulation).
        let t = shifted_time_index as f64 * dt;
        let mass_source_scale = self.mass_source_scale;

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
                        *p += m * amp * scale * mass_source_scale * source_gain;
                    }
                });
                has_sources = true;
            }
        }

        if !has_sources {
            return Ok(());
        }

        // Theorem (PSTD Split-Density Source Injection, Treeby & Cox 2010 Eq. 16–18):
        // k-Wave scales rho_scale = 2·Δt/(n_dim·c₀·Δx) and injects into n_dim density
        // components so total density change = n_dim × s.
        // kwavers always has 3 density components (ρₓ,ρᵧ,ρ_z) in the EOS p=c²·(ρₓ+ρᵧ+ρ_z).
        // For n_dim < 3, the source handler already divided by n_dim (not 3), so each component
        // must receive s·(n_dim/3) to keep the total density injection = n_dim × s_kwave.
        let n_dim_active = {
            let g = &*self.grid;
            [g.nx > 1, g.ny > 1, g.nz > 1]
                .iter()
                .filter(|&&d| d)
                .count()
                .max(1) as f64
        };
        let density_scale = n_dim_active / 3.0;

        match p_mode {
            crate::domain::source::SourceMode::Dirichlet => {
                // Dirichlet: enforce source values directly into all density components.
                // Each component = s·(n_dim/3) so total = n_dim·s = k-Wave's n_dim-component sum.
                Zip::from(&mut self.rhox)
                    .and(&mut self.rhoy)
                    .and(&mut self.rhoz)
                    .and(&self.dpx)
                    .for_each(|rx, ry, rz, &s| {
                        if s.abs() > 0.0 {
                            let v = s * density_scale;
                            *rx = v;
                            *ry = v;
                            *rz = v;
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
                        let v = s * density_scale;
                        *rx += v;
                        *ry += v;
                        *rz += v;
                    });
            }
            crate::domain::source::SourceMode::AdditiveNoCorrection => {
                Zip::from(&mut self.rhox)
                    .and(&mut self.rhoy)
                    .and(&mut self.rhoz)
                    .and(&self.dpx)
                    .for_each(|rx, ry, rz, &s| {
                        let v = s * density_scale;
                        *rx += v;
                        *ry += v;
                        *rz += v;
                    });
            }
        }

        Ok(())
    }

    /// Time step using full k-space pseudospectral method (dispersion-free)
    fn step_forward_kspace(&mut self, dt: f64, time_index: usize) -> KwaversResult<()> {
        // Reuse dpx as source_term scratch buffer (avoids per-step allocation)
        self.dpx.fill(0.0);

        // Apply source_handler sources (density-scaled values)
        if self.source_handler.has_pressure_source() {
            self.source_handler
                .add_pressure_source_into_density(time_index, &mut self.dpx);
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
                            Zip::from(&mut self.dpx).and(mask).for_each(|s, &m| {
                                if m.abs() > 1e-12 {
                                    *s += m * amp;
                                }
                            });
                        }
                        SourceInjectionMode::Additive { scale } => {
                            Zip::from(&mut self.dpx).and(mask).for_each(|s, &m| {
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

        // Temporarily take kspace_operators to break the borrow (avoids .clone() of arrays)
        let ops = self
            .kspace_operators
            .take()
            .ok_or_else(|| {
                KwaversError::Config(crate::core::error::ConfigError::InvalidValue {
                    parameter: "kspace_operators".to_string(),
                    value: "None".to_string(),
                    constraint: "k-space operators must be initialized for FullKSpace method"
                        .to_string(),
                })
            })?;

        // Swap source_term out of dpx to pass as separate argument
        let source_term = std::mem::replace(&mut self.dpx, Array3::zeros((0, 0, 0)));
        self.propagate_kspace(dt, &source_term, &ops)?;
        // Restore both (put full-size array back, discard empty placeholder)
        self.dpx = source_term;
        self.kspace_operators = Some(ops);

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
    use crate::domain::source::{GridSource, SourceMode};
    use crate::solver::forward::pstd::config::{AntiAliasingConfig, BoundaryConfig, PSTDConfig};

    /// Verify that additive pressure source injection produces correct sign pattern.
    ///
    /// Reference: k-Wave Python numpy diagnostic (`diag_source_injection_numpy.py`) confirms
    /// that for a point source at [N/2, N/2, N/2] with N=16:
    /// - p[N/2, N/2, N/2] (source point) > 0
    /// - p[0, N/2, N/2] (off-source) < 0
    ///
    /// If this test fails with p[0,8,8] > 0, the 3D FFT axis ordering does not match
    /// numpy.fftn, causing the spectral source injection to produce incorrect spatial patterns.
    #[test]
    fn test_source_injection_sign_matches_kwave() {
        let n = 16usize;
        let dx = 1e-3_f64;
        let c0 = 1500.0_f64;
        let rho0 = 1000.0_f64;
        let dt = 0.3 * dx / c0; // CFL=0.3 → dt ≈ 2e-7 s
        let src = n / 2; // 8

        let grid = Grid::new(n, n, n, dx, dx, dx).unwrap();
        let medium = HomogeneousMedium::new(rho0, c0, 0.0, 0.0, &grid);

        // Source at [8,8,8] with signal=[0, 1]: step1 → 0, step2 → 1 Pa injection
        let mut p_mask = ndarray::Array3::<f64>::zeros((n, n, n));
        p_mask[[src, src, src]] = 1.0;

        let mut p_signal = ndarray::Array2::<f64>::zeros((1, 2));
        p_signal[[0, 1]] = 1.0; // signal[0]=0, signal[1]=1

        let source = GridSource {
            p_mask: Some(p_mask),
            p_signal: Some(p_signal),
            p_mode: SourceMode::Additive,
            ..GridSource::new_empty()
        };

        let config = PSTDConfig {
            dt,
            nt: 2,
            boundary: BoundaryConfig::None, // No PML for clean 2-step test
            smooth_sources: false,
            ..Default::default()
        };

        let mut solver = PSTDSolver::new(config, grid, &medium, source).unwrap();

        // Step 1: time_index=0, signal[0]=0 → no injection, field stays zero
        solver.step_forward().unwrap();

        // Step 2: time_index=1, signal[1]=1 Pa → injection
        solver.step_forward().unwrap();

        let p_src = solver.fields.p[[src, src, src]];
        let p_off = solver.fields.p[[0, src, src]];

        // Source point should be strongly positive (~0.53 Pa from k-Wave reference)
        assert!(
            p_src > 0.1,
            "p at source [{src},{src},{src}] = {p_src:.6e}, expected ~0.53 Pa (positive)"
        );

        // Off-source point [0,8,8] must be NEGATIVE (k-Wave numpy confirms -4.89e-4 Pa)
        assert!(
            p_off < 0.0,
            "p at [0,{src},{src}] = {p_off:.6e}, expected NEGATIVE (k-Wave: -4.89e-4 Pa). \
             Positive result indicates 3D FFT axis ordering mismatch vs numpy.fftn."
        );
    }

    /// Verify that free wave propagation does not amplify the injected field.
    ///
    /// Root cause of the 2026-03-27 amplitude bug: Nyquist frequency bin was zeroed in
    /// ddx_k_shift_pos/neg operators, which removed ~18% of k-space energy from the
    /// velocity/density gradient computation. This caused a 1.64x amplitude amplification
    /// per free propagation step. This test guards against that regression.
    ///
    /// Reference values from k-Wave binary (N=16, no PML, signal=[0,1,0]):
    ///   step 2 (injection): p[8,8,8] = 0.5344 Pa
    ///   step 3 (free prop): p[8,8,8] = 0.1128 Pa   (ratio 0.211 = decay, not growth)
    ///   step 4 (free prop): p[8,8,8] = -0.3160 Pa  (sign flip, continued propagation)
    #[test]
    fn test_nyquist_not_zeroed_propagation_amplitude() {
        let n = 16usize;
        let dx = 1e-3_f64;
        let c0 = 1500.0_f64;
        let rho0 = 1000.0_f64;
        let dt = 0.3 * dx / c0;
        let src = n / 2;

        let grid = Grid::new(n, n, n, dx, dx, dx).unwrap();
        let medium = HomogeneousMedium::new(rho0, c0, 0.0, 0.0, &grid);

        // signal=[0,1,0,0]: inject once at step 2, then free propagation
        let mut p_mask = ndarray::Array3::<f64>::zeros((n, n, n));
        p_mask[[src, src, src]] = 1.0;
        let mut p_signal = ndarray::Array2::<f64>::zeros((1, 4));
        p_signal[[0, 1]] = 1.0;

        let source = GridSource {
            p_mask: Some(p_mask),
            p_signal: Some(p_signal),
            p_mode: SourceMode::Additive,
            ..GridSource::new_empty()
        };
        let config = PSTDConfig {
            dt,
            nt: 4,
            boundary: BoundaryConfig::None,
            smooth_sources: false,
            ..Default::default()
        };

        let mut solver = PSTDSolver::new(config, grid, &medium, source).unwrap();
        solver.step_forward().unwrap(); // step 1: signal=0, p stays zero
        solver.step_forward().unwrap(); // step 2: inject 1 Pa
        let p_step2 = solver.fields.p[[src, src, src]];
        solver.step_forward().unwrap(); // step 3: free propagation
        let p_step3 = solver.fields.p[[src, src, src]];

        // Injection must be substantial (k-Wave reference: ~0.534 Pa)
        assert!(
            p_step2 > 0.4,
            "step2 p[src] = {p_step2:.4e}, expected ~0.534 Pa"
        );

        // After one free propagation step, amplitude at source must DECREASE
        // (k-Wave: 0.1128 Pa, ratio ~0.211). If Nyquist is zeroed it gives 0.1853 Pa (1.64x)
        // which still passes < p_step2 but let's bound it tightly.
        let ratio = p_step3 / p_step2;
        assert!(
            ratio < 0.3,
            "step3 / step2 ratio = {ratio:.4} at source [{src},{src},{src}], expected ~0.211. \
             Ratio > 0.3 indicates Nyquist zeroing regression (step3 = {p_step3:.4e}, step2 = {p_step2:.4e})."
        );
        assert!(
            p_step3 > 0.0,
            "step3 p[src] = {p_step3:.4e}, expected positive after first free step"
        );
    }

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
