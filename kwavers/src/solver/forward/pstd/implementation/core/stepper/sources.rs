//! Pressure and velocity source injection for `PSTDSolver`.

use super::super::orchestrator::PSTDSolver;
use crate::core::error::KwaversResult;
use crate::domain::source::{SourceField, SourceInjectionMode};
use crate::math::fft::Complex64;
use crate::solver::geometry::Geometry;
use ndarray::{s, Zip};

impl PSTDSolver {
    pub(super) fn apply_pressure_sources(
        &mut self,
        time_index: usize,
        dt: f64,
    ) -> KwaversResult<()> {
        let p_mode = self.source_handler.pressure_mode();
        let mut has_sources = false;
        let source_gain = self.source_gain;
        let shifted_time_index = {
            let shift = self.source_time_shift_samples;
            if shift >= 0 {
                time_index.saturating_add(shift as usize)
            } else {
                time_index.saturating_sub((-shift) as usize)
            }
        };

        self.dpx.fill(0.0);

        if self.source_handler.has_pressure_source() {
            self.source_handler
                .add_pressure_source_into_density(shifted_time_index, &mut self.dpx);
            if source_gain != 1.0 {
                self.dpx.iter_mut().for_each(|v| *v *= source_gain);
            }
            has_sources = true;
        }

        // mass_source_scale = 2·Δt / (N·c₀·Δx_min) converts Pa → density source rate.
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
        //
        // CylindricalAS: only ρₓ (axial) and ρ_z (radial) are updated; ρᵧ must NOT
        // receive source injection (no divergence update → accumulates without decay).
        let is_axisymmetric = self.config.geometry == Geometry::CylindricalAS;
        let (n_dim_active, density_scale) = if is_axisymmetric {
            (2.0_f64, 1.0_f64)
        } else {
            let n = {
                let g = &*self.grid;
                [g.nx > 1, g.ny > 1, g.nz > 1]
                    .iter()
                    .filter(|&&d| d)
                    .count()
                    .max(1) as f64
            };
            (n, n / 3.0)
        };
        let _ = n_dim_active;

        match p_mode {
            crate::domain::source::SourceMode::Dirichlet => {
                if is_axisymmetric {
                    Zip::from(&mut self.rhox)
                        .and(&mut self.rhoz)
                        .and(&self.dpx)
                        .for_each(|rx, rz, &s| {
                            if s.abs() > 0.0 {
                                *rx = s * density_scale;
                                *rz = s * density_scale;
                            }
                        });
                } else {
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
            }
            crate::domain::source::SourceMode::Additive => {
                // R2C: dpx (nx,ny,nz) → p_k (nx,ny,nz_c); source_kappa sliced to nz_c.
                let nz_c = self.p_k.dim().2;
                self.fft.forward_r2c_into(&self.dpx, &mut self.p_k);
                Zip::from(&mut self.p_k)
                    .and(self.source_kappa.slice(s![.., .., ..nz_c]))
                    .for_each(|val, &k| {
                        *val *= Complex64::new(k, 0.0);
                    });
                self.fft
                    .inverse_c2r_into(&self.p_k, &mut self.dpx, &mut self.ux_k);

                if is_axisymmetric {
                    Zip::from(&mut self.rhox)
                        .and(&mut self.rhoz)
                        .and(&self.dpx)
                        .for_each(|rx, rz, &s| {
                            let v = s * density_scale;
                            *rx += v;
                            *rz += v;
                        });
                } else {
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
            crate::domain::source::SourceMode::AdditiveNoCorrection => {
                if is_axisymmetric {
                    Zip::from(&mut self.rhox)
                        .and(&mut self.rhoz)
                        .and(&self.dpx)
                        .for_each(|rx, rz, &s| {
                            let v = s * density_scale;
                            *rx += v;
                            *rz += v;
                        });
                } else {
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
        }

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
}
