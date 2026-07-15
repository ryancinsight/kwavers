//! Pressure and velocity source injection for `PSTDSolver`.

use super::super::orchestrator::PSTDSolver;
use super::ops::{
    add_density_source_components, add_masked_source_term, multiply_complex_by_real_field,
    scale_real_field,
};
use crate::geometry::SolverGeometry;
use kwavers_core::error::KwaversResult;
use kwavers_math::fft::Fft3dInOutExt;
use kwavers_source::{SourceField, SourceInjectionMode};

impl PSTDSolver {
    /// Apply pressure sources.
    /// # Errors
    /// - Returns [`Err`] if an internal constraint is violated.
    ///
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
                scale_real_field(&mut self.dpx, source_gain);
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

                add_masked_source_term(
                    &mut self.dpx,
                    mask,
                    amp * scale * mass_source_scale * source_gain,
                );
                has_sources = true;
            }
        }

        if !has_sources {
            return Ok(());
        }

        // Theorem (PSTD Split-Density Source Injection, Treeby & Cox 2010 Eq. 16–18):
        //
        // k-Wave injects into exactly n_active density components with per-component scale
        //   rho_scale = 2·Δt / (n_active · c₀ · Δx)
        // so that the total density change per step = n_active × rho_scale × s = 2Δt·s/(c₀·Δx).
        //
        // kwavers has 3 split-density arrays (ρₓ,ρᵧ,ρ_z) in the EOS p = c²·(ρₓ+ρᵧ+ρ_z).
        // Each inactive dimension (dim=1) receives no divergence update from the propagator,
        // so injecting into its density array causes monotonic accumulation that grows without
        // bound over all Nt steps — a static pressure bias scaling with Nt × source amplitude.
        //
        // Correctness invariant: inject ONLY into active-dimension density arrays (n_active of them),
        // with density_scale = 1.0 so that:
        //   total EOS contribution per step = n_active × dpx × 1.0
        //   dpx = signal × 2Δt / (n_active · c₀ · Δx)   (from source_handler scaling)
        //   → total = 2Δt·signal / (c₀·Δx)   ← matches k-Wave identically
        //
        // Special case — CylindricalAS: only ρₓ (axial) and ρ_z (radial) are updated by the
        // propagator; ρᵧ must NOT receive injection.
        let is_axisymmetric = self.config.geometry == SolverGeometry::CylindricalAS;

        // Active-dimension flags for the Cartesian non-axisymmetric path.
        let has_y = self.grid.ny > 1;
        let has_z = self.grid.nz > 1;

        match p_mode {
            kwavers_source::SourceMode::Dirichlet => {
                // Density evolves naturally at sensor locations. The direct pressure
                // override p[sensor] = data[t] is applied post-EOS in step_forward
                // via enforce_pressure_dirichlet, matching KWave.jl's
                // time_reversal_boundary_data which sets p after the density→pressure
                // update rather than pre-setting density components.
            }
            kwavers_source::SourceMode::Additive => {
                // R2C: dpx (nx,ny,nz) → p_k (nx,ny,nz_c); source_kappa pre-truncated to nz_c.
                self.fft.forward_r2c_into(&self.dpx, &mut self.p_k);
                multiply_complex_by_real_field(&mut self.p_k, &self.source_kappa);
                self.fft
                    .inverse_c2r_into(&self.p_k, &mut self.dpx, &mut self.ux_k);

                if is_axisymmetric {
                    add_density_source_components(
                        &mut self.rhox,
                        None,
                        Some(&mut self.rhoz),
                        &self.dpx,
                    );
                } else {
                    match (has_y, has_z) {
                        (true, true) => {
                            add_density_source_components(
                                &mut self.rhox,
                                Some(&mut self.rhoy),
                                Some(&mut self.rhoz),
                                &self.dpx,
                            );
                        }
                        (true, false) => {
                            // Quasi-2D (NZ=1): ρ_z never receives z-divergence update;
                            // injecting here causes unbounded accumulation → must be skipped.
                            add_density_source_components(
                                &mut self.rhox,
                                Some(&mut self.rhoy),
                                None,
                                &self.dpx,
                            );
                        }
                        (false, true) => {
                            // XZ plane (NY=1): ρᵧ has no y-divergence update.
                            add_density_source_components(
                                &mut self.rhox,
                                None,
                                Some(&mut self.rhoz),
                                &self.dpx,
                            );
                        }
                        (false, false) => {
                            // 1D (NY=1, NZ=1): only ρₓ participates.
                            add_density_source_components(&mut self.rhox, None, None, &self.dpx);
                        }
                    }
                }
            }
            kwavers_source::SourceMode::AdditiveNoCorrection => {
                if is_axisymmetric {
                    add_density_source_components(
                        &mut self.rhox,
                        None,
                        Some(&mut self.rhoz),
                        &self.dpx,
                    );
                } else {
                    match (has_y, has_z) {
                        (true, true) => {
                            add_density_source_components(
                                &mut self.rhox,
                                Some(&mut self.rhoy),
                                Some(&mut self.rhoz),
                                &self.dpx,
                            );
                        }
                        (true, false) => {
                            // Quasi-2D (NZ=1): ρ_z must not receive injection.
                            add_density_source_components(
                                &mut self.rhox,
                                Some(&mut self.rhoy),
                                None,
                                &self.dpx,
                            );
                        }
                        (false, true) => {
                            // XZ plane (NY=1): ρᵧ must not receive injection.
                            add_density_source_components(
                                &mut self.rhox,
                                None,
                                Some(&mut self.rhoz),
                                &self.dpx,
                            );
                        }
                        (false, false) => {
                            // 1D (NY=1, NZ=1): only ρₓ participates.
                            add_density_source_components(&mut self.rhox, None, None, &self.dpx);
                        }
                    }
                }
            }
        }

        Ok(())
    }

    /// Apply dynamic velocity sources registered via the `Arc<dyn Source>`
    /// API (e.g. elastic mode-isolation tests).
    ///
    /// **Note on scaling.** The k-Wave additive scaling
    /// `2·c₀·Δt/Δα` for velocity sources is applied in the
    /// [`SourceHandler::inject_force_source`] path (used by
    /// `Source.from_velocity_mask` / k-Wave-style `u_mask + u_signal`).
    /// The dynamic-source path here is reserved for the elastic
    /// mode-isolation API where the user-supplied amplitude IS the velocity
    /// to be injected (matching the `DIR-DIR` peak-ratio = 0.99 fixture in
    /// the elastic 2×2 parity study); applying the scaling here would
    /// regress that contract. If a future caller needs k-Wave-style scaling
    /// on the dynamic-source path, bridge it via the `SourceHandler` API
    /// instead.
    pub(crate) fn apply_dynamic_velocity_sources(&mut self, dt: f64) {
        let t = self.time_step_index as f64 * dt;
        for (source, mask) in &self.dynamic_sources {
            let amp = source.amplitude(t);
            if amp.abs() < 1e-12 {
                continue;
            }

            match source.source_type() {
                SourceField::VelocityX => {
                    add_masked_source_term(&mut self.fields.ux, mask, amp);
                }
                SourceField::VelocityY => {
                    add_masked_source_term(&mut self.fields.uy, mask, amp);
                }
                SourceField::VelocityZ => {
                    add_masked_source_term(&mut self.fields.uz, mask, amp);
                }
                SourceField::Pressure => {}
            }
        }
    }
}
