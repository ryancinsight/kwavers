//! Full Westervelt time-step: linear + nonlinear + absorption + artificial
//! viscosity, source injection, history rotation, and conservation-check
//! pipeline.
//!
//! ## Absorption model
//!
//! Classical Stokes-Kirchhoff thermoviscous absorption in the frequency domain
//! gives plane-wave spatial attenuation `α(ω) = δω²/(2c₀³)` [Np m⁻¹], which
//! corresponds to temporal amplitude decay `exp(−α·c·t)` at a fixed spatial
//! location as the wave passes.
//!
//! For an explicit leapfrog scheme, discretizing the third-derivative operator
//! `∂³p/∂t³` with a backward-difference stencil is **unconditionally unstable**:
//! the growth-mode factor `(r−1)³ > 0` for any `r > 1`, so explicit third-
//! derivative terms feed energy into numerical instabilities.
//!
//! The stable replacement is the **multiplicative per-step decay**:
//!
//! ```text
//! p^{n+1} ← p^{n+1} · exp(−α · c · Δt)
//! ```
//!
//! This is the O(Δt) operator-splitting approximation to the exact exponential
//! spatial decay `exp(−α·x) = exp(−α·c·t)`. It is unconditionally stable for
//! all α ≥ 0 and Δt > 0, and converges to the exact Stokes attenuation law in
//! the limit `αcΔt → 0` (Pinton et al. 2009, §IIB).
//!
//! For frequency-dependent power-law absorption `α ∝ fʸ`, use the PSTD
//! fractional-Laplacian path (Treeby & Cox 2010) which applies the exact
//! spectral filter per step.
//!
//! ## Artificial viscosity
//!
//! The term `ν_art · Δt · ∇²p^n` reuses the already-computed Laplacian
//! workspace at the configured stencil order, incurring no additional FD
//! stencil evaluation.

use ndarray::Zip;
use tracing::warn;

use super::WesterveltFdtd;
use kwavers_core::error::KwaversResult;
use kwavers_grid::Grid;
use kwavers_medium::Medium;
use kwavers_source::Source;
use crate::forward::nonlinear::conservation::{ConservationDiagnostics, ViolationSeverity};

impl WesterveltFdtd {
    /// Advance the pressure field by one time step.
    ///
    /// Implements the explicit leapfrog update:
    ///
    /// ```text
    /// p^{n+1} = [2pⁿ − pⁿ⁻¹
    ///           + ((c·Δt)² + ν_art·Δt)·∇²pⁿ        (linear + viscosity)
    ///           + (β·Δt²)/(ρ·c²)·∂²(p²)/∂t²|ⁿ]    (nonlinear)
    ///           · exp(−α·c·Δt)                       (multiplicative absorption)
    /// ```
    ///
    /// # Errors
    /// Propagates any [`KwaversError`] from the Laplacian stencil.
    pub fn update(
        &mut self,
        medium: &dyn Medium,
        grid: &Grid,
        sources: &[Box<dyn Source>],
        t: f64,
        dt: f64,
    ) -> KwaversResult<()> {
        self.calculate_laplacian(grid)?;
        self.calculate_nonlinear_term_into(dt, grid);

        let pressure = &self.pressure;
        let pressure_prev = &self.pressure_prev;
        let laplacian = &self.laplacian;
        let nonlinear_term = &self.nonlinear_term;
        let pressure_next = &mut self.pressure_next;
        let enable_absorption = self.config.enable_absorption;
        let artificial_viscosity = self.config.artificial_viscosity;

        Zip::indexed(pressure_next)
            .and(pressure)
            .and(pressure_prev)
            .and(laplacian)
            .and(nonlinear_term)
            .par_for_each(|(i, j, k), p_next, &p, &p_prev, &lap, &nl| {
                let c = medium.sound_speed(i, j, k);
                let rho = medium.density(i, j, k);
                let beta = 1.0 + medium.nonlinearity(i, j, k) / 2.0;

                // Linear wave + artificial viscosity share the precomputed Laplacian.
                // artificial_viscosity·Δt·∇²p uses the same stencil order as the
                // main wave operator — no redundant stencil recomputation.
                let linear_and_visc = (c * dt).mul_add(c * dt, artificial_viscosity * dt) * lap;

                // Nonlinear coefficient β·Δt²/(ρ·c²)
                let nl_coeff = beta * dt * dt / (rho * c * c);

                // Leapfrog propagation step (Hamilton & Blackstock 1998 §3.5)
                let p_propagated = 2.0f64.mul_add(p, -p_prev) + linear_and_visc + nl_coeff * nl;

                // Multiplicative absorption (Stokes-Kirchhoff, O(Δt) operator splitting):
                //   p *= exp(−α·c·Δt)
                // Stable for all α ≥ 0.  See module-level doc for derivation.
                *p_next = if enable_absorption {
                    let alpha = medium.absorption(i, j, k);
                    p_propagated * (-alpha * c * dt).exp()
                } else {
                    p_propagated
                };
            });

        // Source injection: amplitude × Δt (Pa·s) applied as a pressure impulse.
        for source in sources {
            let amplitude = source.amplitude(t);
            if amplitude.abs() > 1e-12 {
                for position in source.positions() {
                    let i = ((position.0 / grid.dx).round() as usize).min(grid.nx - 1);
                    let j = ((position.1 / grid.dy).round() as usize).min(grid.ny - 1);
                    let k = ((position.2 / grid.dz).round() as usize).min(grid.nz - 1);
                    self.pressure_next[[i, j, k]] += amplitude * dt;
                }
            }
        }

        // History rotation: p2 ← p1 ← p ← p_next.
        // pressure_prev2 is allocated lazily on step 1 (first time p_prev is known
        // alongside the current p) and used by the nonlinear ∂²(p²)/∂t² kernel.
        if let Some(ref mut pp2) = self.pressure_prev2 {
            pp2.assign(&self.pressure_prev);
        } else {
            self.pressure_prev2 = Some(self.pressure_prev.clone());
        }
        std::mem::swap(&mut self.pressure_prev, &mut self.pressure);
        std::mem::swap(&mut self.pressure, &mut self.pressure_next);

        self.current_step += 1;
        self.current_time += dt;
        self.check_conservation_laws();

        Ok(())
    }

    fn check_conservation_laws(&mut self) {
        let should_check = self.conservation_tracker.as_ref().is_some_and(|tracker| {
            self.current_step
                .is_multiple_of(tracker.tolerances.check_interval)
        });
        if !should_check {
            return;
        }

        let (initial_energy, initial_momentum, initial_mass, tolerances) =
            if let Some(ref tracker) = self.conservation_tracker {
                (
                    tracker.initial_energy,
                    tracker.initial_momentum,
                    tracker.initial_mass,
                    tracker.tolerances,
                )
            } else {
                return;
            };

        let diagnostics = self.check_all_conservation(
            initial_energy,
            initial_momentum,
            initial_mass,
            self.current_step,
            self.current_time,
            &tolerances,
        );

        if let Some(ref mut tracker) = self.conservation_tracker {
            for diag in &diagnostics {
                if diag.severity > tracker.max_severity {
                    tracker.max_severity = diag.severity;
                }
            }
            tracker.history.extend(diagnostics.clone());
        }

        for diag in diagnostics {
            match diag.severity {
                ViolationSeverity::Acceptable => {}
                ViolationSeverity::Warning => {
                    warn!("Westervelt FDTD Conservation Warning: {}", diag);
                }
                ViolationSeverity::Error => {
                    warn!("Westervelt FDTD Conservation Error: {}", diag);
                }
                ViolationSeverity::Critical => {
                    warn!("Westervelt FDTD Conservation CRITICAL: {}", diag);
                    warn!("   Solution may be physically invalid!");
                }
            }
        }
    }
}
