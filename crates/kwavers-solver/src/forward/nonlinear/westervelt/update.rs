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

mod pressure;
use tracing::warn;

use super::WesterveltFdtd;
use crate::forward::nonlinear::conservation::{ConservationDiagnostics, ViolationSeverity};
use kwavers_core::error::KwaversResult;
use kwavers_grid::Grid;
use kwavers_medium::Medium;
use kwavers_source::Source;

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
    /// Propagates any [`crate::KwaversError`] from the Laplacian stencil.
    ///
    /// # Panics
    ///
    /// Panics if an internal field loses its contiguous layout.
    pub fn update(
        &mut self,
        medium: &dyn Medium,
        grid: &Grid,
        sources: &[Box<dyn Source>],
        t: f64,
        dt: f64,
    ) -> KwaversResult<()> {
        let dt = match self.config.propagation_direction {
            super::PropagationDirection::Forward => dt,
            super::PropagationDirection::Backward => -dt,
        };

        self.calculate_laplacian(grid)?;
        if self.current_step == 0 {
            self.advance_pressure::<false>(medium, grid, dt);
        } else {
            self.advance_pressure::<true>(medium, grid, dt);
        }

        // Source injection: amplitude × Δt (Pa·s) applied as a pressure impulse.
        // `for_each_position` avoids allocating a temporary position Vec on the
        // timestep path while retaining the open `Source` trait boundary.
        for source in sources {
            let amplitude = source.amplitude(t);
            if amplitude.abs() > 1e-12 {
                let mut inject = |position: (f64, f64, f64)| {
                    let i = ((position.0 / grid.dx).round() as usize).min(grid.nx - 1);
                    let j = ((position.1 / grid.dy).round() as usize).min(grid.ny - 1);
                    let k = ((position.2 / grid.dz).round() as usize).min(grid.nz - 1);
                    self.pressure_next[[i, j, k]] += amplitude * dt;
                };
                source.for_each_position(&mut inject);
            }
        }

        // The old previous field becomes the output buffer for the next step.
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
