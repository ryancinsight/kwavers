//! Full Westervelt time-step: linear + nonlinear + absorption + artificial
//! viscosity, source injection, history rotation, and conservation-check
//! pipeline.

use log::warn;
use ndarray::{Array3, Zip};

use super::WesterveltFdtd;
use crate::core::error::KwaversResult;
use crate::domain::grid::Grid;
use crate::domain::medium::Medium;
use crate::domain::source::Source;
use crate::solver::forward::nonlinear::conservation::{ConservationDiagnostics, ViolationSeverity};

impl WesterveltFdtd {
    /// Update the pressure field for one time step
    pub fn update(
        &mut self,
        medium: &dyn Medium,
        grid: &Grid,
        sources: &[Box<dyn Source>],
        t: f64,
        dt: f64,
    ) -> KwaversResult<()> {
        // Calculate Laplacian of current pressure
        self.calculate_laplacian(grid)?;

        // Calculate nonlinear term
        let nonlinear_term = self.calculate_nonlinear_term(dt, grid);

        // Create new pressure array
        let mut pressure_next = Array3::zeros((grid.nx, grid.ny, grid.nz));

        // Update pressure using Westervelt equation
        Zip::indexed(&mut pressure_next)
            .and(&self.pressure)
            .and(&self.pressure_prev)
            .and(&self.laplacian)
            .and(&nonlinear_term)
            .for_each(|(i, j, k), p_next, &p, &p_prev, &lap, &nl| {
                let x = i as f64 * grid.dx;
                let y = j as f64 * grid.dy;
                let z = k as f64 * grid.dz;

                // Get medium properties
                let c = crate::domain::medium::sound_speed_at(medium, x, y, z, grid);
                let rho = crate::domain::medium::density_at(medium, x, y, z, grid);
                let beta = crate::domain::medium::AcousticProperties::nonlinearity_coefficient(
                    medium, x, y, z, grid,
                );

                // Linear wave propagation term
                let linear_term = (c * dt).powi(2) * lap;

                // Nonlinear term coefficient
                let nl_coeff = beta * dt.powi(2) / (rho * c.powi(2));

                // Absorption term (if enabled)
                let absorption_term = if self.config.enable_absorption {
                    if let Some(ref p_prev2) = self.pressure_prev2 {
                        let alpha =
                            crate::domain::medium::AcousticProperties::absorption_coefficient(
                                medium, x, y, z, grid, 1e6,
                            ); // 1 MHz reference
                               // Theorem (Diffusivity of Sound, Hamilton & Blackstock 1998, Ch. 3 Eq. 3.64):
                               // δ = 2·α·c³ / ω²  where ω = 2π·f_ref (α evaluated at f_ref = 1 MHz).
                               // Leapfrog absorption contribution to p^{n+1}:
                               //   −dt²·(δ/c²)·∂³p/∂t³ ≈ −(δ/c²)·(p−2p''+p'')/ dt
                               // Ref: Hamilton & Blackstock (1998), Nonlinear Acoustics, Academic Press.
                        let f_ref = 1.0e6_f64; // 1 MHz reference frequency (matches alpha query above)
                        let omega_ref = 2.0 * std::f64::consts::PI * f_ref; // ω = 2π·f_ref
                        let delta = 2.0 * alpha * c.powi(3) / omega_ref.powi(2);
                        delta * dt / c.powi(2) * (p - 2.0 * p_prev + p_prev2[[i, j, k]]) / (dt * dt)
                    } else {
                        0.0
                    }
                } else {
                    0.0
                };

                // Artificial viscosity term for numerical stability
                // ∇·(ν ∇p) where ν is artificial viscosity coefficient
                let visc_term = if i > 0
                    && i < grid.nx - 1
                    && j > 0
                    && j < grid.ny - 1
                    && k > 0
                    && k < grid.nz - 1
                {
                    let dx2 = grid.dx * grid.dx;
                    let dy2 = grid.dy * grid.dy;
                    let dz2 = grid.dz * grid.dz;

                    // Laplacian of pressure for viscosity
                    let lap_p = (self.pressure[(i + 1, j, k)] - 2.0 * p
                        + self.pressure[(i - 1, j, k)])
                        / dx2
                        + (self.pressure[(i, j + 1, k)] - 2.0 * p + self.pressure[(i, j - 1, k)])
                            / dy2
                        + (self.pressure[(i, j, k + 1)] - 2.0 * p + self.pressure[(i, j, k - 1)])
                            / dz2;

                    self.config.artificial_viscosity * dt * lap_p
                } else {
                    0.0
                };

                // Update equation: p^{n+1} = 2p^n - p^{n-1} + linear + nonlinear + absorption + viscosity
                *p_next =
                    2.0 * p - p_prev + linear_term - nl_coeff * nl - absorption_term + visc_term;

                // No explicit pressure clamping - allows natural shock formation through nonlinearity
                // Stability maintained through CFL conditions and artificial viscosity
            });

        // Add source contributions
        for source in sources {
            let amplitude = source.amplitude(t);
            if amplitude.abs() > 1e-12 {
                // Source is active if amplitude is non-zero
                let positions = source.positions();
                for position in positions {
                    // Find nearest grid point
                    let i = ((position.0 / grid.dx).round() as usize).min(grid.nx - 1);
                    let j = ((position.1 / grid.dy).round() as usize).min(grid.ny - 1);
                    let k = ((position.2 / grid.dz).round() as usize).min(grid.nz - 1);

                    pressure_next[[i, j, k]] += amplitude * dt;
                }
            }
        }

        // Update pressure history
        if self.config.enable_absorption {
            self.pressure_prev2 = Some(self.pressure_prev.clone());
        }
        self.pressure_prev = self.pressure.clone();
        self.pressure = pressure_next;

        // Update step counters
        self.current_step += 1;
        self.current_time += dt;

        // Conservation diagnostics (if enabled)
        self.check_conservation_laws();

        Ok(())
    }

    /// Check conservation laws and log diagnostics
    ///
    /// Performs conservation checks at configured intervals and logs violations.
    /// Critical violations trigger warnings via tracing infrastructure.
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
