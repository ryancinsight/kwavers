//! `ConservationDiagnostics` trait impl for the KZK solver.
//!
//! All conservation quantities are computed from the physical (real) pressure
//! `Re[p]`.  The imaginary component carries diffraction phase information
//! and does not represent additional acoustic energy.

use super::KZKSolver;
use crate::solver::forward::nonlinear::conservation::ConservationDiagnostics;

impl ConservationDiagnostics for KZKSolver {
    fn calculate_total_energy(&self) -> f64 {
        // Acoustic energy density: E = p²/(2ρ₀c₀²)
        // Total energy: ∫∫∫ E dV
        let mut total_energy = 0.0;
        let rho0 = self.config.rho0;
        let c0 = self.config.c0;
        let factor = 1.0 / (2.0 * rho0 * c0 * c0);
        let dv = self.config.dx * self.config.dx * self.config.dt * c0;

        for i in 0..self.config.nx {
            for j in 0..self.config.ny {
                for t in 0..self.config.nt {
                    let p = self.pressure[[i, j, t]].re;
                    total_energy += p * p * factor * dv;
                }
            }
        }

        total_energy
    }

    fn calculate_total_momentum(&self) -> (f64, f64, f64) {
        // Momentum density: ρ₀ u = p/c₀ (acoustic approximation)
        let mut pz = 0.0;
        let rho0 = self.config.rho0;
        let c0 = self.config.c0;
        let dv = self.config.dx * self.config.dx * self.config.dt * c0;

        for i in 0..self.config.nx {
            for j in 0..self.config.ny {
                for t in 0..self.config.nt {
                    let p = self.pressure[[i, j, t]].re;
                    pz += (rho0 * p / c0) * dv;
                }
            }
        }

        // KZK assumes predominantly z-directed propagation.
        (0.0, 0.0, pz)
    }

    fn calculate_total_mass(&self) -> f64 {
        // For acoustic waves: ρ = ρ₀(1 + p/(ρ₀c₀²))
        let mut total_mass = 0.0;
        let rho0 = self.config.rho0;
        let c0 = self.config.c0;
        let dv = self.config.dx * self.config.dx * self.config.dt * c0;

        for i in 0..self.config.nx {
            for j in 0..self.config.ny {
                for t in 0..self.config.nt {
                    let p = self.pressure[[i, j, t]].re;
                    let rho = rho0 * (1.0 + p / (rho0 * c0 * c0));
                    total_mass += rho * dv;
                }
            }
        }

        total_mass
    }
}
