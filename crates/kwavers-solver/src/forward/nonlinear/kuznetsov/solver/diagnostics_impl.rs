//! `ConservationDiagnostics` trait implementation for `KuznetsovWave`.

use super::wave::KuznetsovWave;
use crate::forward::nonlinear::conservation::ConservationDiagnostics;

impl ConservationDiagnostics for KuznetsovWave {
    fn calculate_total_energy(&self) -> f64 {
        // Acoustic energy density: E = p²/(2ρ₀c₀²)
        // Total energy: ∫∫∫ E dV
        let rho0 = self.medium_properties.rho0;
        let c0 = self.medium_properties.c0;
        let factor = 1.0 / (2.0 * rho0 * c0 * c0);
        let dv = self.grid.dx * self.grid.dy * self.grid.dz;
        let mut total_energy = 0.0;

        for i in 0..self.grid.nx {
            for j in 0..self.grid.ny {
                for k in 0..self.grid.nz {
                    let p = self.pressure_current[[i, j, k]];
                    total_energy += p * p * factor * dv;
                }
            }
        }

        total_energy
    }

    fn calculate_total_momentum(&self) -> (f64, f64, f64) {
        // Momentum density: ρ₀ u where u ≈ ∇p/(ρ₀c₀) (acoustic approximation)
        let rho0 = self.medium_properties.rho0;
        let c0 = self.medium_properties.c0;
        let dv = self.grid.dx * self.grid.dy * self.grid.dz;
        let mut px = 0.0;
        let mut py = 0.0;
        let mut pz = 0.0;

        for i in 1..self.grid.nx - 1 {
            for j in 1..self.grid.ny - 1 {
                for k in 1..self.grid.nz - 1 {
                    let dp_dx = (self.pressure_current[[i + 1, j, k]]
                        - self.pressure_current[[i - 1, j, k]])
                        / (2.0 * self.grid.dx);
                    let dp_dy = (self.pressure_current[[i, j + 1, k]]
                        - self.pressure_current[[i, j - 1, k]])
                        / (2.0 * self.grid.dy);
                    let dp_dz = (self.pressure_current[[i, j, k + 1]]
                        - self.pressure_current[[i, j, k - 1]])
                        / (2.0 * self.grid.dz);

                    px += (rho0 * dp_dx / c0) * dv;
                    py += (rho0 * dp_dy / c0) * dv;
                    pz += (rho0 * dp_dz / c0) * dv;
                }
            }
        }

        (px, py, pz)
    }

    fn calculate_total_mass(&self) -> f64 {
        // For acoustic waves: ρ = ρ₀(1 + p/(ρ₀c₀²))
        let rho0 = self.medium_properties.rho0;
        let c0 = self.medium_properties.c0;
        let dv = self.grid.dx * self.grid.dy * self.grid.dz;
        let mut total_mass = 0.0;

        for i in 0..self.grid.nx {
            for j in 0..self.grid.ny {
                for k in 0..self.grid.nz {
                    let p = self.pressure_current[[i, j, k]];
                    let rho = rho0 * (1.0 + p / (rho0 * c0 * c0));
                    total_mass += rho * dv;
                }
            }
        }

        total_mass
    }
}
