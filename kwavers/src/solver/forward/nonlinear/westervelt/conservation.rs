//! `ConservationDiagnostics` trait impl for the Westervelt FDTD solver.
//!
//! Total acoustic energy `E = ∫ p²/(2ρ₀c₀²) dV`, momentum from pressure
//! gradients in the linear-acoustic limit (`ρ₀ u ≈ p/c₀` magnitude with
//! direction set by ∇p), and mass from the linear pressure-density relation
//! `ρ = ρ₀(1 + p/(ρ₀c₀²))`. All integrals use the cached `medium_properties`
//! center-point ρ₀, c₀ — heterogeneous extension is a future increment.

use super::WesterveltFdtd;
use crate::solver::forward::nonlinear::conservation::ConservationDiagnostics;

impl ConservationDiagnostics for WesterveltFdtd {
    fn calculate_total_energy(&self) -> f64 {
        // Acoustic energy density: E = p²/(2ρ₀c₀²)
        // Total energy: ∫∫∫ E dV
        let mut total_energy = 0.0;
        let rho0 = self.medium_properties.rho0;
        let c0 = self.medium_properties.c0;
        let factor = 1.0 / (2.0 * rho0 * c0 * c0);
        let dv = self.grid.dx * self.grid.dy * self.grid.dz; // Volume element

        for i in 0..self.grid.nx {
            for j in 0..self.grid.ny {
                for k in 0..self.grid.nz {
                    let p = self.pressure[[i, j, k]];
                    total_energy += p * p * factor * dv;
                }
            }
        }

        total_energy
    }

    fn calculate_total_momentum(&self) -> (f64, f64, f64) {
        // Full 3D momentum calculation
        // Momentum density: ρ₀ u where u = ∫ ∇p/(ρ₀) dt (acoustic approximation)
        // For simplicity, use p/c₀ approximation for magnitude
        let mut px = 0.0;
        let mut py = 0.0;
        let mut pz = 0.0;
        let rho0 = self.medium_properties.rho0;
        let c0 = self.medium_properties.c0;
        let dv = self.grid.dx * self.grid.dy * self.grid.dz;

        // Compute momentum from pressure gradients
        for i in 1..self.grid.nx - 1 {
            for j in 1..self.grid.ny - 1 {
                for k in 1..self.grid.nz - 1 {
                    // Pressure gradients (central difference)
                    let dp_dx = (self.pressure[[i + 1, j, k]] - self.pressure[[i - 1, j, k]])
                        / (2.0 * self.grid.dx);
                    let dp_dy = (self.pressure[[i, j + 1, k]] - self.pressure[[i, j - 1, k]])
                        / (2.0 * self.grid.dy);
                    let dp_dz = (self.pressure[[i, j, k + 1]] - self.pressure[[i, j, k - 1]])
                        / (2.0 * self.grid.dz);

                    // Momentum from pressure (acoustic approximation)
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
        // Total mass: ∫∫∫ ρ dV
        let mut total_mass = 0.0;
        let rho0 = self.medium_properties.rho0;
        let c0 = self.medium_properties.c0;
        let dv = self.grid.dx * self.grid.dy * self.grid.dz;

        for i in 0..self.grid.nx {
            for j in 0..self.grid.ny {
                for k in 0..self.grid.nz {
                    let p = self.pressure[[i, j, k]];
                    let rho = rho0 * (1.0 + p / (rho0 * c0 * c0));
                    total_mass += rho * dv;
                }
            }
        }

        total_mass
    }
}
