use super::WesterveltWave;
use crate::solver::forward::nonlinear::conservation::ConservationDiagnostics;

impl ConservationDiagnostics for WesterveltWave {
    fn calculate_total_energy(&self) -> f64 {
        // Acoustic energy density: E = p²/(2ρ₀c₀²); Total energy: ∫∫∫ E dV
        let mut total_energy = 0.0;

        if let (Some(ref grid), Some(ref props)) = (&self.grid_cache, &self.medium_properties) {
            let factor = 1.0 / (2.0 * props.rho0 * props.c0 * props.c0);
            let dv = grid.dx * grid.dy * grid.dz;

            let curr_idx = self.buffer_indices[1];
            let pressure = &self.pressure_buffers[curr_idx];

            for i in 0..grid.nx {
                for j in 0..grid.ny {
                    for k in 0..grid.nz {
                        let p = pressure[[i, j, k]];
                        total_energy += p * p * factor * dv;
                    }
                }
            }
        }

        total_energy
    }

    fn calculate_total_momentum(&self) -> (f64, f64, f64) {
        // Momentum density: ρ₀ u ≈ ρ₀ · ∇p/(ρ₀c₀) dt; full 3D central-difference.
        let mut px = 0.0;
        let mut py = 0.0;
        let mut pz = 0.0;

        if let (Some(ref grid), Some(ref props)) = (&self.grid_cache, &self.medium_properties) {
            let dv = grid.dx * grid.dy * grid.dz;

            let curr_idx = self.buffer_indices[1];
            let pressure = &self.pressure_buffers[curr_idx];

            for i in 1..grid.nx - 1 {
                for j in 1..grid.ny - 1 {
                    for k in 1..grid.nz - 1 {
                        let dp_dx =
                            (pressure[[i + 1, j, k]] - pressure[[i - 1, j, k]]) / (2.0 * grid.dx);
                        let dp_dy =
                            (pressure[[i, j + 1, k]] - pressure[[i, j - 1, k]]) / (2.0 * grid.dy);
                        let dp_dz =
                            (pressure[[i, j, k + 1]] - pressure[[i, j, k - 1]]) / (2.0 * grid.dz);

                        px += (props.rho0 * dp_dx / props.c0) * dv;
                        py += (props.rho0 * dp_dy / props.c0) * dv;
                        pz += (props.rho0 * dp_dz / props.c0) * dv;
                    }
                }
            }
        }

        (px, py, pz)
    }

    fn calculate_total_mass(&self) -> f64 {
        // For acoustic waves: ρ = ρ₀(1 + p/(ρ₀c₀²)); Total mass: ∫∫∫ ρ dV
        let mut total_mass = 0.0;

        if let (Some(ref grid), Some(ref props)) = (&self.grid_cache, &self.medium_properties) {
            let dv = grid.dx * grid.dy * grid.dz;

            let curr_idx = self.buffer_indices[1];
            let pressure = &self.pressure_buffers[curr_idx];

            for i in 0..grid.nx {
                for j in 0..grid.ny {
                    for k in 0..grid.nz {
                        let p = pressure[[i, j, k]];
                        let rho = props.rho0 * (1.0 + p / (props.rho0 * props.c0 * props.c0));
                        total_mass += rho * dv;
                    }
                }
            }
        }

        total_mass
    }
}
