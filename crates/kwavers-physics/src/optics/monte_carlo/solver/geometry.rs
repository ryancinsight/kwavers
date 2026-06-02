//! Voxel geometry helpers for Monte Carlo photon transport.

use super::MonteCarloSolver;
use crate::optics::monte_carlo::photon::Photon;

impl MonteCarloSolver {
    /// Compute the inward surface normal at the voxel face the photon is crossing.
    pub(super) fn voxel_boundary_normal(
        &self,
        pos: [f64; 3],
        dir: [f64; 3],
        i: usize,
        j: usize,
        k: usize,
    ) -> [f64; 3] {
        let x0 = i as f64 * self.grid.dx;
        let y0 = j as f64 * self.grid.dy;
        let z0 = k as f64 * self.grid.dz;
        let x1 = x0 + self.grid.dx;
        let y1 = y0 + self.grid.dy;
        let z1 = z0 + self.grid.dz;

        let tx = if dir[0] > 1e-12 {
            (x1 - pos[0]) / dir[0]
        } else if dir[0] < -1e-12 {
            (x0 - pos[0]) / dir[0]
        } else {
            f64::INFINITY
        };
        let ty = if dir[1] > 1e-12 {
            (y1 - pos[1]) / dir[1]
        } else if dir[1] < -1e-12 {
            (y0 - pos[1]) / dir[1]
        } else {
            f64::INFINITY
        };
        let tz = if dir[2] > 1e-12 {
            (z1 - pos[2]) / dir[2]
        } else if dir[2] < -1e-12 {
            (z0 - pos[2]) / dir[2]
        } else {
            f64::INFINITY
        };

        if tx <= ty && tx <= tz {
            if dir[0] > 0.0 {
                [-1.0, 0.0, 0.0]
            } else {
                [1.0, 0.0, 0.0]
            }
        } else if ty <= tz {
            if dir[1] > 0.0 {
                [0.0, -1.0, 0.0]
            } else {
                [0.0, 1.0, 0.0]
            }
        } else if dir[2] > 0.0 {
            [0.0, 0.0, -1.0]
        } else {
            [0.0, 0.0, 1.0]
        }
    }

    /// Convert position to voxel indices.
    pub(crate) fn position_to_voxel(&self, pos: [f64; 3]) -> Option<(usize, usize, usize)> {
        if pos[0] < 0.0
            || pos[1] < 0.0
            || pos[2] < 0.0
            || pos[0] >= self.grid.dx * self.grid.nx as f64
            || pos[1] >= self.grid.dy * self.grid.ny as f64
            || pos[2] >= self.grid.dz * self.grid.nz as f64
        {
            return None;
        }

        let i = (pos[0] / self.grid.dx).floor() as usize;
        let j = (pos[1] / self.grid.dy).floor() as usize;
        let k = (pos[2] / self.grid.dz).floor() as usize;

        if i < self.grid.nx && j < self.grid.ny && k < self.grid.nz {
            Some((i, j, k))
        } else {
            None
        }
    }

    /// Specular reflection at simulation bounds.
    pub(super) fn handle_boundary(&self, photon: &mut Photon) {
        let bounds = [
            self.grid.dx * self.grid.nx as f64,
            self.grid.dy * self.grid.ny as f64,
            self.grid.dz * self.grid.nz as f64,
        ];

        for (axis, bound) in bounds.iter().enumerate() {
            if photon.position[axis] <= 0.0 {
                photon.direction[axis] = photon.direction[axis].abs();
            } else if photon.position[axis] >= *bound {
                photon.direction[axis] = -photon.direction[axis].abs();
            }
        }
    }
}
