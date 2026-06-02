//! Interface detector iterator.

use crate::grid::Grid;
use crate::medium::Medium;

/// Iterator that finds interfaces in the medium
#[derive(Debug)]
pub struct InterfaceIterator<'a> {
    medium: &'a dyn Medium,
    grid: &'a Grid,
    threshold: f64,
    current: usize,
}

impl<'a> InterfaceIterator<'a> {
    pub fn new(medium: &'a dyn Medium, grid: &'a Grid, threshold: f64) -> Self {
        Self {
            medium,
            grid,
            threshold,
            current: 0,
        }
    }

    fn is_interface(&self, i: usize, j: usize, k: usize) -> bool {
        if i == 0
            || j == 0
            || k == 0
            || i >= self.grid.nx - 1
            || j >= self.grid.ny - 1
            || k >= self.grid.nz - 1
        {
            return false;
        }

        let (x, y, z) = self.grid.indices_to_coordinates(i, j, k);
        let center_density = crate::medium::density_at(self.medium, x, y, z, self.grid);

        let neighbors = [
            (i + 1, j, k),
            (i - 1, j, k),
            (i, j + 1, k),
            (i, j - 1, k),
            (i, j, k + 1),
            (i, j, k - 1),
        ];

        neighbors.iter().any(|&(ni, nj, nk)| {
            let (nx, ny, nz) = self.grid.indices_to_coordinates(ni, nj, nk);
            let neighbor_density =
                crate::medium::density_at(self.medium, nx, ny, nz, self.grid);
            ((neighbor_density - center_density).abs() / center_density) > self.threshold
        })
    }

    fn calculate_density_jump(&self, i: usize, j: usize, k: usize) -> f64 {
        let (x, y, z) = self.grid.indices_to_coordinates(i, j, k);
        let center = crate::medium::density_at(self.medium, x, y, z, self.grid);

        let mut max_jump = 0.0;
        for di in -1..=1 {
            for dj in -1..=1 {
                for dk in -1..=1 {
                    if di == 0 && dj == 0 && dk == 0 {
                        continue;
                    }

                    let ni = (i as i32 + di) as usize;
                    let nj = (j as i32 + dj) as usize;
                    let nk = (k as i32 + dk) as usize;

                    if ni < self.grid.nx && nj < self.grid.ny && nk < self.grid.nz {
                        let (nx, ny, nz) = self.grid.indices_to_coordinates(ni, nj, nk);
                        let neighbor =
                            crate::medium::density_at(self.medium, nx, ny, nz, self.grid);
                        max_jump = f64::max(max_jump, (neighbor - center).abs());
                    }
                }
            }
        }

        max_jump
    }

    fn calculate_normal(&self, i: usize, j: usize, k: usize) -> (f64, f64, f64) {
        let (x, y, z) = self.grid.indices_to_coordinates(i, j, k);

        let dx = if i > 0 && i < self.grid.nx - 1 {
            let (x1, _, _) = self.grid.indices_to_coordinates(i + 1, j, k);
            let (x0, _, _) = self.grid.indices_to_coordinates(i - 1, j, k);
            let d1 = crate::medium::density_at(self.medium, x1, y, z, self.grid);
            let d0 = crate::medium::density_at(self.medium, x0, y, z, self.grid);
            (d1 - d0) / (2.0 * self.grid.dx)
        } else {
            0.0
        };

        let dy = if j > 0 && j < self.grid.ny - 1 {
            let (_, y1, _) = self.grid.indices_to_coordinates(i, j + 1, k);
            let (_, y0, _) = self.grid.indices_to_coordinates(i, j - 1, k);
            let d1 = crate::medium::density_at(self.medium, x, y1, z, self.grid);
            let d0 = crate::medium::density_at(self.medium, x, y0, z, self.grid);
            (d1 - d0) / (2.0 * self.grid.dy)
        } else {
            0.0
        };

        let dz = if k > 0 && k < self.grid.nz - 1 {
            let (_, _, z1) = self.grid.indices_to_coordinates(i, j, k + 1);
            let (_, _, z0) = self.grid.indices_to_coordinates(i, j, k - 1);
            let d1 = crate::medium::density_at(self.medium, x, y, z1, self.grid);
            let d0 = crate::medium::density_at(self.medium, x, y, z0, self.grid);
            (d1 - d0) / (2.0 * self.grid.dz)
        } else {
            0.0
        };

        let mag = (dx * dx + dy * dy + dz * dz).sqrt();
        if mag > 0.0 {
            (dx / mag, dy / mag, dz / mag)
        } else {
            (0.0, 0.0, 0.0)
        }
    }
}

impl<'a> Iterator for InterfaceIterator<'a> {
    type Item = IteratorInterfacePoint;

    fn next(&mut self) -> Option<Self::Item> {
        while self.current < self.grid.size() {
            let k = self.current % self.grid.nz;
            let j = (self.current / self.grid.nz) % self.grid.ny;
            let i = self.current / (self.grid.ny * self.grid.nz);

            self.current += 1;

            if self.is_interface(i, j, k) {
                let (x, y, z) = self.grid.indices_to_coordinates(i, j, k);
                return Some(IteratorInterfacePoint {
                    indices: (i, j, k),
                    position: (x, y, z),
                    density_jump: self.calculate_density_jump(i, j, k),
                    normal: self.calculate_normal(i, j, k),
                });
            }
        }

        None
    }
}

#[derive(Debug, Clone)]
pub struct IteratorInterfacePoint {
    pub indices: (usize, usize, usize),
    pub position: (f64, f64, f64),
    pub density_jump: f64,
    pub normal: (f64, f64, f64),
}
