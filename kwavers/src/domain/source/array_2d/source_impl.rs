//! `impl Source for TransducerArray2D`.

use super::array::TransducerArray2D;
use crate::domain::grid::Grid;
use crate::domain::signal::Signal;
use crate::domain::source::Source;
use ndarray::Array3;

impl Source for TransducerArray2D {
    fn create_mask(&self, grid: &Grid) -> Array3<f64> {
        let grid_ptr: *const Grid = grid;
        let grid_id = grid_ptr as u64;
        if let Some(ref mask) = self.cached_mask {
            if self.cached_grid_id == Some(grid_id) {
                return mask.clone();
            }
        }

        let mut mask = Array3::zeros((grid.nx, grid.ny, grid.nz));

        for (i, element) in self.elements.iter().enumerate() {
            if !element.is_active || !self.active_elements[i] {
                continue;
            }

            let (x, y, z) = element.position;
            let half_width = self.config.element_width / 2.0;
            let half_length = self.config.element_length / 2.0;

            let ix_start = (((x - half_width) - grid.origin[0]) / grid.dx).ceil() as isize;
            let ix_end = (((x + half_width) - grid.origin[0]) / grid.dx).floor() as isize;

            let iy = ((y - grid.origin[1]) / grid.dy).round() as isize;

            let iz_start = (((z - half_length) - grid.origin[2]) / grid.dz).ceil() as isize;
            let iz_end = (((z + half_length) - grid.origin[2]) / grid.dz).floor() as isize;

            for ix in ix_start..=ix_end {
                for iz in iz_start..=iz_end {
                    if ix >= 0
                        && ix < grid.nx as isize
                        && iy >= 0
                        && iy < grid.ny as isize
                        && iz >= 0
                        && iz < grid.nz as isize
                    {
                        mask[[ix as usize, iy as usize, iz as usize]] += element.transmit_weight;
                    }
                }
            }
        }

        mask
    }

    fn amplitude(&self, t: f64) -> f64 {
        match &self.signal {
            Some(signal) => signal.amplitude(t),
            None => 0.0,
        }
    }

    fn get_source_term(&self, t: f64, x: f64, y: f64, z: f64, grid: &Grid) -> f64 {
        let mut total = 0.0;

        for (i, element) in self.elements.iter().enumerate() {
            if !element.is_active || !self.active_elements[i] {
                continue;
            }

            let (ex, ey, ez) = element.position;

            let dx = x - ex;
            let dy = y - ey;
            let dz = z - ez;
            let dist_sq = dx * dx + dy * dy + dz * dz;

            let tol = (element.width.max(element.length) / 2.0).max(grid.dx);

            if dist_sq < tol * tol {
                let delayed_time = t - element.time_delay;
                let amp = self.amplitude(delayed_time);
                total += amp * element.transmit_weight;
            }
        }

        total
    }

    fn positions(&self) -> Vec<(f64, f64, f64)> {
        self.elements
            .iter()
            .enumerate()
            .filter(|(i, e)| e.is_active && self.active_elements[*i])
            .map(|(_, e)| e.position)
            .collect()
    }

    fn signal(&self) -> &dyn Signal {
        self.signal
            .as_ref()
            .expect("TransducerArray2D has no signal set; call set_signal() first")
            .as_ref()
    }
}
