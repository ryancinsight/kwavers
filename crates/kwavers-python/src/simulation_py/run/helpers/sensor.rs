use kwavers_grid::Grid as KwaversGrid;

use super::super::super::Simulation;
use crate::sensor_py::Sensor;
use crate::transducer_array_py::TransducerArray2D;

impl Simulation {
    /// Build a 3D sensor mask from sensor or transducer array settings.
    pub(crate) fn create_sensor_mask(
        grid: &KwaversGrid,
        sensor: Option<&Sensor>,
        transducer: Option<&TransducerArray2D>,
    ) -> ndarray::Array3<bool> {
        let nx = grid.nx;
        let ny = grid.ny;
        let nz = grid.nz;

        if let Some(trans) = transducer {
            let mut mask = ndarray::Array3::<bool>::from_elem((nx, ny, nz), false);
            let width_pts = (trans.inner.element_width() / grid.dx).round() as isize;
            let length_pts = (trans.inner.element_length() / grid.dz).round() as isize;

            for pos in trans.inner.element_positions() {
                let cx = ((pos.0 - grid.origin[0]) / grid.dx).round() as isize;
                let cy = ((pos.1 - grid.origin[1]) / grid.dy).round() as isize;
                let cz = ((pos.2 - grid.origin[2]) / grid.dz).round() as isize;

                let ix_start = cx - (width_pts / 2);
                let ix_end = ix_start + width_pts - 1;
                let iz_start = cz - (length_pts / 2);
                let iz_end = iz_start + length_pts - 1;

                for i in ix_start..=ix_end {
                    for k in iz_start..=iz_end {
                        if i >= 0
                            && i < nx as isize
                            && cy >= 0
                            && cy < ny as isize
                            && k >= 0
                            && k < nz as isize
                        {
                            mask[[i as usize, cy as usize, k as usize]] = true;
                        }
                    }
                }
            }
            return mask;
        }

        let sensor =
            sensor.expect("Simulation must have either a Sensor or TransducerArray2D sensor");

        if let Some(ref mask) = sensor.mask {
            return mask.clone();
        }

        let mut mask = ndarray::Array3::<bool>::from_elem((nx, ny, nz), false);

        if sensor.sensor_type == "grid" {
            mask.fill(true);
            return mask;
        }

        let pos = sensor.position.unwrap_or([
            (nx as f64 * grid.dx) * 0.5,
            (ny as f64 * grid.dy) * 0.5,
            (nz as f64 * grid.dz) * 0.5,
        ]);

        let ix = (pos[0] / grid.dx).round() as isize;
        let iy = (pos[1] / grid.dy).round() as isize;
        let iz = (pos[2] / grid.dz).round() as isize;

        let ix = ix.clamp(0, (nx - 1) as isize) as usize;
        let iy = iy.clamp(0, (ny - 1) as isize) as usize;
        let iz = iz.clamp(0, (nz - 1) as isize) as usize;

        mask[[ix, iy, iz]] = true;
        mask
    }

    /// Build an ordered list of sensor grid indices for a transducer, element by element.
    pub(crate) fn create_transducer_ordered_indices(
        grid: &KwaversGrid,
        trans: &kwavers_transducer::array_2d::TransducerArray2D,
    ) -> Vec<(usize, usize, usize)> {
        let nx = grid.nx;
        let ny = grid.ny;
        let nz = grid.nz;
        let width_pts = (trans.element_width() / grid.dx).round() as isize;
        let length_pts = (trans.element_length() / grid.dz).round() as isize;

        let mut indices = Vec::new();
        for pos in trans.element_positions() {
            let cx = ((pos.0 - grid.origin[0]) / grid.dx).round() as isize;
            let cy = ((pos.1 - grid.origin[1]) / grid.dy).round() as isize;
            let cz = ((pos.2 - grid.origin[2]) / grid.dz).round() as isize;

            let ix_start = cx - (width_pts / 2);
            let iz_start = cz - (length_pts / 2);

            for ii in 0..width_pts {
                for kk in 0..length_pts {
                    let i = ix_start + ii;
                    let k = iz_start + kk;
                    if i >= 0
                        && i < nx as isize
                        && cy >= 0
                        && cy < ny as isize
                        && k >= 0
                        && k < nz as isize
                    {
                        indices.push((i as usize, cy as usize, k as usize));
                    }
                }
            }
        }
        indices
    }
}
