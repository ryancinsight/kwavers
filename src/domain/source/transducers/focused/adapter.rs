use crate::domain::grid::Grid;
use crate::domain::signal::Signal;
use crate::domain::source::transducers::focused::bowl::{BowlConfig, BowlTransducer};
use crate::domain::source::{Source, SourceField};
use ndarray::Array3;
use std::collections::HashMap;
use std::fmt::Debug;
use std::sync::Arc;

#[derive(Debug)]
pub struct FocusedSource {
    pub(crate) transducer: BowlTransducer,
    signal: Arc<dyn Signal>,
    focus_delays: Vec<f64>,
    element_map: HashMap<(usize, usize, usize), Vec<usize>>,
}

impl FocusedSource {
    pub fn new(
        config: BowlConfig,
        signal: Arc<dyn Signal>,
        grid: &Grid,
    ) -> crate::core::error::KwaversResult<Self> {
        let transducer = BowlTransducer::new(config)?;
        let focus_delays = transducer.calculate_focus_delays();

        // Pre-compute grid to element mapping for efficient lookup
        let mut element_map = HashMap::new();
        let tolerance = grid.dx.max(grid.dy).max(grid.dz) * 0.5;
        let _tol2 = tolerance * tolerance; // unused? used in get_source_term optimization if we store it.
                                           // But here we just build the map.
                                           // Actually, we build map based on grid.position_to_indices.

        for (i, &pos) in transducer.element_positions.iter().enumerate() {
            // Find grid index closest to element
            if let Some((ix, iy, iz)) = grid.position_to_indices(pos[0], pos[1], pos[2]) {
                element_map
                    .entry((ix, iy, iz))
                    .or_insert_with(Vec::new)
                    .push(i);
            }
        }

        Ok(Self {
            transducer,
            signal,
            focus_delays,
            element_map,
        })
    }
}

impl Source for FocusedSource {
    fn create_mask(&self, grid: &Grid) -> Array3<f64> {
        let mut mask = Array3::zeros((grid.nx, grid.ny, grid.nz));

        for (key, indices) in &self.element_map {
            let (ix, iy, iz) = *key;
            if ix < grid.nx && iy < grid.ny && iz < grid.nz {
                let mut weight_sum = 0.0;
                for &idx in indices {
                    weight_sum += self.transducer.element_areas[idx];
                }
                mask[(ix, iy, iz)] = weight_sum;
            }
        }
        mask
    }

    fn amplitude(&self, t: f64) -> f64 {
        self.signal.amplitude(t)
    }

    fn positions(&self) -> Vec<(f64, f64, f64)> {
        self.transducer
            .element_positions
            .iter()
            .map(|p| (p[0], p[1], p[2]))
            .collect()
    }

    fn signal(&self) -> &dyn Signal {
        self.signal.as_ref()
    }

    fn get_source_term(&self, t: f64, x: f64, y: f64, z: f64, grid: &Grid) -> f64 {
        // Find grid index for (x, y, z)
        if let Some((ix, iy, iz)) = grid.position_to_indices(x, y, z) {
            if let Some(indices) = self.element_map.get(&(ix, iy, iz)) {
                let mut sum_val = 0.0;
                for &i in indices {
                    let delay = self.focus_delays[i];
                    let weight = self.transducer.element_areas[i];
                    sum_val += weight * self.signal.amplitude(t - delay);
                }
                return sum_val;
            }
        }
        0.0
    }

    fn focal_point(&self) -> Option<(f64, f64, f64)> {
        let f = self.transducer.config.focus;
        Some((f[0], f[1], f[2]))
    }
}
