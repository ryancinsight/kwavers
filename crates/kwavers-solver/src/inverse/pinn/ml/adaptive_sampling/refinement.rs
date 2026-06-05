use super::{AdaptiveCollocationSampler, HighResidualRegion};
use burn::tensor::{backend::AutodiffBackend, Tensor};
use kwavers_core::error::KwaversResult;
use rand::Rng;

impl<B: AutodiffBackend> AdaptiveCollocationSampler<B> {
    /// Adaptive refinement.
    /// # Errors
    /// - Returns [`Err`] if an internal constraint is violated.
    ///
    pub(super) fn adaptive_refinement(&mut self) -> KwaversResult<()> {
        let priorities_data: Vec<f32> = self.priorities.to_data().to_vec().unwrap_or_default();
        let points_data: Vec<f32> = self.active_points.to_data().to_vec().unwrap_or_default();

        let mut indexed_priorities: Vec<(usize, f32)> = priorities_data
            .iter()
            .enumerate()
            .map(|(i, &p)| (i, p))
            .collect();
        indexed_priorities.sort_by(|a, b| b.1.total_cmp(&a.1));

        let refinement_count =
            (self.total_points as f64 * self.strategy.refinement_fraction) as usize;
        let high_priority_indices: Vec<usize> = indexed_priorities
            .iter()
            .take(refinement_count)
            .map(|(idx, _)| *idx)
            .collect();

        let coarsening_count =
            (self.total_points as f64 * self.strategy.coarsening_fraction) as usize;
        let low_priority_indices: Vec<usize> = indexed_priorities
            .iter()
            .rev()
            .take(coarsening_count)
            .map(|(idx, _)| *idx)
            .collect();

        let mut new_points = Vec::new();
        let mut new_priorities = Vec::new();

        for (i, &priority) in priorities_data.iter().enumerate().take(self.total_points) {
            if !high_priority_indices.contains(&i) && !low_priority_indices.contains(&i) {
                let base_idx = i * 3;
                if base_idx + 2 < points_data.len() {
                    new_points.push(points_data[base_idx]);
                    new_points.push(points_data[base_idx + 1]);
                    new_points.push(points_data[base_idx + 2]);
                    new_priorities.push(priority);
                }
            }
        }

        let mut refined_points = Vec::new();
        for &idx in &high_priority_indices {
            let base_idx = idx * 3;
            if base_idx + 2 < points_data.len() {
                let parent_x = points_data[base_idx];
                let parent_y = points_data[base_idx + 1];
                let parent_t = points_data[base_idx + 2];

                let offsets = [-0.125, 0.125];
                for &dx in &offsets {
                    for &dy in &offsets {
                        for &dt in &offsets {
                            let child_x = (parent_x + dx as f32).clamp(0.0, 1.0);
                            let child_y = (parent_y + dy as f32).clamp(0.0, 1.0);
                            let child_t = (parent_t + dt as f32).clamp(0.0, 1.0);
                            refined_points.extend_from_slice(&[child_x, child_y, child_t]);
                        }
                    }
                }
            }
        }

        new_points.extend_from_slice(&refined_points);
        new_priorities.extend(vec![1.0; refined_points.len() / 3]);

        let new_total = new_points.len() / 3;
        if new_total > self.total_points {
            new_points.truncate(self.total_points * 3);
            new_priorities.truncate(self.total_points);
        } else if new_total < self.total_points {
            let deficit = self.total_points - new_total;
            let high_residual_regions = self.identify_high_residual_regions();

            for region in high_residual_regions.into_iter().take(deficit) {
                let mut rng = rand::thread_rng();
                let x =
                    (region.center_x + (rng.gen::<f32>() - 0.5) * region.size_x).clamp(0.0, 1.0);
                let y =
                    (region.center_y + (rng.gen::<f32>() - 0.5) * region.size_y).clamp(0.0, 1.0);
                let t =
                    (region.center_t + (rng.gen::<f32>() - 0.5) * region.size_t).clamp(0.0, 1.0);

                new_points.extend_from_slice(&[x, y, t]);
                new_priorities.push(0.7);
            }
        }

        let device = self.active_points.device();
        self.active_points = Tensor::from_data(new_points.as_slice(), &device);
        self.priorities = Tensor::from_data(new_priorities.as_slice(), &device);

        self.stats.points_refined = refined_points.len() / 3;
        self.stats.points_coarsened = coarsening_count;

        Ok(())
    }

    pub(super) fn identify_high_residual_regions(&self) -> Vec<HighResidualRegion> {
        const GRID_SIZE: usize = 4;
        const MIN_POINTS_PER_REGION: usize = 3;
        const TOP_REGION_FRACTION: f32 = 0.3;

        let points_data = self.active_points.clone().into_data();
        let points_vec: Vec<f32> = points_data.to_vec().unwrap_or_default();
        let priorities_data = self.priorities.clone().into_data();
        let priorities_vec: Vec<f32> = priorities_data.to_vec().unwrap_or_default();

        if points_vec.len() < 3 || points_vec.len() / 3 != priorities_vec.len() {
            return self.create_fallback_regions();
        }

        let num_points = points_vec.len() / 3;

        let mut grid_cells: std::collections::HashMap<
            (usize, usize, usize),
            (Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>),
        > = std::collections::HashMap::new();

        for i in 0..num_points {
            let x = points_vec[i * 3];
            let y = points_vec[i * 3 + 1];
            let t = points_vec[i * 3 + 2];
            let priority = priorities_vec[i];

            let gx = ((x * GRID_SIZE as f32).floor() as usize).min(GRID_SIZE - 1);
            let gy = ((y * GRID_SIZE as f32).floor() as usize).min(GRID_SIZE - 1);
            let gt = ((t * GRID_SIZE as f32).floor() as usize).min(GRID_SIZE - 1);

            let cell = grid_cells.entry((gx, gy, gt)).or_insert((
                Vec::new(),
                Vec::new(),
                Vec::new(),
                Vec::new(),
            ));
            cell.0.push(x);
            cell.1.push(y);
            cell.2.push(t);
            cell.3.push(priority);
        }

        let mut regions = Vec::new();

        for ((_gx, _gy, _gt), (xs, ys, ts, prios)) in grid_cells.iter() {
            if xs.len() < MIN_POINTS_PER_REGION {
                continue;
            }

            let mean_x: f32 = xs.iter().sum::<f32>() / xs.len() as f32;
            let mean_y: f32 = ys.iter().sum::<f32>() / ys.len() as f32;
            let mean_t: f32 = ts.iter().sum::<f32>() / ts.len() as f32;
            let mean_priority: f32 = prios.iter().sum::<f32>() / prios.len() as f32;

            let cell_size = 1.0 / GRID_SIZE as f32;

            regions.push(HighResidualRegion {
                center_x: mean_x,
                center_y: mean_y,
                center_t: mean_t,
                size_x: cell_size * 1.5,
                size_y: cell_size * 1.5,
                size_t: cell_size * 1.5,
                residual_magnitude: mean_priority,
            });
        }

        regions.sort_by(|a, b| b.residual_magnitude.total_cmp(&a.residual_magnitude));

        if !regions.is_empty() {
            let keep_count = ((regions.len() as f32 * TOP_REGION_FRACTION).ceil() as usize)
                .max(2)
                .min(regions.len());
            regions.truncate(keep_count);
        }

        regions
    }

    pub(super) fn create_fallback_regions(&self) -> Vec<HighResidualRegion> {
        let mut regions = Vec::new();

        for i in 0..2 {
            for j in 0..2 {
                for k in 0..2 {
                    regions.push(HighResidualRegion {
                        center_x: (i as f32 + 0.5) / 2.0,
                        center_y: (j as f32 + 0.5) / 2.0,
                        center_t: (k as f32 + 0.5) / 2.0,
                        size_x: 0.5,
                        size_y: 0.5,
                        size_t: 0.5,
                        residual_magnitude: 1.0,
                    });
                }
            }
        }

        regions
    }
}
