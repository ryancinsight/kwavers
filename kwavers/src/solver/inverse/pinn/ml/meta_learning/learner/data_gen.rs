use super::MetaLearner;
use crate::core::error::KwaversResult;
use crate::solver::inverse::pinn::ml::meta_learning::types::{PhysicsTask, TaskData};
use burn::tensor::backend::AutodiffBackend;
use std::f64::consts::PI;

impl<B: AutodiffBackend> MetaLearner<B> {
    /// Generate task data.
    /// # Errors
    /// - Returns [`Err`] if an internal constraint is violated.
    ///
    pub(super) fn generate_task_data(&self, task: &PhysicsTask) -> KwaversResult<TaskData> {
        let collocation_points = self.generate_collocation_points(task.geometry.as_ref());
        let boundary_data =
            self.generate_boundary_data(task.geometry.as_ref(), &task.boundary_conditions);
        let initial_data = self.generate_initial_conditions(task);

        Ok(TaskData {
            collocation_points,
            boundary_data,
            initial_data,
        })
    }

    fn generate_collocation_points(
        &self,
        geometry: &crate::solver::inverse::pinn::ml::BurnWave2dGeometry,
    ) -> Vec<(f64, f64, f64)> {
        let mut points = Vec::new();
        let num_points = 1000;

        for _ in 0..num_points {
            let x = rand::random::<f64>() * 2.0 - 1.0;
            let y = rand::random::<f64>() * 2.0 - 1.0;
            let t = rand::random::<f64>() * 1.0;

            if geometry.contains(x, y) {
                points.push((x, y, t));
            }
        }

        points
    }

    fn generate_boundary_data(
        &self,
        geometry: &crate::solver::inverse::pinn::ml::BurnWave2dGeometry,
        conditions: &[crate::solver::inverse::pinn::ml::BoundaryCondition2D],
    ) -> Vec<(f64, f64, f64, f64)> {
        let condition_count = conditions.len().max(1);
        let (x_min, x_max, y_min, _y_max) = geometry.bounding_box();
        let span_x = (x_max - x_min).abs();
        let base_count = 200;
        let mut data = Vec::new();

        let push_point = |points: &mut Vec<(f64, f64, f64, f64)>, x: f64, y: f64, t: f64| {
            let idx = points.len() % condition_count;
            let bc = conditions
                .get(idx)
                .copied()
                .unwrap_or(crate::solver::inverse::pinn::ml::BoundaryCondition2D::Dirichlet);
            let bc_value = match bc {
                crate::solver::inverse::pinn::ml::BoundaryCondition2D::Dirichlet => 0.0,
                crate::solver::inverse::pinn::ml::BoundaryCondition2D::Neumann => 0.0,
                crate::solver::inverse::pinn::ml::BoundaryCondition2D::Periodic => 0.0,
                crate::solver::inverse::pinn::ml::BoundaryCondition2D::Absorbing => 0.0,
            };
            points.push((x, y, t, bc_value));
        };

        match geometry {
            crate::solver::inverse::pinn::ml::BurnWave2dGeometry::Rectangular {
                x_min,
                x_max,
                y_min,
                y_max,
            } => {
                let (x_min, x_max, y_min, y_max) = (*x_min, *x_max, *y_min, *y_max);
                let n = (base_count / 4).max(25);
                for i in 0..n {
                    let s = i as f64 / (n - 1) as f64;
                    let t = s;
                    let x = x_min + (x_max - x_min) * s;
                    push_point(&mut data, x, y_min, t);
                    push_point(&mut data, x, y_max, t);
                }
                for i in 0..n {
                    let s = i as f64 / (n - 1) as f64;
                    let t = s;
                    let y = y_min + (y_max - y_min) * s;
                    push_point(&mut data, x_min, y, t);
                    push_point(&mut data, x_max, y, t);
                }
            }
            crate::solver::inverse::pinn::ml::BurnWave2dGeometry::Circular {
                x_center,
                y_center,
                radius,
            } => {
                let (x_center, y_center, radius) = (*x_center, *y_center, *radius);
                let n = base_count.max(100);
                for i in 0..n {
                    let s = i as f64 / (n - 1) as f64;
                    let theta = 2.0 * PI * s;
                    let x = x_center + radius * theta.cos();
                    let y = y_center + radius * theta.sin();
                    push_point(&mut data, x, y, s);
                }
            }
            crate::solver::inverse::pinn::ml::BurnWave2dGeometry::LShaped {
                x_min,
                x_max,
                y_min,
                y_max,
                notch_x,
                notch_y,
            } => {
                let (x_min, x_max, y_min, y_max, notch_x, notch_y) =
                    (*x_min, *x_max, *y_min, *y_max, *notch_x, *notch_y);
                let n = (base_count / 6).max(25);
                for i in 0..n {
                    let s = i as f64 / (n - 1) as f64;
                    let t = s;
                    let x = x_min + (x_max - x_min) * s;
                    push_point(&mut data, x, y_min, t);
                    push_point(&mut data, x, y_max, t);
                }
                for i in 0..n {
                    let s = i as f64 / (n - 1) as f64;
                    let t = s;
                    let y = y_min + (y_max - y_min) * s;
                    push_point(&mut data, x_min, y, t);
                    if y <= notch_y {
                        push_point(&mut data, x_max, y, t);
                    }
                }
                for i in 0..n {
                    let s = i as f64 / (n - 1) as f64;
                    let t = s;
                    let x = notch_x + (x_max - notch_x) * s;
                    push_point(&mut data, x, notch_y, t);
                }
                for i in 0..n {
                    let s = i as f64 / (n - 1) as f64;
                    let t = s;
                    let y = notch_y + (y_max - notch_y) * s;
                    push_point(&mut data, notch_x, y, t);
                }
            }
            crate::solver::inverse::pinn::ml::BurnWave2dGeometry::Polygonal { vertices, holes } => {
                let sample_edges = |poly: &[(f64, f64)], points: &mut Vec<(f64, f64, f64, f64)>| {
                    if poly.len() < 2 {
                        return;
                    }
                    let n_edges = poly.len();
                    let n = (base_count / n_edges.max(1)).max(10);
                    for i in 0..n_edges {
                        let (x0, y0) = poly[i];
                        let (x1, y1) = poly[(i + 1) % n_edges];
                        for j in 0..n {
                            let s = j as f64 / (n - 1) as f64;
                            let t = s;
                            let x = x0 + (x1 - x0) * s;
                            let y = y0 + (y1 - y0) * s;
                            push_point(points, x, y, t);
                        }
                    }
                };
                sample_edges(vertices, &mut data);
                for hole in holes {
                    sample_edges(hole, &mut data);
                }
            }
            crate::solver::inverse::pinn::ml::BurnWave2dGeometry::ParametricCurve {
                x_func,
                y_func,
                t_min,
                t_max,
                ..
            } => {
                let (t_min, t_max) = (*t_min, *t_max);
                let n = base_count.max(100);
                for i in 0..n {
                    let s = i as f64 / (n - 1) as f64;
                    let t_param = t_min + (t_max - t_min) * s;
                    let x = x_func(t_param);
                    let y = y_func(t_param);
                    push_point(&mut data, x, y, s);
                }
            }
            crate::solver::inverse::pinn::ml::BurnWave2dGeometry::AdaptiveMesh {
                base_geometry,
                ..
            } => {
                data.extend(self.generate_boundary_data(base_geometry.as_ref(), conditions));
            }
            crate::solver::inverse::pinn::ml::BurnWave2dGeometry::MultiRegion {
                regions, ..
            } => {
                for (region, _) in regions {
                    data.extend(self.generate_boundary_data(region, conditions));
                }
            }
        }

        if data.is_empty() {
            let n = base_count.max(50);
            for i in 0..n {
                let s = i as f64 / (n - 1) as f64;
                let x = x_min + span_x * s;
                push_point(&mut data, x, y_min, s);
            }
        }

        data
    }

    /// Generate initial condition data for meta-learning tasks.
    ///
    /// Samples 200 points within geometry using three IC patterns (selected by task ID hash):
    /// 1. Gaussian pulse: u₀(x,y) = A·exp(-r²/(2σ²))
    /// 2. Plane wave: u₀(x,y) = A·sin(k·r) with wave velocity initial condition
    /// 3. Delta pulse: tighter Gaussian
    fn generate_initial_conditions(&self, task: &PhysicsTask) -> Vec<(f64, f64, f64, f64, f64)> {
        let geometry = &task.geometry;
        let n_ic = 200;
        let (x_samples, y_samples) = geometry.sample_points(n_ic);
        let (x_min, x_max, y_min, y_max) = geometry.bounding_box();
        let span_x = (x_max - x_min).abs().max(1e-12);
        let span_y = (y_max - y_min).abs().max(1e-12);
        let center_x = (x_min + x_max) * 0.5;
        let center_y = (y_min + y_max) * 0.5;
        let min_span = span_x.min(span_y).max(1e-12);
        let base_sigma = 0.15 * min_span;
        let delta_sigma = 0.05 * min_span;
        let amplitude = 1.0;
        let mut hash: u64 = 0;
        for byte in task.id.as_bytes() {
            hash = hash.wrapping_mul(131).wrapping_add(*byte as u64);
        }
        let pattern = (hash % 3) as u8;
        let kx = 2.0 * PI / span_x;
        let ky = 2.0 * PI / span_y;
        let k_norm = (kx * kx + ky * ky).sqrt();
        let omega = task.physics_params.wave_speed * k_norm;
        let mut data = Vec::with_capacity(n_ic);

        for i in 0..n_ic {
            let x = x_samples[i];
            let y = y_samples[i];
            let t = 0.0;
            let (u0, v0) = match pattern {
                0 => {
                    let dx = x - center_x;
                    let dy = y - center_y;
                    let r2 = dx * dx + dy * dy;
                    let u0 = amplitude * (-r2 / (2.0 * base_sigma * base_sigma)).exp();
                    (u0, 0.0)
                }
                1 => {
                    let phase = kx * (x - center_x) + ky * (y - center_y);
                    let u0 = amplitude * phase.sin();
                    let v0 = if matches!(
                        task.pde_type,
                        crate::solver::inverse::pinn::ml::meta_learning::types::PdeType::Wave
                            | crate::solver::inverse::pinn::ml::meta_learning::types::PdeType::Acoustic
                            | crate::solver::inverse::pinn::ml::meta_learning::types::PdeType::Elastic
                    ) {
                        -omega * amplitude * phase.cos()
                    } else {
                        0.0
                    };
                    (u0, v0)
                }
                _ => {
                    let dx = x - center_x;
                    let dy = y - center_y;
                    let r2 = dx * dx + dy * dy;
                    let u0 = amplitude * (-r2 / (2.0 * delta_sigma * delta_sigma)).exp();
                    (u0, 0.0)
                }
            };
            data.push((x, y, t, u0, v0));
        }

        data
    }
}
