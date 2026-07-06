use super::{TestPoint, TransferLearner};
use kwavers_core::error::KwaversResult;

impl<B: coeus_ops::BackendOps<f32> + coeus_ops::CpuBackend + Default> TransferLearner<B>
where
    B::DeviceBuffer<f32>:
        coeus_core::CpuAddressableStorage<f32> + coeus_core::CpuAddressableStorageMut<f32>,
{
    /// Evaluate model accuracy on geometry
    /// # Errors
    /// - Propagates any [`KwaversError`] returned by called functions.
    ///
    pub(super) fn evaluate_accuracy(
        &self,
        model: &crate::inverse::pinn::ml::BurnPINN2DWave<B>,
        geometry: &crate::inverse::pinn::ml::BurnWave2dGeometry,
        conditions: &[crate::inverse::pinn::ml::BoundaryCondition2D],
    ) -> KwaversResult<f32> {
        let test_points = self.generate_test_points(geometry, 1000)?;
        let mut total_residual = 0.0;
        let mut total_boundary_error = 0.0;

        for point in &test_points {
            let x_arr = ndarray::Array1::from_vec(vec![point.x]);
            let y_arr = ndarray::Array1::from_vec(vec![point.y]);
            let t_arr = ndarray::Array1::from_vec(vec![0.0]);
            let _prediction = model.predict(&x_arr, &y_arr, &t_arr)?;
            let residual = self.compute_pde_residual(model, point.x, point.y, 0.0)?;
            total_residual += residual * residual;
        }

        for condition in conditions {
            let boundary_error = self.evaluate_boundary_condition(model, condition, geometry)?;
            total_boundary_error += boundary_error * boundary_error;
        }

        let pde_accuracy = 1.0 / (1.0 + total_residual.sqrt() / test_points.len() as f64);
        let boundary_accuracy = 1.0 / (1.0 + total_boundary_error.sqrt() / conditions.len() as f64);

        let overall_accuracy = 0.7 * pde_accuracy + 0.3 * boundary_accuracy;

        Ok(overall_accuracy as f32)
    }

    /// Generate test points within geometry for evaluation
    /// # Errors
    /// - Returns [`Err`] if an internal constraint is violated.
    ///
    pub(super) fn generate_test_points(
        &self,
        geometry: &crate::inverse::pinn::ml::BurnWave2dGeometry,
        num_points: usize,
    ) -> KwaversResult<Vec<TestPoint>> {
        let mut points = Vec::with_capacity(num_points);
        let (x_min, x_max, y_min, y_max) = geometry.bounding_box();

        for i in 0..num_points {
            let x = x_min + (x_max - x_min) * (i as f64 / num_points as f64);
            let y = y_min + (y_max - y_min) * ((i as f64 * 1.618) % 1.0);

            if geometry.contains(x, y) {
                points.push(TestPoint { x, y });
            }
        }

        Ok(points)
    }

    /// Compute PDE residual at a point (simplified wave equation: ∂²u/∂t² = c²∇²u)
    /// # Errors
    /// - Propagates any [`KwaversError`] returned by called functions.
    ///
    pub(super) fn compute_pde_residual(
        &self,
        model: &crate::inverse::pinn::ml::BurnPINN2DWave<B>,
        x: f64,
        y: f64,
        t: f64,
    ) -> KwaversResult<f64> {
        let eps = 1e-6;

        let x_arr = ndarray::Array1::from_vec(vec![x]);
        let y_arr = ndarray::Array1::from_vec(vec![y]);
        let t_arr = ndarray::Array1::from_vec(vec![t]);

        let u_center = model.predict(&x_arr, &y_arr, &t_arr)?;
        let u_val = u_center[[0, 0]];

        let x_plus = ndarray::Array1::from_vec(vec![x + eps]);
        let u_x_plus = model.predict(&x_plus, &y_arr, &t_arr)?[[0, 0]];

        let x_minus = ndarray::Array1::from_vec(vec![x - eps]);
        let u_x_minus = model.predict(&x_minus, &y_arr, &t_arr)?[[0, 0]];

        let y_plus = ndarray::Array1::from_vec(vec![y + eps]);
        let u_y_plus = model.predict(&x_arr, &y_plus, &t_arr)?[[0, 0]];

        let y_minus = ndarray::Array1::from_vec(vec![y - eps]);
        let u_y_minus = model.predict(&x_arr, &y_minus, &t_arr)?[[0, 0]];

        let laplacian = (u_x_plus - 2.0 * u_val + u_x_minus) / (eps * eps)
            + (u_y_plus - 2.0 * u_val + u_y_minus) / (eps * eps);

        Ok(laplacian.abs())
    }

    /// Evaluate boundary condition satisfaction by sampling boundary points.
    ///
    /// Computes the mean squared BC violation across uniformly sampled boundary points.
    ///
    /// For Dirichlet (u = 0): ε_BC = (1/N) Σ |u_model(x_bc, y_bc, 0)|²
    /// For Neumann (∂u/∂n = 0): ε_BC = (1/N) Σ |∂u_model/∂n|²
    /// For Periodic/Absorbing: Returns 0.0 (no simple pointwise residual)
    /// # Errors
    /// - Propagates any [`KwaversError`] returned by called functions.
    ///
    pub(super) fn evaluate_boundary_condition(
        &self,
        model: &crate::inverse::pinn::ml::BurnPINN2DWave<B>,
        condition: &crate::inverse::pinn::ml::BoundaryCondition2D,
        geometry: &crate::inverse::pinn::ml::BurnWave2dGeometry,
    ) -> KwaversResult<f64> {
        use crate::inverse::pinn::ml::BoundaryCondition2D;

        let n_bc = 100;
        let (x_min, x_max, y_min, y_max) = geometry.bounding_box();
        let eps = 1e-5;

        match condition {
            BoundaryCondition2D::Dirichlet => {
                let mut total_violation = 0.0;
                let mut count = 0;

                let boundary_points: Vec<(f64, f64)> = (0..n_bc)
                    .flat_map(|i| {
                        let frac = i as f64 / (n_bc - 1) as f64;
                        let x = x_min + (x_max - x_min) * frac;
                        let y = y_min + (y_max - y_min) * frac;
                        vec![(x, y_min), (x, y_max), (x_min, y), (x_max, y)]
                    })
                    .collect();

                for (x, y) in &boundary_points {
                    let x_arr = ndarray::Array1::from_vec(vec![*x]);
                    let y_arr = ndarray::Array1::from_vec(vec![*y]);
                    let t_arr = ndarray::Array1::from_vec(vec![0.0]);
                    let u = model.predict(&x_arr, &y_arr, &t_arr)?[[0, 0]];
                    total_violation += u * u;
                    count += 1;
                }

                Ok((total_violation / count as f64).sqrt())
            }
            BoundaryCondition2D::Neumann => {
                let mut total_violation = 0.0;
                let mut count = 0;

                for i in 0..n_bc {
                    let frac = i as f64 / (n_bc - 1) as f64;
                    let x = x_min + (x_max - x_min) * frac;
                    let y = y_min + (y_max - y_min) * frac;

                    let x_arr = ndarray::Array1::from_vec(vec![x]);
                    let y0 = ndarray::Array1::from_vec(vec![y_min]);
                    let y_eps = ndarray::Array1::from_vec(vec![y_min + eps]);
                    let t_arr = ndarray::Array1::from_vec(vec![0.0]);
                    let u0 = model.predict(&x_arr, &y0, &t_arr)?[[0, 0]];
                    let u_eps = model.predict(&x_arr, &y_eps, &t_arr)?[[0, 0]];
                    let dudn = (u_eps - u0) / eps;
                    total_violation += dudn * dudn;
                    count += 1;

                    let x0 = ndarray::Array1::from_vec(vec![x_min]);
                    let x_eps_arr = ndarray::Array1::from_vec(vec![x_min + eps]);
                    let y_arr = ndarray::Array1::from_vec(vec![y]);
                    let u0 = model.predict(&x0, &y_arr, &t_arr)?[[0, 0]];
                    let u_eps = model.predict(&x_eps_arr, &y_arr, &t_arr)?[[0, 0]];
                    let dudn = (u_eps - u0) / eps;
                    total_violation += dudn * dudn;
                    count += 1;
                }

                Ok((total_violation / count as f64).sqrt())
            }
            BoundaryCondition2D::Periodic | BoundaryCondition2D::Absorbing => Ok(0.0),
        }
    }
}
