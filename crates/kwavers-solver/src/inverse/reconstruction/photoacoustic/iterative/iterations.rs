//! SIRT, ART, OSEM iteration steps and regularization for [`IterativeMethods`].

use super::IterativeAlgorithm;
use super::IterativeMethods;
use kwavers_core::error::KwaversResult;
use kwavers_core::utils::iterators::apply_inplace;
use moirai_parallel::{enumerate_mut_with, Adaptive};
use ndarray::{Array1, Array2};

impl IterativeMethods {
    /// One SIRT step: x ← x + λ · Aᵀ(y − Ax).
    /// # Errors
    /// - Returns [`Err`] if an internal constraint is violated.
    ///
    pub(super) fn sirt_iteration(
        &self,
        a: &Array2<f64>,
        x: &Array1<f64>,
        y: &Array1<f64>,
    ) -> KwaversResult<Array1<f64>> {
        let ax = a.dot(x);
        let residual = y - &ax;
        let update = a.t().dot(&residual);
        Ok(x + self.relaxation_factor * &update)
    }

    /// One ART step: sequential Kaczmarz row projections.
    ///
    /// For each row i: x ← x + λ · (yᵢ − aᵢᵀx) / ‖aᵢ‖² · aᵢ
    /// # Errors
    /// - Returns [`Err`] if an internal constraint is violated.
    ///
    pub(super) fn art_iteration(
        &self,
        a: &Array2<f64>,
        x: &mut Array1<f64>,
        y: &Array1<f64>,
    ) -> KwaversResult<()> {
        for (i, row) in a.rows().into_iter().enumerate() {
            let ax_i = row.dot(x);
            let residual = y[i] - ax_i;
            let row_norm_sq = row.dot(&row);

            if row_norm_sq > 0.0 {
                let update_factor = self.relaxation_factor * residual / row_norm_sq;
                if let (Some(x_values), Some(row_values)) =
                    (x.as_slice_memory_order_mut(), row.as_slice_memory_order())
                {
                    enumerate_mut_with::<Adaptive, _, _>(x_values, |idx, x_value| {
                        *x_value += update_factor * row_values[idx];
                    });
                } else {
                    for (x_value, &a_value) in x.iter_mut().zip(row.iter()) {
                        *x_value += update_factor * a_value;
                    }
                }
            }
        }
        Ok(())
    }

    /// One OSEM step: ordered-subset EM update with positivity constraint.
    ///
    /// Divides measurements into `subsets` ordered subsets, updating x after
    /// each subset via x ← x · (Aₛᵀ(yₛ/Aₛx)) / sensitivity_s.
    /// # Errors
    /// - Returns [`Err`] if an internal constraint is violated.
    ///
    pub(super) fn osem_iteration(
        &self,
        a: &Array2<f64>,
        x: &mut Array1<f64>,
        y: &Array1<f64>,
        subsets: usize,
    ) -> KwaversResult<()> {
        let (n_measurements, n_voxels) = a.dim();

        apply_inplace(x, |v| v.max(1e-10));

        let subset_size = n_measurements.div_ceil(subsets);

        for subset_idx in 0..subsets {
            let start_idx = subset_idx * subset_size;
            let end_idx = ((subset_idx + 1) * subset_size).min(n_measurements);

            let a_subset = a.slice(ndarray::s![start_idx..end_idx, ..]);
            let y_subset = y.slice(ndarray::s![start_idx..end_idx]);

            let sensitivity = a_subset.sum_axis(ndarray::Axis(0));
            let forward_proj = a_subset.dot(x);

            let mut ratio = Array1::zeros(end_idx - start_idx);
            for i in 0..ratio.len() {
                if forward_proj[i] > 1e-10 {
                    ratio[i] = y_subset[i] / forward_proj[i];
                }
            }

            let correction = a_subset.t().dot(&ratio);

            for i in 0..n_voxels {
                if sensitivity[i] > 1e-10 {
                    x[i] *= correction[i] / sensitivity[i];
                }
            }
        }

        Ok(())
    }

    /// Proximal gradient regularization step.
    ///
    /// Applies a gradient descent step on the 3-D Laplacian smoothness penalty:
    /// x ← x − λ · (−Δx), then enforces non-negativity for OSEM.
    /// # Errors
    /// - Returns [`Err`] if an internal constraint is violated.
    ///
    pub(super) fn apply_regularization(&self, x: &mut Array1<f64>) -> KwaversResult<()> {
        if self.regularization_parameter <= 0.0 {
            return Ok(());
        }

        let n = x.len();
        let grid_size_est = (n as f64).cbrt() as usize;

        let mut grad_reg = Array1::zeros(n);

        for idx in 0..n {
            let (i, j, k) = self.linear_to_3d_index(idx, [grid_size_est; 3]);
            let mut laplacian = -6.0 * x[idx];
            let mut count = 0;

            for (di, dj, dk) in &[
                (-1, 0, 0),
                (1, 0, 0),
                (0, -1, 0),
                (0, 1, 0),
                (0, 0, -1),
                (0, 0, 1),
            ] {
                let ni = (i as i32 + di) as usize;
                let nj = (j as i32 + dj) as usize;
                let nk = (k as i32 + dk) as usize;

                if ni < grid_size_est && nj < grid_size_est && nk < grid_size_est {
                    let neighbor_idx = ni * grid_size_est * grid_size_est + nj * grid_size_est + nk;
                    if neighbor_idx < n {
                        laplacian += x[neighbor_idx];
                        count += 1;
                    }
                }
            }

            if count > 0 {
                grad_reg[idx] = -laplacian / f64::from(count);
            }
        }

        *x = &*x - self.regularization_parameter * grad_reg;

        if matches!(self.algorithm, IterativeAlgorithm::OSEM { .. }) {
            apply_inplace(x, |v| v.max(0.0));
        }

        Ok(())
    }
}
