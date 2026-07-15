//! SIRT, ART, OSEM iteration steps and regularization for [`IterativeMethods`].

use super::IterativeAlgorithm;
use super::IterativeMethods;
use kwavers_core::error::KwaversResult;
use leto::{Array1, Array2};
use moirai_parallel::{enumerate_mut_with, Adaptive};

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
        let n_rows = a.shape()[0];
        // ax = A·x
        let mut ax = Array1::<f64>::zeros(n_rows);
        leto_ops::matvec(&a.view(), &x.view(), &mut ax.view_mut())
            .expect("invariant: SIRT A·x conforms");
        // residual = y − ax
        let mut residual = Array1::<f64>::zeros(n_rows);
        for i in 0..n_rows {
            residual[i] = y[i] - ax[i];
        }
        // update = Aᵀ·residual  (transposed view; matvec handles the strides)
        let at = a
            .transpose([1, 0])
            .expect("invariant: SIRT transpose valid");
        let mut update = Array1::<f64>::zeros(at.shape()[0]);
        leto_ops::matvec(&at, &residual.view(), &mut update.view_mut())
            .expect("invariant: SIRT Aᵀ·residual conforms");
        // x ← x + λ·update
        let mut out = Array1::<f64>::zeros(x.len());
        for i in 0..x.len() {
            out[i] = x[i] + self.relaxation_factor * update[i];
        }
        Ok(out)
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
        let n_rows = a.shape()[0];
        for i in 0..n_rows {
            let row = a
                .index_axis::<1>(0, i)
                .expect("invariant: ART row index in range");
            let ax_i = leto_ops::dot(&row, &x.view()).expect("invariant: ART aᵢ·x conforms");
            let residual = y[i] - ax_i;
            let row_norm_sq = leto_ops::dot(&row, &row).expect("invariant: ART ‖aᵢ‖² conforms");

            if row_norm_sq > 0.0 {
                let update_factor = self.relaxation_factor * residual / row_norm_sq;
                if let (Some(x_values), Some(row_values)) = (x.as_slice_mut(), row.as_slice()) {
                    enumerate_mut_with::<Adaptive, _, _>(x_values, |idx, x_value| {
                        *x_value += update_factor * row_values[idx];
                    });
                } else {
                    for k in 0..x.len() {
                        x[k] += update_factor * row[k];
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
        let [n_measurements, n_voxels] = a.shape();

        for value in x.iter_mut() {
            *value = value.max(1e-10);
        }

        let subset_size = n_measurements.div_ceil(subsets);

        for subset_idx in 0..subsets {
            let start_idx = subset_idx * subset_size;
            let end_idx = ((subset_idx + 1) * subset_size).min(n_measurements);

            let a_subset = a
                .slice_with::<2>(&s![start_idx..end_idx, ..])
                .expect("invariant: OSEM measurement-row subset in range");
            let y_subset = y
                .slice_with::<1>(&s![start_idx..end_idx])
                .expect("invariant: OSEM measurement subset in range");
            let n_sub = end_idx - start_idx;

            // sensitivity[j] = Σ_i A_subset[i, j]  (column sums)
            let mut sensitivity = Array1::<f64>::zeros(n_voxels);
            for ii in 0..n_sub {
                for j in 0..n_voxels {
                    sensitivity[j] += a_subset[[ii, j]];
                }
            }

            // forward_proj = A_subset · x
            let mut forward_proj = Array1::<f64>::zeros(n_sub);
            leto_ops::matvec(&a_subset, &x.view(), &mut forward_proj.view_mut())
                .expect("invariant: OSEM A_s·x conforms");

            let mut ratio = Array1::<f64>::zeros(n_sub);
            for i in 0..ratio.len() {
                if forward_proj[i] > 1e-10 {
                    ratio[i] = y_subset[i] / forward_proj[i];
                }
            }

            // correction = A_subsetᵀ · ratio  (transposed view)
            let at = a_subset
                .transpose([1, 0])
                .expect("invariant: OSEM subset transpose valid");
            let mut correction = Array1::<f64>::zeros(n_voxels);
            leto_ops::matvec(&at, &ratio.view(), &mut correction.view_mut())
                .expect("invariant: OSEM A_sᵀ·ratio conforms");

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

        let mut grad_reg = Array1::<f64>::zeros(n);

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

        for idx in 0..n {
            x[idx] -= self.regularization_parameter * grad_reg[idx];
        }

        if matches!(self.algorithm, IterativeAlgorithm::OSEM { .. }) {
            for value in x.iter_mut() {
                *value = value.max(0.0);
            }
        }

        Ok(())
    }
}
