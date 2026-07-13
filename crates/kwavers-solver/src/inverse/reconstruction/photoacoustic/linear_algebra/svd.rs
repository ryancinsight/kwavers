//! Truncated SVD solver via power iteration.

use leto::{Array1, Array2, ArrayView1};

use kwavers_core::error::KwaversResult;

use super::PhotoacousticLinearSolver;

impl PhotoacousticLinearSolver {
    /// Solve using truncated SVD for ill-conditioned problems.
    ///
    /// Computes the pseudo-inverse solution discarding singular values below
    /// `truncation * σ_max`. Uses the power-iteration SVD internally.
    ///
    /// # Errors
    /// Returns `Err` when the power-iteration SVD fails.
    pub fn solve_truncated_svd(
        &self,
        a: &Array2<f64>,
        b: ArrayView1<f64>,
        truncation: f64,
    ) -> KwaversResult<Array1<f64>> {
        let (u, s, vt) = self.power_iteration_svd(a)?;

        let s_max = s.iter().copied().fold(0.0, f64::max);
        let threshold = truncation * s_max;

        let mut x = Array1::<f64>::zeros(vt.shape()[0]);
        for (i, &s_val) in s.iter().enumerate() {
            if s_val > threshold {
                let ui = u
                    .index_axis::<1>(1, i)
                    .expect("invariant: SVD U column index in range");
                let vi = vt
                    .index_axis::<1>(0, i)
                    .expect("invariant: SVD Vᵀ row index in range");
                let coeff = leto_ops::dot(&ui, &b).expect("invariant: SVD uᵢ·b conforms") / s_val;
                for j in 0..x.len() {
                    x[j] += vi[j] * coeff;
                }
            }
        }

        Ok(x)
    }

    /// Power-iteration SVD: computes `(U, s, Vᵀ)` for an `m × n` matrix `A`.
    ///
    /// Runs 100 power iterations per singular triplet. For production workloads
    /// prefer LAPACK `DGESVD`; this implementation suffices for moderate
    /// problem sizes in the photoacoustic reconstruction pipeline.
    ///
    /// # Errors
    /// Always returns `Ok`; the signature matches callers that propagate errors.
    fn power_iteration_svd(
        &self,
        a: &Array2<f64>,
    ) -> KwaversResult<(Array2<f64>, Vec<f64>, Array2<f64>)> {
        let [m, n] = a.shape();
        let k = m.min(n);

        // AᵀA  (n×n dense)
        let at = a.transpose([1, 0]).expect("invariant: SVD transpose valid");
        let mut ata = Array2::<f64>::zeros((n, n));
        leto_ops::matmul(&at, &a.view(), &mut ata.view_mut()).expect("invariant: SVD AᵀA conforms");

        let mut v = Array2::eye(n);
        let mut s = vec![0.0; k];

        for i in 0..k {
            let mut vi = Array1::<f64>::ones(n);
            for _ in 0..100 {
                // vi ← AᵀA·vi, then normalise
                let mut vi_next = Array1::<f64>::zeros(n);
                leto_ops::matvec(&ata.view(), &vi.view(), &mut vi_next.view_mut())
                    .expect("invariant: SVD AᵀA·v conforms");
                let norm = leto_ops::dot(&vi_next.view(), &vi_next.view())
                    .expect("invariant: SVD ‖v‖ conforms")
                    .sqrt();
                for j in 0..n {
                    vi_next[j] /= norm;
                }
                vi = vi_next;
            }

            let mut avi = Array1::<f64>::zeros(m);
            leto_ops::matvec(&a.view(), &vi.view(), &mut avi.view_mut())
                .expect("invariant: SVD A·v conforms");
            s[i] = leto_ops::dot(&avi.view(), &avi.view())
                .expect("invariant: SVD singular value conforms")
                .sqrt();

            for j in 0..n {
                v[[j, i]] = vi[j];
            }
        }

        // U = AV / S, column-wise.
        let mut u = Array2::<f64>::zeros((m, k));
        for i in 0..k {
            if s[i] > 1e-10 {
                let vi = v
                    .index_axis::<1>(1, i)
                    .expect("invariant: SVD V column index in range");
                let mut ui = Array1::<f64>::zeros(m);
                leto_ops::matvec(&a.view(), &vi, &mut ui.view_mut())
                    .expect("invariant: SVD A·vᵢ conforms");
                for j in 0..m {
                    u[[j, i]] = ui[j] / s[i];
                }
            }
        }

        Ok((
            u,
            s,
            v.transpose([1, 0])
                .expect("invariant: SVD Vᵀ transpose valid")
                .to_contiguous(),
        ))
    }
}
