//! Truncated SVD solver via power iteration.

use leto::{
    Array1,
    Array2,
    ArrayView1,
};

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

        let mut x = Array1::zeros(vt.nrows());
        for (i, &s_val) in s.iter().enumerate() {
            if s_val > threshold {
                let ui = u.column(i);
                let vi = vt.row(i);
                x += &(vi.to_owned() * (ui.dot(&b) / s_val));
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
        let (m, n) = a.dim();
        let k = m.min(n);

        let ata = a.t().dot(a);

        let mut v = Array2::eye(n);
        let mut s = vec![0.0; k];

        for i in 0..k {
            let mut vi = Array1::ones(n);
            for _ in 0..100 {
                vi = ata.dot(&vi);
                vi /= vi.dot(&vi).sqrt();
            }

            let avi = a.dot(&vi);
            s[i] = avi.dot(&avi).sqrt();

            for j in 0..n {
                v[[j, i]] = vi[j];
            }
        }

        // U = AV / S, column-wise.
        let mut u = Array2::zeros((m, k));
        for i in 0..k {
            if s[i] > 1e-10 {
                let vi = v.column(i);
                let ui = a.dot(&vi) / s[i];
                for j in 0..m {
                    u[[j, i]] = ui[j];
                }
            }
        }

        Ok((u, s, v.t().to_owned()))
    }
}
