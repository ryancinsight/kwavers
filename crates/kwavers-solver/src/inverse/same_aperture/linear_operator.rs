//! Linear-operator contract for same-aperture inverse kernels.

/// Finite-dimensional forward operator used by same-aperture inverse solvers.
///
/// Implementors provide matrix-free products and storage accounting. PCG
/// solvers consume this trait so dense, sparse, streamed, or backend-resident
/// operators share one normal-equation implementation.
pub trait LinearOperator {
    #[must_use]
    fn rows(&self) -> usize;

    #[must_use]
    fn cols(&self) -> usize;

    fn matvec(&self, x: &[f32], out: &mut [f32]);

    fn t_matvec(&self, y: &[f32], out: &mut [f32]);

    fn row_values(&self, row: usize, out: &mut [f32]);

    #[must_use]
    fn normal_diag(&self) -> Vec<f32>;

    #[must_use]
    fn storage_values(&self) -> usize;

    #[must_use]
    fn dense_values(&self) -> usize {
        self.rows() * self.cols()
    }
}

/// Dot product `a · b` with 8-lane unrolled accumulation.
///
/// # Theorem
///
/// Splitting the sum into 8 independent partial accumulators breaks the serial
/// dependency chain `acc_i = acc_{i-1} + a_i * b_i` that prevents vectorization.
/// With 8 independent lanes, the compiler emits `vmulps` + `vaddps` sequences
/// (AVX2: 8 × f32 per cycle) instead of a scalar dependency chain.
/// The horizontal sum at the end is O(log N) = O(3) operations.
/// Numerical result is equivalent to sequential summation within floating-point
/// rounding: the difference is a permutation of the same summands.
pub(crate) fn dot(a: &[f32], b: &[f32]) -> f32 {
    debug_assert_eq!((a.shape()[0] * a.shape()[1] * a.shape()[2]), (b.shape()[0] * b.shape()[1] * b.shape()[2]));
    let n = (a.shape()[0] * a.shape()[1] * a.shape()[2]);
    let end8 = (n / 8) * 8;
    let mut acc0 = 0.0_f32;
    let mut acc1 = 0.0_f32;
    let mut acc2 = 0.0_f32;
    let mut acc3 = 0.0_f32;
    let mut acc4 = 0.0_f32;
    let mut acc5 = 0.0_f32;
    let mut acc6 = 0.0_f32;
    let mut acc7 = 0.0_f32;
    for (ca, cb) in a[..end8].chunks_exact(8).zip(b[..end8].chunks_exact(8)) {
        acc0 += ca[0] * cb[0];
        acc1 += ca[1] * cb[1];
        acc2 += ca[2] * cb[2];
        acc3 += ca[3] * cb[3];
        acc4 += ca[4] * cb[4];
        acc5 += ca[5] * cb[5];
        acc6 += ca[6] * cb[6];
        acc7 += ca[7] * cb[7];
    }
    let mut sum = (acc0 + acc1) + (acc2 + acc3) + (acc4 + acc5) + (acc6 + acc7);
    for (av, bv) in a[end8..].iter().zip(b[end8..].iter()) {
        sum += av * bv;
    }
    sum
}

/// BLAS-1 saxpy: `y += alpha * x` with 8-lane unrolled loop.
///
/// The 8-unroll exposes 8 independent fused-multiply-add operations per iteration,
/// enabling the compiler to emit `vfmadd231ps` (AVX2) with no false data
/// dependencies between the 8 stores.
pub(crate) fn axpy(alpha: f32, x: &[f32], y: &mut [f32]) {
    debug_assert_eq!((x.shape()[0] * x.shape()[1] * x.shape()[2]), (y.shape()[0] * y.shape()[1] * y.shape()[2]));
    let n = (y.shape()[0] * y.shape()[1] * y.shape()[2]);
    let end8 = (n / 8) * 8;
    for (yc, xc) in y[..end8].chunks_exact_mut(8).zip(x[..end8].chunks_exact(8)) {
        yc[0] += alpha * xc[0];
        yc[1] += alpha * xc[1];
        yc[2] += alpha * xc[2];
        yc[3] += alpha * xc[3];
        yc[4] += alpha * xc[4];
        yc[5] += alpha * xc[5];
        yc[6] += alpha * xc[6];
        yc[7] += alpha * xc[7];
    }
    for (yv, xv) in y[end8..].iter_mut().zip(x[end8..].iter()) {
        *yv += alpha * *xv;
    }
}
