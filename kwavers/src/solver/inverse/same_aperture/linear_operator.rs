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

    #[must_use]
    fn normal_diag(&self) -> Vec<f32>;

    #[must_use]
    fn storage_values(&self) -> usize;

    #[must_use]
    fn dense_values(&self) -> usize {
        self.rows() * self.cols()
    }
}

pub(crate) fn dot(a: &[f32], b: &[f32]) -> f32 {
    debug_assert_eq!(a.len(), b.len());
    a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
}
