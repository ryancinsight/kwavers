//! Per-receiver Born terms shared by prediction and the discrete adjoint.

use super::super::born_kernel::fields::FieldProducts;
use super::super::born_kernel::{integrate, project_adjoint};
use kwavers_math::fft::Complex64;

#[cfg(test)]
mod tests;

pub(super) struct BornReceiverTerms<'a> {
    pub(super) potential_volume: &'a [f64],
    pub(super) derivative_volume: &'a [f64],
}

impl BornReceiverTerms<'_> {
    pub(super) fn predict<P: FieldProducts<f64>>(
        &self,
        direct: Complex64,
        fields: &P,
    ) -> Complex64 {
        let scattered = integrate::<_, _, { super::super::born_kernel::VOXELS_PER_BLOCK }>(
            fields,
            self.potential_volume,
        );
        direct + scattered
    }

    /// Adds the adjoint in the same C-order used by `voxel_centers`.
    /// For shape `(nx, ny, nz)`, voxel `(ix, iy, iz)` has offset
    /// `(ix * ny + iy) * nz + iz`. The freshly allocated gradient and every
    /// term vector share that order and length. Borrowing its dense slice once
    /// removes repeated coordinate division and checked layout projection.
    /// A block keeps its gradient, incident field and weights local while
    /// visiting receivers in their original order. Interleaving independent
    /// voxels changes no scalar expression or per-voxel addition sequence.
    pub(super) fn accumulate_adjoint<const K: usize, P: FieldProducts<f64>>(
        &self,
        gradient: &mut [f64],
        receivers: &[(P, Complex64)],
    ) {
        const { assert!(K > 0, "Born block capacity must be positive") };
        assert_eq!(gradient.len(), self.derivative_volume.len());
        for (fields, _) in receivers {
            assert_eq!(gradient.len(), fields.len());
        }
        let mut derivatives = [Complex64::new(0.0, 0.0); K];
        let mut accumulated = [0.0; K];
        for (block, (gradient, weights)) in gradient
            .chunks_mut(K)
            .zip(self.derivative_volume.chunks(K))
            .enumerate()
        {
            let len = gradient.len();
            accumulated[..len].copy_from_slice(gradient);
            let start = block * K;
            for &(ref fields, residual) in receivers {
                project_adjoint::<_, _, K, true>(
                    fields,
                    start..start + len,
                    weights,
                    residual,
                    &mut derivatives,
                    &mut accumulated,
                );
            }
            gradient.copy_from_slice(&accumulated[..len]);
        }
    }
}
