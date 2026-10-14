//! Per-receiver Born terms shared by prediction and the discrete adjoint.

use super::super::born_kernel::{integrate, project_adjoint, weighted_product};
use kwavers_math::fft::Complex64;

#[cfg(test)]
mod tests;

pub(super) struct BornReceiverTerms<'a> {
    pub(super) incident: &'a [Complex64],
    pub(super) potential_volume: &'a [f64],
    pub(super) derivative_volume: &'a [f64],
}

impl BornReceiverTerms<'_> {
    pub(super) fn predict(&self, direct: Complex64, green: &[Complex64]) -> Complex64 {
        let scattered = integrate::<_, { super::super::born_kernel::VOXELS_PER_BLOCK }>(
            self.incident,
            green,
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
    pub(super) fn accumulate_adjoint<const K: usize>(
        &self,
        gradient: &mut [f64],
        scaled_incident: &[Complex64],
        receivers: &[(&[Complex64], Complex64)],
    ) {
        debug_assert_eq!(gradient.len(), scaled_incident.len());
        debug_assert_eq!(gradient.len(), self.derivative_volume.len());
        let receivers: Vec<_> = receivers
            .iter()
            .map(|&(green, residual)| {
                assert_eq!(gradient.len(), green.len());
                let (blocks, tail) = green.as_chunks::<K>();
                (blocks, tail, residual)
            })
            .collect();
        let (gradient, gradient_tail) = gradient.as_chunks_mut::<K>();
        let (incident, incident_tail) = scaled_incident.as_chunks::<K>();
        let (weights, weights_tail) = self.derivative_volume.as_chunks::<K>();
        for (block, ((gradient, incident), weights)) in
            gradient.iter_mut().zip(incident).zip(weights).enumerate()
        {
            let (mut accumulated, incident, weights) = (*gradient, *incident, *weights);
            for &(green, _, residual) in &receivers {
                let green = green[block];
                for lane in 0..K {
                    let derivative = weighted_product(incident[lane], green[lane], weights[lane]);
                    accumulated[lane] += project_adjoint(derivative, residual);
                }
            }
            *gradient = accumulated;
        }
        for lane in 0..gradient_tail.len() {
            let mut accumulated = gradient_tail[lane];
            for &(_, green, residual) in &receivers {
                let derivative =
                    weighted_product(incident_tail[lane], green[lane], weights_tail[lane]);
                accumulated += project_adjoint(derivative, residual);
            }
            gradient_tail[lane] = accumulated;
        }
    }
}
