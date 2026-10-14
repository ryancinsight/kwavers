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
    /// removes repeated coordinate division and checked layout projection;
    /// complex arithmetic and receiver/voxel addition order remain identical.
    pub(super) fn accumulate_adjoint<const K: usize>(
        &self,
        gradient: &mut [f64],
        scaled_incident: &[Complex64],
        residual: Complex64,
        green: &[Complex64],
    ) {
        debug_assert_eq!(gradient.len(), scaled_incident.len());
        debug_assert_eq!(gradient.len(), self.derivative_volume.len());
        debug_assert_eq!(gradient.len(), green.len());
        let (gradient, gradient_tail) = gradient.as_chunks_mut::<K>();
        let (incident, incident_tail) = scaled_incident.as_chunks::<K>();
        let (green, green_tail) = green.as_chunks::<K>();
        let (weights, weights_tail) = self.derivative_volume.as_chunks::<K>();
        for (((gradient, incident), green), weights) in
            gradient.iter_mut().zip(incident).zip(green).zip(weights)
        {
            let (incident, green, weights) = (*incident, *green, *weights);
            for lane in 0..K {
                let derivative = weighted_product(incident[lane], green[lane], weights[lane]);
                gradient[lane] += project_adjoint(derivative, residual);
            }
        }
        for lane in 0..gradient_tail.len() {
            let derivative =
                weighted_product(incident_tail[lane], green_tail[lane], weights_tail[lane]);
            gradient_tail[lane] += project_adjoint(derivative, residual);
        }
    }
}
