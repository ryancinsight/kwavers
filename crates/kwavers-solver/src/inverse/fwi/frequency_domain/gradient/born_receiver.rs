//! Per-receiver Born terms shared by prediction and the discrete adjoint.

use super::super::forward::outgoing_green;
use kwavers_math::fft::Complex64;
use kwavers_physics::acoustics::imaging::modalities::ultrasound::frequency_domain_fwi::helmholtz_slowness_derivative;
use kwavers_transducer::transducers::ElementPosition;

#[cfg(test)]
mod tests;

pub(super) struct BornReceiverTerms<'a> {
    pub(super) centers: &'a [(usize, ElementPosition)],
    pub(super) incident: &'a [Complex64],
    pub(super) potential: &'a [f64],
    pub(super) slowness: &'a [f64],
    pub(super) omega: f64,
    pub(super) reference_wavenumber: f64,
    pub(super) min_distance: f64,
    pub(super) cell_volume: f64,
}

impl BornReceiverTerms<'_> {
    pub(super) fn fill_green(&self, receiver: ElementPosition, scratch: &mut Vec<Complex64>) {
        scratch.clear();
        scratch.extend(self.centers.iter().map(|(_, point)| {
            outgoing_green(
                *point,
                receiver,
                self.reference_wavenumber,
                self.min_distance,
            )
        }));
    }

    pub(super) fn predict(
        &self,
        sources: &[ElementPosition],
        receiver: ElementPosition,
        green: &[Complex64],
    ) -> Complex64 {
        let direct = sources
            .iter()
            .map(|&source| {
                outgoing_green(
                    source,
                    receiver,
                    self.reference_wavenumber,
                    self.min_distance,
                )
            })
            .sum::<Complex64>();
        let scattered = self
            .incident
            .iter()
            .zip(self.potential.iter())
            .zip(green.iter())
            .map(|((&incident, &potential), &value)| {
                incident * value * (potential * self.cell_volume)
            })
            .sum::<Complex64>();
        direct + scattered
    }

    /// Adds the adjoint in the same C-order used by `voxel_centers`.
    /// For shape `(nx, ny, nz)`, voxel `(ix, iy, iz)` has offset
    /// `(ix * ny + iy) * nz + iz`. The freshly allocated gradient and every
    /// term vector share that order and length. Borrowing its dense slice once
    /// removes repeated coordinate division and checked layout projection;
    /// complex arithmetic and receiver/voxel addition order remain identical.
    pub(super) fn accumulate_adjoint(
        &self,
        gradient: &mut [f64],
        source_scale: Complex64,
        residual: Complex64,
        green: &[Complex64],
    ) {
        debug_assert_eq!(gradient.len(), self.centers.len());
        for ((grad, (&incident, &slowness)), &value) in gradient
            .iter_mut()
            .zip(self.incident.iter().zip(self.slowness.iter()))
            .zip(green.iter())
        {
            let derivative = source_scale
                * incident
                * value
                * (helmholtz_slowness_derivative(self.omega, slowness) * self.cell_volume);
            *grad += (derivative.conj() * residual).re;
        }
    }
}
