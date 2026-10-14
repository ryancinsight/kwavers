//! Per-receiver Born terms shared by prediction and the discrete adjoint.

use super::super::forward::outgoing_green;
use kwavers_math::fft::Complex64;
use kwavers_physics::acoustics::imaging::modalities::ultrasound::frequency_domain_fwi::helmholtz_slowness_derivative;
use kwavers_transducer::transducers::ElementPosition;
use leto::ArrayViewMut3;

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

    pub(super) fn accumulate_adjoint(
        &self,
        gradient: &mut ArrayViewMut3<'_, f64>,
        source_scale: Complex64,
        residual: Complex64,
        green: &[Complex64],
    ) {
        let [_, ny, nz] = gradient.shape();
        for (((linear_index, _), (&incident, &slowness)), &value) in self
            .centers
            .iter()
            .zip(self.incident.iter().zip(self.slowness.iter()))
            .zip(green.iter())
        {
            let derivative = source_scale
                * incident
                * value
                * (helmholtz_slowness_derivative(self.omega, slowness) * self.cell_volume);
            let ix = linear_index / (ny * nz);
            let rem = linear_index % (ny * nz);
            gradient[[ix, rem / nz, rem % nz]] += (derivative.conj() * residual).re;
        }
    }
}
