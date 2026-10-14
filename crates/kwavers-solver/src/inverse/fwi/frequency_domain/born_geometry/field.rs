//! Fixed-reference Green fields and C-order voxel coordinates.
//!
//! G(x,r;k) = exp(i k d)/(4 pi d), d = max(|x-r|, h/2).
//! Incident pressure sums sources in their supplied order. Coordinates follow
//! the same (ix * ny + iy) * nz + iz layout as the dense model and gradient.

use aequitas::systems::si::quantities::Length;
use aequitas::systems::si::units::Meter;
use kwavers_core::constants::numerical::FOUR_PI;
use kwavers_math::fft::Complex64;
use kwavers_transducer::transducers::ElementPosition;

pub(super) fn incident_field(
    sources: &[ElementPosition],
    centers: &[(usize, ElementPosition)],
    wavenumber: f64,
    min_distance: f64,
) -> Vec<Complex64> {
    centers
        .iter()
        .map(|(_, point)| {
            sources
                .iter()
                .map(|&source| outgoing_green(source, *point, wavenumber, min_distance))
                .sum()
        })
        .collect()
}

pub(super) fn outgoing_green(
    source: ElementPosition,
    receiver: ElementPosition,
    wavenumber: f64,
    min_distance: f64,
) -> Complex64 {
    let dx = source.x.in_unit::<Meter>() - receiver.x.in_unit::<Meter>();
    let dy = source.y.in_unit::<Meter>() - receiver.y.in_unit::<Meter>();
    let dz = source.z.in_unit::<Meter>() - receiver.z.in_unit::<Meter>();
    let distance = (dx * dx + dy * dy + dz * dz).sqrt().max(min_distance);
    Complex64::from_polar(1.0 / (FOUR_PI * distance), wavenumber * distance)
}

pub(in super::super) fn voxel_centers(
    dimensions: (usize, usize, usize),
    spacing_m: f64,
) -> Vec<(usize, ElementPosition)> {
    let (nx, ny, nz) = dimensions;
    let cx = 0.5 * nx as f64;
    let cy = 0.5 * ny as f64;
    let cz = 0.5 * nz as f64;
    let mut centers = Vec::with_capacity(nx * ny * nz);
    for ix in 0..nx {
        for iy in 0..ny {
            for iz in 0..nz {
                centers.push((
                    ix * ny * nz + iy * nz + iz,
                    ElementPosition {
                        x: Length::from_unit::<Meter>((ix as f64 + 0.5 - cx) * spacing_m),
                        y: Length::from_unit::<Meter>((iy as f64 + 0.5 - cy) * spacing_m),
                        z: Length::from_unit::<Meter>((iz as f64 + 0.5 - cz) * spacing_m),
                    },
                ));
            }
        }
    }
    centers
}
