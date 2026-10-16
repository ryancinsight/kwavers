//! Analytical Rayleigh oracle for the focused rectangular-element array in brain.

use super::{attenuation_np_m, GridSpec, LinearArraySpec, FREQUENCY_HZ, SOURCE_CONVENTION_FACTOR};
use anyhow::{anyhow, Result};
use eunomia::Complex64;
use eunomia::FloatElement;
use kwavers_transducer::transducers::physics::{
    rayleigh_pressure, PlanarAperture, PlanarApertureShape, RayleighIntegralSpec,
};
use leto::Array2;
use std::f64::consts::TAU;

/// Gauss–Legendre order per element edge; the element is 0.45 λ wide, so 8 nodes resolve it.
pub const ELEMENT_QUADRATURE_ORDER: usize = 8;

/// Rayleigh apertures for every element with the focusing phase of the drive.
///
/// The drive is `sin(ω(t − τ_i))`, whose phasor under the `e^{−iωt}` convention of
/// the Rayleigh kernel is `i·e^{+iωτ_i}`; the common factor `i` does not change
/// magnitudes and is dropped.
///
/// # Errors
///
/// Returns an error if an aperture cannot be constructed.
pub fn element_apertures(
    array: &LinearArraySpec,
    focus_m: [f64; 3],
    amplitude_pa: f64,
) -> Result<Vec<PlanarAperture>> {
    let delays = array.focus_delays_s(focus_m);
    let omega = TAU * FREQUENCY_HZ;
    array
        .x_m
        .iter()
        .zip(&delays)
        .map(|(&x, &tau)| {
            PlanarAperture::oriented(
                [x, array.y_m, array.z_m],
                [0.0, 0.0, 1.0],
                [1.0, 0.0, 0.0],
                PlanarApertureShape::Rectangle {
                    width_m: array.element_width_m,
                    height_m: array.element_height_m,
                },
                Complex64::from_polar(amplitude_pa, omega * tau),
            )
            .map_err(|error| anyhow!("{error:?}"))
        })
        .collect()
}

/// Brain propagation: power-law attenuation at the drive frequency, `e^{−α r}` per path.
fn spec() -> Result<RayleighIntegralSpec> {
    let wavenumber = TAU * FREQUENCY_HZ / super::SOUND_SPEED_M_S;
    RayleighIntegralSpec::new(
        wavenumber,
        attenuation_np_m(),
        ELEMENT_QUADRATURE_ORDER,
        ELEMENT_QUADRATURE_ORDER,
    )
    .map_err(|error| anyhow!("{error:?}"))
}

/// Oracle pressure amplitude on the x–z plane at elevation index `j`.
///
/// Values carry [`SOURCE_CONVENTION_FACTOR`] so they compare directly with the
/// additive-source PSTD run. Layout is `[i, k]` over the grid.
///
/// # Errors
///
/// Returns an error if the Rayleigh evaluation fails.
pub fn xz_plane(grid: &GridSpec, j: usize, apertures: &[PlanarAperture]) -> Result<Array2<f64>> {
    let y = f64::from_count(j) * grid.dx;
    let points: Vec<[f64; 3]> = (0..grid.nx)
        .flat_map(|i| {
            (0..grid.nz).map(move |k| {
                [
                    f64::from_count(i) * grid.dx,
                    y,
                    f64::from_count(k) * grid.dx,
                ]
            })
        })
        .collect();
    let values = amplitudes_at(&points, apertures)?;
    let mut plane = Array2::<f64>::zeros([grid.nx, grid.nz]);
    for (index, value) in values.into_iter().enumerate() {
        plane[[index / grid.nz, index % grid.nz]] = value;
    }
    Ok(plane)
}

/// Oracle pressure amplitude at arbitrary physical points, scaled as in [`xz_plane`].
///
/// # Errors
///
/// Returns an error if the Rayleigh evaluation fails.
pub fn amplitudes_at(points_m: &[[f64; 3]], apertures: &[PlanarAperture]) -> Result<Vec<f64>> {
    let values =
        rayleigh_pressure(points_m, apertures, &spec()?).map_err(|error| anyhow!("{error:?}"))?;
    Ok(values
        .iter()
        .map(|value| SOURCE_CONVENTION_FACTOR * value.norm())
        .collect())
}

/// Oracle amplitude at a single focal point.
///
/// # Errors
///
/// Returns an error if the Rayleigh evaluation fails.
pub fn focal_amplitude(
    array: &LinearArraySpec,
    focus_m: [f64; 3],
    amplitude_pa: f64,
) -> Result<f64> {
    let apertures = element_apertures(array, focus_m, amplitude_pa)?;
    Ok(amplitudes_at(&[focus_m], &apertures)?[0])
}
