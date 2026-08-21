//! 3-D spherical acquisition geometry and source construction.

use super::{Array2, Array3, FwiGeometry, KwaversResult, NX, NY, NZ};
use aequitas::systems::si::quantities::{Frequency, Pressure, Time};
use kwavers_signal::DomainRickerWavelet;
use kwavers_source::{GridSource, SourceMode};
use std::f64::consts::PI;

const PEAK_PRESSURE_PA: f64 = 1.0e5;

/// Generate positions on a sphere with the Fibonacci lattice.
///
/// Positions are rounded to grid coordinates and clamped to the physical
/// domain so the CPML boundary is not used as an acquisition surface.
pub(super) fn fibonacci_sphere_elements(
    n: usize,
    r: f64,
    cx: f64,
    cy: f64,
    cz: f64,
) -> Vec<[usize; 3]> {
    let golden_ratio = (1.0 + 5.0_f64.sqrt()) / 2.0;
    (0..n)
        .map(|i| {
            let y_norm = (2.0 * (i as f64 + 0.5) / n as f64) - 1.0;
            let r_xz = (1.0 - y_norm * y_norm).sqrt();
            let phi = 2.0 * PI * i as f64 / golden_ratio;
            let x_off = r * r_xz * phi.cos();
            let y_off = r * y_norm;
            let z_off = r * r_xz * phi.sin();
            let ix = ((cx + x_off).round() as isize).clamp(6, NX as isize - 7) as usize;
            let iy = ((cy + y_off).round() as isize).clamp(6, NY as isize - 7) as usize;
            let iz = ((cz + z_off).round() as isize).clamp(6, NZ as isize - 7) as usize;
            [ix, iy, iz]
        })
        .collect()
}

fn build_receiver_mask_3d(all_elements: &[[usize; 3]], source_idx: usize) -> Array3<bool> {
    let mut mask = Array3::<bool>::from_elem((NX, NY, NZ), false);
    for (idx, &[ix, iy, iz]) in all_elements.iter().enumerate() {
        if idx != source_idx {
            mask[[ix, iy, iz]] = true;
        }
    }
    mask
}

/// Build one FWI shot with a point source and all other sphere elements as receivers.
pub(super) fn build_shot_3d(
    source_pos: [usize; 3],
    all_elements: &[[usize; 3]],
    shot_idx: usize,
    f0_hz: f64,
    nt: usize,
    dt: f64,
) -> KwaversResult<FwiGeometry> {
    let [ix, iy, iz] = source_pos;
    let mut source_mask = Array3::<f64>::zeros((NX, NY, NZ));
    source_mask[[ix, iy, iz]] = 1.0;

    let wavelet = DomainRickerWavelet::causal(
        Frequency::from_base(f0_hz),
        Pressure::from_base(PEAK_PRESSURE_PA),
    )?;
    let mut p_signal = Array2::<f64>::zeros((1, nt));
    for (t, pressure) in wavelet.samples(Time::from_base(dt), nt)?.enumerate() {
        p_signal[[0, t]] = pressure;
    }

    let mut source = GridSource::new_empty();
    source.p_mask = Some(source_mask);
    source.p_signal = Some(p_signal);
    source.p_mode = SourceMode::Dirichlet;

    Ok(FwiGeometry::new(
        source,
        build_receiver_mask_3d(all_elements, shot_idx),
    ))
}

#[cfg(test)]
mod tests {
    use super::{build_receiver_mask_3d, fibonacci_sphere_elements, NX, NY, NZ};

    #[test]
    fn fibonacci_positions_stay_inside_the_physical_domain() {
        let elements =
            fibonacci_sphere_elements(24, 21.0, NX as f64 / 2.0, NY as f64 / 2.0, NZ as f64 / 2.0);
        assert_eq!(elements.len(), 24);
        assert!(elements.iter().all(|&[ix, iy, iz]| {
            (6..=NX - 7).contains(&ix) && (6..=NY - 7).contains(&iy) && (6..=NZ - 7).contains(&iz)
        }));
    }

    #[test]
    fn receiver_mask_excludes_the_transmitting_element() {
        let elements = [[12, 24, 12], [14, 24, 12], [16, 24, 12]];
        let mask = build_receiver_mask_3d(&elements, 1);
        assert!(!mask[[14, 24, 12]]);
        assert!(mask[[12, 24, 12]]);
        assert!(mask[[16, 24, 12]]);
    }
}
