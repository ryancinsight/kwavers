//! Multi-shot source and receiver geometry for the transcranial example.

use super::config::GridSpec;
use aequitas::systems::si::quantities::{Frequency, Pressure, Time};
use kwavers_core::error::KwaversResult;
use kwavers_signal::DomainRickerWavelet;
use kwavers_solver::inverse::fwi::time_domain::FwiGeometry;
use kwavers_source::{GridSource, SourceMode};
use leto::{Array2, Array3};

/// Peak source pressure in pascals.
const P0_PA: f64 = 1.0e5;

/// Four source positions covering a 120-degree left aperture.
pub(crate) const SOURCE_POSITIONS: [(usize, usize); 4] = [(18, 7), (5, 22), (5, 42), (18, 57)];

/// Build the common receiver aperture on the right side of the bath.
pub(crate) fn build_receiver_mask() -> Array3<bool> {
    let mut sensor_mask =
        Array3::<bool>::from_elem((GridSpec::NX, GridSpec::NY, GridSpec::NZ), false);
    let receiver_x = GridSpec::NX - 5;
    let centre = GridSpec::NZ / 2;
    for offset in 0..8_usize {
        let receiver_z = (centre as isize - 3 + offset as isize) as usize;
        sensor_mask[[receiver_x, 0, receiver_z]] = true;
    }
    sensor_mask
}

/// Build one typed FWI shot, including its provider-owned Ricker samples.
pub(crate) fn build_shot(
    source_x: usize,
    source_z: usize,
    nt: usize,
    dt: f64,
    frequency: f64,
) -> KwaversResult<FwiGeometry> {
    let mut source_mask = Array3::<f64>::zeros((GridSpec::NX, GridSpec::NY, GridSpec::NZ));
    source_mask[[source_x, 0, source_z]] = 1.0;

    let wavelet =
        DomainRickerWavelet::causal(Frequency::from_base(frequency), Pressure::from_base(P0_PA))?;
    let mut pressure_signal = Array2::<f64>::zeros((1, nt));
    for (time_index, pressure) in wavelet.samples(Time::from_base(dt), nt)?.enumerate() {
        pressure_signal[[0, time_index]] = pressure;
    }

    let mut source = GridSource::new_empty();
    source.p_mask = Some(source_mask);
    source.p_signal = Some(pressure_signal);
    source.p_mode = SourceMode::Dirichlet;

    Ok(FwiGeometry::new(source, build_receiver_mask()))
}

#[cfg(test)]
mod tests {
    use super::{build_receiver_mask, SOURCE_POSITIONS};
    use crate::config::GridSpec;

    #[test]
    fn receiver_mask_contains_exactly_eight_samples() {
        let mask = build_receiver_mask();
        assert_eq!(mask.iter().filter(|&&active| active).count(), 8);
        assert!(mask[[GridSpec::NX - 5, 0, GridSpec::NZ / 2 - 3]]);
        assert!(mask[[GridSpec::NX - 5, 0, GridSpec::NZ / 2 + 4]]);
    }

    #[test]
    fn source_aperture_is_four_position_and_in_bounds() {
        assert_eq!(SOURCE_POSITIONS.len(), 4);
        assert!(SOURCE_POSITIONS
            .iter()
            .all(|&(x, z)| { x < GridSpec::NX && z < GridSpec::NZ }));
    }
}
