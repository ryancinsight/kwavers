//! Full-ring source and receiver geometry for the planar seismic workflow.

use aequitas::systems::si::quantities::{Frequency, Pressure, Time};
use kwavers_signal::DomainRickerWavelet;
use kwavers_solver::inverse::fwi::time_domain::FwiGeometry;
use kwavers_source::{GridSource, SourceMode};
use leto::{Array2, Array3};

use super::{KwaversResult, NX, NY, NZ};

/// Centre frequency of the Ricker source wavelet [Hz].
pub(super) const F0_HZ: f64 = 150_000.0;

/// Peak source pressure [Pa].
pub(super) const P0_PA: f64 = 1.0e5;

/// Number of active elements in the FWI full-ring section.
pub(super) const FWI_ACTIVE_ELEMENTS: usize = 16;

/// Number of transmit sources on the full-ring section.
pub(super) const N_SHOTS: usize = 8;

/// Number of receivers for each shot.
pub(super) const N_RECEIVERS: usize = FWI_ACTIVE_ELEMENTS - 1;

/// Active transducer element positions sampled from a full-ring array.
pub(super) const ACTIVE_TRANSDUCER_POSITIONS: [(usize, usize); FWI_ACTIVE_ELEMENTS] = [
    (52, 32),
    (50, 40),
    (46, 46),
    (40, 50),
    (32, 52),
    (24, 50),
    (18, 46),
    (14, 40),
    (12, 32),
    (14, 24),
    (18, 18),
    (24, 14),
    (32, 12),
    (40, 14),
    (46, 18),
    (50, 24),
];

/// Transmit subset indexes into [`ACTIVE_TRANSDUCER_POSITIONS`].
pub(super) const TRANSMIT_ELEMENT_INDICES: [usize; N_SHOTS] = [0, 2, 4, 6, 8, 10, 12, 14];

/// Build the receiver mask on the same full-ring transducer section.
pub(super) fn build_receiver_mask(source_element_index: usize) -> Array3<bool> {
    let mut mask = Array3::<bool>::from_elem((NX, NY, NZ), false);
    for (element_index, &(ix, iz)) in ACTIVE_TRANSDUCER_POSITIONS.iter().enumerate() {
        if element_index != source_element_index {
            mask[[ix, 0, iz]] = true;
        }
    }
    mask
}

/// Return the transmit coordinates used by the current FWI run.
pub(super) fn transmit_positions() -> Vec<(usize, usize)> {
    TRANSMIT_ELEMENT_INDICES
        .iter()
        .map(|&idx| ACTIVE_TRANSDUCER_POSITIONS[idx])
        .collect()
}

/// Build one FWI source and receiver geometry.
pub(super) fn build_shot(
    source_element_index: usize,
    f0_hz: f64,
    nt: usize,
    dt: f64,
) -> KwaversResult<FwiGeometry> {
    let (ix, iz) = ACTIVE_TRANSDUCER_POSITIONS[source_element_index];
    let mut source_mask = Array3::<f64>::zeros((NX, NY, NZ));
    source_mask[[ix, 0, iz]] = 1.0;

    let wavelet =
        DomainRickerWavelet::causal(Frequency::from_base(f0_hz), Pressure::from_base(P0_PA))?;
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
        build_receiver_mask(source_element_index),
    ))
}

#[cfg(test)]
mod tests {
    use super::super::R_HEAD;
    use super::*;

    #[test]
    fn full_ring_transducer_section_stays_outside_skull_and_cpml() {
        let cx = NX as f64 / 2.0;
        let cz = NZ as f64 / 2.0;
        let mut has_superior = false;
        let mut has_inferior = false;
        for &(ix, iz) in &ACTIVE_TRANSDUCER_POSITIONS {
            let r = ((ix as f64 - cx).powi(2) + (iz as f64 - cz).powi(2)).sqrt();
            assert!(
                r > R_HEAD,
                "element ({ix},{iz}) must be outside skull radius {R_HEAD}, got {r}"
            );
            assert!(
                (10..54).contains(&ix) && (10..54).contains(&iz),
                "element ({ix},{iz}) must stay inside CPML-free physical domain"
            );
            has_superior |= iz < NZ / 2;
            has_inferior |= iz > NZ / 2;
        }
        assert!(
            has_superior && has_inferior,
            "full-ring section must cover both z hemispheres"
        );
    }

    #[test]
    fn receiver_mask_excludes_only_transmitting_element() {
        for &source_index in &TRANSMIT_ELEMENT_INDICES {
            let mask = build_receiver_mask(source_index);
            let active = mask.iter().filter(|&&v| v).count();
            assert_eq!(active, N_RECEIVERS);
            let (sx, sz) = ACTIVE_TRANSDUCER_POSITIONS[source_index];
            assert!(!mask[[sx, 0, sz]]);
            for (idx, &(ix, iz)) in ACTIVE_TRANSDUCER_POSITIONS.iter().enumerate() {
                assert_eq!(mask[[ix, 0, iz]], idx != source_index);
            }
        }
    }
}
