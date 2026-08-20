//! Tests for the transmission-acquisition seam (ADR 115).

use kwavers_transducer::transducers::ElementPosition;

use super::super::acquisition::{RingAcquisition, TransmissionAcquisition};
use kwavers_physics::acoustics::imaging::modalities::ultrasound::frequency_domain_fwi::MultiRowRingArray;

use aequitas::systems::si::quantities::Length;
use aequitas::systems::si::units::Meter;

fn ring(circumferential: usize, rows: usize) -> MultiRowRingArray {
    MultiRowRingArray::new(
        circumferential,
        rows,
        Length::from_unit::<Meter>(0.02),
        Length::from_unit::<Meter>(0.001),
    )
    .expect("valid ring geometry")
}

/// ADR 115 promised a behaviour-preserving substitution, so the seam must
/// report exactly what the concrete array reported — not merely something
/// within a tolerance.
///
/// This is the bitwise oracle stated in the ADR, pinned at the seam boundary
/// rather than by keeping the pre-seam solver alongside the new one: a retained
/// duplicate path is the very thing the substitution exists to avoid, and the
/// solver above the seam consumes nothing but these four answers.
#[test]
fn ring_acquisition_reports_exactly_what_the_array_reports() {
    for (circumferential, rows) in [(4usize, 1usize), (8, 2), (16, 3)] {
        let array = ring(circumferential, rows);
        let acquisition = RingAcquisition::new(&array);

        assert_eq!(
            acquisition.transmission_count(),
            array.circumferential_elements(),
            "transmit count for {circumferential}x{rows}"
        );
        assert_eq!(acquisition.receiver_count(), array.element_count());

        for transmit in 0..acquisition.transmission_count() {
            assert_eq!(
                acquisition.sources(transmit),
                array.cylindrical_source(transmit).as_slice(),
                "sources for transmit {transmit} of {circumferential}x{rows}"
            );
            assert_eq!(
                acquisition.receivers(transmit),
                array.elements(),
                "receivers for transmit {transmit}"
            );
        }
    }
}

/// A ring is rotationally symmetric, so every transmit sees the same receivers.
///
/// Worth pinning because it is the property the pre-seam code silently relied
/// on when it hoisted one receiver set out of the transmit loop, and it is
/// exactly the property a rotation stage does not have.
#[test]
fn ring_receivers_do_not_depend_on_the_transmit() {
    let array = ring(8, 2);
    let acquisition = RingAcquisition::new(&array);
    let first = acquisition.receivers(0);
    for transmit in 1..acquisition.transmission_count() {
        assert_eq!(acquisition.receivers(transmit), first);
    }
}

/// A second implementor, to show the seam is general rather than a rename of
/// the ring's API.
///
/// Two opposed rows that swap roles between transmits: receivers genuinely
/// differ per transmit, which no ring array can exercise. This is the shape
/// FWI-024-D needs, without the rotation-stage geometry that is its own
/// increment.
struct OpposedPair {
    near: Vec<ElementPosition>,
    far: Vec<ElementPosition>,
}

impl TransmissionAcquisition for OpposedPair {
    fn transmission_count(&self) -> usize {
        2
    }

    fn receiver_count(&self) -> usize {
        self.far.len()
    }

    fn sources(&self, transmit: usize) -> &[ElementPosition] {
        if transmit == 0 {
            &self.near
        } else {
            &self.far
        }
    }

    fn receivers(&self, transmit: usize) -> &[ElementPosition] {
        if transmit == 0 {
            &self.far
        } else {
            &self.near
        }
    }
}

#[test]
fn an_acquisition_may_move_its_receivers_between_transmits() {
    let at = |z: f64| ElementPosition {
        x: Length::from_unit::<Meter>(0.0),
        y: Length::from_unit::<Meter>(0.0),
        z: Length::from_unit::<Meter>(z),
    };
    let near = vec![at(0.0); 3];
    let far = vec![at(0.05); 3];
    let acquisition = OpposedPair {
        near: near.clone(),
        far: far.clone(),
    };

    assert_eq!(acquisition.transmission_count(), 2);
    assert_eq!(acquisition.receiver_count(), 3);
    assert_eq!(acquisition.sources(0), near.as_slice());
    assert_eq!(acquisition.receivers(0), far.as_slice());
    // The half that distinguishes this from a ring: transmit 1 records on a
    // different set of positions than transmit 0.
    assert_eq!(acquisition.sources(1), far.as_slice());
    assert_eq!(acquisition.receivers(1), near.as_slice());
    assert_ne!(acquisition.receivers(0), acquisition.receivers(1));
}

/// The seam is used as `&dyn`, so it must stay dyn-compatible.
///
/// A generic method added later would compile everywhere except the operator
/// trait that carries it, and the failure would surface as an unrelated
/// dyn-compatibility error in `Config`. This fails at the seam instead.
#[test]
fn the_seam_is_dyn_compatible() {
    let array = ring(4, 1);
    let ring_acquisition = RingAcquisition::new(&array);
    let erased: &dyn TransmissionAcquisition = &ring_acquisition;
    assert_eq!(erased.transmission_count(), 4);
    assert_eq!(erased.sources(0), array.cylindrical_source(0).as_slice());
}
