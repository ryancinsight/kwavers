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

// ── RotatingAcquisition ────────────────────────────────────────────────────

use super::super::acquisition::RotatingAcquisition;
use kwavers_physics::acoustics::imaging::modalities::ultrasound::frequency_domain_fwi::RotatingOpposedLinearArray;
use std::f64::consts::TAU;

fn rotating(n: usize, views: usize) -> RotatingOpposedLinearArray {
    RotatingOpposedLinearArray::new(n, 1.5e-3, 0.1, views).expect("rotating array")
}

#[test]
fn rotating_acquisition_counts_match_geometry() {
    let arr = rotating(8, 180);
    let acq = RotatingAcquisition::new(&arr);
    assert_eq!(acq.transmission_count(), 8 * 180);
    assert_eq!(acq.receiver_count(), 16);
    assert_eq!(acq.sources(0).len(), 1);
    assert_eq!(acq.receivers(0).len(), 16);
}

/// Verifies the core per-transmit indexing: receivers depend on the view but
/// not on which element within that view fires, because the whole array
/// rotates together.
#[test]
fn rotating_acquisition_receivers_constant_within_view() {
    let n = 4_usize;
    let arr = rotating(n, 6);
    let acq = RotatingAcquisition::new(&arr);
    // All n transmits of view 0 should see the same receiver set.
    let ref_rx = acq.receivers(0).to_vec();
    for elem in 1..n {
        assert_eq!(acq.receivers(elem), ref_rx.as_slice());
    }
    // View 1 (transmits n..2n) should differ from view 0.
    assert_ne!(acq.receivers(n), ref_rx.as_slice());
}

/// The seam must be dyn-compatible even with the rotating acquisition.
#[test]
fn rotating_acquisition_is_dyn_compatible() {
    let arr = rotating(4, 8);
    let acq = RotatingAcquisition::new(&arr);
    let erased: &dyn TransmissionAcquisition = &acq;
    assert_eq!(erased.transmission_count(), 4 * 8);
    assert_eq!(erased.sources(0).len(), 1);
}

/// Round-trip: positions rotated by +step then back by −step reproduce the
/// originals within floating-point tolerance.  This is the acceptance oracle
/// from ADR 116.
#[test]
fn rotating_acquisition_round_trip_identity() {
    let n = 4_usize;
    let views = 8_usize;
    let arr = rotating(n, views);
    let acq = RotatingAcquisition::new(&arr);
    let step = TAU / views as f64;

    for elem in 0..n {
        let src0 = acq.sources(elem)[0]; // view 0, element `elem`
        let src1 = acq.sources(n + elem)[0]; // view 1, element `elem`
        let cos_neg = (-step).cos();
        let sin_neg = (-step).sin();
        let bx = src1.x.in_unit::<Meter>() * cos_neg - src1.y.in_unit::<Meter>() * sin_neg;
        let by = src1.x.in_unit::<Meter>() * sin_neg + src1.y.in_unit::<Meter>() * cos_neg;
        assert!(
            (bx - src0.x.in_unit::<Meter>()).abs() <= 1.0e-12,
            "elem {elem} round-trip x: {bx} vs {}",
            src0.x.in_unit::<Meter>()
        );
        assert!(
            (by - src0.y.in_unit::<Meter>()).abs() <= 1.0e-12,
            "elem {elem} round-trip y: {by} vs {}",
            src0.y.in_unit::<Meter>()
        );
    }
}
