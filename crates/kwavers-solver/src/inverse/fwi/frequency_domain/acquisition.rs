//! What frequency-domain FWI needs to know about an acquisition.
//!
//! The inversion previously took `&MultiRowRingArray` in every entry point,
//! reaching through ring-specific API in all six modules. What it actually
//! needs is four questions — how many transmit events, which elements fire on
//! each, where the receivers are, and how many — and none of the answers
//! require the acquisition to be a ring.
//!
//! Making that explicit is what lets a transmission-USCT acquisition (two
//! opposed linear arrays on a rotation stage) drive the same inversion, per
//! ADR 115.
//!
//! # Receivers are indexed by transmit
//!
//! A rotation stage moves the receivers along with the sources, so receiver
//! coordinates differ from view to view. A ring array is the special case where
//! rotational symmetry makes every transmit's receiver set identical, and it
//! returns the same slice each time — that coincidence is the only reason a
//! single fixed receiver set worked before.
//!
//! The receiver *count* is constant across transmits. It is the row width of
//! every observation matrix, and a ragged acquisition would make observed and
//! predicted rows incomparable.

use kwavers_transducer::transducers::ElementPosition;

use kwavers_physics::acoustics::imaging::modalities::ultrasound::frequency_domain_fwi::{
    MultiRowRingArray, RotatingOpposedLinearArray,
};

/// The acquisition geometry a frequency-domain inversion reads.
///
/// Deliberately dyn-compatible: it is passed through
/// [`super::operator::HelmholtzForwardOperator::predict_receiver_rows`], and
/// `Config` stores that operator as `Arc<dyn HelmholtzForwardOperator>`. See
/// ADR 115 for why the seam is `&dyn` rather than a generic parameter.
pub trait TransmissionAcquisition {
    /// Number of transmit events.
    fn transmission_count(&self) -> usize;

    /// Receivers recorded per transmit.
    ///
    /// Constant across transmits by contract; see the module docs.
    fn receiver_count(&self) -> usize;

    /// Elements firing on `transmit`.
    ///
    /// # Panics
    ///
    /// May panic when `transmit >= self.transmission_count()`. Callers loop
    /// over `0..transmission_count()`, so an out-of-range index is a caller
    /// defect rather than an input to validate on every transmit of every
    /// frequency of every iteration.
    fn sources(&self, transmit: usize) -> &[ElementPosition];

    /// Receiver positions for `transmit`.
    ///
    /// # Panics
    ///
    /// Same contract as [`Self::sources`].
    fn receivers(&self, transmit: usize) -> &[ElementPosition];
}

/// A [`MultiRowRingArray`] viewed as an acquisition.
///
/// Borrows the array and precomputes the per-transmit source layout. The
/// array's own `cylindrical_source` builds a fresh `Vec` per call, and it is
/// called inside the transmit loop of a solver that runs per frequency per
/// iteration; the seam should not institutionalise an allocation the algorithm
/// never needed.
///
/// The precomputed buffer is a reordering of the array's elements, not a
/// second copy of anything larger: transmit `q` fires circumferential element
/// `q` in every row, so the layout is `rows` positions per transmit.
#[derive(Debug)]
pub struct RingAcquisition<'a> {
    array: &'a MultiRowRingArray,
    /// Flat `[transmission_count * rows]` source positions.
    sources: Vec<ElementPosition>,
    rows: usize,
}

impl<'a> RingAcquisition<'a> {
    /// View `array` as an acquisition, precomputing its transmit columns.
    #[must_use]
    pub fn new(array: &'a MultiRowRingArray) -> Self {
        let transmissions = array.circumferential_elements();
        let mut sources = Vec::with_capacity(transmissions * array.rows());
        for transmit in 0..transmissions {
            sources.extend_from_slice(&array.cylindrical_source(transmit));
        }
        Self {
            array,
            sources,
            rows: array.rows(),
        }
    }
}

impl TransmissionAcquisition for RingAcquisition<'_> {
    fn transmission_count(&self) -> usize {
        self.array.circumferential_elements()
    }

    fn receiver_count(&self) -> usize {
        self.array.element_count()
    }

    fn sources(&self, transmit: usize) -> &[ElementPosition] {
        let start = transmit * self.rows;
        &self.sources[start..start + self.rows]
    }

    fn receivers(&self, _transmit: usize) -> &[ElementPosition] {
        // Rotationally symmetric: every transmit sees the same receiver set.
        self.array.elements()
    }
}

/// A [`RotatingOpposedLinearArray`] viewed as an acquisition.
///
/// Owns the rotating array by reference and delegates to its pre-computed
/// position data (ADR 116).
#[derive(Debug)]
pub struct RotatingAcquisition<'a> {
    array: &'a RotatingOpposedLinearArray,
}

impl<'a> RotatingAcquisition<'a> {
    /// View `array` as an acquisition.
    #[must_use]
    pub fn new(array: &'a RotatingOpposedLinearArray) -> Self {
        Self { array }
    }
}

impl TransmissionAcquisition for RotatingAcquisition<'_> {
    fn transmission_count(&self) -> usize {
        self.array.transmission_count()
    }

    fn receiver_count(&self) -> usize {
        self.array.receiver_count()
    }

    fn sources(&self, transmit: usize) -> &[ElementPosition] {
        self.array.transmit_sources(transmit)
    }

    fn receivers(&self, transmit: usize) -> &[ElementPosition] {
        self.array.transmit_receivers(transmit)
    }
}
