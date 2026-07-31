//! Speed-shift acquisition samples.

use aequitas::systems::si::quantities::Time;
use kwavers_solver::inverse::same_aperture::PlanarPoint;

/// One measured differential travel-time shift.
#[derive(Clone, Copy, Debug)]
pub struct SoundSpeedShiftSample {
    /// Transmit point in the imaging plane.
    pub transmitter: PlanarPoint,
    /// Receive point in the imaging plane.
    pub receiver: PlanarPoint,
    /// Observed minus reference travel time.
    pub time_shift: Time,
}

impl SoundSpeedShiftSample {
    /// Construct a measured shift sample.
    #[must_use]
    pub fn new(transmitter: PlanarPoint, receiver: PlanarPoint, time_shift: Time) -> Self {
        Self {
            transmitter,
            receiver,
            time_shift,
        }
    }
}
