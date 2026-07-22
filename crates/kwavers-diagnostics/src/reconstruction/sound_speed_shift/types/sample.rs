//! Speed-shift acquisition samples.

use kwavers_solver::inverse::same_aperture::PlanarPoint;

/// One measured differential travel-time shift.
#[derive(Clone, Copy, Debug)]
pub struct SoundSpeedShiftSample {
    /// Transmit point in the imaging plane.
    pub transmitter: PlanarPoint,
    /// Receive point in the imaging plane.
    pub receiver: PlanarPoint,
    /// Observed minus reference travel time `s`.
    pub time_shift_s: f64,
}

impl SoundSpeedShiftSample {
    /// Construct a measured shift sample.
    #[must_use]
    pub fn new(transmitter: PlanarPoint, receiver: PlanarPoint, time_shift_s: f64) -> Self {
        Self {
            transmitter,
            receiver,
            time_shift_s,
        }
    }
}
