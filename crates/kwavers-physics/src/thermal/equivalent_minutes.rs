//! Validated CEM43 equivalent-time quantities.

use aequitas::systems::si::quantities::Time;
use eunomia::FloatElement;
use kwavers_core::error::{KwaversError, KwaversResult};

/// Cumulative equivalent minutes at 43 °C (CEM43).
///
/// CEM43 is a clinical thermal-dose metric, not absorbed dose. It is stored as
/// an [`aequitas::systems::si::quantities::Time`] in base seconds so the
/// provider remains the source of dimensional arithmetic while this type keeps
/// the 43 °C equivalence semantic and the non-negative finite invariant.
#[repr(transparent)]
#[derive(Clone, Copy, Debug, PartialEq, PartialOrd)]
pub struct CumulativeEquivalentMinutes<T = f64>(Time<T>);

impl<T: FloatElement> CumulativeEquivalentMinutes<T> {
    /// Construct a zero CEM43 value.
    #[must_use]
    pub fn zero() -> Self {
        Self(Time::from_base(T::ZERO))
    }

    /// Construct CEM43 from equivalent minutes after validating its domain.
    ///
    /// # Errors
    ///
    /// Returns [`KwaversError::InvalidInput`] when `minutes` is non-finite or
    /// negative.
    pub fn try_from_minutes(minutes: T) -> KwaversResult<Self> {
        if !minutes.is_finite() || minutes < T::ZERO {
            return Err(KwaversError::InvalidInput(format!(
                "CEM43 equivalent minutes must be finite and non-negative, got {minutes:?}"
            )));
        }
        Self::try_from_time(Time::from_base(minutes * T::from_f64(60.0)))
    }

    /// Construct CEM43 from an equivalent-time value in base seconds.
    ///
    /// # Errors
    ///
    /// Returns [`KwaversError::InvalidInput`] when `time` is non-finite or
    /// negative.
    pub fn try_from_time(time: Time<T>) -> KwaversResult<Self> {
        let seconds = time.into_base();
        if !seconds.is_finite() || seconds < T::ZERO {
            return Err(KwaversError::InvalidInput(format!(
                "CEM43 equivalent time must be finite and non-negative, got {seconds:?} s"
            )));
        }
        Ok(Self(Time::from_base(seconds)))
    }

    /// Return the underlying Aequitas time in base seconds.
    #[must_use]
    pub fn as_time(self) -> Time<T> {
        self.0
    }

    /// Return the value in equivalent minutes.
    #[must_use]
    pub fn as_minutes(self) -> T {
        self.0.into_base() / T::from_f64(60.0)
    }
}

#[cfg(test)]
mod tests {
    use super::CumulativeEquivalentMinutes;

    #[test]
    fn round_trips_minutes_through_aequitas_time() {
        let value = CumulativeEquivalentMinutes::try_from_minutes(2.5).unwrap();

        assert_eq!(value.as_time().into_base(), 150.0);
        assert_eq!(value.as_minutes(), 2.5);
    }

    #[test]
    fn rejects_invalid_equivalent_time() {
        assert!(CumulativeEquivalentMinutes::try_from_minutes(-1.0).is_err());
        assert!(CumulativeEquivalentMinutes::try_from_minutes(f64::NAN).is_err());
        assert!(CumulativeEquivalentMinutes::try_from_minutes(f64::INFINITY).is_err());
    }
}
