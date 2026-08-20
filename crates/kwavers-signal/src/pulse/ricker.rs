//! Ricker wavelet implementation.

use crate::Signal;
use aequitas::systems::si::quantities::{Frequency, Pressure, Time};
use kwavers_core::error::{KwaversError, KwaversResult, ValidationError};
use std::f64::consts::PI;

const CAUSAL_PEAK_CYCLES: f64 = 1.5;

/// Ricker wavelet (Mexican hat wavelet)
///
/// Commonly used in seismic and ultrasound applications
/// Second derivative of Gaussian function
///
/// Reference: Ricker, N. (1953). "The form and laws of propagation of seismic wavelets"
#[derive(Debug, Clone)]
pub struct DomainRickerWavelet {
    peak_frequency: Frequency<f64>,
    peak_time: Time<f64>,
    amplitude: Pressure<f64>,
}

impl DomainRickerWavelet {
    /// Construct a causal Ricker wavelet with a 1.5-cycle peak delay.
    ///
    /// The delay gives the wavelet three half-cycles of build-up before its
    /// pressure maximum, matching the causal source convention used by the
    /// seismic propagators.
    ///
    /// # Errors
    ///
    /// Returns [`KwaversError::Validation`] when the peak frequency is not
    /// finite and positive or the pressure amplitude is not finite and
    /// non-negative.
    ///
    /// # Examples
    ///
    /// ```
    /// use aequitas::systems::si::quantities::{Frequency, Pressure, Time};
    /// use kwavers_signal::DomainRickerWavelet;
    ///
    /// # fn main() -> Result<(), Box<dyn std::error::Error>> {
    /// let wavelet = DomainRickerWavelet::causal(
    ///     Frequency::from_base(150_000.0),
    ///     Pressure::from_base(100_000.0),
    /// )?;
    /// let samples: Vec<_> = wavelet.samples(Time::from_base(1.0e-7), 64)?.collect();
    /// assert_eq!(samples.len(), 64);
    /// # Ok(())
    /// # }
    /// ```
    pub fn causal(peak_frequency: Frequency<f64>, amplitude: Pressure<f64>) -> KwaversResult<Self> {
        let frequency_hz = *peak_frequency.as_base();
        validate_parameter("peak_frequency", frequency_hz, true)?;
        Self::new(
            peak_frequency,
            Time::from_base(CAUSAL_PEAK_CYCLES / frequency_hz),
            amplitude,
        )
    }

    /// Construct a Ricker wavelet from SI-typed physical parameters.
    ///
    /// # Errors
    ///
    /// Returns [`KwaversError::Validation`] when the peak frequency is not
    /// finite and positive, the peak time is not finite and non-negative, or
    /// the pressure amplitude is not finite and non-negative.
    pub fn new(
        peak_frequency: Frequency<f64>,
        peak_time: Time<f64>,
        amplitude: Pressure<f64>,
    ) -> KwaversResult<Self> {
        validate_parameter("peak_frequency", *peak_frequency.as_base(), true)?;
        validate_parameter("peak_time", *peak_time.as_base(), false)?;
        validate_parameter("amplitude", *amplitude.as_base(), false)?;

        Ok(Self {
            peak_frequency,
            peak_time,
            amplitude,
        })
    }

    /// Iterate over uniformly sampled pressure amplitudes without allocating.
    ///
    /// Sample `i` is evaluated at `i * time_step` for `i` in
    /// `0..sample_count`. The returned iterator is monomorphized at the caller
    /// and owns a copy of the wavelet's three scalar quantities.
    ///
    /// # Errors
    ///
    /// Returns [`KwaversError::Validation`] when `time_step` is not finite and
    /// positive or when `sample_count` cannot be represented by the exact
    /// integer-to-`f64` conversion used for sample times.
    pub fn samples(
        &self,
        time_step: Time<f64>,
        sample_count: usize,
    ) -> KwaversResult<impl ExactSizeIterator<Item = f64>> {
        let time_step_seconds = *time_step.as_base();
        validate_parameter("time_step", time_step_seconds, true)?;
        let sample_count = u32::try_from(sample_count).map_err(|_| {
            KwaversError::Validation(ValidationError::InvalidParameter {
                parameter: "sample_count".to_owned(),
                reason: "must not exceed u32::MAX for exact sample-time conversion".to_owned(),
            })
        })?;
        let wavelet = self.clone();

        Ok((0..sample_count)
            .map(move |index| wavelet.amplitude(f64::from(index) * time_step_seconds)))
    }

    /// Compute the Ricker wavelet value
    /// r(t) = A * (1 - 2π²f²τ²) * exp(-π²f²τ²)
    /// where τ = t - `t_peak`
    fn ricker_value(&self, t: f64) -> f64 {
        let tau = t - *self.peak_time.as_base();
        let f = *self.peak_frequency.as_base();
        // Normalize before multiplying by π so a valid high-frequency wavelet
        // never evaluates the peak as infinity times zero.
        let normalized_time = tau * f;
        if !normalized_time.is_finite() {
            return 0.0;
        }
        let arg = PI * normalized_time;
        if !arg.is_finite() {
            return 0.0;
        }
        let arg_squared = arg * arg;
        if !arg_squared.is_finite() {
            return 0.0;
        }

        2.0f64.mul_add(-arg_squared, 1.0) * (-arg_squared).exp()
    }
}

impl Signal for DomainRickerWavelet {
    fn amplitude(&self, t: f64) -> f64 {
        *self.amplitude.as_base() * self.ricker_value(t)
    }

    fn frequency(&self, t: f64) -> f64 {
        // Instantaneous frequency of Ricker wavelet
        // Peak at center, decreases away from center
        let peak_frequency = *self.peak_frequency.as_base();
        let tau = (t - *self.peak_time.as_base()).abs();
        let normalized_time = tau * peak_frequency;
        if !normalized_time.is_finite() {
            return 0.0;
        }
        let arg = PI * normalized_time;
        if !arg.is_finite() {
            return 0.0;
        }
        let decay_factor = (-(arg * arg)).exp();
        peak_frequency * decay_factor
    }

    fn phase(&self, _t: f64) -> f64 {
        0.0 // Ricker wavelet is real-valued
    }

    fn duration(&self) -> Option<f64> {
        // Effective duration (99% energy)
        Some(4.0 / *self.peak_frequency.as_base())
    }

    fn clone_box(&self) -> Box<dyn Signal> {
        Box::new(self.clone())
    }
}

fn validate_parameter(
    name: &'static str,
    value: f64,
    strictly_positive: bool,
) -> KwaversResult<()> {
    let valid_sign = if strictly_positive {
        value > 0.0
    } else {
        value >= 0.0
    };
    if value.is_finite() && valid_sign {
        return Ok(());
    }

    Err(KwaversError::Validation(ValidationError::InvalidValue {
        parameter: name.to_owned(),
        value,
        reason: if strictly_positive {
            "must be finite and greater than zero".to_owned()
        } else {
            "must be finite and non-negative".to_owned()
        },
    }))
}

const _: () = assert!(std::mem::size_of::<DomainRickerWavelet>() == 3 * std::mem::size_of::<f64>());

#[cfg(test)]
mod tests {
    use super::*;

    const ROUNDING_BOUND: f64 = 16.0 * f64::EPSILON;

    #[test]
    fn sampled_wavelet_matches_ricker_reference_values() {
        let wavelet = DomainRickerWavelet::new(
            Frequency::from_base(1.0),
            Time::from_base(2.0),
            Pressure::from_base(3.0),
        )
        .expect("reference parameters are valid");

        let samples: Vec<_> = wavelet
            .samples(Time::from_base(1.0), 4)
            .expect("reference sampling is valid")
            .collect();
        let expected_at_one = -3.0 * (-PI * PI).exp() * (2.0 * PI * PI - 1.0);

        assert_eq!(samples.len(), 4);
        assert!((samples[1] - expected_at_one).abs() <= ROUNDING_BOUND * 3.0);
        assert_eq!(samples[2], 3.0);
        assert!((samples[3] - expected_at_one).abs() <= ROUNDING_BOUND * 3.0);
    }

    #[test]
    fn causal_constructor_places_peak_after_one_and_a_half_cycles() {
        let wavelet =
            DomainRickerWavelet::causal(Frequency::from_base(2.0), Pressure::from_base(4.0))
                .expect("reference parameters are valid");

        assert_eq!(wavelet.amplitude(0.75), 4.0);
    }

    #[test]
    fn invalid_physical_parameters_return_typed_errors() {
        assert_invalid_parameter(
            DomainRickerWavelet::new(
                Frequency::from_base(0.0),
                Time::from_base(0.0),
                Pressure::from_base(1.0),
            ),
            "peak_frequency",
        );
        assert_invalid_parameter(
            DomainRickerWavelet::new(
                Frequency::from_base(1.0),
                Time::from_base(-f64::EPSILON),
                Pressure::from_base(1.0),
            ),
            "peak_time",
        );
        assert_invalid_parameter(
            DomainRickerWavelet::new(
                Frequency::from_base(1.0),
                Time::from_base(0.0),
                Pressure::from_base(-f64::EPSILON),
            ),
            "amplitude",
        );

        let wavelet = DomainRickerWavelet::new(
            Frequency::from_base(1.0),
            Time::from_base(0.0),
            Pressure::from_base(1.0),
        )
        .expect("wavelet parameters are valid");
        assert_invalid_parameter(wavelet.samples(Time::from_base(f64::NAN), 1), "time_step");
    }

    #[test]
    fn extreme_finite_frequency_preserves_peak_and_decays_without_nan() {
        let wavelet = DomainRickerWavelet::new(
            Frequency::from_base(f64::MAX),
            Time::from_base(0.0),
            Pressure::from_base(1.0),
        )
        .expect("maximum finite frequency is valid");

        assert_eq!(wavelet.amplitude(0.0), 1.0);
        assert_eq!(wavelet.frequency(0.0), f64::MAX);
        assert_eq!(wavelet.amplitude(1.0), 0.0);
        assert_eq!(wavelet.frequency(1.0), 0.0);
    }

    #[test]
    fn zero_pressure_produces_zero_samples() {
        let wavelet =
            DomainRickerWavelet::causal(Frequency::from_base(2.0), Pressure::from_base(0.0))
                .expect("zero pressure is a valid boundary value");

        assert!(wavelet
            .samples(Time::from_base(0.25), 8)
            .expect("sampling parameters are valid")
            .all(|sample| sample == 0.0));
    }

    fn assert_invalid_parameter<T>(result: KwaversResult<T>, expected_parameter: &str) {
        assert!(matches!(
            result,
            Err(KwaversError::Validation(ValidationError::InvalidValue {
                parameter,
                ..
            })) if parameter == expected_parameter
        ));
    }
}
