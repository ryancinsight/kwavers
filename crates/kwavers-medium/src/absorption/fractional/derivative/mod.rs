//! Physical time derivatives of homogeneous SI data.
mod spectral;
mod time;

use aequitas::systems::si::{dimensions, quantities::Dimensionless};
use aequitas_runtime::{Exponent, RuntimeDimension};
use eunomia::RealField;
use kwavers_core::error::{KwaversError, KwaversResult};

/// Causal finite-history and periodic Fourier derivatives of real order α.
/// Each result carries the input dimension divided by time to the power α.
///
/// ```
/// use aequitas::systems::si::quantities::{Dimensionless, Pressure, Time};
/// use aequitas_runtime::Exponent;
/// use kwavers_medium::absorption::FractionalDerivative;
///
/// let derivative = FractionalDerivative::new(Dimensionless::from_base(0.5), 1)?;
/// let echo = Pressure::from_base(&[2.0_f64, 0.0][..]);
/// let result = derivative.apply(echo, Time::from_base(4.0))?;
/// assert_eq!(result.as_base(), &[1.0, -0.5]);
/// assert_eq!(Some(&result.dimension().axes()[2]), Exponent::ratio(-5, 2).as_ref());
/// # Ok::<(), kwavers_core::error::KwaversError>(())
/// ```
///
/// ```compile_fail,E0308
/// use aequitas::systems::si::quantities::{Dimensionless, Pressure, Time};
/// use eunomia::Complex;
/// use kwavers_medium::absorption::FractionalDerivative;
/// let d = FractionalDerivative::new(Dimensionless::from_base(0.5), 0).unwrap();
/// let bins = [Complex::new(1.0, 0.0)];
/// let _ = d.apply_frequency_domain(Pressure::from_base(&bins[..]), &[Time::from_base(1.0)]);
/// ```
#[derive(Debug)]
pub struct FractionalDerivative<T = f64> {
    order: Dimensionless<T>,
    time_power: RuntimeDimension,
    max_lag: usize,
}

impl<T: RealField> FractionalDerivative<T> {
    /// Select a derivative order and maximum number of prior samples used by
    /// the causal method. Negative order denotes fractional integration.
    ///
    /// # Errors
    /// Rejects nonfinite orders.
    pub fn new(order: Dimensionless<T>, max_lag: usize) -> KwaversResult<Self> {
        let exponent = Exponent::from_scalar(order.into_base())
            .ok_or_else(|| invalid("Fractional order must be finite"))?;
        Ok(Self {
            order,
            time_power: RuntimeDimension::of::<dimensions::Time>().power(&exponent),
            max_lag,
        })
    }
}

fn invalid(message: &'static str) -> KwaversError {
    KwaversError::InvalidInput(message.into())
}

/// Apply a positive power-law scale to a signed scalar. If materializing
/// the scale alone loses its range, combine logarithms before exponentiation.
enum Scale<T> {
    Direct(T),
    Log(T),
}

impl<T: RealField> Scale<T> {
    fn new(base: T, order: T, log_base: impl FnOnce() -> T) -> Self {
        let half = T::ONE.scale_binary(-1);
        let factor = if order == half {
            base.sqrt()
        } else if order == -half {
            T::ONE / base.sqrt()
        } else if order == T::ONE {
            base
        } else if order == -T::ONE {
            T::ONE / base
        } else {
            base.powf(order)
        };
        if base.is_finite()
            && base * T::EPSILON != T::ZERO
            && factor.is_finite()
            && factor > T::ZERO
            && factor * T::EPSILON != T::ZERO
        {
            Self::Direct(factor)
        } else {
            Self::Log(order * log_base())
        }
    }

    fn apply(&self, value: T, binary_shift: i32) -> KwaversResult<T> {
        if value == T::ZERO {
            return Ok(value);
        }
        let scaled = match *self {
            Self::Direct(factor) if binary_shift == 0 => value * factor,
            Self::Direct(factor) => {
                // Normalize separately so neither the restored input nor
                // the scaled intermediate has to fit before the last shift.
                let e = value.binary_exponent().expect("finite nonzero value");
                let f = factor.binary_exponent().expect("finite positive factor");
                (value.scale_binary(-e) * factor.scale_binary(-f))
                    .scale_binary(binary_shift + e + f)
            }
            Self::Log(log) => {
                let shift = T::from_integer(i64::from(binary_shift));
                (shift.scalar_fmadd(T::LN_2, log) + value.abs().ln())
                    .exp()
                    .copysign(value)
            }
        };
        if scaled.is_finite() {
            Ok(scaled)
        } else {
            Err(invalid("Fractional derivative exceeds the scalar range"))
        }
    }
}
