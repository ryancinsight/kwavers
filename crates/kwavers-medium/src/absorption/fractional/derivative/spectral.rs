use super::{invalid, FractionalDerivative, Scale};
use aequitas::systems::si::quantities::Frequency;
use aequitas_runtime::RuntimeQuantity;
use eunomia::{Complex, RealField};
use kwavers_core::error::KwaversResult;

/// Reduce the order in quarter turns. Exact integer orders therefore give
/// exact coordinate rotations, including orders too large to multiply by π.
fn direction<T: RealField>(order: T) -> Complex<T> {
    let four = T::from_count(4);
    let magnitude = order.abs();
    let turn = magnitude - (magnitude / four).floor() * four;
    let q = turn.round();
    let delta = (turn - q) * T::FRAC_PI_2;
    let (c, s) = (delta.cos(), delta.sin());
    let (re, im) = if q == T::ONE {
        (-s, c)
    } else if q == T::from_count(2) {
        (-c, -s)
    } else if q == T::from_count(3) {
        (s, -c)
    } else {
        (c, s)
    };
    Complex::new(re, if order < T::ZERO { -im } else { im })
}

#[inline]
fn rotate<T: RealField>(sample: Complex<T>, phase: Complex<T>, f: T) -> Complex<T> {
    sample * Complex::new(phase.re, if f < T::ZERO { -phase.im } else { phase.im })
}

/// With 0 < α ≤ 1, |ω| ≤ √MAX and sample components ≤ √MAX/4,
/// even the sum of both rotated components times |ω|^α fits. The lower
/// bound keeps nonzero input components and frequencies in normal range.
/// A buffer outside this envelope goes through the scaled per-bin kernel.
fn bounded<T: RealField>(values: &[Complex<T>], frequencies: &[Frequency<T>]) -> bool {
    let root = T::max_value().sqrt();
    let low = T::ONE / root;
    let high = root / T::TAU;
    let within = |x: T| {
        let x = x.abs();
        x == T::ZERO || (low..=high).contains(&x)
    };
    values.iter().zip(frequencies).fold(true, |ok, (s, f)| {
        ok & within(s.re) & within(s.im) & within(f.into_base())
    })
}

fn regular_spectrum<T: RealField>(
    values: &[Complex<T>],
    frequencies: &[Frequency<T>],
    phase: Complex<T>,
    magnitude: impl Fn(T) -> T,
) -> Vec<Complex<T>> {
    values
        .iter()
        .zip(frequencies)
        .map(|(&s, f)| {
            let f = f.into_base();
            rotate(s, phase, f) * magnitude(T::TAU * f.abs())
        })
        .collect()
}

impl<T: RealField> FractionalDerivative<T> {
    /// Multiply signed Fourier coefficients by (i·2πf)^α on the principal
    /// branch. DC is zero for positive order and unchanged for order zero.
    /// Frequencies are supplied in the spectrum's storage order.
    ///
    /// # Errors
    /// Rejects unequal axis lengths, nonfinite values, a negative order at
    /// DC, and unrepresentable outputs. Input coefficients are borrowed.
    pub fn apply_frequency_domain<'a>(
        &self,
        spectrum: impl Into<RuntimeQuantity<&'a [Complex<T>]>>,
        frequencies: &[Frequency<T>],
    ) -> KwaversResult<RuntimeQuantity<Vec<Complex<T>>>> {
        let spectrum = spectrum.into();
        let values = *spectrum.as_base();
        if values.len() != frequencies.len() {
            return Err(invalid(
                "Fractional derivative requires one frequency per coefficient",
            ));
        }
        let order = self.order.into_base();
        let phase = direction(order);
        let dimension = spectrum.dimension().quotient(&self.time_power);
        if order > T::ZERO && order <= T::ONE && bounded(values, frequencies) {
            let result = if order == T::ONE.scale_binary(-1) {
                regular_spectrum(values, frequencies, phase, T::sqrt)
            } else if order == T::ONE {
                regular_spectrum(values, frequencies, phase, |omega| omega)
            } else {
                regular_spectrum(values, frequencies, phase, |omega| omega.powf(order))
            };
            return Ok(RuntimeQuantity::from_base(result, dimension));
        }
        let result = values
            .iter()
            .zip(frequencies)
            .map(|(&sample, &frequency)| {
                let f = frequency.into_base();
                if !f.is_finite() || !sample.re.is_finite() || !sample.im.is_finite() {
                    return Err(invalid(
                        "Fractional derivative frequencies and coefficients must be finite",
                    ));
                }
                if order == T::ZERO {
                    return Ok(sample);
                }
                if f == T::ZERO {
                    if order < T::ZERO {
                        return Err(invalid("Negative fractional order is singular at DC"));
                    }
                    return Ok(Complex::new(T::ZERO, T::ZERO));
                }
                let rotated = rotate(sample, phase, f);
                // A unit rotation can grow one component by √2. Recover
                // that component before a small multiplier brings it in range.
                let largest = sample.re.abs().max_scalar(sample.im.abs());
                let (rotated, shift) = if largest * T::EPSILON == T::ZERO {
                    // Resolve the direction before rounding either tiny
                    // component. A large multiplier can revive both.
                    let e = largest.binary_exponent().unwrap_or(0); // zero needs no shift
                    let unit = Complex::new(sample.re.scale_binary(-e), sample.im.scale_binary(-e));
                    (rotate(unit, phase, f), e)
                } else if rotated.re.is_finite() && rotated.im.is_finite() {
                    (rotated, 0)
                } else {
                    let half = Complex::new(sample.re.scale_binary(-1), sample.im.scale_binary(-1));
                    (rotate(half, phase, f), 1)
                };
                let f = f.abs();
                let scale = Scale::new(T::TAU * f, order, || T::TAU.ln() + f.ln());
                Ok(Complex::new(
                    scale.apply(rotated.re, shift)?,
                    scale.apply(rotated.im, shift)?,
                ))
            })
            .collect::<KwaversResult<Vec<_>>>()?;
        Ok(RuntimeQuantity::from_base(result, dimension))
    }
}
