use super::{invalid, FractionalDerivative, Scale};
use aequitas::systems::si::quantities::Time;
use aequitas_runtime::RuntimeQuantity;
use eunomia::RealField;
use kwavers_core::error::KwaversResult;

impl<T: RealField> FractionalDerivative<T> {
    /// Apply the causal Grünwald–Letnikov operator with zero prehistory.
    /// At index n the result is Δt^(−α) Σ wₖ xₙ₋ₖ over retained lags,
    /// with w₀=1 and wₖ/wₖ₋₁=(k−1−α)/k. A nonnegative integer order
    /// terminates the binomial series exactly.
    ///
    /// # Errors
    /// Rejects invalid samples, a nonpositive or nonfinite timestep, or an
    /// unrepresentable coefficient, convolution or output.
    pub fn apply<'a>(
        &self,
        signal: impl Into<RuntimeQuantity<&'a [T]>>,
        timestep: Time<T>,
    ) -> KwaversResult<RuntimeQuantity<Vec<T>>> {
        let dt = timestep.into_base();
        if !dt.is_finite() || dt <= T::ZERO {
            return Err(invalid(
                "Fractional derivative timestep must be positive and finite",
            ));
        }
        let signal = signal.into();
        let values = *signal.as_base();
        if values.iter().any(|x| !x.is_finite()) {
            return Err(invalid("Fractional derivative sample must be finite"));
        }
        let dimension = signal.dimension().quotient(&self.time_power);
        let order = self.order.into_base();
        if order == T::ZERO || values.is_empty() {
            return Ok(RuntimeQuantity::from_base(values.to_vec(), dimension));
        }
        let mut weights = vec![T::ONE];
        for k in 1..=self.max_lag.min(values.len() - 1) {
            let numerator = T::from_count(k - 1) - order;
            if numerator == T::ZERO {
                break;
            }
            let next = weights[k - 1] * (numerator / T::from_count(k));
            if !next.is_finite() || next == T::ZERO {
                return Err(invalid(
                    "Fractional binomial coefficient exceeds scalar range",
                ));
            }
            weights.push(next);
        }
        let scale = Scale::new(dt, -order, || dt.ln());
        let mut result = vec![T::ZERO; values.len()];
        // Sweep adjacent outputs together so the CPU can apply each lag to
        // independent SIMD lanes. Retain ascending lag order for every sum.
        const TILE: usize = 256;
        for (block, output) in result.chunks_mut(TILE).enumerate() {
            let start = block * TILE;
            for (k, &weight) in weights.iter().take(start + output.len()).enumerate() {
                let skip = k.saturating_sub(start);
                let source = start + skip - k;
                for (sum, &x) in output[skip..].iter_mut().zip(&values[source..]) {
                    *sum = weight.scalar_fmadd(x, *sum);
                }
            }
        }
        for (i, sum) in result.iter_mut().enumerate() {
            *sum = if sum.is_finite() && sum.abs() * T::EPSILON != T::ZERO {
                scale.apply(*sum, 0)
            } else {
                // Scale all terms by the largest product exponent. The
                // normalized sum fits even when the dimensional sum does
                // not. Restore the common shift with the timestep factor.
                let terms = weights
                    .iter()
                    .zip(values[..=i].iter().rev())
                    .filter_map(|(&w, &x)| {
                        let ew = w.binary_exponent()?;
                        let ex = x.binary_exponent()?;
                        Some((w.scale_binary(-ew), x.scale_binary(-ex), ew + ex))
                    });
                let shift = terms.clone().map(|(_, _, e)| e).max().unwrap_or(0);
                let normalized = terms.fold(T::ZERO, |acc, (w, x, e)| {
                    w.scale_binary(e - shift).scalar_fmadd(x, acc)
                });
                scale.apply(normalized, shift)
            }?;
        }
        Ok(RuntimeQuantity::from_base(result, dimension))
    }
}
