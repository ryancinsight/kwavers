use super::{density_fraction, GasHeat, HeatGeometry, HeatPressure};
use crate::forward::bubble_dynamics::gas_transfer::DiffusiveBubbleError;
use eunomia::RealField;

impl<T: RealField> GasHeat<T> {
    /// Advance the gas heat equation in y=r/R with fixed surface temperature.
    ///
    /// The ideal gas identity is `(p/p₀) 3∫ y²/(1+q) dy = (n/n₀)(R₀/R)^3`,
    /// where q=T/T₀−1. Continuity and the gas energy balance give the relative
    /// advection speed `u/R−yU/R = −y ṗ/(3γp) + α₀ q_y/((p/p₀)R²)−yU/R`.
    /// See Prosperetti, Crum & Commander (1988), doi:10.1121/1.396198.
    ///
    /// At frozen midpoint coefficients the heat update is linear in the pressure
    /// increment. Factor the tridiagonal system once for both right-hand sides,
    /// then enforce the nonlinear endpoint inventory using a scalar Newton solve.
    /// The caller repeats this solve with updated midpoint coefficients.
    pub(in crate::forward::bubble_dynamics::gas_transfer) fn trial(
        &mut self,
        geometry: HeatGeometry<T>,
        tolerance: T,
    ) -> Result<HeatPressure<T>, DiffusiveBubbleError> {
        let half = T::from_f64(0.5);
        let two = T::from_count(2);
        let mid_pressure = T::ONE + half * (self.old_pressure + self.trial_pressure);
        let rate = (self.trial_pressure - self.old_pressure) / geometry.dt;
        let diffusion = self.diffusivity / (mid_pressure * geometry.radius * geometry.radius);
        let thermal_source = (self.gamma - T::ONE) / (self.gamma * mid_pressure);
        let intervals = self.accepted.len();
        let inverse = self.inverse_step;
        let advection = geometry.velocity / geometry.radius
            + rate / (T::from_count(3) * self.gamma * mid_pressure);
        let midpoint = |j: usize| {
            if j == intervals {
                T::ZERO
            } else {
                half * (self.accepted[j] + self.trial[j])
            }
        };
        for j in 0..intervals {
            let temperature = T::ONE + midpoint(j);
            let center = temperature * diffusion * inverse * inverse;
            let (left, right) = if j == 0 {
                // The even spherical temperature has ∇²q(0)=3q_yy(0).
                (T::ZERO, T::from_count(6) * center)
            } else {
                let y = T::from_count(j) / inverse;
                let gradient = (midpoint(j + 1) - midpoint(j - 1)) * half * inverse;
                let drift = (two * temperature * diffusion / y + y * advection
                    - diffusion * gradient)
                    * half
                    * inverse;
                (center - drift, center + drift)
            };
            let left = left * half * geometry.dt;
            let right = right * half * geometry.dt;
            let previous = j.checked_sub(1);
            let pivot = T::ONE + left + right + previous.map_or(T::ZERO, |k| left * self.upper[k]);
            if !pivot.is_finite() || pivot <= T::ZERO {
                return Err(DiffusiveBubbleError::InvalidState("gas heat pivot"));
            }
            self.upper[j] = -right / pivot;
            self.base[j] =
                (self.accepted[j] + previous.map_or(T::ZERO, |k| left * self.base[k])) / pivot;
            self.response[j] = (half * thermal_source * temperature
                + previous.map_or(T::ZERO, |k| left * self.response[k]))
                / pivot;
        }
        for j in (0..intervals - 1).rev() {
            self.base[j] = self.base[j] - self.upper[j] * self.base[j + 1];
            self.response[j] = self.response[j] - self.upper[j] * self.response[j + 1];
        }
        for j in 0..intervals {
            self.base[j] = two * self.base[j] - self.accepted[j];
            self.response[j] = two * self.response[j];
        }

        let iso = density_fraction(
            self.radius,
            geometry.endpoint_radius,
            geometry.endpoint_gas_fraction,
        );
        let increment = self.increment(iso, tolerance)?;
        let endpoint = self.old_pressure + increment;
        let mut change = (endpoint - self.trial_pressure).abs();
        for j in 0..intervals {
            let value = self.base[j] + increment * self.response[j];
            change = change.max((value - self.trial[j]).abs());
            self.trial[j] = value;
        }
        self.trial_pressure = endpoint;
        Ok(HeatPressure {
            old: self.old_pressure,
            endpoint,
            settled: change <= tolerance,
        })
    }

    fn increment(&self, iso: T, tolerance: T) -> Result<T, DiffusiveBubbleError> {
        let mut increment = self.trial_pressure - self.old_pressure;
        for _ in 0..16 {
            let pressure = T::ONE + self.old_pressure + increment;
            if !pressure.is_finite() || pressure <= T::ZERO {
                return Err(DiffusiveBubbleError::InvalidState("gas heat pressure"));
            }
            let mut integral = T::ZERO;
            let mut derivative = T::ZERO;
            for j in 0..self.accepted.len() {
                let q = self.base[j] + increment * self.response[j];
                let temperature = T::ONE + q;
                if !temperature.is_finite() || temperature <= T::ZERO {
                    return Err(DiffusiveBubbleError::InvalidState("gas temperature"));
                }
                let weight = self.weights[j] / temperature;
                integral -= weight * q;
                derivative -= weight * self.response[j] / temperature;
            }
            // Use I−1 and the pressure perturbation to preserve sub-ppm changes.
            let residual = self.old_pressure + increment - iso + pressure * integral;
            if residual.abs() <= tolerance {
                return Ok(increment);
            }
            let slope = T::ONE + integral + pressure * derivative;
            if !slope.is_finite() || slope <= T::ZERO {
                return Err(DiffusiveBubbleError::InvalidState(
                    "gas heat pressure derivative",
                ));
            }
            increment -= residual / slope;
        }
        Err(DiffusiveBubbleError::NoConvergence)
    }
}
