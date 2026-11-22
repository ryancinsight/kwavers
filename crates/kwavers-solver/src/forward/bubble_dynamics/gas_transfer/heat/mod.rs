use super::{DiffusiveBubbleError, GasHeatTransfer};
use eunomia::RealField;

mod midpoint;

/// Trial wall geometry and inventory for one temperature step.
pub(super) struct HeatGeometry<T> {
    pub(super) dt: T,
    pub(super) radius: T,
    pub(super) velocity: T,
    pub(super) endpoint_radius: T,
    pub(super) endpoint_gas_fraction: T,
}

/// Dimensionless gas pressure increments relative to the initial dry pressure.
pub(super) struct HeatPressure<T> {
    pub(super) old: T,
    pub(super) endpoint: T,
    pub(super) settled: bool,
}

pub(super) struct GasHeat<T> {
    gamma: T,
    diffusivity: T,
    radius: T,
    inverse_step: T,
    weights: Vec<T>,
    accepted: Vec<T>,
    trial: Vec<T>,
    upper: Vec<T>,
    base: Vec<T>,
    response: Vec<T>,
    old_pressure: T,
    trial_pressure: T,
}

impl<T: RealField> GasHeat<T> {
    pub(super) fn new(config: GasHeatTransfer<T>, radius: T) -> Result<Self, DiffusiveBubbleError> {
        let gamma = config.heat_capacity_ratio;
        let diffusivity = config.reference_diffusivity.into_base();
        let count = config.radial_intervals;
        if !gamma.is_finite()
            || gamma <= T::ONE
            || !diffusivity.is_finite()
            || diffusivity <= T::ZERO
            || count < 4
            || !count.is_multiple_of(2)
        {
            return Err(DiffusiveBubbleError::InvalidState("gas heat parameters"));
        }
        let inverse_step = T::from_count(count);
        if !inverse_step.is_finite() {
            return Err(DiffusiveBubbleError::InvalidState("gas heat grid"));
        }
        let weights = (0..count)
            .map(|j| {
                let y = T::from_count(j) / inverse_step;
                // Simpson weights for 3 times the spherical volume integral.
                T::from_count(if j.is_multiple_of(2) { 2 } else { 4 }) * y * y / inverse_step
            })
            .collect();
        Ok(Self {
            gamma,
            diffusivity,
            radius,
            inverse_step,
            weights,
            accepted: vec![T::ZERO; count],
            trial: vec![T::ZERO; count],
            upper: vec![T::ZERO; count],
            base: vec![T::ZERO; count],
            response: vec![T::ZERO; count],
            old_pressure: T::ZERO,
            trial_pressure: T::ZERO,
        })
    }

    pub(super) fn pressure_fraction(&self, radius: T, gas_fraction: T) -> T {
        let integral = self
            .weights
            .iter()
            .zip(&self.accepted)
            .fold(T::ZERO, |sum, (&w, &q)| sum - w * q / (T::ONE + q));
        (density_fraction(self.radius, radius, gas_fraction) - integral) / (T::ONE + integral)
    }

    pub(super) fn begin(&mut self, radius: T, gas_fraction: T) {
        self.trial.copy_from_slice(&self.accepted);
        self.old_pressure = self.pressure_fraction(radius, gas_fraction);
        self.trial_pressure = self.old_pressure;
    }

    pub(super) fn commit(&mut self) {
        std::mem::swap(&mut self.accepted, &mut self.trial);
    }
}

fn density_fraction<T: RealField>(reference: T, radius: T, gas_fraction: T) -> T {
    let relative_gap = (reference - radius) / radius;
    let three = T::from_count(3);
    let volume = relative_gap * (three + relative_gap * (three + relative_gap));
    volume + gas_fraction * (T::ONE + volume)
}
