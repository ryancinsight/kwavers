//! Thermal-response quantity conversion.

use aequitas::systems::si::quantities::{
    MolarEnergy, MolarHeatCapacity, ReciprocalTime, ThermodynamicTemperature, Time,
};
use asclepius::{
    response::thermal::{ArrheniusDamage, Cem43, TemperatureSamples},
    DamageIntegral, EquivalentExposure,
};
use kwavers_core::constants::{
    fundamental::GAS_CONSTANT, numerical::SECONDS_PER_MINUTE, thermodynamic::KELVIN_OFFSET_C,
};

pub(crate) fn cem43_cumulative(temperatures_c: &[f64], step_s: f64) -> Result<Vec<f64>, String> {
    let observation = TemperatureSamples::new(
        temperatures_c
            .iter()
            .copied()
            .map(|value| ThermodynamicTemperature::from_base(value + KELVIN_OFFSET_C)),
        Time::from_base(step_s),
    )
    .map_err(|source| source.to_string())?;
    let mut exposure = vec![EquivalentExposure::zero(); temperatures_c.len()];
    Cem43::canonical()
        .cumulative_into(observation, &mut exposure)
        .map_err(|source| source.to_string())?;
    Ok(exposure
        .into_iter()
        .map(|value| value.get().into_base() / SECONDS_PER_MINUTE)
        .collect())
}

pub(crate) fn cem43_independent_exposures(
    temperatures_c: &[f64],
    duration_s: f64,
) -> Result<Vec<f64>, String> {
    let law = Cem43::canonical();
    let step = Time::from_base(duration_s);
    temperatures_c
        .iter()
        .copied()
        .map(|temperature_c| {
            law.increment(
                ThermodynamicTemperature::from_base(temperature_c + KELVIN_OFFSET_C),
                step,
            )
            .map(|exposure| exposure.get().into_base() / SECONDS_PER_MINUTE)
            .map_err(|source| source.to_string())
        })
        .collect()
}

pub(crate) fn arrhenius_damage_integral(
    temperatures_c: &[f64],
    step_s: f64,
    frequency_factor_per_s: f64,
    activation_energy_j_per_mol: f64,
) -> Result<f64, String> {
    let law = arrhenius_law(frequency_factor_per_s, activation_energy_j_per_mol)?;
    let observation = TemperatureSamples::new(
        temperatures_c
            .iter()
            .copied()
            .map(|value| ThermodynamicTemperature::from_base(value + KELVIN_OFFSET_C)),
        Time::from_base(step_s),
    )
    .map_err(|source| source.to_string())?;
    law.evaluate_uniform(observation)
        .map(DamageIntegral::get)
        .map_err(|source| source.to_string())
}

pub(crate) fn arrhenius_cumulative(
    temperatures_c: &[f64],
    step_s: f64,
    frequency_factor_per_s: f64,
    activation_energy_j_per_mol: f64,
) -> Result<Vec<f64>, String> {
    Ok(arrhenius_damage_history(
        temperatures_c,
        step_s,
        frequency_factor_per_s,
        activation_energy_j_per_mol,
    )?
    .into_iter()
    .map(DamageIntegral::get)
    .collect())
}

pub(crate) fn arrhenius_cumulative_kelvin(
    temperatures_k: &[f64],
    step_s: f64,
    frequency_factor_per_s: f64,
    activation_energy_j_per_mol: f64,
) -> Result<Vec<f64>, String> {
    Ok(arrhenius_damage_history_from(
        temperatures_k
            .iter()
            .copied()
            .map(ThermodynamicTemperature::from_base),
        temperatures_k.len(),
        step_s,
        frequency_factor_per_s,
        activation_energy_j_per_mol,
    )?
    .into_iter()
    .map(DamageIntegral::get)
    .collect())
}

pub(crate) fn arrhenius_kill_probability(
    temperatures_c: &[f64],
    step_s: f64,
    frequency_factor_per_s: f64,
    activation_energy_j_per_mol: f64,
) -> Result<Vec<f64>, String> {
    Ok(arrhenius_damage_history(
        temperatures_c,
        step_s,
        frequency_factor_per_s,
        activation_energy_j_per_mol,
    )?
    .into_iter()
    .map(|damage| ArrheniusDamage::kill_probability(damage).get())
    .collect())
}

pub(crate) fn arrhenius_steady_kill_probability(
    temperatures_c: &[f64],
    duration_s: f64,
    frequency_factor_per_s: f64,
    activation_energy_j_per_mol: f64,
) -> Result<Vec<f64>, String> {
    let law = arrhenius_law(frequency_factor_per_s, activation_energy_j_per_mol)?;
    let step = Time::from_base(duration_s);
    temperatures_c
        .iter()
        .copied()
        .map(|temperature_c| {
            law.increment(
                ThermodynamicTemperature::from_base(temperature_c + KELVIN_OFFSET_C),
                step,
            )
            .map(|damage| ArrheniusDamage::kill_probability(damage).get())
            .map_err(|source| source.to_string())
        })
        .collect()
}

fn arrhenius_damage_history(
    temperatures_c: &[f64],
    step_s: f64,
    frequency_factor_per_s: f64,
    activation_energy_j_per_mol: f64,
) -> Result<Vec<DamageIntegral<f64>>, String> {
    arrhenius_damage_history_from(
        temperatures_c
            .iter()
            .copied()
            .map(|value| ThermodynamicTemperature::from_base(value + KELVIN_OFFSET_C)),
        temperatures_c.len(),
        step_s,
        frequency_factor_per_s,
        activation_energy_j_per_mol,
    )
}

fn arrhenius_damage_history_from<I>(
    temperatures: I,
    length: usize,
    step_s: f64,
    frequency_factor_per_s: f64,
    activation_energy_j_per_mol: f64,
) -> Result<Vec<DamageIntegral<f64>>, String>
where
    I: ExactSizeIterator<Item = ThermodynamicTemperature<f64>>,
{
    let law = arrhenius_law(frequency_factor_per_s, activation_energy_j_per_mol)?;
    let observation = TemperatureSamples::new(temperatures, Time::from_base(step_s))
        .map_err(|source| source.to_string())?;
    let mut damage = vec![DamageIntegral::zero(); length];
    law.cumulative_into(observation, &mut damage)
        .map_err(|source| source.to_string())?;
    Ok(damage)
}

fn arrhenius_law(
    frequency_factor_per_s: f64,
    activation_energy_j_per_mol: f64,
) -> Result<ArrheniusDamage<f64>, String> {
    ArrheniusDamage::new(
        ReciprocalTime::from_base(frequency_factor_per_s),
        MolarEnergy::from_base(activation_energy_j_per_mol),
        MolarHeatCapacity::from_base(GAS_CONSTANT),
    )
    .map_err(|source| source.to_string())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cem43_adapters_match_canonical_reference_cases() {
        let independent = cem43_independent_exposures(&[42.0, 43.0, 44.0], 60.0)
            .expect("canonical CEM43 observations");
        assert_eq!(independent, [0.25, 1.0, 2.0]);

        let cumulative =
            cem43_cumulative(&[42.0, 43.0, 44.0], 60.0).expect("canonical CEM43 history");
        assert_eq!(cumulative, [0.25, 1.25, 3.25]);
    }

    #[test]
    fn arrhenius_adapters_preserve_steady_rate_and_cumulative_damage() {
        let temperature_k = 300.0;
        let activation_energy = GAS_CONSTANT * temperature_k;
        let cumulative = arrhenius_cumulative_kelvin(
            &[temperature_k, temperature_k],
            2.0,
            core::f64::consts::E,
            activation_energy,
        )
        .expect("valid unit-rate Arrhenius history");

        // A=e and E_a/(RT)=1 give an analytical rate of one per second.
        // Six elementary floating operations plus exp require at most this
        // conservative first-order rounding envelope at these unit-scale values.
        let bound = 16.0 * f64::EPSILON;
        assert!((cumulative[0] - 2.0).abs() <= bound);
        assert!((cumulative[1] - 4.0).abs() <= bound);
    }

    #[test]
    fn response_adapters_reject_empty_or_invalid_observations() {
        assert!(cem43_cumulative(&[], 60.0).is_err());
        assert!(cem43_cumulative(&[43.0], 0.0).is_err());
        assert!(cem43_cumulative(&[f64::NAN], 60.0).is_err());
        assert!(arrhenius_damage_integral(&[], 1.0, 1.0, 1.0).is_err());
    }
}
