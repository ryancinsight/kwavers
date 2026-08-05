use super::*;
use aequitas::systems::si::{
    quantities::{
        Dimensionless, Frequency, Intensity, Length, MassDensity, Pressure, ReciprocalLength,
        ThermodynamicTemperature, Time, Velocity,
    },
    units::{
        Hertz, Kelvin, KilogramPerCubicMeter, Meter, MeterPerSecond, Pascal, PerMeter, Second,
        WattPerSquareMeter,
    },
};
use eunomia::assert_relative_eq;
use kwavers_core::constants::fundamental::{
    DENSITY_TISSUE, SOUND_SPEED_TISSUE, SOUND_SPEED_WATER_SIM,
};
use kwavers_core::constants::numerical::MHZ_TO_HZ;
use kwavers_core::constants::thermodynamic::{
    BODY_TEMPERATURE_C, BODY_TEMPERATURE_K, KELVIN_OFFSET_C,
};
use leto::Array3;

#[test]
fn test_acoustic_heating_source() {
    let source = AcousticHeatingSource::new(
        ReciprocalLength::from_unit::<PerMeter>(0.5),
        Intensity::from_unit::<WattPerSquareMeter>(1e4),
    );
    let power = source.power().into_base();
    assert_eq!(power, 10_000.0);
}

#[test]
fn test_heating_depth_attenuation() {
    let source = AcousticHeatingSource::new(
        ReciprocalLength::from_unit::<PerMeter>(0.5),
        Intensity::from_unit::<WattPerSquareMeter>(1e4),
    );
    let power_0 = source
        .power_at_depth(Length::from_unit::<Meter>(0.0))
        .into_base();
    let power_1cm = source
        .power_at_depth(Length::from_unit::<Meter>(0.01))
        .into_base();

    assert_relative_eq!(power_0, 10_000.0, epsilon = 8.0 * f64::EPSILON);
    let expected_1cm = 10_000.0_f64 * (-0.01_f64).exp();
    // The typed expression performs two products plus exp; 16 ulps bounds the
    // first-order f64 rounding error without hiding a formula error.
    assert_relative_eq!(
        power_1cm,
        expected_1cm,
        epsilon = 16.0 * f64::EPSILON * expected_1cm.abs()
    );
    assert!(power_1cm < power_0);
}

#[test]
fn test_temperature_coefficients_soft_tissue() {
    let coeff = TemperatureCoefficients::soft_tissue();

    // Base properties at 37°C
    let c0 = Velocity::from_unit::<MeterPerSecond>(SOUND_SPEED_TISSUE);
    let rho0 = MassDensity::from_unit::<KilogramPerCubicMeter>(DENSITY_TISSUE);
    let alpha0 = ReciprocalLength::from_unit::<PerMeter>(0.5);
    let reference_temperature = ThermodynamicTemperature::from_unit::<Kelvin>(BODY_TEMPERATURE_K);
    let temperature = ThermodynamicTemperature::from_unit::<Kelvin>(40.0 + KELVIN_OFFSET_C);

    // At 40°C (3°C higher)
    let c_40 = coeff.sound_speed(c0, temperature, reference_temperature);
    let rho_40 = coeff.density(rho0, temperature, reference_temperature);
    let alpha_40 = coeff.absorption(alpha0, temperature, reference_temperature);

    // Sound speed increases
    assert!(c_40.into_base() > c0.into_base());
    // Density decreases
    assert!(rho_40.into_base() < rho0.into_base());
    // Absorption increases
    assert!(alpha_40.into_base() > alpha0.into_base());
}

#[test]
fn test_acoustic_streaming_velocity() {
    let streaming = AcousticStreaming::new(
        Intensity::from_unit::<WattPerSquareMeter>(1e3),
        Velocity::from_unit::<MeterPerSecond>(SOUND_SPEED_WATER_SIM),
        MassDensity::from_unit::<KilogramPerCubicMeter>(DENSITY_TISSUE),
    );
    let v = streaming.velocity().into_base();
    let expected_velocity = 1e3_f64 / (DENSITY_TISSUE * SOUND_SPEED_WATER_SIM.powi(2));
    assert_relative_eq!(
        v,
        expected_velocity,
        epsilon = 32.0 * f64::EPSILON * expected_velocity.abs()
    );

    let expected_power = 1e3_f64.powi(2) / (DENSITY_TISSUE * SOUND_SPEED_WATER_SIM.powi(3));
    assert_relative_eq!(
        streaming.power().into_base(),
        expected_power,
        epsilon = 32.0 * f64::EPSILON * expected_power.abs()
    );
}

#[test]
fn test_nonlinear_heating() {
    let nl = NonlinearHeating::new(
        Dimensionless::from_base(5.0),
        Pressure::from_unit::<Pascal>(1e5),
        Velocity::from_unit::<MeterPerSecond>(SOUND_SPEED_WATER_SIM),
        MassDensity::from_unit::<KilogramPerCubicMeter>(DENSITY_TISSUE),
        Frequency::from_unit::<Hertz>(MHZ_TO_HZ),
    );
    let power = nl.power_density_gradient().into_base();
    assert!(power > 0.0);

    let shock = nl.shock_parameter().into_base();
    assert!(shock > 0.0);
}

#[test]
fn test_nonlinear_regime_detection() {
    // Linear regime
    let nl_linear = NonlinearHeating::new(
        Dimensionless::from_base(5.0),
        Pressure::from_unit::<Pascal>(1e4),
        Velocity::from_unit::<MeterPerSecond>(SOUND_SPEED_WATER_SIM),
        MassDensity::from_unit::<KilogramPerCubicMeter>(DENSITY_TISSUE),
        Frequency::from_unit::<Hertz>(MHZ_TO_HZ),
    );
    assert!(!nl_linear.is_nonlinear_significant());

    // Nonlinear regime
    let nl_nonlinear = NonlinearHeating::new(
        Dimensionless::from_base(5.0),
        Pressure::from_unit::<Pascal>(5e5),
        Velocity::from_unit::<MeterPerSecond>(SOUND_SPEED_WATER_SIM),
        MassDensity::from_unit::<KilogramPerCubicMeter>(DENSITY_TISSUE),
        Frequency::from_unit::<Hertz>(MHZ_TO_HZ),
    );
    assert!(nl_nonlinear.is_nonlinear_significant());
}

#[test]
fn test_thermal_acoustic_coupling() {
    let mut coupling = ThermalAcousticCoupling::new(
        ReciprocalLength::from_unit::<PerMeter>(0.5),
        Intensity::from_unit::<WattPerSquareMeter>(1e4),
        TemperatureCoefficients::soft_tissue(),
    );
    coupling.initialize((5, 5, 5));

    let temperature = Array3::from_elem((5, 5, 5), BODY_TEMPERATURE_C);
    let intensity = Array3::from_elem((5, 5, 5), 1e4);

    coupling
        .update(
            &temperature,
            &intensity,
            ThermodynamicTemperature::from_unit::<Kelvin>(BODY_TEMPERATURE_K),
            Time::from_unit::<Second>(0.1),
        )
        .unwrap();

    let energy_density = coupling.total_energy_density().into_base();
    assert!(energy_density > 0.0);
}

#[test]
fn test_coupling_temperature_effects_on_properties() {
    let coupling = ThermalAcousticCoupling::new(
        ReciprocalLength::from_unit::<PerMeter>(0.5),
        Intensity::from_unit::<WattPerSquareMeter>(1e4),
        TemperatureCoefficients::soft_tissue(),
    );

    let c0 = Velocity::from_unit::<MeterPerSecond>(SOUND_SPEED_TISSUE);
    let hot_temperature = ThermodynamicTemperature::from_unit::<Kelvin>(45.0 + KELVIN_OFFSET_C);
    let reference_temperature = ThermodynamicTemperature::from_unit::<Kelvin>(BODY_TEMPERATURE_K);
    let c_hot = coupling.sound_speed_at_temperature(c0, hot_temperature, reference_temperature);
    let rho0 = MassDensity::from_unit::<KilogramPerCubicMeter>(DENSITY_TISSUE);
    let rho_hot = coupling.density_at_temperature(rho0, hot_temperature, reference_temperature);

    // Temperature increases both sound speed and decreases density
    assert!(c_hot.into_base() > c0.into_base());
    assert!(rho_hot.into_base() < rho0.into_base());
}

#[test]
fn test_temperature_coefficient_variants() {
    let soft = TemperatureCoefficients::soft_tissue();
    let water = TemperatureCoefficients::water();
    let blood = TemperatureCoefficients::blood();
    let bone = TemperatureCoefficients::bone();

    // Each should have different coefficients
    assert_ne!(
        soft.sound_speed_coefficient.into_base(),
        water.sound_speed_coefficient.into_base()
    );
    assert_ne!(
        blood.absorption_coefficient.into_base(),
        bone.absorption_coefficient.into_base()
    );
}

#[test]
fn test_acoustic_heating_zero_absorption() {
    let source = AcousticHeatingSource::new(
        ReciprocalLength::from_unit::<PerMeter>(0.0),
        Intensity::from_unit::<WattPerSquareMeter>(1e5),
    );
    assert_eq!(source.power().into_base(), 0.0);
}

#[test]
fn test_acoustic_heating_zero_intensity() {
    let source = AcousticHeatingSource::new(
        ReciprocalLength::from_unit::<PerMeter>(0.5),
        Intensity::from_unit::<WattPerSquareMeter>(0.0),
    );
    assert_eq!(source.power().into_base(), 0.0);
}
