use super::ElasticPropertyData;
use kwavers_core::constants::fundamental::{DENSITY_WATER_NOMINAL, SOUND_SPEED_WATER_SIM};

#[test]
fn test_elastic_engineering_conversion() {
    let density = 7850.0;
    let youngs = 200e9;
    let poisson = 0.3;

    let elastic = ElasticPropertyData::from_engineering(density, youngs, poisson);

    assert!((elastic.youngs_modulus() - youngs).abs() / youngs < 1e-10);
    assert!((elastic.poisson_ratio() - poisson).abs() < 1e-10);
}

#[test]
fn test_elastic_wave_speeds() {
    let steel = ElasticPropertyData::steel();

    let cp = steel.p_wave_speed();
    assert!(
        cp > 5000.0 && cp < 7000.0,
        "P-wave speed {} out of expected range",
        cp
    );

    let cs = steel.s_wave_speed();
    assert!(cs < cp);
    assert!(
        cs > 2500.0 && cs < 4000.0,
        "S-wave speed {} out of expected range",
        cs
    );
}

#[test]
fn test_elastic_poisson_bounds() {
    let density = DENSITY_WATER_NOMINAL;

    assert!(ElasticPropertyData::try_from_engineering(density, 1e9, 0.5).is_err());
    assert!(ElasticPropertyData::try_from_engineering(density, 1e9, -1.0).is_err());

    let ep = ElasticPropertyData::try_from_engineering(density, 1e9, 0.3).unwrap();
    assert!(ep.density > 0.0);
}

#[test]
fn test_elastic_moduli_relations() {
    let elastic = ElasticPropertyData::from_engineering(7850.0, 200e9, 0.3);

    let e = elastic.youngs_modulus();
    let nu = elastic.poisson_ratio();
    let k = elastic.bulk_modulus();
    let mu = elastic.shear_modulus();

    let k_expected = e / (3.0 * (1.0 - 2.0 * nu));
    assert!((k - k_expected).abs() / k < 1e-10);

    let mu_expected = e / (2.0 * (1.0 + nu));
    assert!((mu - mu_expected).abs() / mu < 1e-10);
}

#[test]
fn test_elastic_from_wave_speeds() {
    let density = 7850.0;
    let cp = 5960.0;
    let cs = 3220.0;

    let elastic = ElasticPropertyData::from_wave_speeds(density, cp, cs);

    assert!((elastic.p_wave_speed() - cp).abs() < 1e-6);
    assert!((elastic.s_wave_speed() - cs).abs() < 1e-6);
    assert_eq!(elastic.density, density);
    assert!(elastic.lambda > 0.0);
    assert!(elastic.mu > 0.0);
}

#[test]
fn test_elastic_from_wave_speeds_validation() {
    let density = DENSITY_WATER_NOMINAL;

    assert!(
        ElasticPropertyData::try_from_wave_speeds(density, SOUND_SPEED_WATER_SIM, 1600.0).is_err()
    );
    assert!(
        ElasticPropertyData::try_from_wave_speeds(-1000.0, SOUND_SPEED_WATER_SIM, 1000.0).is_err()
    );
    assert!(
        ElasticPropertyData::try_from_wave_speeds(density, -SOUND_SPEED_WATER_SIM, 1000.0).is_err()
    );
    assert!(
        ElasticPropertyData::try_from_wave_speeds(density, SOUND_SPEED_WATER_SIM, -1000.0).is_err()
    );

    let ep =
        ElasticPropertyData::try_from_wave_speeds(density, SOUND_SPEED_WATER_SIM, 1000.0).unwrap();
    assert!(ep.density > 0.0);
}

#[test]
fn test_elastic_wave_speed_round_trip() {
    let original = ElasticPropertyData::from_engineering(2700.0, 69e9, 0.33);

    let cp = original.p_wave_speed();
    let cs = original.s_wave_speed();

    let reconstructed = ElasticPropertyData::from_wave_speeds(original.density, cp, cs);

    assert!((reconstructed.lambda - original.lambda).abs() / original.lambda < 1e-10);
    assert!((reconstructed.mu - original.mu).abs() / original.mu < 1e-10);
    assert!(
        (reconstructed.youngs_modulus() - original.youngs_modulus()).abs()
            / original.youngs_modulus()
            < 1e-10
    );
    assert!((reconstructed.poisson_ratio() - original.poisson_ratio()).abs() < 1e-10);
}
