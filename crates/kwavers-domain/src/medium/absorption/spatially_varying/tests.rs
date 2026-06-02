use super::*;
use kwavers_core::constants::numerical::MHZ_TO_HZ;
use kwavers_core::constants::thermodynamic::BODY_TEMPERATURE_K;
use kwavers_core::error::KwaversResult;
use ndarray::Array3;

#[test]
fn test_uniform_absorption() -> KwaversResult<()> {
    let absorption = SpatiallyVaryingAbsorption::uniform(10, 10, 10, 0.5, 1.1)?;

    let alpha = absorption.absorption_at_point(5, 5, 5, MHZ_TO_HZ);
    assert!((alpha - 0.5).abs() < 1e-10);

    Ok(())
}

#[test]
fn test_frequency_dependence() -> KwaversResult<()> {
    let absorption = SpatiallyVaryingAbsorption::uniform(5, 5, 5, 1.0, 1.5)?;

    let alpha_1mhz = absorption.absorption_at_point(0, 0, 0, MHZ_TO_HZ);
    let alpha_2mhz = absorption.absorption_at_point(0, 0, 0, 2.0 * MHZ_TO_HZ);

    let expected_ratio = 2.0_f64.powf(1.5);
    let actual_ratio = alpha_2mhz / alpha_1mhz;

    assert!((actual_ratio - expected_ratio).abs() < 1e-10);

    Ok(())
}

#[test]
fn test_spherical_inclusion() -> KwaversResult<()> {
    let mut absorption = SpatiallyVaryingAbsorption::uniform(20, 20, 20, 0.5, 1.0)?;

    absorption.add_spherical_inclusion((0.5, 0.5, 0.5), 0.3, 2.0, 1.5, 0.1, 0.1, 0.1);

    let alpha_center = absorption.absorption_at_point(5, 5, 5, MHZ_TO_HZ);
    assert!((alpha_center - 2.0).abs() < 1e-10);

    let alpha_far = absorption.absorption_at_point(15, 15, 15, MHZ_TO_HZ);
    assert!((alpha_far - 0.5).abs() < 1e-10);

    Ok(())
}

#[test]
fn test_compute_absorption_field() -> KwaversResult<()> {
    let absorption = SpatiallyVaryingAbsorption::uniform(8, 8, 8, 0.75, 1.1)?;

    let field = absorption.compute_absorption_field(2.0 * MHZ_TO_HZ);

    let expected = 0.75 * 2.0_f64.powf(1.1);
    for &val in field.iter() {
        assert!((val - expected).abs() < 1e-10);
    }

    Ok(())
}

#[test]
fn test_temperature_dependence() -> KwaversResult<()> {
    let absorption = SpatiallyVaryingAbsorption::uniform(5, 5, 5, 1.0, 1.0)?;

    let mut temp_field = Array3::from_elem((5, 5, 5), BODY_TEMPERATURE_K);
    temp_field[[2, 2, 2]] = 320.15;

    let absorption = absorption.with_temperature_dependence(temp_field, 0.01)?;

    let alpha_ref = absorption.absorption_at_point(0, 0, 0, MHZ_TO_HZ);
    let alpha_hot = absorption.absorption_at_point(2, 2, 2, MHZ_TO_HZ);

    let expected_ratio = 1.1;
    let actual_ratio = alpha_hot / alpha_ref;

    assert!((actual_ratio - expected_ratio).abs() < 1e-10);

    Ok(())
}

#[test]
fn test_validation() -> KwaversResult<()> {
    let good = SpatiallyVaryingAbsorption::uniform(3, 3, 3, 0.5, 1.2)?;
    good.validate().unwrap();

    let mut bad_alpha = Array3::from_elem((3, 3, 3), 0.5);
    bad_alpha[[1, 1, 1]] = -0.1;
    let bad = SpatiallyVaryingAbsorption::new(bad_alpha, Array3::from_elem((3, 3, 3), 1.0), MHZ_TO_HZ);
    assert!(bad.is_err());

    Ok(())
}

#[test]
fn test_statistics() -> KwaversResult<()> {
    let mut absorption = SpatiallyVaryingAbsorption::uniform(10, 10, 10, 1.0, 1.2)?;
    absorption.set_region(0..5, 0..5, 0..5, 0.5, 1.0)?;
    absorption.set_region(5..10, 5..10, 5..10, 2.0, 1.5)?;

    let stats = absorption.statistics();

    assert!((stats.alpha_0_min - 0.5).abs() < 1e-10);
    assert!((stats.alpha_0_max - 2.0).abs() < 1e-10);
    assert!((stats.gamma_min - 1.0).abs() < 1e-10);
    assert!((stats.gamma_max - 1.5).abs() < 1e-10);

    Ok(())
}
