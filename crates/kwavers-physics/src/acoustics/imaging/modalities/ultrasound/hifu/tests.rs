use super::*;
use aequitas::systems::si::quantities::{
    Frequency, Intensity, Length, Power, ThermodynamicTemperature, Time,
};
use kwavers_core::constants::fundamental::{DENSITY_WATER_NOMINAL, SOUND_SPEED_WATER_SIM};
use kwavers_core::constants::numerical::MHZ_TO_HZ;
use kwavers_core::error::{KwaversError, ValidationError};
use kwavers_grid::Grid;
use kwavers_imaging::ultrasound::hifu::{
    DomainHIFUTransducer, DomainHIFUTreatmentPlan, HifuTargetShape, HifuTreatmentProtocol,
    TreatmentPhase, TreatmentTarget,
};
use kwavers_medium::homogeneous::HomogeneousMedium;
use leto::Array3;

#[test]
fn hifu_pressure_field_is_centered_at_geometric_focus_depth(
) -> kwavers_core::error::KwaversResult<()> {
    let grid = Grid::new(9, 9, 17, 0.001, 0.001, 0.001)?;
    let medium = HomogeneousMedium::new(
        DENSITY_WATER_NOMINAL,
        SOUND_SPEED_WATER_SIM,
        0.5,
        1.0,
        &grid,
    );
    let transducer = DomainHIFUTransducer::new_single_element(
        Frequency::from_base(MHZ_TO_HZ),
        Power::from_base(50.0),
        Length::from_base(0.010),
        Length::from_base(0.004),
    );

    let pressure = compute_pressure_field(&transducer, &grid, &medium)?;
    let center = (grid.nx / 2, grid.ny / 2, 10);
    let focus = pressure[[center.0, center.1, center.2]];
    let lateral = pressure[[center.0 + 3, center.1, center.2]];
    let corner = pressure[[0, 0, center.2]];

    assert!(
        focus > lateral,
        "focused Rayleigh-Sommerfeld field must exceed lateral same-depth value: focus={focus:e}, lateral={lateral:e}"
    );
    assert!(
        focus > corner,
        "focused Rayleigh-Sommerfeld field must be centered laterally, not pinned to grid corner: focus={focus:e}, corner={corner:e}"
    );
    Ok(())
}

#[test]
fn hifu_pressure_field_is_laterally_symmetric() -> kwavers_core::error::KwaversResult<()> {
    let grid = Grid::new(9, 9, 13, 0.001, 0.001, 0.001)?;
    let medium = HomogeneousMedium::new(
        DENSITY_WATER_NOMINAL,
        SOUND_SPEED_WATER_SIM,
        0.5,
        1.0,
        &grid,
    );
    let transducer = DomainHIFUTransducer::new_single_element(
        Frequency::from_base(MHZ_TO_HZ),
        Power::from_base(25.0),
        Length::from_base(0.008),
        Length::from_base(0.004),
    );

    let pressure = compute_pressure_field(&transducer, &grid, &medium)?;
    let left = pressure[[2, 4, 8]];
    let right = pressure[[6, 4, 8]];

    assert!(
        (left - right).abs() < 1.0e-8 * left.max(right).max(1.0),
        "centered aperture must produce symmetric lateral pressure: left={left:e}, right={right:e}"
    );
    Ok(())
}

#[test]
fn hifu_intensity_uses_peak_pressure_half_impedance_formula(
) -> kwavers_core::error::KwaversResult<()> {
    let grid = Grid::new(5, 5, 9, 0.001, 0.001, 0.001)?;
    let medium = HomogeneousMedium::new(
        DENSITY_WATER_NOMINAL,
        SOUND_SPEED_WATER_SIM,
        0.5,
        1.0,
        &grid,
    );
    let transducer = DomainHIFUTransducer::new_single_element(
        Frequency::from_base(MHZ_TO_HZ),
        Power::from_base(10.0),
        Length::from_base(0.006),
        Length::from_base(0.003),
    );

    let pressure = compute_pressure_field(&transducer, &grid, &medium)?;
    let intensity = compute_intensity_field(&transducer, &grid, &medium)?;
    let p = pressure[[2, 2, 6]];
    let expected = p * p / (2.0 * DENSITY_WATER_NOMINAL * SOUND_SPEED_WATER_SIM);

    assert!(
        (intensity[[2, 2, 6]] - expected).abs() < expected * 1.0e-12,
        "HIFU intensity must be p_peak^2/(2 rho c)"
    );
    Ok(())
}

#[test]
fn cem43_reference_temperatures_match_sapareto_dewey() {
    let grid = Grid::new(3, 1, 1, 0.005, 0.005, 0.005).unwrap();
    let mut thermal_dose = HifuThermalDose::new(&grid);
    let temperatures = Array3::from_shape_vec(grid.dimensions(), vec![42.0, 43.0, 44.0])
        .expect("test shape matches values");
    thermal_dose
        .add_temperature_measurement(temperatures.clone(), Time::from_base(0.0))
        .unwrap();
    thermal_dose
        .add_temperature_measurement(temperatures, Time::from_base(60.0))
        .unwrap();
    assert_eq!(thermal_dose.dose_at(0, 0, 0).unwrap().as_minutes(), 0.25);
    assert_eq!(thermal_dose.dose_at(1, 0, 0).unwrap().as_minutes(), 1.0);
    assert_eq!(thermal_dose.dose_at(2, 0, 0).unwrap().as_minutes(), 2.0);
}

#[test]
fn thermal_dose_uses_seconds_and_detects_ablation_threshold() {
    let grid = Grid::new(4, 4, 4, 0.005, 0.005, 0.005).unwrap();
    let mut thermal_dose = HifuThermalDose::new(&grid);

    thermal_dose
        .add_temperature_measurement(
            Array3::from_elem(grid.dimensions(), 55.0),
            Time::from_base(0.0),
        )
        .unwrap();
    thermal_dose
        .add_temperature_measurement(
            Array3::from_elem(grid.dimensions(), 55.0),
            Time::from_base(60.0),
        )
        .unwrap();

    let dose_center = thermal_dose.dose_at(2, 2, 2).unwrap().as_minutes();
    let expected = 4096.0;
    assert!(
        (dose_center - expected).abs() < 1.0e-9,
        "one minute at 55 deg C must contribute 4096 CEM43 minutes, got {dose_center:e}"
    );
    assert!(thermal_dose.ablation_threshold_reached()[[2, 2, 2]]);
}

#[test]
fn treatment_plan_validation_accepts_target_inside_focal_access_region(
) -> kwavers_core::error::KwaversResult<()> {
    let target = TreatmentTarget {
        center: [
            Length::from_base(0.0),
            Length::from_base(0.0),
            Length::from_base(0.08),
        ],
        dimensions: [
            Length::from_base(0.01),
            Length::from_base(0.01),
            Length::from_base(0.01),
        ],
        shape: HifuTargetShape::Sphere,
    };
    let protocol = HifuTreatmentProtocol {
        total_duration: Time::from_base(30.0),
        pulse_duration: Time::from_base(5.0),
        prf: Frequency::from_base(1.0),
        cooling_period: Time::from_base(10.0),
        phases: vec![TreatmentPhase {
            name: "Heating".to_string(),
            duration: Time::from_base(20.0),
            power: Power::from_base(50.0),
            focus_offset: [
                Length::from_base(0.0),
                Length::from_base(0.0),
                Length::from_base(0.0),
            ],
        }],
    };

    let plan = DomainHIFUTreatmentPlan::new(target, protocol);
    let transducer = DomainHIFUTransducer::new_single_element(
        Frequency::from_base(MHZ_TO_HZ),
        Power::from_base(50.0),
        Length::from_base(0.08),
        Length::from_base(0.04),
    );

    plan.validate(&transducer)?;
    Ok(())
}

#[test]
fn treatment_plan_validation_applies_si_temperature_and_intensity_limits(
) -> kwavers_core::error::KwaversResult<()> {
    let target = TreatmentTarget {
        center: [
            Length::from_base(0.0),
            Length::from_base(0.0),
            Length::from_base(0.08),
        ],
        dimensions: [
            Length::from_base(0.01),
            Length::from_base(0.01),
            Length::from_base(0.01),
        ],
        shape: HifuTargetShape::Sphere,
    };
    let protocol = HifuTreatmentProtocol {
        total_duration: Time::from_base(30.0),
        pulse_duration: Time::from_base(5.0),
        prf: Frequency::from_base(1.0),
        cooling_period: Time::from_base(10.0),
        phases: Vec::new(),
    };
    let transducer = DomainHIFUTransducer::new_single_element(
        Frequency::from_base(MHZ_TO_HZ),
        Power::from_base(50.0),
        Length::from_base(0.08),
        Length::from_base(0.04),
    );

    let mut temperature_plan = DomainHIFUTreatmentPlan::new(target.clone(), protocol.clone());
    temperature_plan.safety.max_temperature = ThermodynamicTemperature::from_base(373.16);
    let temperature_error = temperature_plan
        .validate(&transducer)
        .expect_err("temperature above the SI safety limit must be rejected");
    match temperature_error {
        KwaversError::Validation(ValidationError::InvalidValue {
            parameter, value, ..
        }) => {
            assert_eq!(parameter, "safety.max_temperature");
            assert_eq!(value, 373.16);
        }
        other => panic!("unexpected validation error: {other:?}"),
    }

    let mut intensity_plan = DomainHIFUTreatmentPlan::new(target, protocol);
    intensity_plan.safety.max_intensity = Intensity::from_base(1.0e7 + 1.0);
    let intensity_error = intensity_plan
        .validate(&transducer)
        .expect_err("intensity above the SI safety limit must be rejected");
    match intensity_error {
        KwaversError::Validation(ValidationError::InvalidValue {
            parameter, value, ..
        }) => {
            assert_eq!(parameter, "safety.max_intensity");
            assert_eq!(value, 1.0e7 + 1.0);
        }
        other => panic!("unexpected validation error: {other:?}"),
    }
    Ok(())
}
