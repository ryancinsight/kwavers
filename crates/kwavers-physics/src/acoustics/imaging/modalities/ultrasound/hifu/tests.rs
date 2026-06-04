use super::*;
use kwavers_core::constants::fundamental::{DENSITY_WATER_NOMINAL, SOUND_SPEED_WATER_SIM};
use kwavers_core::constants::numerical::MHZ_TO_HZ;
use kwavers_grid::Grid;
use kwavers_domain::imaging::ultrasound::hifu::{
    DomainHIFUTransducer, DomainHIFUTreatmentPlan, HifuTargetShape, HifuTreatmentProtocol,
    TreatmentPhase, TreatmentTarget,
};
use kwavers_medium::homogeneous::HomogeneousMedium;
use ndarray::Array3;

#[test]
fn hifu_pressure_field_is_centered_at_geometric_focus_depth() -> kwavers_core::error::KwaversResult<()> {
    let grid = Grid::new(9, 9, 17, 0.001, 0.001, 0.001)?;
    let medium = HomogeneousMedium::new(
        DENSITY_WATER_NOMINAL,
        SOUND_SPEED_WATER_SIM,
        0.5,
        1.0,
        &grid,
    );
    let transducer = DomainHIFUTransducer::new_single_element(MHZ_TO_HZ, 50.0, 0.010, 0.004);

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
    let transducer = DomainHIFUTransducer::new_single_element(MHZ_TO_HZ, 25.0, 0.008, 0.004);

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
fn hifu_intensity_uses_peak_pressure_half_impedance_formula() -> kwavers_core::error::KwaversResult<()> {
    let grid = Grid::new(5, 5, 9, 0.001, 0.001, 0.001)?;
    let medium = HomogeneousMedium::new(
        DENSITY_WATER_NOMINAL,
        SOUND_SPEED_WATER_SIM,
        0.5,
        1.0,
        &grid,
    );
    let transducer = DomainHIFUTransducer::new_single_element(MHZ_TO_HZ, 10.0, 0.006, 0.003);

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
    assert!((thermal_dose::cem43_increment_minutes(43.0, 1.0) - 1.0).abs() < 1.0e-12);
    assert!((thermal_dose::cem43_increment_minutes(44.0, 1.0) - 2.0).abs() < 1.0e-12);
    assert!((thermal_dose::cem43_increment_minutes(42.0, 1.0) - 0.25).abs() < 1.0e-12);
}

#[test]
fn thermal_dose_uses_seconds_and_detects_ablation_threshold() {
    let grid = Grid::new(4, 4, 4, 0.005, 0.005, 0.005).unwrap();
    let mut thermal_dose = HifuThermalDose::new(&grid);

    thermal_dose.add_temperature_measurement(Array3::from_elem(grid.dimensions(), 55.0), 0.0);
    thermal_dose.add_temperature_measurement(Array3::from_elem(grid.dimensions(), 55.0), 60.0);

    let dose_center = thermal_dose.dose_at(2, 2, 2);
    let expected = 4096.0;
    assert!(
        (dose_center - expected).abs() < 1.0e-9,
        "one minute at 55 deg C must contribute 4096 CEM43 minutes, got {dose_center:e}"
    );
    assert!(thermal_dose.ablation_threshold_reached()[[2, 2, 2]]);
}

#[test]
fn treatment_plan_validation_accepts_target_inside_focal_access_region() -> kwavers_core::error::KwaversResult<()>
{
    let target = TreatmentTarget {
        center: [0.0, 0.0, 0.08],
        dimensions: [0.01, 0.01, 0.01],
        shape: HifuTargetShape::Sphere,
    };
    let protocol = HifuTreatmentProtocol {
        total_duration: 30.0,
        pulse_duration: 5.0,
        prf: 1.0,
        cooling_period: 10.0,
        phases: vec![TreatmentPhase {
            name: "Heating".to_string(),
            duration: 20.0,
            power: 50.0,
            focus_offset: [0.0, 0.0, 0.0],
        }],
    };

    let plan = DomainHIFUTreatmentPlan::new(target, protocol);
    let transducer = DomainHIFUTransducer::new_single_element(MHZ_TO_HZ, 50.0, 0.08, 0.04);

    plan.validate(&transducer)?;
    Ok(())
}
