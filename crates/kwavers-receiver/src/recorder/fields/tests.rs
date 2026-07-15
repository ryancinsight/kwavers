//! Unit tests for [`SensorRecordField`] and [`SensorRecordSpec`].

use super::{SensorRecordField, SensorRecordSpec};

#[test]
fn pressure_only_spec_has_no_velocity_or_stats() {
    let spec = SensorRecordSpec::pressure_only();
    assert!(spec.records_pressure());
    assert!(!spec.needs_any_velocity());
    assert!(!spec.needs_pressure_stats());
}

#[test]
fn velocity_time_series_detection() {
    let spec = SensorRecordSpec::from_fields(&[
        SensorRecordField::Pressure,
        SensorRecordField::VelocityX,
        SensorRecordField::VelocityY,
    ]);
    assert!(spec.records_ux());
    assert!(spec.records_uy());
    assert!(!spec.records_uz());
    assert!(spec.needs_velocity_time_series());
    assert!(!spec.needs_velocity_stats());
}

#[test]
fn stats_detection_pressure_and_velocity() {
    let spec = SensorRecordSpec::from_fields(&[
        SensorRecordField::Pressure,
        SensorRecordField::PressureMax,
        SensorRecordField::VelocityMaxX,
    ]);
    assert!(spec.needs_pressure_stats());
    assert!(spec.needs_velocity_stats());
    assert!(spec.needs_ux_stats());
    assert!(!spec.needs_uy_stats());
    assert!(!spec.needs_uz_stats());
    assert!(!spec.needs_velocity_time_series());
}

#[test]
fn kwave_name_roundtrip_for_representative_fields() {
    assert_eq!(SensorRecordField::Pressure.kwave_name(), "p");
    assert_eq!(SensorRecordField::VelocityX.kwave_name(), "ux");
    assert_eq!(
        SensorRecordField::VelocityNonStaggeredX.kwave_name(),
        "ux_non_staggered"
    );
    assert_eq!(SensorRecordField::IntensityAvgX.kwave_name(), "I_avg_x");
}

#[test]
fn field_classification_correctness() {
    assert!(SensorRecordField::VelocityX.is_velocity());
    assert!(SensorRecordField::VelocityNonStaggeredZ.is_velocity());
    assert!(!SensorRecordField::Pressure.is_velocity());
    // IntensityX requires velocity recording but is NOT classified as a velocity field
    assert!(!SensorRecordField::IntensityX.is_velocity());
}

#[test]
fn intensity_spec_requires_pressure_and_matching_velocity_component() {
    let spec = SensorRecordSpec::from_fields(&[
        SensorRecordField::IntensityX,
        SensorRecordField::IntensityAvgX,
    ]);

    assert!(spec.needs_intensity());
    assert!(spec.records_pressure());
    assert!(!spec.records_ux());
    assert!(spec.records_intensity_x());
    assert!(!spec.records_uy());
    assert!(!spec.records_uz());
    assert!(!spec.records_intensity_y());
    assert!(!spec.records_intensity_z());
    assert!(!spec.needs_velocity_stats());
    assert!(!spec.needs_pressure_stats());
    assert!(!SensorRecordField::IntensityX.needs_velocity_time_series());
    assert!(!spec.needs_velocity_time_series());
    assert!(spec.needs_any_velocity());
}
