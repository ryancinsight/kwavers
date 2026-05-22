use crate::core::constants::fundamental::SOUND_SPEED_TISSUE;
use ndarray::Array2;

use super::super::{
    select_transmit_schedule, AnatomyKind, DeviceLayout, Point2, PreparedTheranosticSlice,
    TransmitScheduleConfig, TransmitScheduleStrategy,
};

#[test]
fn uniform_transmit_schedule_selects_equispaced_sequence() {
    let layout = fixture_layout();
    let prepared = fixture_prepared();
    let schedule = select_transmit_schedule(
        &layout,
        &prepared,
        TransmitScheduleConfig {
            strategy: TransmitScheduleStrategy::Uniform,
            budget: Some(4),
        },
    )
    .unwrap();

    assert_eq!(schedule.active_indices, vec![0, 2, 4, 6]);
    assert_eq!(schedule.effective_budget(), 4);
    assert_eq!(schedule.total_element_count, 8);
    assert!((schedule.budget_fraction() - 0.5).abs() < f64::EPSILON);
}

#[test]
fn patient_adaptive_transmit_schedule_starts_with_target_sensitive_element() {
    let layout = fixture_layout();
    let prepared = fixture_prepared();
    let schedule = select_transmit_schedule(
        &layout,
        &prepared,
        TransmitScheduleConfig {
            strategy: TransmitScheduleStrategy::PatientAdaptive,
            budget: Some(3),
        },
    )
    .unwrap();

    assert_eq!(schedule.active_indices[0], 7);
    assert_eq!(schedule.active_indices.len(), 3);
    assert_eq!(
        schedule
            .active_indices
            .iter()
            .copied()
            .collect::<std::collections::BTreeSet<_>>()
            .len(),
        3
    );
}

#[test]
fn transmit_schedule_rejects_impossible_budget() {
    let err = TransmitScheduleConfig {
        strategy: TransmitScheduleStrategy::Uniform,
        budget: Some(9),
    }
    .validate(8)
    .unwrap_err();

    assert!(format!("{err}").contains("transmit_budget"));
}

fn fixture_layout() -> DeviceLayout {
    let therapy_elements = (0..8)
        .map(|idx| Point2 {
            x_m: idx as f64 - 3.5,
            y_m: 0.0,
        })
        .collect();
    DeviceLayout {
        therapy_elements,
        imaging_receivers: Vec::new(),
        focus_m: Point2 { x_m: 3.0, y_m: 0.0 },
        skin_contact_m: Point2 { x_m: 3.5, y_m: 0.0 },
        model_name: "fixture".to_owned(),
    }
}

fn fixture_prepared() -> PreparedTheranosticSlice {
    let mut target = Array2::<bool>::from_elem((9, 9), false);
    target[[8, 4]] = true;
    PreparedTheranosticSlice {
        anatomy: AnatomyKind::Kidney,
        ct_hu: Array2::zeros((9, 9)),
        label: Array2::zeros((9, 9)),
        sound_speed_m_s: Array2::from_elem((9, 9), SOUND_SPEED_TISSUE),
        attenuation_np_per_m_mhz: Array2::from_elem((9, 9), 0.5),
        body_mask: Array2::from_elem((9, 9), true),
        organ_mask: Array2::from_elem((9, 9), true),
        target_mask: target,
        spacing_m: 1.0,
        source_slice_index: 0,
        source_dimensions: [9, 9],
        source_spacing_m: [1.0, 1.0],
        crop_bounds_index: [0, 8, 0, 8],
    }
}
