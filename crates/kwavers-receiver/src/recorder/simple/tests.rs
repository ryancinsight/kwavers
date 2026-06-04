use super::SensorRecorder;
use crate::recorder::config::RecordingMode;
use ndarray::{Array1, Array3};

#[test]
fn pressure_view_exposes_recorded_prefix_without_clone() {
    let mut mask = Array3::from_elem((3, 1, 1), false);
    mask[[1, 0, 0]] = true;
    let mut recorder = SensorRecorder::new(Some(&mask), (3, 1, 1), 2).unwrap();

    let mut p0 = Array3::zeros((3, 1, 1));
    p0[[1, 0, 0]] = 7.0;
    recorder.record_step(&p0).unwrap();

    let full = recorder.pressure_data_view().unwrap();
    assert_eq!(full.dim(), (1, 2));
    assert_eq!(full[[0, 0]], 7.0);
    assert_eq!(full[[0, 1]], 0.0);

    let recorded = recorder.recorded_pressure_view().unwrap();
    assert_eq!(recorded.dim(), (1, 1));
    assert_eq!(recorded[[0, 0]], 7.0);
}

#[test]
fn checkpoint_view_matches_owned_checkpoint_state() {
    let mut mask = Array3::from_elem((3, 1, 1), false);
    mask[[1, 0, 0]] = true;
    let mut recorder = SensorRecorder::new(Some(&mask), (3, 1, 1), 2).unwrap();

    let mut p0 = Array3::zeros((3, 1, 1));
    p0[[1, 0, 0]] = 13.0;
    recorder.record_step(&p0).unwrap();

    let (view, view_next, view_expected) = recorder.checkpoint_state_view().unwrap();
    let (owned, owned_next, owned_expected) = recorder.checkpoint_state().unwrap();

    assert_eq!(view_next, owned_next);
    assert_eq!(view_expected, owned_expected);
    assert_eq!(view.dim(), owned.dim());
    assert_eq!(view[[0, 0]], owned[[0, 0]]);
}

#[test]
fn pressure_stat_fill_reuses_caller_storage() {
    let mut mask = Array3::from_elem((3, 1, 1), false);
    mask[[0, 0, 0]] = true;
    mask[[2, 0, 0]] = true;
    let modes = [RecordingMode::AllStatistics];
    let mut recorder = SensorRecorder::with_modes(Some(&mask), (3, 1, 1), 2, &modes).unwrap();

    let p0 = Array3::from_shape_vec((3, 1, 1), vec![1.0, 9.0, -2.0]).unwrap();
    let p1 = Array3::from_shape_vec((3, 1, 1), vec![4.0, 8.0, 6.0]).unwrap();
    recorder.record_step(&p0).unwrap();
    recorder.record_step(&p1).unwrap();

    let mut out = Array1::zeros(2);
    recorder.fill_p_max(&mut out).unwrap();
    assert_eq!(out.as_slice().unwrap(), &[4.0, 6.0]);

    recorder.fill_p_min(&mut out).unwrap();
    assert_eq!(out.as_slice().unwrap(), &[1.0, -2.0]);

    recorder.fill_p_final(&mut out).unwrap();
    assert_eq!(out.as_slice().unwrap(), &[4.0, 6.0]);

    recorder.fill_p_rms(&mut out).unwrap();
    let expected_0 = ((1.0_f64.powi(2) + 4.0_f64.powi(2)) / 2.0).sqrt();
    let expected_1 = ((-2.0_f64).powi(2) + 6.0_f64.powi(2)).sqrt() / 2.0_f64.sqrt();
    assert!((out[0] - expected_0).abs() < 1e-14);
    assert!((out[1] - expected_1).abs() < 1e-14);

    let mut wrong_len = Array1::zeros(1);
    assert!(recorder.fill_p_max(&mut wrong_len).is_err());
}

#[test]
fn pressure_stat_fill_errors_when_stats_not_requested() {
    let mut mask = Array3::from_elem((1, 1, 1), false);
    mask[[0, 0, 0]] = true;
    let recorder = SensorRecorder::new(Some(&mask), (1, 1, 1), 1).unwrap();

    let mut out = Array1::zeros(1);
    assert!(recorder.fill_p_max(&mut out).is_err());
    assert_eq!(recorder.extract_p_max(), None);
}
