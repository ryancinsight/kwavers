use super::ivus_chapter_metrics;
use super::IvusChapterMetricsInput;

#[test]
fn ivus_chapter_metrics_match_masked_means_and_areas() {
    let x = [-1.0e-3, -1.0e-3, 1.0e-3, 1.0e-3];
    let y = [-1.0e-3, 1.0e-3, -1.0e-3, 1.0e-3];
    let lumen = [true, false, false, false];
    let eel = [true, true, true, false];
    let plaque = [false, true, false, false];
    let bmode = [0.20, 0.70, 0.50, 0.10];

    let metrics = ivus_chapter_metrics(IvusChapterMetricsInput {
        x_m: &x,
        y_m: &y,
        lumen_mask: &lumen,
        eel_mask: &eel,
        plaque_mask: &plaque,
        bmode_cartesian: &bmode,
        sound_speed_m_s: 1540.0,
        imaging_frequency_hz: 20.0e6,
        therapy_frequency_hz: 1.5e6,
        bmode_dynamic_range_db: 60.0,
        therapy_mechanical_index: 0.25,
        therapy_peak_delta_t_c: 0.04,
        therapy_target_to_offtarget_deposition_ratio: 101.0,
    })
    .unwrap();

    assert!((metrics.imaging_wavelength_um - 77.0).abs() < 1.0e-12);
    assert!((metrics.therapy_wavelength_mm - 1540.0 / 1.5e6 * 1.0e3).abs() < 1.0e-12);
    assert!((metrics.lumen_area_mm2 - 4.0).abs() < 1.0e-12);
    assert!((metrics.plaque_area_mm2 - 4.0).abs() < 1.0e-12);
    assert_eq!(metrics.bmode_dynamic_range_db, 60.0);
    assert_eq!(metrics.bmode_mean_lumen_intensity, 0.20);
    assert!((metrics.bmode_mean_wall_intensity - 0.60).abs() < 1.0e-12);
    assert_eq!(metrics.therapy_mechanical_index, 0.25);
    assert_eq!(metrics.therapy_peak_delta_t_c, 0.04);
    assert_eq!(metrics.therapy_target_to_offtarget_deposition_ratio, 101.0);
}

#[test]
fn ivus_chapter_metrics_rejects_empty_masks() {
    let err = ivus_chapter_metrics(IvusChapterMetricsInput {
        x_m: &[-1.0e-3, -1.0e-3, 1.0e-3, 1.0e-3],
        y_m: &[-1.0e-3, 1.0e-3, -1.0e-3, 1.0e-3],
        lumen_mask: &[false, false, false, false],
        eel_mask: &[true, true, true, true],
        plaque_mask: &[true, false, false, false],
        bmode_cartesian: &[0.0, 0.0, 0.0, 0.0],
        sound_speed_m_s: 1540.0,
        imaging_frequency_hz: 20.0e6,
        therapy_frequency_hz: 1.5e6,
        bmode_dynamic_range_db: 60.0,
        therapy_mechanical_index: 0.25,
        therapy_peak_delta_t_c: 0.04,
        therapy_target_to_offtarget_deposition_ratio: 101.0,
    })
    .unwrap_err();
    assert!(err.contains("lumen and plaque masks"));
}
