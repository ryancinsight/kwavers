//! Tests for multi-modal fusion quality assessment

use super::*;
use crate::domain::imaging::ultrasound::elastography::ElasticityMap;
use ndarray::Array3;

#[test]
fn test_calculate_image_metrics() {
    let mut data = Array3::<f64>::from_elem((10, 10, 1), 10.0);
    // Add some noise to background
    data[[0, 1, 0]] = 9.0;
    data[[1, 0, 0]] = 11.0;

    // Add signal (top 10% = 10 pixels) on diagonal
    for i in 0..10 {
        data[[i, i, 0]] = 50.0;
    }

    let metrics = calculate_image_metrics(&data);

    assert!(metrics.snr > 1.0);
    assert!(metrics.cnr > 1.0);
}

#[test]
fn test_compute_pa_quality() {
    let mut data = Array3::<f64>::from_elem((10, 10, 1), 1.0);
    data[[0, 1, 0]] = 0.9;
    data[[1, 0, 0]] = 1.1;
    data[[5, 5, 0]] = 100.0; // Peak

    let quality = compute_pa_quality(&data);
    assert!(quality > 0.0 && quality <= 1.0);

    let flat_data = Array3::<f64>::from_elem((10, 10, 1), 1.0);
    let flat_quality = compute_pa_quality(&flat_data);
    assert!(quality > flat_quality);
}

#[test]
fn test_compute_elastography_quality() {
    let youngs = Array3::<f64>::from_elem((10, 10, 1), 1000.0);
    let shear = Array3::<f64>::zeros((10, 10, 1));
    let speed = Array3::<f64>::zeros((10, 10, 1));

    let mut youngs = youngs;
    youngs[[0, 1, 0]] = 900.0;
    youngs[[1, 0, 0]] = 1100.0;

    let mut map = ElasticityMap {
        youngs_modulus: youngs,
        shear_modulus: shear,
        shear_wave_speed: speed,
    };

    map.youngs_modulus[[5, 5, 0]] = 5000.0;

    let quality = compute_elastography_quality(&map);
    assert!(quality > 0.0 && (0.0..=1.0).contains(&quality));

    let mut bad_map = map.clone();
    bad_map.youngs_modulus[[9, 0, 0]] = -100.0;

    let bad_quality = compute_elastography_quality(&bad_map);
    assert!(bad_quality < quality);
}

#[test]
fn test_compute_optical_quality_visible_light() {
    let intensity = Array3::<f64>::from_elem((8, 8, 4), 100.0);
    let wavelength = 550e-9;

    let quality = compute_optical_quality(&intensity, wavelength);

    assert!((0.0..=1.0).contains(&quality));
    assert!(quality > 0.6);
}

#[test]
fn test_compute_optical_quality_infrared() {
    let intensity = Array3::<f64>::from_elem((8, 8, 4), 100.0);
    let wavelength = 1000e-9;

    let quality = compute_optical_quality(&intensity, wavelength);

    assert!((0.0..=1.0).contains(&quality));
}

#[test]
fn test_estimate_modality_noise() {
    let mut data = Array3::<f64>::from_elem((8, 8, 4), 100.0);
    data[[0, 0, 0]] = 110.0;
    data[[1, 1, 1]] = 90.0;

    let noise = estimate_modality_noise(&data);

    assert!(noise > 0.0);
}

#[test]
fn test_bayesian_fusion_weighted_average() {
    let values = vec![1.0, 3.0];
    let weights = vec![0.75, 0.25];

    let (mean, _uncertainty) = bayesian_fusion_single_voxel(&values, &weights);

    let expected = (1.0 * 0.75 + 3.0 * 0.25) / (0.75 + 0.25);
    assert!((mean - expected).abs() < 1e-10);
}

#[test]
fn test_compute_fusion_uncertainty() {
    let data1 = Array3::<f64>::from_elem((4, 4, 2), 1.0);
    let data2 = Array3::<f64>::from_elem((4, 4, 2), 3.0);
    let data3 = Array3::<f64>::from_elem((4, 4, 2), 2.0);

    let modality_data = vec![&data1, &data2, &data3];
    let weights = vec![1.0, 1.0, 1.0];

    let uncertainty = compute_fusion_uncertainty(&modality_data, &weights);

    assert_eq!(uncertainty.dim(), (4, 4, 2));

    let first_uncertainty = uncertainty[[0, 0, 0]];
    for value in uncertainty.iter() {
        assert!((value - first_uncertainty).abs() < 1e-10);
    }
    assert!(first_uncertainty > 0.0);
}

#[test]
fn test_compute_confidence_map() {
    let quality_scores = vec![0.8, 0.9, 0.7];
    let uncertainty = Array3::<f64>::from_elem((4, 4, 2), 0.2);

    let confidence = compute_confidence_map(&quality_scores, &uncertainty);

    let avg_quality = (0.8 + 0.9 + 0.7) / 3.0;
    let expected_confidence = avg_quality * (1.0 - 0.2);

    for value in confidence.iter() {
        assert!((value - expected_confidence).abs() < 1e-10);
    }
}
