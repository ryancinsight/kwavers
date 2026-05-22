use burn::backend::NdArray;
use ndarray::{Array2, Array3};

use crate::core::constants::fundamental::SOUND_SPEED_WATER_SIM;

use super::*;

type TestBackend = NdArray;

#[test]
fn test_burn_beamformer_creation() {
    let device = Default::default();
    let _beamformer: BurnDasBeamformer<TestBackend> = BurnDasBeamformer::new(device);
}

#[test]
fn test_distance_computation() {
    let device = Default::default();
    let beamformer: BurnDasBeamformer<TestBackend> = BurnDasBeamformer::new(device);

    let sensor_pos = Array2::from_shape_vec((1, 3), vec![0.0, 0.0, 0.0]).unwrap();
    let sensor_tensor = beamformer.array_to_tensor_2d(&sensor_pos).unwrap();

    let focal = Array2::from_shape_vec((1, 3), vec![3.0, 4.0, 0.0]).unwrap();
    let focal_tensor = beamformer.array_to_tensor_2d(&focal).unwrap();

    let distances = beamformer.compute_distances_batch(&sensor_tensor, &focal_tensor);
    let dist_data = distances.into_data();
    let dist_vec = dist_data.as_slice::<f32>().unwrap();

    assert!((dist_vec[0] - 5.0).abs() < 1e-5);
}

#[test]
fn test_single_focal_point_beamforming() {
    let device = Default::default();
    let beamformer: BurnDasBeamformer<TestBackend> = BurnDasBeamformer::new(device);

    let rf_data = Array3::from_shape_fn((2, 3, 10), |(i, j, k)| (i + j + k) as f64 * 0.1);
    let sensor_pos = Array2::from_shape_vec((2, 3), vec![0.0, 0.0, 0.0, 0.01, 0.0, 0.0]).unwrap();
    let focal_points = Array2::from_shape_vec((1, 3), vec![0.005, 0.0, 0.01]).unwrap();

    let result = beamformer.beamform(&rf_data, &sensor_pos, &focal_points, None, 1e6, SOUND_SPEED_WATER_SIM);

    let output = result.unwrap();
    assert_eq!(output.shape(), &[1, 3, 1]);
    assert!((output[[0, 0, 0]] - 1.5).abs() < 1.0e-5);
    assert!((output[[0, 1, 0]] - 1.7).abs() < 1.0e-5);
    assert!((output[[0, 2, 0]] - 1.9).abs() < 1.0e-5);
}

#[test]
fn test_apodization() {
    let device = Default::default();
    let beamformer: BurnDasBeamformer<TestBackend> = BurnDasBeamformer::new(device);

    let rf_data = Array3::from_shape_vec(
        (3, 1, 5),
        vec![
            1.0, 2.0, 3.0, 4.0, 5.0, 2.0, 4.0, 6.0, 8.0, 10.0, 3.0, 6.0, 9.0, 12.0, 15.0,
        ],
    )
    .unwrap();

    let sensor_pos =
        Array2::from_shape_vec((3, 3), vec![0.0, 0.0, 0.0, 0.01, 0.0, 0.0, 0.02, 0.0, 0.0])
            .unwrap();

    let focal_points = Array2::from_shape_vec((1, 3), vec![0.0, 0.0, 0.0]).unwrap();
    let apod_weights = vec![0.5, 1.0, 0.5];

    let result = beamformer.beamform(
        &rf_data,
        &sensor_pos,
        &focal_points,
        Some(&apod_weights),
        1e6,
        SOUND_SPEED_WATER_SIM,
    );

    let output = result.unwrap();
    assert_eq!(output.shape(), &[1, 1, 1]);
    assert!((output[[0, 0, 0]] - 0.5).abs() < 1.0e-5);
}

#[test]
fn test_invalid_input_dimensions() {
    let device = Default::default();
    let beamformer: BurnDasBeamformer<TestBackend> = BurnDasBeamformer::new(device);

    let rf_data = Array3::zeros((3, 2, 10));
    let sensor_pos = Array2::zeros((2, 3));
    let focal_points = Array2::zeros((1, 3));

    let result = beamformer.beamform(&rf_data, &sensor_pos, &focal_points, None, 1e6, SOUND_SPEED_WATER_SIM);

    assert!(result.is_err());
}

#[test]
fn test_cpu_wrapper() {
    let rf_data = Array3::ones((2, 2, 10));
    let sensor_pos = Array2::zeros((2, 3));
    let focal_points = Array2::zeros((1, 3));

    let result = beamform_cpu(&rf_data, &sensor_pos, &focal_points, None, 1e6, SOUND_SPEED_WATER_SIM);
    let output = result.unwrap();
    assert_eq!(output.shape(), &[1, 2, 1]);
    assert_eq!(output[[0, 0, 0]], 2.0);
    assert_eq!(output[[0, 1, 0]], 2.0);
}

#[test]
fn test_array_tensor_conversion() {
    let device = Default::default();
    let beamformer: BurnDasBeamformer<TestBackend> = BurnDasBeamformer::new(device);

    let array = Array3::from_shape_fn((2, 3, 4), |(i, j, k)| (i + j + k) as f64);
    let tensor = beamformer.array_to_tensor_3d(&array).unwrap();
    let reconstructed = beamformer.tensor_to_array_3d(&tensor).unwrap();

    assert_eq!(array.shape(), reconstructed.shape());
    for (a, b) in array.iter().zip(reconstructed.iter()) {
        assert!((a - b).abs() < 1e-5);
    }
}

#[test]
fn test_multiple_focal_points() {
    let device = Default::default();
    let beamformer: BurnDasBeamformer<TestBackend> = BurnDasBeamformer::new(device);

    let rf_data = Array3::ones((4, 5, 20));
    let sensor_pos = Array2::zeros((4, 3));

    let focal_points = Array2::from_shape_vec(
        (3, 3),
        vec![0.0, 0.0, 0.01, 0.01, 0.0, 0.01, 0.0, 0.01, 0.01],
    )
    .unwrap();

    let result = beamformer.beamform(&rf_data, &sensor_pos, &focal_points, None, 1e6, SOUND_SPEED_WATER_SIM);

    let output = result.unwrap();
    assert_eq!(output.shape(), &[3, 5, 1]);
    assert!(output.iter().all(|value| (*value - 4.0).abs() < 1.0e-5));
}
