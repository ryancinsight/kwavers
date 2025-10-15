//! Integration tests for photoacoustic reconstruction filters
//!
//! Tests extracted from filters module to maintain GRASP compliance (<500 lines/module).

use kwavers::solver::reconstruction::FilterType;
use kwavers::solver::reconstruction::photoacoustic::{
    PhotoacousticAlgorithm, PhotoacousticConfig, Filters,
};
use ndarray::Array2;
use std::f64::consts::PI;

/// Helper to create test configuration
fn create_test_config() -> PhotoacousticConfig {
PhotoacousticConfig {
    algorithm: PhotoacousticAlgorithm::FilteredBackProjection,
    sensor_positions: vec![[0.0, 0.0, 0.0]],
    grid_size: [64, 64, 64],
    sound_speed: 1500.0,
    sampling_frequency: 1e6,
    envelope_detection: false,
    bandpass_filter: None,
    regularization_parameter: 0.01,
}
}

#[test]
fn test_hamming_filter_creation() {
let config = create_test_config();
let filters = Filters::new(&config);

// Create Hamming filter with known size
let n = 128;
let filter = filters.create_hamming_filter(n);
    
    // Verify filter properties
    assert_eq!(filter.len(), n);
    
    // Hamming window should have specific characteristics
    // - DC component (center) should be near maximum
    // - Edges should be attenuated
    let center = filter[n / 2];
    let edge = filter[0].max(filter[n - 1]);
    
    assert!(center > edge, "Center of Hamming filter should be larger than edges");
    
    // All values should be non-negative
    for &val in filter.iter() {
        assert!(val >= 0.0, "Hamming filter should have non-negative values");
    }
}

#[test]
fn test_hann_filter_creation() {
    let config = create_test_config();
    let filters = Filters::new(&config);
    
    // Create Hann filter with known size
    let n = 128;
    let filter = filters.create_hann_filter(n);
    
    // Verify filter properties
    assert_eq!(filter.len(), n);
    
    // Hann window should smoothly taper to edges
    let center = filter[n / 2];
    let quarter = filter[n / 4];
    let edge = filter[0].max(filter[n - 1]);
    
    assert!(center > quarter, "Center should be larger than quarter point");
    assert!(quarter > edge, "Quarter point should be larger than edge");
    
    // All values should be non-negative
    for &val in filter.iter() {
        assert!(val >= 0.0, "Hann filter should have non-negative values");
    }
}

#[test]
fn test_apply_hamming_filter() {
    let config = create_test_config();
    let mut filters = Filters::new(&config);
    filters.set_filter_type(FilterType::Hamming);
    
    // Create test data with known frequency content
    let n_samples = 64;
    let n_sensors = 4;
    let mut data = Array2::zeros((n_samples, n_sensors));
    
    // Fill with a simple sine wave
    for i in 0..n_samples {
        let t = i as f64;
        for j in 0..n_sensors {
            data[[i, j]] = (2.0 * PI * t / 8.0).sin();
        }
    }
    
    // Apply Hamming filter
    let result = filters.apply_fbp_filter(&data);
    assert!(result.is_ok(), "Hamming filter should apply successfully");
    
    let filtered = result.unwrap();
    assert_eq!(filtered.dim(), data.dim(), "Output dimensions should match input");
}

#[test]
fn test_apply_hann_filter() {
    let config = create_test_config();
    let mut filters = Filters::new(&config);
    filters.set_filter_type(FilterType::Hann);
    
    // Create test data
    let n_samples = 64;
    let n_sensors = 4;
    let mut data = Array2::zeros((n_samples, n_sensors));
    
    // Fill with a simple sine wave
    for i in 0..n_samples {
        let t = i as f64;
        for j in 0..n_sensors {
            data[[i, j]] = (2.0 * PI * t / 8.0).sin();
        }
    }
    
    // Apply Hann filter
    let result = filters.apply_fbp_filter(&data);
    assert!(result.is_ok(), "Hann filter should apply successfully");
    
    let filtered = result.unwrap();
    assert_eq!(filtered.dim(), data.dim(), "Output dimensions should match input");
}

#[test]
fn test_none_filter_no_change() {
    let config = create_test_config();
    let mut filters = Filters::new(&config);
    filters.set_filter_type(FilterType::None);
    
    // Create test data
    let n_samples = 32;
    let n_sensors = 2;
    let mut data = Array2::zeros((n_samples, n_sensors));
    
    // Fill with known values
    for i in 0..n_samples {
        for j in 0..n_sensors {
            data[[i, j]] = (i + j) as f64;
        }
    }
    
    // Apply None filter - should return unchanged data
    let result = filters.apply_fbp_filter(&data);
    assert!(result.is_ok(), "None filter should apply successfully");
    
    let filtered = result.unwrap();
    
    // Verify data is unchanged
    for i in 0..n_samples {
        for j in 0..n_sensors {
            assert_eq!(
                filtered[[i, j]], 
                data[[i, j]],
                "None filter should not modify data"
            );
        }
    }
}

#[test]
fn test_filter_type_exhaustive() {
    // This test ensures all FilterType variants are handled
    let config = create_test_config();
    let mut filters = Filters::new(&config);
    
    let data = Array2::from_elem((32, 2), 1.0);
    
    // Test each filter type
    let filter_types = [
        FilterType::RamLak,
        FilterType::SheppLogan,
        FilterType::Cosine,
        FilterType::Hamming,
        FilterType::Hann,
        FilterType::None,
    ];
    
    for filter_type in &filter_types {
        filters.set_filter_type(filter_type.clone());
        let result = filters.apply_fbp_filter(&data);
        assert!(
            result.is_ok(), 
            "Filter type {:?} should apply successfully", 
            filter_type
        );
    }
}

#[test]
fn test_filter_window_properties() {
    // Verify that window functions have expected mathematical properties
    let config = create_test_config();
    let filters = Filters::new(&config);
    let n = 64;
    
    // Test Hamming window
    let hamming = filters.create_hamming_filter(n);
    
    // Hamming coefficients: 0.54 - 0.46*cos(2πn/(N-1))
    // At n=0: 0.54 - 0.46*cos(0) = 0.54 - 0.46 = 0.08
    // At n=(N-1)/2: 0.54 - 0.46*cos(π) = 0.54 + 0.46 = 1.0
    let center_idx = n / 2;
    let center_ram_lak = 0.5; // Ram-Lak at center
    let expected_hamming = center_ram_lak * 1.0; // Hamming window ≈ 1.0 at center
    
    assert!(
        (hamming[center_idx] - expected_hamming).abs() < 0.2,
        "Hamming filter center should be close to expected value"
    );
    
    // Test Hann window
    let hann = filters.create_hann_filter(n);
    
    // Hann coefficients: 0.5 * (1 - cos(2πn/(N-1)))
    // At n=0: 0.5 * (1 - cos(0)) = 0.5 * (1 - 1) = 0.0
    // At n=(N-1)/2: 0.5 * (1 - cos(π)) = 0.5 * (1 - (-1)) = 1.0
    let expected_hann = center_ram_lak * 1.0;
    
    assert!(
        (hann[center_idx] - expected_hann).abs() < 0.2,
        "Hann filter center should be close to expected value"
    );
}
