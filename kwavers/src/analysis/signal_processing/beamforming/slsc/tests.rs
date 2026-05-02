use super::*;
use ndarray::Array2;
use num_complex::Complex64;

#[test]
fn test_slsc_default_config() {
    let slsc = SlscBeamformer::new();
    assert_eq!(slsc.config().max_lag, 10);
    assert!(slsc.config().normalize);
}

#[test]
fn test_slsc_with_config() {
    let config = SlscConfig::with_max_lag(20);
    let slsc = SlscBeamformer::with_config(config);
    assert_eq!(slsc.config().max_lag, 20);
}

#[test]
fn test_slsc_process_simple() {
    let n_elements = 4;
    let n_samples = 10;
    let data = Array2::from_elem((n_elements, n_samples), Complex64::new(1.0, 0.0));

    let slsc = SlscBeamformer::new();
    let result = slsc.process(&data).expect("SLSC processing should succeed");

    assert_eq!(result.len(), n_samples);
    for &val in result.iter() {
        assert!((0.0..=1.0).contains(&val), "Coherence should be in [0, 1]");
    }
}

#[test]
fn test_slsc_rejects_single_element() {
    let data = Array2::from_elem((1, 10), Complex64::new(1.0, 0.0));
    let slsc = SlscBeamformer::new();
    let result = slsc.process(&data);
    assert!(result.is_err());
}

#[test]
fn test_lag_weighting_uniform() {
    let w = LagWeighting::Uniform;
    assert_eq!(w.weight(1, 10), 1.0);
    assert_eq!(w.weight(5, 10), 1.0);
}

#[test]
fn test_lag_weighting_triangular() {
    let w = LagWeighting::Triangular;
    assert_eq!(w.weight(0, 10), 1.0);
    assert_eq!(w.weight(5, 10), 0.5);
    assert_eq!(w.weight(10, 10), 0.0);
}

#[test]
fn test_slsc_grid_processing() {
    let n_elements = 8;
    let height = 10;
    let width = 20;
    let n_pixels = height * width;

    let data = Array2::from_elem((n_elements, n_pixels), Complex64::new(1.0, 0.0));
    let slsc = SlscBeamformer::new();
    let result = slsc
        .process_grid(&data, (height, width))
        .expect("Grid processing should succeed");

    assert_eq!(result.shape(), &[height, width]);
}
