use super::{Avx512Config, Avx512StencilProcessor};
use ndarray::Array3;

#[test]
fn test_avx512_processor_creation() {
    let config = Avx512Config::default();
    let result = Avx512StencilProcessor::new(32, 32, 32, config);
    match result {
        Ok(processor) => {
            assert_eq!(processor.nx, 32);
            assert_eq!(processor.ny, 32);
            assert_eq!(processor.nz, 32);
        }
        Err(e) => {
            println!("AVX-512 not available: {}", e);
        }
    }
}

#[test]
fn test_avx512_invalid_dimensions() {
    let config = Avx512Config::default();
    let result = Avx512StencilProcessor::new(2, 32, 32, config);
    assert!(result.is_err());
}

#[test]
fn test_avx512_invalid_tile_size() {
    let mut config = Avx512Config::default();
    config.tile_size = 7;
    let result = Avx512StencilProcessor::new(32, 32, 32, config);
    assert!(result.is_err());
}

#[test]
fn test_pressure_update_dimensions() {
    let config = Avx512Config::default();
    if let Ok(processor) = Avx512StencilProcessor::new(16, 16, 16, config) {
        let p_curr = Array3::zeros((16, 16, 16));
        let p_prev = Array3::zeros((16, 16, 16));
        let u_div = Array3::zeros((16, 16, 16));
        let result = processor.update_pressure_avx512(&p_curr, &p_prev, &u_div);
        assert!(result.is_ok());
    }
}

#[test]
fn test_pressure_update_mismatch() {
    let config = Avx512Config::default();
    if let Ok(processor) = Avx512StencilProcessor::new(16, 16, 16, config) {
        let p_curr = Array3::zeros((16, 16, 16));
        let p_prev = Array3::zeros((12, 12, 12));
        let u_div = Array3::zeros((16, 16, 16));
        let result = processor.update_pressure_avx512(&p_curr, &p_prev, &u_div);
        assert!(result.is_err());
    }
}
