//! Tests for AVX-512 pressure field update.

use super::super::SimdAvx512Config;
use super::super::SimdAvx512StencilProcessor;
use ndarray::Array3;

#[test]
fn test_pressure_update_dimensions() {
    let config = SimdAvx512Config::default();
    if let Ok(processor) = SimdAvx512StencilProcessor::new(16, 16, 16, config) {
        let p_curr = Array3::zeros((16, 16, 16));
        let p_prev = Array3::zeros((16, 16, 16));
        let u_div = Array3::zeros((16, 16, 16));

        let result = processor.update_pressure_avx512(&p_curr, &p_prev, &u_div);
        result.unwrap();
    }
}

#[test]
fn test_pressure_update_mismatch() {
    let config = SimdAvx512Config::default();
    if let Ok(processor) = SimdAvx512StencilProcessor::new(16, 16, 16, config) {
        let p_curr = Array3::zeros((16, 16, 16));
        let p_prev = Array3::zeros((12, 12, 12)); // Mismatch
        let u_div = Array3::zeros((16, 16, 16));

        let result = processor.update_pressure_avx512(&p_curr, &p_prev, &u_div);
        assert!(result.is_err());
    }
}
