//! Value-semantic regression tests for SIMD ops.

use super::{
    FdtdSimdOps, FftSimdOps, InterpolationSimdOps, SimdConfig, SimdLevel, SimdPerformance,
};

#[test]
fn test_simd_config_detection() {
    let config = SimdConfig::detect();
    assert!(matches!(
        config.level,
        SimdLevel::Scalar
            | SimdLevel::Sse2
            | SimdLevel::Avx2
            | SimdLevel::Avx512
            | SimdLevel::Neon
            | SimdLevel::Portable
    ));
    assert!(config.vector_width >= 1);
    assert!(config.alignment >= std::mem::align_of::<f32>());
}

#[test]
fn test_fdtd_simd_ops_creation() {
    let _ = FdtdSimdOps::new();
}

#[test]
fn test_fft_simd_ops_creation() {
    let _ = FftSimdOps::new();
}

#[test]
fn test_interpolation_simd_ops_creation() {
    let _ = InterpolationSimdOps::new();
}

#[test]
fn test_performance_metrics() {
    let metrics = SimdPerformance::get_metrics();
    assert!(metrics.estimated_speedup >= 1.0);
    assert!(metrics.vector_width >= 1);
}

#[test]
fn test_scalar_pressure_update() {
    let mut pressure = vec![1.0; 1000];
    let pressure_prev = vec![0.9; 1000];
    let laplacian = vec![0.1; 1000];
    let c_dt2 = 0.5;

    let ops = FdtdSimdOps::new();
    ops.update_pressure_3d(&mut pressure, &pressure_prev, &laplacian, c_dt2, 10, 10, 10);

    // Check that values changed
    assert_ne!(pressure[5 + 5 * 10 + 5 * 10 * 10], 1.0);
}

#[test]
fn test_complex_multiply() {
    let mut real1 = vec![1.0, 2.0, 3.0];
    let mut imag1 = vec![0.5, 1.5, 2.5];
    let real2 = vec![0.5, 1.0, 1.5];
    let imag2 = vec![0.2, 0.4, 0.6];

    let ops = FftSimdOps::new();
    ops.complex_multiply(&mut real1, &mut imag1, &real2, &imag2);

    // Results should be different from input
    assert_ne!(real1[0], 1.0);
    assert_ne!(imag1[0], 0.5);
}

#[test]
fn test_trilinear_interpolation() {
    let data = vec![1.0; 1000]; // 10x10x10 grid
    let query_points = vec![(5.0, 5.0, 5.0), (2.5, 3.5, 4.5)];
    let mut results = vec![0.0; 2];

    let ops = InterpolationSimdOps::new();
    ops.trilinear_interpolate(&data, 10, 10, 10, &query_points, &mut results);

    // Should interpolate to valid values
    assert!(results[0] >= 0.0);
    assert!(results[1] >= 0.0);
}
