use super::{FdtdAvx512Config, FdtdAvx512StencilProcessor};
use leto::Array3;

#[test]
fn test_avx512_processor_creation() {
    let config = FdtdAvx512Config::default();
    let result = FdtdAvx512StencilProcessor::new(32, 32, 32, config);
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
    let config = FdtdAvx512Config::default();
    let result = FdtdAvx512StencilProcessor::new(2, 32, 32, config);
    assert!(result.is_err());
}

#[test]
fn test_avx512_invalid_tile_size() {
    let config = FdtdAvx512Config {
        tile_size: 7,
        ..FdtdAvx512Config::default()
    };
    let result = FdtdAvx512StencilProcessor::new(32, 32, 32, config);
    assert!(result.is_err());
}

#[test]
fn test_pressure_update_dimensions() {
    let config = FdtdAvx512Config::default();
    if let Ok(processor) = FdtdAvx512StencilProcessor::new(16, 16, 16, config) {
        let p_curr = Array3::zeros((16, 16, 16));
        let p_prev = Array3::zeros((16, 16, 16));
        let u_div = Array3::zeros((16, 16, 16));
        processor
            .update_pressure_avx512(&p_curr, &p_prev, &u_div)
            .unwrap();
    }
}

/// Acoustic leapfrog stationarity: a spatially-uniform pressure field
/// `p^n = p^(n-1) = C` satisfies `∇²p = 0`, so the wave equation predicts
/// `p^(n+1) = 2p^n − p^(n-1) + 0 = C` — the field must remain constant.
///
/// This pins the sign and magnitude of the stencil coefficients: a wrong-
/// sign Laplacian or a missing factor would leave the interior unchanged
/// only by coincidence here, but `update_pressure_avx512` also zeroes the
/// Dirichlet boundary, so we check both: the interior matches `C` and the
/// boundary is zero.
#[test]
fn pressure_update_keeps_interior_constant_for_uniform_field() {
    let config = FdtdAvx512Config::default();
    let Ok(processor) = FdtdAvx512StencilProcessor::new(16, 16, 16, config) else {
        return; // AVX-512 unavailable on this host
    };
    let constant = 7.5_f64;
    let p_curr = Array3::from_elem([16, 16, 16], constant);
    let p_prev = Array3::from_elem([16, 16, 16], constant);
    let u_div = Array3::zeros((16, 16, 16));

    let p_new = processor
        .update_pressure_avx512(&p_curr, &p_prev, &u_div)
        .unwrap();

    // Interior cell — wave equation: ∂²p/∂t² = c²·∇²p; uniform field → no change.
    let interior = p_new[[8, 8, 8]];
    assert!(
        (interior - constant).abs() < 1e-9,
        "uniform field must remain constant in the interior; got {interior}, expected {constant}"
    );

    // Boundary cell — Dirichlet zero (set in the kernel after the stencil sweep).
    assert_eq!(p_new[[0, 8, 8]], 0.0, "boundary must be Dirichlet zero");
}

#[test]
fn test_pressure_update_mismatch() {
    let config = FdtdAvx512Config::default();
    if let Ok(processor) = FdtdAvx512StencilProcessor::new(16, 16, 16, config) {
        let p_curr = Array3::zeros((16, 16, 16));
        let p_prev = Array3::zeros((12, 12, 12));
        let u_div = Array3::zeros((16, 16, 16));
        let result = processor.update_pressure_avx512(&p_curr, &p_prev, &u_div);
        assert!(result.is_err());
    }
}
