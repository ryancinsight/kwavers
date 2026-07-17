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
/// This checks the leapfrog recurrence, complete interior coverage, and the
/// zero-Dirichlet boundary policy. A non-uniform reference case is required to
/// establish Laplacian coefficient sign or magnitude.
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

    // Eight additions, two products, and one fused add bound the roundoff by
    // a small multiple of ε·|C|; 32ε covers the rounded coefficient setup.
    let tolerance = 32.0 * f64::EPSILON * constant.abs();
    for x in 1..15 {
        for y in 1..15 {
            for z in 1..15 {
                let interior = p_new[[x, y, z]];
                assert!(
                    (interior - constant).abs() <= tolerance,
                    "interior [{x}, {y}, {z}] = {interior}, expected {constant}"
                );
            }
        }
    }

    // Boundary cell — Dirichlet zero (set in the kernel after the stencil sweep).
    assert_eq!(p_new[[0, 8, 8]], 0.0, "boundary must be Dirichlet zero");
}

/// A linear pressure field has the exact centred difference `p[+1] - p[-1] =
/// 2` on every interior point. This independently checks each axis stride,
/// complete vector/tail coverage, and the momentum-update sign.
#[test]
fn velocity_update_matches_linear_pressure_gradient() {
    let config = FdtdAvx512Config::default();
    let Ok(processor) = FdtdAvx512StencilProcessor::new(16, 16, 16, config) else {
        return;
    };

    let expected = 2.0 * processor.velocity_coeff;
    let tolerance = 8.0 * f64::EPSILON * expected.abs().max(1.0);
    for dim in 0..3 {
        let pressure = Array3::from_shape_fn([16, 16, 16], |[x, y, z]| match dim {
            0 => x as f64,
            1 => y as f64,
            2 => z as f64,
            _ => unreachable!(),
        });
        let mut velocity = Array3::zeros((16, 16, 16));
        processor
            .update_velocity_avx512(&mut velocity, &pressure, dim)
            .unwrap();

        for x in 1..15 {
            for y in 1..15 {
                for z in 1..15 {
                    let value = velocity[[x, y, z]];
                    assert!(
                        (value - expected).abs() <= tolerance,
                        "dim={dim}, [{x}, {y}, {z}] = {value}, expected {expected}"
                    );
                }
            }
        }
    }
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
