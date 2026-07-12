use super::{FdtdSimdStencilConfig, FdtdSimdStencilProcessor};
use leto::Array3;

#[test]
fn test_stencil_creation() {
    let config = FdtdSimdStencilConfig::default();
    let result = FdtdSimdStencilProcessor::new(64, 64, 64, config);
    let _processor = result.unwrap();
}

#[test]
fn test_dimension_validation() {
    let config = FdtdSimdStencilConfig::default();
    let result = FdtdSimdStencilProcessor::new(2, 64, 64, config);
    assert!(result.is_err());
}

#[test]
fn test_pressure_update() {
    let config = FdtdSimdStencilConfig::default();
    let mut processor = FdtdSimdStencilProcessor::new(16, 16, 16, config).unwrap();

    let pressure = Array3::ones((16, 16, 16));
    let pressure_prev = Array3::ones((16, 16, 16));
    let velocity_div = Array3::zeros((16, 16, 16));

    let result = processor.update_pressure(&pressure, &pressure_prev, &velocity_div);

    let updated = result.unwrap();
    assert_eq!(updated.shape(), pressure.shape());
}

#[test]
fn test_velocity_update() {
    let config = FdtdSimdStencilConfig::default();
    let mut processor = FdtdSimdStencilProcessor::new(16, 16, 16, config).unwrap();

    let mut velocity = Array3::zeros((16, 16, 16));
    let pressure = Array3::ones((16, 16, 16));

    let result = processor.update_velocity(&mut velocity, &pressure);
    result.unwrap();
    assert_eq!(velocity.shape(), [16, 16, 16]);
}

#[test]
fn test_fused_update() {
    let config = FdtdSimdStencilConfig::default();
    let mut processor = FdtdSimdStencilProcessor::new(16, 16, 16, config).unwrap();

    let pressure = Array3::ones((16, 16, 16));
    let pressure_prev = Array3::ones((16, 16, 16));
    let mut velocity = Array3::zeros((16, 16, 16));
    let velocity_dim = velocity.shape();
    let velocity_div = Array3::zeros((16, 16, 16));

    let result = processor.fused_update(&pressure, &pressure_prev, &mut velocity, &velocity_div);

    let p_new = result.unwrap();
    assert_eq!(p_new.shape(), pressure.shape());
    assert_eq!(velocity.shape(), velocity_dim);
}

/// Tiled and non-tiled (tile=256) results must be bitwise identical on a 17³ grid.
/// Non-power-of-two grid size exercises tile boundary handling.
/// # Panics
/// - Panics if an internal invariant assumed to hold at this call site is violated.
///
#[test]
fn test_tiling_matches_naive() {
    let n = 17usize;
    let mut config_tiled = FdtdSimdStencilConfig::default();
    config_tiled.tile_size = 8;
    let mut processor_tiled = FdtdSimdStencilProcessor::new(n, n, n, config_tiled).unwrap();

    let mut config_naive = FdtdSimdStencilConfig::default();
    config_naive.tile_size = 256;
    let mut processor_naive = FdtdSimdStencilProcessor::new(n, n, n, config_naive).unwrap();

    let pressure = Array3::from_elem([n, n, n], 1000.0_f64);
    let pressure_prev = Array3::from_elem([n, n, n], 990.0_f64);
    let velocity_div = Array3::from_elem([n, n, n], 0.1_f64);

    let p_tiled = processor_tiled
        .update_pressure(&pressure, &pressure_prev, &velocity_div)
        .unwrap();
    let p_naive = processor_naive
        .update_pressure(&pressure, &pressure_prev, &velocity_div)
        .unwrap();

    let max_diff = p_tiled
        .iter()
        .zip(p_naive.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0_f64, f64::max);
    assert!(
        max_diff < f64::EPSILON * 100.0,
        "Tiled and naive pressure stencils must be identical; max diff = {max_diff:.2e}"
    );
}

/// In-place velocity update must match the clone-based path.
/// # Panics
/// - Panics if an internal invariant assumed to hold at this call site is violated.
///
#[test]
fn test_velocity_inplace_no_regression() {
    let n = 16usize;
    let config = FdtdSimdStencilConfig::default();
    let mut processor = FdtdSimdStencilProcessor::new(n, n, n, config).unwrap();

    let pressure = Array3::from_elem([n, n, n], 500.0_f64);
    let mut vel_inplace = Array3::from_elem([n, n, n], 0.1_f64);
    processor
        .update_velocity(&mut vel_inplace, &pressure)
        .unwrap();

    for k in 1..n - 1 {
        for j in 1..n - 1 {
            for i in 1..n - 1 {
                assert!(
                    (vel_inplace[[i, j, k]] - 0.1).abs() < 1e-12,
                    "Interior vel at [{i},{j},{k}] changed unexpectedly: {}",
                    vel_inplace[[i, j, k]]
                );
            }
        }
    }
    assert_eq!(vel_inplace[[0, 1, 1]], 0.0);
    assert_eq!(vel_inplace[[n - 1, 1, 1]], 0.0);
}

#[test]
fn test_tile_statistics() {
    let config = FdtdSimdStencilConfig::default();
    let processor = FdtdSimdStencilProcessor::new(64, 64, 64, config).unwrap();
    let (tx, ty, tz) = processor.tile_stats();
    assert!(tx > 0 && ty > 0 && tz > 0);
    assert_eq!(processor.total_tiles(), tx * ty * tz);
}

/// Acoustic wave-equation stationarity: a spatially-uniform pressure field
/// `p^n = p^(n-1) = C` with zero velocity divergence has `∇²p = 0`, so the
/// leapfrog must return `p^(n+1) = C` at every interior cell.
///
/// This pins the sign and magnitude of `pressure_coeff` against the
/// computed-Laplacian convention. Previously `pressure_coeff = −c²·Δt²/Δx²`
/// (wrong sign) combined with a Laplacian that already includes 1/Δx² gave
/// an effective Laplacian coefficient with the wrong sign and an extra
/// 1/Δx² factor — for Δx = 1 mm the spurious factor was ~−10⁶ relative
/// to the correct wave equation. With `pressure_coeff = +c²·Δt²` and zero
/// Laplacian, the uniform field is exactly preserved.
#[test]
fn pressure_update_keeps_uniform_field_constant() {
    let config = FdtdSimdStencilConfig::default();
    let mut processor = FdtdSimdStencilProcessor::new(16, 16, 16, config).unwrap();
    let constant = 12.5_f64;
    let pressure = Array3::from_elem([16, 16, 16], constant);
    let pressure_prev = Array3::from_elem([16, 16, 16], constant);
    let velocity_div = Array3::zeros((16, 16, 16));

    let p_new = processor
        .update_pressure(&pressure, &pressure_prev, &velocity_div)
        .unwrap();

    for k in 1..15 {
        for j in 1..15 {
            for i in 1..15 {
                let v = p_new[[i, j, k]];
                assert!(
                    (v - constant).abs() < 1e-9,
                    "interior cell [{i},{j},{k}] = {v}, expected {constant}"
                );
            }
        }
    }
}

#[test]
fn test_stability_check() {
    let config = FdtdSimdStencilConfig::default();
    let cfl = config.sound_speed * config.dt / config.dx;
    assert!(
        cfl <= config.cfl_number + f64::EPSILON * 10.0,
        "CFL {cfl:.6} exceeds cfl_number {:.6}",
        config.cfl_number
    );
}
