use super::*;
use ndarray::Array3;

#[test]
fn test_simd_config_detection() {
    init_simd();
    let config = get_simd_config();
    println!("Detected SIMD level: {:?}", config.level);
    assert!(config.enabled || config.level == SimdLevel::Scalar);
}

#[test]
fn test_strategy_selection() {
    let best = StencilStrategy::select_best();
    println!("Best strategy: {}", best.as_str());
    assert!(best.is_available());
}

#[test]
fn test_dispatcher_creation() {
    let result = FdtdStencilDispatcher::new(32, 32, 32, -1.0, -1.0);
    assert!(result.is_ok());
}

#[test]
fn test_dispatcher_invalid_dimensions() {
    let result = FdtdStencilDispatcher::new(2, 32, 32, -1.0, -1.0);
    assert!(result.is_err());
}

#[test]
fn test_dispatcher_strategy_override() {
    let result =
        FdtdStencilDispatcher::with_strategy(32, 32, 32, -1.0, -1.0, StencilStrategy::Scalar);
    assert!(result.is_ok());
    assert_eq!(result.unwrap().strategy(), StencilStrategy::Scalar);
}

#[test]
fn test_pressure_update_scalar() {
    let mut dispatcher =
        FdtdStencilDispatcher::with_strategy(16, 16, 16, -1.0, -1.0, StencilStrategy::Scalar)
            .unwrap();

    let p_curr = Array3::from_elem((16, 16, 16), 1.0_f64);
    let p_prev = Array3::from_elem((16, 16, 16), 0.5_f64);
    let u_div = Array3::zeros((16, 16, 16));

    let result = dispatcher.update_pressure(&p_curr, &p_prev, &u_div);
    let result = result.unwrap();

    for k in 1..15 {
        for j in 1..15 {
            for i in 1..15 {
                assert_eq!(result[[i, j, k]], 1.5);
            }
        }
    }

    assert_eq!(result[[0, 0, 0]], 0.0);
    assert_eq!(result[[15, 15, 15]], 0.0);
}

#[test]
fn test_metrics() {
    let dispatcher = FdtdStencilDispatcher::new(32, 32, 32, -1.0, -1.0).unwrap();
    let metrics = dispatcher.metrics();
    println!(
        "Strategy: {}, SIMD Level: {:?}, Vector Width: {}",
        metrics.selected_strategy.as_str(),
        metrics.simd_level,
        metrics.vector_width
    );
}
