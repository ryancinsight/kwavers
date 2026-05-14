//! `infer_grid` integration and batch-size invariance tests.

use super::super::config::ParamFieldPINNConfig;
use super::super::forward::{infer_grid, GridQueryParams};
use super::super::network::ParamFieldPINNNetwork;
use super::B;

#[test]
fn test_infer_grid_shape_and_finiteness() {
    let cfg = ParamFieldPINNConfig::default();
    let device = Default::default();
    let net = ParamFieldPINNNetwork::<B>::new(&cfg, &device).unwrap();

    let params = GridQueryParams {
        shape: (8, 6, 6),
        focus_idx: (4, 3, 3),
        dx_m: 1.0e-3,
        f0: 0.75e6,
        pnp: 22.5e6,
        coord_half_m: (4.0e-3, 3.0e-3, 3.0e-3),
        f0_range: (0.5e6, 1.0e6),
        pnp_range: (15.0e6, 30.0e6),
        output_transforms: super::super::target_transform::OutputTransforms::linear(
            30.0e6, 30.0e6, 21.0e6,
        )
        .unwrap(),
        batch_size: 64,
    };
    let (p_min, p_max, p_rms) = infer_grid(&net, &params, &device).unwrap();
    assert_eq!(p_min.dim(), (8, 6, 6));
    assert_eq!(p_max.dim(), (8, 6, 6));
    assert_eq!(p_rms.dim(), (8, 6, 6));
    for v in p_min.iter().chain(p_max.iter()).chain(p_rms.iter()) {
        assert!(
            v.is_finite(),
            "untrained network produced non-finite output"
        );
    }
}

#[test]
fn test_infer_grid_rejects_focus_out_of_bounds() {
    let cfg = ParamFieldPINNConfig::default();
    let device = Default::default();
    let net = ParamFieldPINNNetwork::<B>::new(&cfg, &device).unwrap();

    let mut params = GridQueryParams::default();
    params.shape = (4, 4, 4);
    params.focus_idx = (4, 0, 0); // out of bounds
    params.coord_half_m = (1e-3, 1e-3, 1e-3);
    assert!(infer_grid(&net, &params, &device).is_err());
}

#[test]
fn test_infer_grid_batch_size_invariance() {
    // The network is deterministic, so streaming the same grid in
    // different batch sizes must produce identical outputs.
    let cfg = ParamFieldPINNConfig::default();
    let device = Default::default();
    let net = ParamFieldPINNNetwork::<B>::new(&cfg, &device).unwrap();

    let make_params = |bs: usize| GridQueryParams {
        shape: (5, 4, 4),
        focus_idx: (2, 2, 2),
        dx_m: 1.0e-3,
        f0: 0.6e6,
        pnp: 18.0e6,
        coord_half_m: (3.0e-3, 2.0e-3, 2.0e-3),
        f0_range: (0.5e6, 1.0e6),
        pnp_range: (15.0e6, 30.0e6),
        output_transforms: super::super::target_transform::OutputTransforms::linear(
            30.0e6, 30.0e6, 21.0e6,
        )
        .unwrap(),
        batch_size: bs,
    };
    let (a_min, a_max, a_rms) = infer_grid(&net, &make_params(7), &device).unwrap();
    let (b_min, b_max, b_rms) = infer_grid(&net, &make_params(80), &device).unwrap();
    let close = |a: f64, b: f64| (a - b).abs() < 1e-6_f64.max(1e-6 * a.abs().max(b.abs()));
    for ((va, vb), label) in a_min
        .iter()
        .zip(b_min.iter())
        .map(|p| (p, "p_min"))
        .chain(a_max.iter().zip(b_max.iter()).map(|p| (p, "p_max")))
        .chain(a_rms.iter().zip(b_rms.iter()).map(|p| (p, "p_rms")))
    {
        assert!(
            close(*va, *vb),
            "batch-size invariance violated in {label}: {va} vs {vb}"
        );
    }
}
