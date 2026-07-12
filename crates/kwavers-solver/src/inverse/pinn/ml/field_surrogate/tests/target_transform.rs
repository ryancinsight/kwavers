//! Integration tests for the signed-log1p target transform (Phase C-8).
//!
//! The standalone math is exercised by the `target_transform` module's
//! internal `#[cfg(test)]` block; these tests verify the surrounding
//! plumbing — that the sampler emits transformed targets in `[-1, 1]`,
//! that the round-trip through `infer_grid` recovers the original Pa
//! to within the transform's f32 round-trip tolerance, and that
//! sub-`p_eps` pressures are compressed instead of vanishing.

use leto::Array3;

use super::super::config::ParamFieldPINNConfig;
use super::super::forward::{infer_grid, GridQueryParams};
use super::super::network::ParamFieldPINNNetwork;
use super::super::target_transform::{OutputTransforms, TargetTransform};
use super::super::KernelCubeSampler;
use super::B;
use kwavers_core::constants::numerical::{MHZ_TO_HZ, MPA_TO_PA};
use kwavers_physics::field_surrogate::FocalKernel;

fn gaussian_kernel(f0: f64, pnp: f64) -> FocalKernel {
    let (nx, ny, nz) = (12usize, 8usize, 8usize);
    let focus = (nx / 2, ny / 2, nz / 2);
    let mut field = Array3::<f64>::zeros((nx, ny, nz));
    for ([i, j, k], v) in field.indexed_iter_mut().unwrap() {
        let dx = (i as f64 - focus.0 as f64) / (nx as f64 / 4.0);
        let dy = (j as f64 - focus.1 as f64) / (ny as f64 / 4.0);
        let dz = (k as f64 - focus.2 as f64) / (nz as f64 / 4.0);
        let r2 = dx * dx + dy * dy + dz * dz;
        *v = pnp * (-0.5 * r2).exp();
    }
    FocalKernel::new(field, 1.0e-3, focus, f0, pnp, MPA_TO_PA, 2.0e-3, 5.0e-3)
}

fn cube() -> Vec<FocalKernel> {
    vec![
        gaussian_kernel(0.5 * MHZ_TO_HZ, 15.0 * MPA_TO_PA),
        gaussian_kernel(0.5 * MHZ_TO_HZ, 30.0 * MPA_TO_PA),
        gaussian_kernel(MHZ_TO_HZ, 15.0 * MPA_TO_PA),
        gaussian_kernel(MHZ_TO_HZ, 30.0 * MPA_TO_PA),
    ]
}

#[test]
fn sampler_default_uses_linear_transform() {
    let sampler = KernelCubeSampler::new(&cube(), None);
    match sampler.output_transforms.p_max {
        TargetTransform::Linear { scale_pa } => {
            // Default scale equals the per-channel maximum (30 MPa here).
            assert!((scale_pa - 30.0 * MPA_TO_PA as f32).abs() < 1.0);
        }
        TargetTransform::SignedLog1p { .. } => {
            panic!("default sampler must use linear transform");
        }
    }
}

#[test]
fn sampler_with_signed_log1p_emits_targets_in_unit_interval() {
    let kernels = cube();
    let p_max_pa = 30.0 * MPA_TO_PA as f32;
    let transforms =
        OutputTransforms::signed_log1p(p_max_pa, p_max_pa, p_max_pa * 0.7, 1.0e-3).unwrap();
    let sampler =
        KernelCubeSampler::with_transforms(&kernels, None, transforms).expect("sampler build");

    let batch = sampler.batch::<B>(0, 512);
    let host: Vec<f32> = batch.targets.tensor.as_slice().to_vec();

    // All transformed targets must lie in [-1, 1] (the network's
    // output space) and the per-batch maximum magnitude must reach a
    // substantial fraction of 1 — otherwise the transform has not
    // engaged dynamic-range compression.
    let mut max_abs = 0.0_f32;
    for &t in &host {
        assert!(t.is_finite(), "non-finite target {t}");
        assert!(t.abs() <= 1.0 + 1e-5, "target out of [-1, 1]: {t}");
        max_abs = max_abs.max(t.abs());
    }
    assert!(
        max_abs > 0.9,
        "signed-log1p sampler did not produce near-peak targets: max_abs={max_abs}"
    );
}

#[test]
fn signed_log1p_lifts_sub_epsilon_targets_above_linear() {
    // At |p| = 1e-3 · p_max_pa the linear transform emits |t| = 1e-3;
    // the signed-log1p transform with eps_ratio = 1e-3 emits
    // |t| = ln(2)/t_max ≈ 0.1. This contrast is what closes the
    // focal-peak underprediction.
    let p_max_pa = 30.0 * MPA_TO_PA as f32;
    let lin = OutputTransforms::linear(p_max_pa, p_max_pa, p_max_pa * 0.7).unwrap();
    let log = OutputTransforms::signed_log1p(p_max_pa, p_max_pa, p_max_pa * 0.7, 1.0e-3).unwrap();
    let p_probe = p_max_pa * 1.0e-3;
    let n_lin = lin.p_max.forward(p_probe);
    let n_log = log.p_max.forward(p_probe);
    assert!(
        n_log > 50.0 * n_lin,
        "signed-log1p must lift sub-epsilon targets: linear={n_lin}, log={n_log}"
    );
}

#[test]
fn infer_grid_signed_log1p_round_trips_through_untrained_network() {
    // An untrained network produces arbitrary `[-1, 1]` outputs; the
    // round-trip we exercise here is the inverse-transform path
    // itself — given any network output in `[-1, 1]`, the inferred
    // physical Pa must be finite and lie within the transform's
    // calibrated range.
    let cfg = ParamFieldPINNConfig::default();
    let net = ParamFieldPINNNetwork::<B>::new(&cfg).unwrap();
    let p_max_pa = 30.0 * MPA_TO_PA as f32;
    let transforms =
        OutputTransforms::signed_log1p(p_max_pa, p_max_pa, p_max_pa * 0.7, 1.0e-3).unwrap();
    let params = GridQueryParams {
        shape: (6, 4, 4),
        focus_idx: (3, 2, 2),
        dx_m: 1.0e-3,
        f0: 0.75 * MHZ_TO_HZ,
        pnp: 22.5 * MPA_TO_PA,
        coord_half_m: (3.0e-3, 2.0e-3, 2.0e-3),
        f0_range: (0.5 * MHZ_TO_HZ, MHZ_TO_HZ),
        pnp_range: (15.0 * MPA_TO_PA, 30.0 * MPA_TO_PA),
        output_transforms: transforms,
        batch_size: 64,
    };
    let (pmin, pmax, prms) = infer_grid(&net, &params).unwrap();
    // `TargetTransform::inverse` clamps its input to `[-1, 1]` before
    // inverting, so |p| saturates at |p|_max when the (unbounded,
    // linear-output-layer) untrained network emits |t_norm| >= 1. At
    // that boundary `T⁻¹(±1)` is exactly `±|p|_max` in real arithmetic,
    // but f32 `ln`/`exp_m1` carries ~1 ULP relative round-trip error
    // (f32::EPSILON ≈ 1.19e-7); allow a 10x-margin relative tolerance
    // on top of the exact bound rather than an unattainable exact `<=`.
    let bound_pa = p_max_pa as f64 * (1.0 + 10.0 * f32::EPSILON as f64);
    for v in pmin.iter().chain(pmax.iter()).chain(prms.iter()) {
        assert!(v.is_finite(), "non-finite Pa from infer_grid: {v}");
        assert!(v.abs() <= bound_pa, "infer_grid Pa exceeds |p|_max: {v}");
    }
}
