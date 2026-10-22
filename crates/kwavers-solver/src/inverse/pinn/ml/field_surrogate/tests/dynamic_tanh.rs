//! Dynamic Tanh activation tests.

use coeus_autograd::Var;
use kwavers_core::error::KwaversResult;

use crate::inverse::pinn::be;

use super::super::dynamic_tanh::DynamicTanh;
use super::B;

fn var_col(backend: B, values: &[f32], cols: usize) -> KwaversResult<Var<f32, B>> {
    be(Var::new(
        be(coeus_tensor::Tensor::from_slice_on(
            vec![values.len() / cols, cols],
            values,
            &backend,
        ))?,
        false,
    ))
}

#[test]
fn test_default_init_recovers_vanilla_tanh() -> KwaversResult<()> {
    // `DynamicTanh::new` initialises (alpha, gamma, beta) = (1, 1, 0), so at
    // construction the activation reduces to `tanh(x)`.
    let backend = B::default();
    let dyt = DynamicTanh::<B>::new()?;
    let xs: [f32; 5] = [-2.0, -0.5, 0.0, 0.5, 2.0];
    let input = var_col(backend, &xs, 1)?;
    let out = dyt.forward(&input)?;
    let out_vec = out.tensor.as_slice();
    for (i, &x) in xs.iter().enumerate() {
        let expected = x.tanh();
        let diff = (out_vec[i] - expected).abs();
        assert!(
            diff < 1e-5,
            "vanilla-tanh init mismatch at x={x}: dyt={}, tanh(x)={expected}",
            out_vec[i]
        );
    }
    Ok(())
}

#[test]
fn test_custom_init_scalars_round_trip() -> KwaversResult<()> {
    let dyt = DynamicTanh::<B>::with_init(0.7, 1.3, -0.2)?;
    let (alpha, gamma, beta) = dyt.scalars();
    assert!((alpha - 0.7).abs() < 1e-6);
    assert!((gamma - 1.3).abs() < 1e-6);
    assert!((beta + 0.2).abs() < 1e-6);
    Ok(())
}

#[test]
fn test_forward_applies_affine_then_tanh_then_affine() -> KwaversResult<()> {
    // With (alpha, gamma, beta) = (2, 3, -1) the output at x = 0 is exactly
    // beta = -1, and at x = 1 it is 3 * tanh(2) - 1.
    let backend = B::default();
    let dyt = DynamicTanh::<B>::with_init(2.0, 3.0, -1.0)?;
    let input = var_col(backend, &[0.0_f32, 1.0_f32], 1)?;
    let out_vec = dyt.forward(&input)?.tensor.as_slice().to_vec();
    assert!(
        (out_vec[0] + 1.0).abs() < 1e-5,
        "x=0 case: got {}, expected -1",
        out_vec[0]
    );
    let expected = 3.0 * 2.0_f32.tanh() - 1.0;
    assert!(
        (out_vec[1] - expected).abs() < 1e-4,
        "x=1 case: got {}, expected {expected}",
        out_vec[1]
    );
    Ok(())
}

#[test]
fn test_forward_shape_preserved() -> KwaversResult<()> {
    let backend = B::default();
    let dyt = DynamicTanh::<B>::new()?;
    let n = 17;
    let f = 11;
    let data: Vec<f32> = (0..n * f).map(|i| (i as f32).mul_add(0.01, -0.5)).collect();
    let input = var_col(backend, &data, f)?;
    let out = dyt.forward(&input)?;
    assert_eq!(out.tensor.shape(), &[n, f]);
    Ok(())
}

#[test]
fn test_autodiff_gradient_flow_through_dyt() -> KwaversResult<()> {
    // Gradients must reach both the input and the (alpha, gamma, beta)
    // parameters through the DyT expression; none may be NaN or infinite.
    let backend = B::default();
    let dyt = DynamicTanh::<B>::new()?;
    let input = be(Var::new(
        be(coeus_tensor::Tensor::from_slice_on(
            vec![3, 1],
            &[0.3_f32, 0.6, -0.4],
            &backend,
        ))?,
        true,
    ))?;

    for p in dyt.parameters() {
        be(p.zero_grad())?;
    }
    let out = dyt.forward(&input)?;
    let loss = be(coeus_autograd::mean(&be(coeus_autograd::mul(&out, &out))?))?;
    be(loss.backward())?;

    for p in dyt.parameters() {
        let grad = p.grad().ok_or_else(|| {
            kwavers_core::error::KwaversError::InvalidInput(
                "invariant: a DyT parameter on the forward path has a gradient".into(),
            )
        })?;
        assert!(grad.as_slice().iter().all(|g| g.is_finite()));
    }
    let input_grad = input.grad().ok_or_else(|| {
        kwavers_core::error::KwaversError::InvalidInput(
            "invariant: the DyT input requires grad and lies on the forward path".into(),
        )
    })?;
    assert!(input_grad.as_slice().iter().all(|g| g.is_finite()));
    Ok(())
}
