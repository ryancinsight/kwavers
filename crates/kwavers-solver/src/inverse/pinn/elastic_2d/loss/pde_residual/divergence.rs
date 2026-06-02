//! Stress divergence computation via automatic differentiation.

#[cfg(feature = "pinn")]
use burn::tensor::{backend::AutodiffBackend, Tensor};

/// Compute stress divergence ∇·σ using automatic differentiation.
///
/// Divergence of stress tensor in 2D:
/// - div_x = ∂σ_xx/∂x + ∂σ_xy/∂y
/// - div_y = ∂σ_xy/∂x + ∂σ_yy/∂y
///
/// Appears in the elastic wave equation: ρ ∂²u/∂t² = ∇·σ
#[cfg(feature = "pinn")]
pub fn compute_stress_divergence<B: AutodiffBackend>(
    sigma_xx: Tensor<B, 2>,
    sigma_xy: Tensor<B, 2>,
    sigma_yy: Tensor<B, 2>,
    x: Tensor<B, 2>,
    y: Tensor<B, 2>,
) -> (Tensor<B, 2>, Tensor<B, 2>) {
    let grads_sxx = sigma_xx.clone().backward();
    let dsxx_dx = x
        .grad(&grads_sxx)
        .map(|g| Tensor::<B, 2>::from_data(g.into_data(), &Default::default()))
        .unwrap_or_else(|| x.zeros_like());

    let grads_sxy = sigma_xy.clone().backward();
    let dsxy_dy = y
        .grad(&grads_sxy)
        .map(|g| Tensor::<B, 2>::from_data(g.into_data(), &Default::default()))
        .unwrap_or_else(|| y.zeros_like());
    let dsxy_dx = x
        .grad(&grads_sxy)
        .map(|g| Tensor::<B, 2>::from_data(g.into_data(), &Default::default()))
        .unwrap_or_else(|| x.zeros_like());

    let grads_syy = sigma_yy.backward();
    let dsyy_dy = y
        .grad(&grads_syy)
        .map(|g| Tensor::<B, 2>::from_data(g.into_data(), &Default::default()))
        .unwrap_or_else(|| y.zeros_like());

    let div_x = dsxx_dx + dsxy_dy;
    let div_y = dsxy_dx + dsyy_dy;

    (div_x, div_y)
}
