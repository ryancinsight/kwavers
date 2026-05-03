//! Spatial displacement gradient computation via automatic differentiation.

#[cfg(feature = "pinn")]
use burn::tensor::{backend::AutodiffBackend, Tensor};

/// Compute spatial gradients of displacement using automatic differentiation.
///
/// Computes ∂u/∂x, ∂u/∂y, ∂v/∂x, ∂v/∂y via backpropagation through the
/// network with respect to input coordinates.
///
/// ## Arguments
///
/// * `u` - x-displacement \[N, 1\]
/// * `v` - y-displacement \[N, 1\]
/// * `x` - x-coordinates \[N, 1\] (must support autodiff)
/// * `y` - y-coordinates \[N, 1\] (must support autodiff)
///
/// ## Returns
///
/// `(∂u/∂x, ∂u/∂y, ∂v/∂x, ∂v/∂y)`
#[cfg(feature = "pinn")]
pub fn compute_displacement_gradients<B: AutodiffBackend>(
    u: Tensor<B, 2>,
    v: Tensor<B, 2>,
    x: Tensor<B, 2>,
    y: Tensor<B, 2>,
) -> (Tensor<B, 2>, Tensor<B, 2>, Tensor<B, 2>, Tensor<B, 2>) {
    let grads = u.clone().backward();
    let dudx = x
        .grad(&grads)
        .map(|g| Tensor::<B, 2>::from_data(g.into_data(), &Default::default()))
        .unwrap_or_else(|| x.zeros_like());

    let grads_u_y = u.backward();
    let dudy = y
        .grad(&grads_u_y)
        .map(|g| Tensor::<B, 2>::from_data(g.into_data(), &Default::default()))
        .unwrap_or_else(|| y.zeros_like());

    let grads_v_x = v.clone().backward();
    let dvdx = x
        .grad(&grads_v_x)
        .map(|g| Tensor::<B, 2>::from_data(g.into_data(), &Default::default()))
        .unwrap_or_else(|| x.zeros_like());

    let grads_v_y = v.backward();
    let dvdy = y
        .grad(&grads_v_y)
        .map(|g| Tensor::<B, 2>::from_data(g.into_data(), &Default::default()))
        .unwrap_or_else(|| y.zeros_like());

    (dudx, dudy, dvdx, dvdy)
}
