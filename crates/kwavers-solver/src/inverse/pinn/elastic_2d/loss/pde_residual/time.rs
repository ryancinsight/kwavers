//! Time derivative computation (velocity and acceleration) via automatic differentiation.

#[cfg(feature = "pinn")]
use burn::tensor::{backend::AutodiffBackend, Tensor};

/// Compute time derivatives ∂u/∂t and ∂²u/∂t² using automatic differentiation.
///
/// - Velocity: v = ∂u/∂t  (first autodiff pass)
/// - Acceleration: a = ∂²u/∂t² = ∂v/∂t  (second autodiff pass)
///
/// ## Arguments
///
/// * `u` - Displacement \[N, 2\]
/// * `t` - Time coordinates \[N, 1\] (must support autodiff)
///
/// ## Returns
///
/// `(velocity, acceleration)` each \[N, 2\]
#[cfg(feature = "pinn")]
pub fn compute_time_derivatives<B: AutodiffBackend>(
    u: Tensor<B, 2>,
    t: Tensor<B, 2>,
) -> (Tensor<B, 2>, Tensor<B, 2>) {
    let grads_u = u.clone().backward();
    let velocity = t
        .grad(&grads_u)
        .map(|g| Tensor::<B, 2>::from_data(g.into_data(), &Default::default()))
        .unwrap_or_else(|| u.zeros_like());

    let grads_velocity = velocity.clone().backward();
    let acceleration = t
        .grad(&grads_velocity)
        .map(|g| Tensor::<B, 2>::from_data(g.into_data(), &Default::default()))
        .unwrap_or_else(|| velocity.zeros_like());

    (velocity, acceleration)
}
