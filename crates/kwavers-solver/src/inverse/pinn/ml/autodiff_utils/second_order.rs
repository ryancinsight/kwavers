//! Second-order spatial autodiff utilities: ∂²u/∂xᵢ², ∇²u, and ∇(∇·u).
//!
//! # References
//! - Raissi et al. (2019): "Physics-informed neural networks". *J. Comput. Phys.*, 378, 686–707.
//!
//! ## Weight-gradient contract
//!
//! Every function here returns a **`Var`**, not a plain `Tensor`: each value
//! is built purely from finite-difference arithmetic on top of independent
//! forward passes (`forward_fn(&Var::new(perturbed_input, false))`), never
//! through a `Var::grad()` extraction. `Var::grad()` reads a plain
//! accumulator buffer (`Var`'s `grad: Option<Arc<GradBuffer<T,B>>>` field
//! has no `BackwardNode`/`.creator`), so any value built from it is
//! permanently detached from the network's weight graph — a subsequent
//! `.backward()` on a loss containing that value contributes exactly zero
//! gradient to the weights, silently making a "physics loss" term
//! training-inert. Pure multi-forward-pass FD combination avoids this
//! entirely: each perturbed forward pass shares the *same* weight `Var`
//! leaves (`self.layer.weight`, etc.), so ordinary FD arithmetic (`add`,
//! `sub`, `scalar_mul`) on their outputs preserves a live `.creator` chain
//! back to those weights, verified empirically (coeus-nn diagnostic,
//! since removed): an FD combination of forward passes yields a nonzero
//! weight gradient after `backward()`, while combining a `.grad()`-derived
//! value the same way yields exactly zero.

use coeus_autograd::Var;

/// 2D gradient component pair `(∂/∂x, ∂/∂y)`, each `[batch, 1]`.
type GradientPair2D<B> = (Var<f32, B>, Var<f32, B>);

/// Compute second-order spatial derivative ∂²u/∂xᵢ² via central finite differences.
///
/// # Arguments
/// - `forward_fn`: Forward pass function.
/// - `input`: Input tensor `[batch, 3]`.
/// - `output_component`: Output component (0 for u_x, 1 for u_y).
/// - `spatial_dim`: Spatial dimension to differentiate twice (1 for x, 2 for y).
///
/// # Returns
/// `Var` `[batch, 1]` containing the second derivative, still connected to
/// the network's weight graph (see module-level weight-gradient contract).
///
/// # Mathematical Note
/// Central finite-difference approximation (ε = 1e-4):
/// ```text
/// ∂²u/∂xᵢ² ≈ (u(xᵢ+ε) − 2u(xᵢ) + u(xᵢ−ε)) / ε²
/// ```
/// Truncation error O(ε²).
/// # Errors
/// - Returns [`Err`] if an internal constraint is violated.
pub fn compute_second_derivative_2d<B, F>(
    forward_fn: F,
    input: &coeus_tensor::Tensor<f32, B>,
    output_component: usize,
    spatial_dim: usize,
) -> Result<Var<f32, B>, kwavers_core::error::KwaversError>
where
    B: coeus_ops::BackendOps<f32> + coeus_ops::CpuBackend + Default,
    B::DeviceBuffer<f32>: coeus_core::CpuAddressableStorage<f32>,
    F: Fn(&Var<f32, B>) -> Var<f32, B>,
{
    if !(1..=2).contains(&spatial_dim) {
        return Err(kwavers_core::error::KwaversError::InvalidInput(format!(
            "spatial_dim must be 1 (x) or 2 (y), got {}",
            spatial_dim
        )));
    }

    let eps = 1e-4_f32;
    let batch = input.shape()[0];
    let backend = B::default();

    let raw = input.as_slice();
    let mut plus = raw.to_vec();
    let mut minus = raw.to_vec();
    for row in 0..batch {
        plus[row * 3 + spatial_dim] += eps;
        minus[row * 3 + spatial_dim] -= eps;
    }
    let input_plus = coeus_tensor::Tensor::from_slice_on(vec![batch, 3], &plus, &backend);
    let input_minus = coeus_tensor::Tensor::from_slice_on(vec![batch, 3], &minus, &backend);

    let output = forward_fn(&Var::new(input.clone(), false));
    let u = coeus_autograd::slice(&output, &[(0, batch), (output_component, output_component + 1)]);

    let output_plus = forward_fn(&Var::new(input_plus, false));
    let u_plus = coeus_autograd::slice(
        &output_plus,
        &[(0, batch), (output_component, output_component + 1)],
    );

    let output_minus = forward_fn(&Var::new(input_minus, false));
    let u_minus = coeus_autograd::slice(
        &output_minus,
        &[(0, batch), (output_component, output_component + 1)],
    );

    let two_u = coeus_autograd::scalar_mul(&u, 2.0);
    let d2u = coeus_autograd::scalar_mul(
        &coeus_autograd::sub(&coeus_autograd::add(&u_plus, &u_minus), &two_u),
        1.0 / (eps * eps),
    );
    Ok(d2u)
}

/// Compute scalar Laplacian ∇²u = ∂²u/∂x² + ∂²u/∂y².
///
/// # Arguments
/// - `forward_fn`: Forward pass function.
/// - `input`: Input tensor `[batch, 3]`.
/// - `output_component`: Output component.
///
/// # Returns
/// `Var` `[batch, 1]` containing the Laplacian.
///
/// # Mathematical Note
/// ```text
/// ∇²u = ∂²u/∂x² + ∂²u/∂y²
/// ```
/// # Errors
/// - Propagates any [`KwaversError`] returned by called functions.
pub fn compute_laplacian_2d<B, F>(
    forward_fn: F,
    input: &coeus_tensor::Tensor<f32, B>,
    output_component: usize,
) -> Result<Var<f32, B>, kwavers_core::error::KwaversError>
where
    B: coeus_ops::BackendOps<f32> + coeus_ops::CpuBackend + Default,
    B::DeviceBuffer<f32>: coeus_core::CpuAddressableStorage<f32>,
    F: Fn(&Var<f32, B>) -> Var<f32, B> + Clone,
{
    let d2u_dx2 = compute_second_derivative_2d(forward_fn.clone(), input, output_component, 1)?;
    let d2u_dy2 = compute_second_derivative_2d(forward_fn, input, output_component, 2)?;
    Ok(coeus_autograd::add(&d2u_dx2, &d2u_dy2))
}

/// Compute gradient of divergence ∇(∇·u) = [∂(∇·u)/∂x, ∂(∇·u)/∂y] for the P-wave term.
///
/// # Arguments
/// - `forward_fn`: Forward pass function.
/// - `input`: Input tensor `[batch, 3]` with columns `[t, x, y]`; `output_component`
///   0 is `u_x`, 1 is `u_y`.
///
/// # Returns
/// Tuple `(∂(∇·u)/∂x, ∂(∇·u)/∂y)`.
///
/// # Mathematical Specification
/// Elastic wave P-wave term (Achenbach 1973), expanded directly into raw
/// second partials of the displacement components (never routed through
/// `compute_divergence_2d`'s `Var::grad()` extraction — see module-level
/// weight-gradient contract):
/// ```text
/// ∇·u = ∂u_x/∂x + ∂u_y/∂y
/// ∂(∇·u)/∂x = ∂²u_x/∂x² + ∂²u_y/∂x∂y
/// ∂(∇·u)/∂y = ∂²u_x/∂x∂y + ∂²u_y/∂y²
/// ```
/// The mixed partial ∂²uᵢ/∂x∂y uses the standard 4-point central stencil:
/// ```text
/// ∂²f/∂x∂y ≈ [f(x+ε,y+ε) − f(x+ε,y−ε) − f(x−ε,y+ε) + f(x−ε,y−ε)] / (4ε²)
/// ```
/// Truncation error O(ε²), matching [`compute_second_derivative_2d`].
/// # Errors
/// - Propagates any [`KwaversError`] returned by called functions.
pub fn compute_gradient_of_divergence_2d<B, F>(
    forward_fn: F,
    input: &coeus_tensor::Tensor<f32, B>,
) -> Result<GradientPair2D<B>, kwavers_core::error::KwaversError>
where
    B: coeus_ops::BackendOps<f32> + coeus_ops::CpuBackend + Default,
    B::DeviceBuffer<f32>: coeus_core::CpuAddressableStorage<f32>,
    F: Fn(&Var<f32, B>) -> Var<f32, B> + Clone,
{
    let d2ux_dx2 = compute_second_derivative_2d(forward_fn.clone(), input, 0, 1)?;
    let d2uy_dy2 = compute_second_derivative_2d(forward_fn.clone(), input, 1, 2)?;
    let d2ux_dxdy = compute_mixed_partial_2d(forward_fn.clone(), input, 0)?;
    let d2uy_dxdy = compute_mixed_partial_2d(forward_fn, input, 1)?;

    let ddiv_dx = coeus_autograd::add(&d2ux_dx2, &d2uy_dxdy);
    let ddiv_dy = coeus_autograd::add(&d2ux_dxdy, &d2uy_dy2);

    Ok((ddiv_dx, ddiv_dy))
}

/// Compute mixed partial ∂²u/∂x∂y for output component `output_component`
/// via the standard 4-point central finite-difference stencil, entirely on
/// forward-pass outputs (weight-gradient-preserving, see module docs).
fn compute_mixed_partial_2d<B, F>(
    forward_fn: F,
    input: &coeus_tensor::Tensor<f32, B>,
    output_component: usize,
) -> Result<Var<f32, B>, kwavers_core::error::KwaversError>
where
    B: coeus_ops::BackendOps<f32> + coeus_ops::CpuBackend + Default,
    B::DeviceBuffer<f32>: coeus_core::CpuAddressableStorage<f32>,
    F: Fn(&Var<f32, B>) -> Var<f32, B>,
{
    let eps = 1e-4_f32;
    let batch = input.shape()[0];
    let backend = B::default();
    let raw = input.as_slice();

    let perturb = |dx: f32, dy: f32| {
        let mut v = raw.to_vec();
        for row in 0..batch {
            v[row * 3 + 1] += dx;
            v[row * 3 + 2] += dy;
        }
        coeus_tensor::Tensor::from_slice_on(vec![batch, 3], &v, &backend)
    };
    let component_of = |output: &Var<f32, B>| {
        coeus_autograd::slice(output, &[(0, batch), (output_component, output_component + 1)])
    };

    let u_pp = component_of(&forward_fn(&Var::new(perturb(eps, eps), false)));
    let u_pm = component_of(&forward_fn(&Var::new(perturb(eps, -eps), false)));
    let u_mp = component_of(&forward_fn(&Var::new(perturb(-eps, eps), false)));
    let u_mm = component_of(&forward_fn(&Var::new(perturb(-eps, -eps), false)));

    let sum_diag = coeus_autograd::add(&u_pp, &u_mm);
    let sum_anti = coeus_autograd::add(&u_pm, &u_mp);
    let numerator = coeus_autograd::sub(&sum_diag, &sum_anti);
    Ok(coeus_autograd::scalar_mul(&numerator, 1.0 / (4.0 * eps * eps)))
}
