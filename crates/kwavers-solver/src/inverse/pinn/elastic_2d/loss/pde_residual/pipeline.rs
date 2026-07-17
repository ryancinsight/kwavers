//! Elastic-wave PDE residual for `ElasticPINN2D`.
//!
//! Thin adapter over `autodiff_utils::elastic`:
//! `ElasticPINN2D::forward` takes three separate leaf `Var`s `(x, y, t)`,
//! while the shared autodiff utility expects a single `forward_fn(&Var) ->
//! Var` closure over a combined `[batch, 3]` input (columns `[t, x, y]`, the
//! convention used throughout `autodiff_utils`). This module only performs
//! that column-slicing adaptation; the physics and finite-difference
//! numerics live entirely in `autodiff_utils::elastic`.

use coeus_autograd::Var;

use crate::inverse::pinn::elastic_2d::model::ElasticPINN2D;
use crate::inverse::pinn::ml::autodiff_utils::compute_elastic_wave_residual_2d;

/// Elastic-wave PDE residual components `(residual_x, residual_y)`, each `[N, 1]`.
type ElasticResidualPair<B> = (Var<f32, B>, Var<f32, B>);

/// Compute elastic wave equation PDE residual: R = ρ ∂²u/∂t² − ∇·σ
///
/// Where σ is computed from displacement via:
/// - strain: ε = ∇_s u (symmetric gradient)
/// - stress: σ = λ tr(ε) I + 2μ ε (Hooke's law)
///
/// ## Arguments
///
/// * `model` - the network; `forward_fn` re-evaluates it at several
///   perturbed collocation points for the finite-difference second-order
///   terms, so gradient contributions correctly flow back to `model`'s
///   weights (see `autodiff_utils::second_order`'s module-level
///   weight-gradient contract).
/// * `x`, `y`, `t` - Coordinates `[N, 1]` (leaf `Var`s; values only — the
///   combined tensor is rebuilt internally for the FD perturbations)
/// * `rho` - Density (kg/m³)
/// * `lambda` - First Lamé parameter (Pa)
/// * `mu` - Shear modulus (Pa)
///
/// ## Returns
///
/// `(residual_x, residual_y)` — PDE residuals per component `[N, 1]`, live
/// `Var`s safe to square, mean, and `.backward()` as a trained loss term.
/// # Errors
/// - Propagates any [`kwavers_core::error::KwaversError`] returned by called functions.
pub fn compute_elastic_wave_pde_residual<B>(
    model: &ElasticPINN2D<B>,
    x: &Var<f32, B>,
    y: &Var<f32, B>,
    t: &Var<f32, B>,
    rho: f64,
    lambda: f64,
    mu: f64,
) -> Result<ElasticResidualPair<B>, kwavers_core::error::KwaversError>
where
    B: coeus_ops::BackendOps<f32> + coeus_ops::CpuBackend + Default,
    B::DeviceBuffer<f32>:
        coeus_core::CpuAddressableStorage<f32> + coeus_core::CpuAddressableStorageMut<f32>,
{
    let batch = x.tensor.shape()[0];
    let backend = B::default();

    let mut combined = vec![0.0_f32; batch * 3];
    let t_slice = t.tensor.as_slice();
    let x_slice = x.tensor.as_slice();
    let y_slice = y.tensor.as_slice();
    for row in 0..batch {
        combined[row * 3] = t_slice[row];
        combined[row * 3 + 1] = x_slice[row];
        combined[row * 3 + 2] = y_slice[row];
    }
    let input = coeus_tensor::Tensor::from_slice_on(vec![batch, 3], &combined, &backend);

    let forward_fn = |combined_input: &Var<f32, B>| -> Var<f32, B> {
        let n = combined_input.tensor.shape()[0];
        let t = coeus_autograd::slice(combined_input, &[(0, n), (0, 1)]);
        let x = coeus_autograd::slice(combined_input, &[(0, n), (1, 2)]);
        let y = coeus_autograd::slice(combined_input, &[(0, n), (2, 3)]);
        model.forward(&x, &y, &t)
    };

    compute_elastic_wave_residual_2d(forward_fn, &input, rho, lambda, mu)
}
