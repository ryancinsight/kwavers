//! Matrix-free finite-difference Hessian action.

use super::objective::Objective;
use super::vector::max_abs;
use kwavers_core::error::KwaversResult;
use leto::Array3;

#[cfg(test)]
mod tests;

/// Matrix-free Hessian action `H v`, by central-free forward difference of the
/// exact adjoint gradient:
///
/// ```text
/// H v ≈ [g(m + ε v) − g(m)] / ε,   ε = fd_epsilon · reference_slowness / max|v|
/// ```
///
/// Scaling `ε` by `max|v|` keeps the largest slowness perturbation at
/// `fd_epsilon · reference_slowness` regardless of how `v` is normalized, so the
/// difference stays in the regime where the linearization dominates round-off.
/// No Jacobian is assembled, so this works for any
/// [`super::super::operator::HelmholtzForwardOperator`] (single-scatter Born or CBS).
///
/// Both consumers use this one implementation: [`super::super::gauss_newton`] for the
/// Newton-CG inner solve, and [`super::super::inversion`] for the curvature term of its
/// line search.
/// `gradient0` is the current gradient from `objective.evaluate(slowness)`;
/// each consumer retains that model/gradient pair until accepting a new pair.
///
/// # Errors
/// Propagates forward/adjoint evaluation errors from [`Objective::evaluate`].
pub(in super::super) fn hessian_vector(
    slowness: &Array3<f64>,
    gradient0: &Array3<f64>,
    direction: &Array3<f64>,
    objective: &Objective<'_>,
    reference_slowness: f64,
    fd_epsilon: f64,
) -> KwaversResult<Array3<f64>> {
    let scale = max_abs(direction);
    if scale <= f64::EPSILON {
        let shape = slowness.shape();
        return Ok(Array3::zeros([shape[0], shape[1], shape[2]]));
    }
    // Keep the largest slowness perturbation at fd_epsilon · reference_slowness.
    let eps = fd_epsilon * reference_slowness / scale;
    let mut perturbed = slowness.clone();
    for (m, &v) in perturbed.iter_mut().zip(direction.iter()) {
        *m += eps * v;
    }
    super::super::inversion::clamp_slowness(&mut perturbed, objective.config);
    // If projection leaves a valid model unchanged, repeatable g(m) gives
    // (g(m) - g(m)) / eps = +0 for finite g(m) and finite positive eps.
    // Compare every bit; retain evaluation for invalid/nonfinite inputs and
    // operators whose elementary functions permit varying precision.
    if objective.has_repeatable_gradient()
        && eps.is_finite()
        && eps > 0.0
        && gradient0.iter().all(|value| value.is_finite())
        && perturbed
            .iter()
            .zip(slowness.iter())
            .all(|(&trial, &base)| {
                base.is_finite() && base > 0.0 && trial.to_bits() == base.to_bits()
            })
    {
        return Ok(Array3::zeros(slowness.shape()));
    }
    let (_objective, gradient1) = objective.evaluate(&perturbed)?;
    let mut hv = gradient1;
    for (h, &g0) in hv.iter_mut().zip(gradient0.iter()) {
        *h = (*h - g0) / eps;
    }
    Ok(hv)
}
