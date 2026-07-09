//! Matrix-free Gauss-Newton (truncated Newton-CG) inversion.
//!
//! The nonlinear-conjugate-gradient loop in [`super::inversion`] scales the
//! steepest-descent direction by a fixed slowness step and backtracks; when the
//! model is already close to the truth the gradient is small, the trial steps
//! fall below the objective's numerical-decrease threshold, and no step is
//! accepted (a *differential* monitor starting from a known background recovers
//! nothing). A Newton step solves the normal equations `H p = -g` and lands a
//! correctly-sized step in one shot, independent of the gradient magnitude.
//!
//! This is matrix-free: the Gauss-Newton/Hessian action `H v` is obtained by a
//! finite difference of the exact adjoint gradient,
//! `H v ≈ [g(m + ε v) − g(m)] / ε`, so it works for **any**
//! [`super::operator::HelmholtzForwardOperator`] (single-scatter Born or CBS)
//! without assembling a Jacobian. The inner solve is Steihaug-truncated conjugate
//! gradients with Levenberg-Marquardt damping; the outer step uses backtracking
//! from the full Newton step.
//!
//! References: Nocedal & Wright (2006) *Numerical Optimization* §7.1 (Newton-CG,
//! Steihaug); Métivier et al. (2013) truncated-Newton FWI.

use super::gradient::{dot, max_abs, objective_and_gradient};
use super::inversion::clamp_slowness;
use super::types::{
    Config, FrequencyObservation, InversionResult, FREQUENCY_DOMAIN_FWI_SOLVER_MODEL,
};
use kwavers_core::error::KwaversResult;
use kwavers_physics::acoustics::imaging::modalities::ultrasound::frequency_domain_fwi::{
    slowness_to_sound_speed, sound_speed_to_slowness, MultiRowRingArray,
};
use leto::Array3;

/// Gauss-Newton / Newton-CG tuning.
#[derive(Clone, Copy, Debug)]
pub struct GaussNewtonConfig {
    /// Inner conjugate-gradient iterations per Newton step.
    pub cg_iterations: usize,
    /// Initial Levenberg-Marquardt damping `λ` in `(H + λI) p = -g`.
    pub lm_damping: f64,
    /// Relative slowness perturbation for the finite-difference Hessian action,
    /// as a fraction of the reference slowness.
    pub fd_epsilon: f64,
    /// Maximum LM damping increases per outer Newton step before giving up.
    pub max_lm_tries: usize,
}

impl Default for GaussNewtonConfig {
    fn default() -> Self {
        Self {
            cg_iterations: 8,
            lm_damping: 1.0e-3,
            fd_epsilon: 1.0e-3,
            max_lm_tries: 12,
        }
    }
}

/// LM damping increase factor when a trial step fails to reduce the objective.
const LM_INCREASE: f64 = 4.0;
/// LM damping decrease factor after a successful step (toward Gauss-Newton).
const LM_DECREASE: f64 = 0.5;
/// Lower bound on LM damping (keeps the operator nonsingular).
const LM_MIN: f64 = 1.0e-12;

/// Gauss-Newton inversion. Same contract as [`super::inversion::invert`] but with
/// Newton-CG steps that engage near-truth residuals.
///
/// `config.iterations` bounds the outer Newton steps.
///
/// # Errors
/// Propagates forward/adjoint evaluation errors from [`objective_and_gradient`].
pub fn invert_gauss_newton(
    observations: &[FrequencyObservation],
    array: &MultiRowRingArray,
    initial_sound_speed_m_s: &Array3<f64>,
    config: &Config,
    gn: &GaussNewtonConfig,
) -> KwaversResult<InversionResult> {
    let mut slowness = sound_speed_to_slowness(initial_sound_speed_m_s)?;
    let (mut objective, mut gradient) =
        objective_and_gradient(&slowness, observations, array, config)?;
    let mut history = vec![objective];
    let reference_slowness = 1.0 / config.reference_sound_speed_m_s;
    let mut lambda = gn.lm_damping.max(LM_MIN);

    for _outer in 0..config.iterations {
        if max_abs(&gradient) <= f64::EPSILON {
            break;
        }

        // Levenberg-Marquardt: solve (H + λI) p = -g, increasing λ until the full
        // step reduces the objective. Large λ → small, well-scaled steepest-descent
        // step (engages near-truth residuals); small λ → Gauss-Newton step (fast
        // far from truth). This adapts the step scale without a separate line
        // search, fixing the negative-curvature / tiny-gradient stall.
        let mut accepted = None;
        for _ in 0..gn.max_lm_tries {
            let step = newton_cg(
                &slowness,
                &gradient,
                observations,
                array,
                config,
                gn,
                reference_slowness,
                lambda,
            )?;
            if max_abs(&step) <= f64::EPSILON {
                lambda *= LM_INCREASE;
                continue;
            }
            let mut candidate = slowness.clone();
            for (s, &p) in candidate.iter_mut().zip(step.iter()) {
                *s += p;
            }
            clamp_slowness(&mut candidate, config);
            let (candidate_objective, candidate_gradient) =
                objective_and_gradient(&candidate, observations, array, config)?;
            if candidate_objective < objective {
                accepted = Some((candidate, candidate_objective, candidate_gradient));
                lambda = (lambda * LM_DECREASE).max(LM_MIN);
                break;
            }
            lambda *= LM_INCREASE;
        }

        let Some((candidate, candidate_objective, candidate_gradient)) = accepted else {
            break;
        };
        slowness = candidate;
        objective = candidate_objective;
        gradient = candidate_gradient;
        history.push(objective);
    }

    Ok(InversionResult {
        sound_speed_m_s: slowness_to_sound_speed(&slowness)?,
        objective_history: history,
        frequencies_used: (observations.shape()[0] * observations.shape()[1] * observations.shape()[2]),
        transmissions_used: observations
            .first()
            .map(|obs| obs.observed_pressure.shape()[0])
            .unwrap_or(0),
        receivers_used: array.element_count(),
        model_family: FREQUENCY_DOMAIN_FWI_SOLVER_MODEL,
    })
}

/// Gauss-Newton/Hessian action `H v ≈ [g(m + ε v) − g(m)] / ε` with a relative
/// finite-difference step (no Jacobian assembly).
// allow(too_many_arguments): each parameter is a distinct mathematical input to the
// matrix-free Hessian action (current model, base gradient, probe direction, the
// forward-problem triple, FD step). Bundling them into a struct would hide the
// finite-difference formula rather than clarify it.
#[allow(clippy::too_many_arguments)]
fn hessian_vector(
    slowness: &Array3<f64>,
    gradient0: &Array3<f64>,
    direction: &Array3<f64>,
    observations: &[FrequencyObservation],
    array: &MultiRowRingArray,
    config: &Config,
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
    clamp_slowness(&mut perturbed, config);
    let (_objective, gradient1) = objective_and_gradient(&perturbed, observations, array, config)?;
    let mut hv = gradient1;
    for (h, &g0) in hv.iter_mut().zip(gradient0.iter()) {
        *h = (*h - g0) / eps;
    }
    Ok(hv)
}

/// Steihaug-truncated conjugate gradients solving `(H + λI) p = -g`.
///
/// On negative curvature it truncates: the CG iterate so far, or **zeros** on the
/// first iteration to signal the caller to increase `λ` (rather than returning an
/// unscaled steepest-descent step that the line search would reject near truth).
// allow(too_many_arguments): distinct Newton-CG inputs (model, gradient, the
// forward-problem triple, GN/regularisation parameters) — see hessian_vector.
#[allow(clippy::too_many_arguments)]
fn newton_cg(
    slowness: &Array3<f64>,
    gradient: &Array3<f64>,
    observations: &[FrequencyObservation],
    array: &MultiRowRingArray,
    config: &Config,
    gn: &GaussNewtonConfig,
    reference_slowness: f64,
    lambda: f64,
) -> KwaversResult<Array3<f64>> {
    let shape = slowness.shape();
    let mut p = Array3::<f64>::zeros([shape[0], shape[1], shape[2]]);
    // Residual r = -g - (H+λI)p, with p = 0 → r = -g.
    let mut r = gradient.mapv(|g| -g);
    let mut direction = r.clone();
    let mut rs_old = dot(&r, &r);
    if rs_old.sqrt() <= f64::EPSILON {
        return Ok(p);
    }
    let rs_initial = rs_old;

    for _ in 0..gn.cg_iterations {
        let mut hd = hessian_vector(
            slowness,
            gradient,
            &direction,
            observations,
            array,
            config,
            reference_slowness,
            gn.fd_epsilon,
        )?;
        if lambda > 0.0 {
            for (h, &d) in hd.iter_mut().zip(direction.iter()) {
                *h += lambda * d;
            }
        }
        let d_hd = dot(&direction, &hd);
        if d_hd <= 0.0 {
            // Negative curvature: keep the CG iterate; on the first iteration
            // return zeros so the LM loop increases λ.
            return Ok(p);
        }
        let alpha = rs_old / d_hd;
        for (pv, &d) in p.iter_mut().zip(direction.iter()) {
            *pv += alpha * d;
        }
        for (rv, &h) in r.iter_mut().zip(hd.iter()) {
            *rv -= alpha * h;
        }
        let rs_new = dot(&r, &r);
        if rs_new <= 1.0e-12 * rs_initial {
            break;
        }
        let beta = rs_new / rs_old;
        for (dv, &rv) in direction.iter_mut().zip(r.iter()) {
            *dv = rv + beta * *dv;
        }
        rs_old = rs_new;
    }
    Ok(p)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::inverse::fwi::frequency_domain::{simulate_frequency_observation, Config};
    use leto::Array3;

    fn ring(n_elem: usize, diameter: f64) -> MultiRowRingArray {
        MultiRowRingArray::new(n_elem, 1, diameter, 0.0).unwrap()
    }

    /// From the EXACT background (where NLCG accepts no step), Gauss-Newton must
    /// reduce the objective and recover a positive Δc at the inclusion — the
    /// near-truth engagement the monitor needs.
    #[test]
    fn gauss_newton_engages_near_truth_where_nlcg_stalls() {
        let n = 10;
        let centre = n / 2;
        let spacing = 1.0e-3;
        let array = ring(16, 0.024);
        let config = Config {
            reference_sound_speed_m_s: 1500.0,
            spacing_m: spacing,
            iterations: 6,
            min_sound_speed_m_s: 1400.0,
            max_sound_speed_m_s: 1700.0,
            estimate_source_scaling: false,
            ..Config::default()
        };

        // Background homogeneous; perturbed has a +60 m/s inclusion at the centre.
        let background = Array3::from_elem([n, n, 1], 1500.0);
        let mut perturbed = background.clone();
        for i in centre - 1..=centre + 1 {
            for j in centre - 1..=centre + 1 {
                perturbed[[i, j, 0]] = 1560.0;
            }
        }
        let observations: Vec<FrequencyObservation> = [3.0e5, 5.0e5]
            .iter()
            .map(|&f| {
                FrequencyObservation::new(
                    f,
                    simulate_frequency_observation(&perturbed, &array, f, &config).unwrap(),
                )
            })
            .collect();

        // Objective at the exact background start.
        let start_slowness = sound_speed_to_slowness(&background).unwrap();
        let (obj_start, _) =
            objective_and_gradient(&start_slowness, &observations, &array, &config).unwrap();

        let gn = GaussNewtonConfig::default();
        let result = invert_gauss_newton(&observations, &array, &background, &config, &gn).unwrap();

        let obj_end = *result.objective_history.last().unwrap();
        eprintln!(
            "GN objective {obj_start:.4e} -> {obj_end:.4e} ({} steps); centre Δc {:+.2}",
            (result.objective_history.shape()[0] * result.objective_history.shape()[1] * result.objective_history.shape()[2]),
            result.sound_speed_m_s[[centre, centre, 0]] - 1500.0
        );
        assert!(
            obj_end < obj_start,
            "Gauss-Newton must reduce the objective from the exact background: {obj_start} -> {obj_end}"
        );
        let centre_dc = result.sound_speed_m_s[[centre, centre, 0]] - 1500.0;
        assert!(
            centre_dc > 0.0,
            "Gauss-Newton must recover a positive Δc at the inclusion, got {centre_dc}"
        );
    }
}
