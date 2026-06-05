use super::super::super::config::NewtonKrylovConfig;
use super::super::super::residual_metric::norm;
use super::super::*;
use crate::integration::nonlinear::GMRESConfig;
use kwavers_core::error::{KwaversError, ValidationError};
use kwavers_field::UnifiedFieldType;
use ndarray::Array3;

/// Backtracking line search reuses the candidate-state workspace.
///
/// The `Density` block has no physics rate in the monolithic residual, so
/// `F(u) = u - u_prev`.  With `u_prev = 0` and `du = -u/2`, the first
/// candidate at `alpha = 1` gives `F(u + du) = u/2`, which satisfies the
/// sufficient-decrease test exactly.  The workspace must therefore contain the
/// half-state candidate and keep the same allocation across repeated calls.
#[test]
fn test_line_search_trial_workspace_reuses_buffer_and_refreshes_values() {
    let mut coupler = MonolithicCoupler::new(NewtonKrylovConfig::default(), GMRESConfig::default());
    let dims = (3, 3, 3);
    let field_order = vec![UnifiedFieldType::Density];
    let u_prev = Array3::zeros(dims);

    let u = Array3::from_elem(dims, 2.0);
    let du = Array3::from_elem(dims, -1.0);
    let f = coupler
        .compute_residual(&u, &u_prev, 1e-6, dims, &field_order)
        .unwrap();
    let alpha = coupler
        .line_search(&u, &du, &f, &u_prev, 1e-6, dims, &field_order)
        .unwrap();
    assert_eq!(alpha, 1.0);

    let trial = coupler.line_search_state_scratch.as_ref().unwrap();
    let first_ptr = trial.as_ptr();
    assert!(trial.iter().all(|&value| (value - 1.0).abs() < 1e-15));

    let u = Array3::from_elem(dims, 4.0);
    let du = Array3::from_elem(dims, -2.0);
    let f = coupler
        .compute_residual(&u, &u_prev, 1e-6, dims, &field_order)
        .unwrap();
    let alpha = coupler
        .line_search(&u, &du, &f, &u_prev, 1e-6, dims, &field_order)
        .unwrap();
    assert_eq!(alpha, 1.0);

    let trial = coupler.line_search_state_scratch.as_ref().unwrap();
    assert_eq!(trial.as_ptr(), first_ptr);
    assert!(trial.iter().all(|&value| (value - 2.0).abs() < 1e-15));
}

/// Adaptive line search uses the configured maximum trial step.
///
/// For the identity residual `F(u)=u-u_prev`, `u=2`, and `du=-2`, a configured
/// `alpha_max=0.5` gives the accepted candidate `u+alpha*du = 1`.  Returning
/// `1.0` here would prove the configuration field is dead solver state.
#[test]
fn test_line_search_uses_configured_initial_alpha() {
    let config = NewtonKrylovConfig {
        line_search_parameter: 0.5,
        ..NewtonKrylovConfig::default()
    };
    let mut coupler = MonolithicCoupler::new(config, GMRESConfig::default());
    let dims = (3, 3, 3);
    let field_order = vec![UnifiedFieldType::Density];
    let u_prev = Array3::zeros(dims);
    let u = Array3::from_elem(dims, 2.0);
    let du = Array3::from_elem(dims, -2.0);
    let f = coupler
        .compute_residual(&u, &u_prev, 1e-6, dims, &field_order)
        .unwrap();

    let alpha = coupler
        .line_search(&u, &du, &f, &u_prev, 1e-6, dims, &field_order)
        .unwrap();

    assert_eq!(alpha, 0.5);
    let trial = coupler.line_search_state_scratch.as_ref().unwrap();
    assert!(trial.iter().all(|&value| (value - 1.0).abs() < 1e-15));
}

/// Invalid line-search step bounds are rejected before residual evaluation.
#[test]
fn test_line_search_rejects_invalid_configured_alpha() {
    let config = NewtonKrylovConfig {
        line_search_parameter: 0.0,
        ..NewtonKrylovConfig::default()
    };
    let mut coupler = MonolithicCoupler::new(config, GMRESConfig::default());
    let dims = (3, 3, 3);
    let field_order = vec![UnifiedFieldType::Density];
    let u_prev = Array3::zeros(dims);
    let u = Array3::from_elem(dims, 2.0);
    let du = Array3::from_elem(dims, -1.0);
    let f = coupler
        .compute_residual(&u, &u_prev, 1e-6, dims, &field_order)
        .unwrap();

    let error = coupler
        .line_search(&u, &du, &f, &u_prev, 1e-6, dims, &field_order)
        .unwrap_err();

    match error {
        KwaversError::Validation(ValidationError::InvalidValue {
            parameter,
            value,
            reason,
        }) => {
            assert_eq!(parameter, "NewtonKrylovConfig::line_search_parameter");
            assert_eq!(value, 0.0);
            assert_eq!(reason, "must be finite and in (0, 1]");
        }
        other => panic!("expected line-search validation error, got {other:?}"),
    }
}

/// Failed backtracking returns the final evaluated step, not an untested step.
///
/// For the identity residual `F(u)=u-u_prev`, `u=1`, and `du=1`, every
/// positive trial step increases the residual norm.  With five backtracking
/// trials from `alpha_max=1`, the last evaluated candidate is `alpha=1/16`.
/// Returning `1/32` would apply a state never checked against the residual.
#[test]
fn test_line_search_fallback_returns_last_evaluated_alpha() {
    let mut coupler = MonolithicCoupler::new(NewtonKrylovConfig::default(), GMRESConfig::default());
    let dims = (3, 3, 3);
    let field_order = vec![UnifiedFieldType::Density];
    let u_prev = Array3::zeros(dims);
    let u = Array3::from_elem(dims, 1.0);
    let du = Array3::from_elem(dims, 1.0);
    let f = coupler
        .compute_residual(&u, &u_prev, 1e-6, dims, &field_order)
        .unwrap();

    let alpha = coupler
        .line_search(&u, &du, &f, &u_prev, 1e-6, dims, &field_order)
        .unwrap();

    assert_eq!(alpha, 1.0 / 16.0);
    let trial = coupler.line_search_state_scratch.as_ref().unwrap();
    assert!(trial
        .iter()
        .all(|&value| (value - (1.0 + alpha)).abs() < 1e-15));
}

/// Jacobian-vector products reuse the perturbed-state workspace.
///
/// The `Density` block has no rate term, so its residual derivative is the
/// identity operator.  The finite-difference JVP must therefore return the
/// direction vector, while the scratch buffer stores the most recent
/// `u + eps * v` candidate and keeps the same allocation across calls.
#[test]
fn test_jvp_state_workspace_reuses_buffer_and_returns_identity_derivative() {
    let mut coupler = MonolithicCoupler::new(NewtonKrylovConfig::default(), GMRESConfig::default());
    let dims = (3, 3, 3);
    let field_order = vec![UnifiedFieldType::Density];
    let u_prev = Array3::zeros(dims);
    let dt = 1e-6;

    let u = Array3::from_elem(dims, 2.0);
    let v = Array3::from_elem(dims, 0.25);
    let eps = 1e-8 * (1.0 + norm(&u));
    let jv = coupler
        .jacobian_vector_product(&v, &u, &u_prev, dt, dims, &field_order)
        .unwrap();
    assert!(jv.iter().all(|&value| (value - 0.25).abs() < 1e-8));

    let state = coupler.jvp_state_scratch.as_ref().unwrap();
    let first_ptr = state.as_ptr();
    let expected_state = 2.0 + eps * 0.25;
    assert!(state
        .iter()
        .all(|&value| (value - expected_state).abs() < 1e-15));

    let u = Array3::from_elem(dims, 3.0);
    let v = Array3::from_elem(dims, 0.5);
    let eps = 1e-8 * (1.0 + norm(&u));
    let jv = coupler
        .jacobian_vector_product(&v, &u, &u_prev, dt, dims, &field_order)
        .unwrap();
    assert!(jv.iter().all(|&value| (value - 0.5).abs() < 1e-8));

    let state = coupler.jvp_state_scratch.as_ref().unwrap();
    let expected_state = 3.0 + eps * 0.5;
    assert_eq!(state.as_ptr(), first_ptr);
    assert!(state
        .iter()
        .all(|&value| (value - expected_state).abs() < 1e-15));
}
